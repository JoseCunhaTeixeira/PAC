"""The quality of a run's dispersion images and picks: sigpipe's measures of each window's image
(sigpipe.masw.quality.image: the coherent band, peaks on the grid's edges, competing ridges) and
what the picked curve's file gives (its points, band, wavelengths and uncertainties); for a run
the assistant made, its image and curve checks (G2 on the image, G3 on the M0 pick, G4 along the
line), with the attempts of the phase shift and of the picking, and who picked each window: the
assistant automatically, or a person by hand in PAC. Shown along the line at each window's
middle: the dispersion strip of Visualization, and the selected window's card."""

from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from masw.io.pick_origin import Origin, pick_origin
from masw.io.quality.files import folder_path, fundamental, wavelengths
from masw.io.quality.log import (
    Attempt,
    AttemptSummary,
    GateResult,
    Metric,
    QCLog,
    config_section,
    read_log,
)
from masw.io.quality.view import (
    Card,
    Cell,
    GateView,
    Overview,
    Part,
    PartLegend,
    PartState,
    Sentence,
    Setting,
    Status,
    Track,
    flag_text,
    gate_view,
    measured_status,
    number,
    plural,
    span,
    verdict_sentence,
    verdict_status,
    warnings,
    worst,
)
from sigpipe.algorithms.picking.dispersion.curve import (
    max_resolvable_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import load_curves
from sigpipe.masw.quality.image import aliased, coherent_columns, competing_ridges, edge_peaks
from sigpipe.masw.runs import load_image, window_folders, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE

# The units the attempts' overrides are said in.
UNITS = {"fmin": "Hz", "fmax": "Hz", "vmin": "m/s", "vmax": "m/s", "wavelength_step": "m"}


class ImageThresholds(BaseModel):
    """G2's limits: the assistant's defaults, or those the run's checks used."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    coherent_level: float = 0.3  # of the way from the noise floor to 1
    min_coherent_columns: float = 0.5
    edge_share: float = 0.02
    max_edge_columns: float = 0.2
    competing_ratio: float = 0.7
    competing_separation: float = 0.15
    max_competing_columns: float = 0.7
    vmin_floor: float = 30.0


class PickThresholds(BaseModel):
    """The limits of G3's measures of the pick itself."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    min_sharpness: float = 0.8
    min_prominence: float = 0.5
    min_on_data: float = 0.6
    max_constant_wavelength: float = 0.4


class CurveThresholds(BaseModel):
    """G3's limits: the assistant's defaults, or those the run's checks used."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    metrics: PickThresholds = Field(default_factory=PickThresholds)
    max_jump: float = 0.3
    min_points: int = 4
    max_uncertainty: float = 0.5  # median uncertainty over the velocity


class DispersionThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: ImageThresholds
    curve: CurveThresholds


class CurveStats(BaseModel):
    """What a picked curve's file gives."""

    model_config = ConfigDict(frozen=True)

    label: str
    n_points: int
    band_hz: tuple[float, float]
    wavelength_m: tuple[float, float]
    uncertainty: float | None  # median over the velocity; None: no uncertainty saved


class DispersionCard(Card):
    """A window's card: its image's coherent band, its picks and who made them."""

    attempts: tuple[AttemptSummary, ...] = ()
    picked_by: Origin | None  # None: nothing picked
    curves: tuple[CurveStats, ...]
    band_hz: tuple[float, float] | None  # the image's coherent band
    wavelength_limits_m: tuple[float | None, float | None]  # what the window resolves


def thresholds_of(run_folder: Path) -> DispersionThresholds:
    return DispersionThresholds(
        image=ImageThresholds.model_validate(config_section(run_folder, "image")),
        curve=CurveThresholds.model_validate(config_section(run_folder, "curve")),
    )


def dispersion_overview(folder: str) -> Overview:
    """Each window at its middle: the checks' verdicts on its image and pick, or its pick's
    measures against their limits, and the longest wavelength it picked."""
    run_folder = folder_path(folder)
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    cells: list[Cell] = []
    origins: Counter[Origin] = Counter()
    by_hand: set[str] = set()
    present: set[tuple[int, PartState]] = set()  # each part's states on the line
    for unit in window_folders(run_folder):
        window = run_folder / unit
        curve = fundamental(window)
        picked_by = pick_origin(window, log, curve is not None)
        if picked_by is not None:
            origins[picked_by] += 1
        if picked_by == "hand":
            by_hand.add(unit)
        g2, g3, g4 = _results(log, unit, picked_by)
        stats = curve_stats(curve) if curve is not None else None
        image, picks = _states(g2, g3, g4, curve, picked_by, thresholds, None)
        # The image's state only when the assistant checked the images.
        parts = (image, picks) if log is not None else (picks,)
        present.update(enumerate(parts))
        # Its image and its curve apart, as its cell's parts, then its curve's figures.
        hover = [f"xmid {number(xmid_of(unit), 4)} m"]
        if log is not None:
            hover.append(_part_line("Image", image, IMAGE_STATES, g2))
        curve_states = CURVE_STATES if log is not None else MEASURED_CURVE_STATES
        hover.append(_part_line("Curve", picks, curve_states, g3, g4))
        if stats is not None:
            hover.append(
                f"{stats.label}: {stats.n_points} points, {span(*stats.band_hz, 'Hz')}, "
                f"λ {span(*stats.wavelength_m, 'm')}"
                + (
                    f", uncertainty {stats.uncertainty:.0%}"
                    if stats.uncertainty is not None
                    else ""
                )
            )
        cells.append(
            Cell(
                key=unit,
                x=xmid_of(unit),
                status=_status(image, picks),
                hover=tuple(hover),
                value=stats.wavelength_m[1] if stats is not None else None,
                parts=parts,
            )
        )
    picked = sum(origins.values())
    summary = f"{picked} of {plural(len(cells), 'window')} picked"
    if log is None or not origins["auto"]:
        summary += " by hand"
    elif not origins["hand"]:
        summary += " automatically"
    else:
        summary += f" ({origins['auto']} automatically, {origins['hand']} by hand)"
    if picked < len(cells):
        summary += f" · {len(cells) - picked} without a curve"
    paco = log is not None
    return Overview(
        paco=paco,
        summary=summary,
        legend={
            "pass": "passed the image and curve checks" if paco else "picked, within the limits",
            "warn": "kept, still flagged" if paco else "a measure beyond its limit",
            "fail": "rejected",
            "none": "no curve",
        },
        cells=tuple(cells),
        parts=_legends(paco, present),
        track=Track(label="Longest wavelength picked (m)", short="Longest λ", kind="value"),
        settings=_picking_settings(log, origins, by_hand),
    )


# Each part's states, in words: the image's (G2), the curve's (G3, G4; or by hand).
IMAGE_STATES: dict[PartState, str] = {
    "pass": "passed",
    "warn": "flagged",
    "fail": "rejected",
    "none": "not checked",
}
CURVE_STATES: dict[PartState, str] = {
    "pass": "passed",
    "warn": "flagged",
    "fail": "rejected",
    "hand": "by hand",
    "none": "no curve",
}
# A run PAC made: its curves' measures against the assistant's limits, not judged.
MEASURED_CURVE_STATES: dict[PartState, str] = {
    "pass": "within the limits",
    "warn": "a measure beyond its limit",
    "hand": "by hand",
    "none": "no curve",
}


def _states(
    g2: GateResult | None,
    g3: GateResult | None,
    g4: GateResult | None,
    curve: DispersionCurve | None,
    picked_by: Origin | None,
    thresholds: DispersionThresholds,
    lambda_min: float | None,
) -> tuple[PartState, PartState]:
    """A window's image and curve apart: its image by G2's verdict (none: not checked); its curve
    by hand, none (no curve), by G3's and G4's verdicts, or by its measures against their
    limits."""
    image: PartState = verdict_status(g2) if g2 is not None else "none"
    if curve is None:
        return image, "none"
    if picked_by == "hand":
        return image, "hand"
    if g3 is not None or g4 is not None:
        return image, verdict_status(g3, g4)
    return image, measured_status(curve_metrics(curve, thresholds.curve, lambda_min))


def _part_line(
    part: str, state: PartState, states: Mapping[PartState, str], *results: GateResult | None
) -> str:
    """One of a window's parts on its cell's hover: its state and, judged, the gates that judged
    it with their flags ("Image: flagged (G2: competing ridges)"); no curve said so."""
    if part == "Curve" and state == "none":
        return "No curve"
    said = f"{part}: {states[state]}"
    judged = [one for one in results if one is not None]
    if judged and state != "hand":
        flags = dict.fromkeys(flag_text(flag.name) for one in judged for flag in one.flags)
        said += f" ({', '.join(one.gate for one in judged)}"
        said += f": {', '.join(flags)})" if flags else ")"
    return said


def _status(image: PartState, curve: PartState) -> Status:
    """A window as one state: a curve by hand, passed as it is; without a curve, none, unless its
    image was rejected; else the worse of its image and its curve."""
    if curve == "hand":
        return "pass"
    if curve == "none":
        return "fail" if image == "fail" else "none"
    return worst(cast(Status, image), cast(Status, curve))


def _legends(paco: bool, present: set[tuple[int, PartState]]) -> tuple[PartLegend, ...]:
    """The cells' parts' legends, each state on the line said: the image's when the assistant
    checked the images, then the curve's."""
    curve = CURVE_STATES if paco else MEASURED_CURVE_STATES
    parts = (("Image", IMAGE_STATES), ("Curve", curve)) if paco else (("Curve", curve),)
    return tuple(
        PartLegend(
            title=title,
            legend={state: said for state, said in states.items() if (i, state) in present},
        )
        for i, (title, states) in enumerate(parts)
    )


def dispersion_card(folder: str, xmid: float) -> DispersionCard:
    run_folder = folder_path(folder)
    unit = f"xmid_{xmid:.2f}"
    window = run_folder / unit
    if not window.is_dir():
        raise ValueError(f"No window for folder={folder}, xmid={xmid}")
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    image = load_image(window) if (window / IMAGE_FILE).exists() else None
    saved = load_curves(window)
    curves = saved.dispersion_curves if saved is not None else ()
    m0 = fundamental(window)
    picked_by = pick_origin(window, log, bool(curves))
    g2, g3, g4 = _results(log, unit, picked_by)
    lambda_min = min_resolvable_wavelength(image.acquisition) if image is not None else None
    lambda_max = max_resolvable_wavelength(image.acquisition) if image is not None else None
    measured_image = image_metrics(image, thresholds.image) if image is not None else ()
    image_measures = g2.metrics if g2 is not None else measured_image
    band, share = (
        (g2.kept.band_hz, _metric(g2.metrics, "coherent_columns"))
        if g2 is not None
        else coherent_band(image, thresholds.image)
        if image is not None
        else (None, None)
    )
    curve_measures = (
        g3.metrics
        if g3 is not None
        else curve_metrics(m0, thresholds.curve, lambda_min)
        if m0 is not None
        else ()
    )
    said: list[Sentence] = []
    # A pick changed by hand leaves the assistant's curve checks and picking attempts behind:
    # none of them is of the pick now in the window.
    judged = log is not None and picked_by != "hand"
    by_hand = picked_by == "hand" and m0 is not None
    verdict = (
        Sentence(mark="pass", text="Picked by hand.")
        if by_hand
        else verdict_sentence("window", g2, g3, g4)
        if judged
        else None
    )
    if image is None:
        said.append(Sentence(mark="fail", text="No dispersion image."))
    elif band is None:
        said.append(Sentence(mark="warn", text="No coherent band in the image."))
    else:
        coherent = next((one for one in image_measures if one.name == "coherent_columns"), None)
        said.append(
            Sentence(
                mark="pass" if coherent is None or coherent.passed else "warn",
                text=f"Its image is coherent from {span(*band, 'Hz')}"
                + (f", over {share:.0%} of its columns." if share is not None else "."),
            )
        )
    stats = [curve_stats(curve) for curve in curves]
    if not by_hand:
        said += _picks(stats, picked_by, g3, curve_measures, lambda_min, judged)
    if log is not None:
        said += _again(log.of(unit, "phase_shift"), "Imaged again")
        if judged:
            said += _again(log.of(unit, "picking"), "Picked again")
    said += warnings(g2, g3, g4)
    gates = [gate_view("G2", g2, measured_image)]
    if by_hand:
        # The user's curve: passed as it is, and not checked along the line.
        gates += [
            GateView(gate="G3", verdict="pass", metrics=(), by_hand=True),
            GateView(gate="G4", verdict=None, metrics=(), by_hand=True),
        ]
    else:
        gates.append(gate_view("G3", g3, curve_measures))
        if g4 is not None:
            gates.append(gate_view("G4", g4))
    # Its image and its curve apart, as its cell's parts: the image's only when the assistant
    # checked the images.
    image_state, curve_state = _states(g2, g3, g4, m0, picked_by, thresholds, lambda_min)
    parts = (Part(label="image", state=image_state),) if log is not None else ()
    return DispersionCard(
        key=unit,
        status=_status(image_state, curve_state),
        title=f"xmid {number(xmid, 4)} m",
        verdict=verdict,
        sentences=tuple(said),
        gates=tuple(gates),
        parts=(*parts, Part(label="curve", state=curve_state)),
        attempts=(
            log.summaries(unit, "phase_shift") + (log.summaries(unit, "picking") if judged else ())
            if log is not None
            else ()
        ),
        picked_by=picked_by,
        curves=tuple(stats),
        band_hz=band,
        wavelength_limits_m=(lambda_min, lambda_max),
    )


def _results(
    log: QCLog | None, unit: str, picked_by: Origin | None
) -> tuple[GateResult | None, GateResult | None, GateResult | None]:
    """G2's, G3's and G4's latest results on the window; G3's and G4's none once its pick was
    changed by hand after them."""
    if log is None:
        return None, None, None
    g2 = log.result(unit, "phase_shift", "G2")
    if picked_by == "hand":
        return g2, None, None
    return g2, log.result(unit, "picking", "G3"), log.result(unit, "picking", "G4")


def _picks(
    stats: list[CurveStats],
    picked_by: Origin | None,
    g3: GateResult | None,
    measures: tuple[Metric, ...],
    lambda_min: float | None,
    paco: bool,
) -> list[Sentence]:
    """What was picked, by whom, and how deep it reaches."""
    if not stats:
        return [Sentence(mark="fail" if paco else "info", text="No curve picked.")]
    first, *others = stats
    by = "automatically by the assistant" if picked_by == "auto" else "by hand"
    text = (
        f"{first.label} picked {by}: {first.n_points} points, {span(*first.band_hz, 'Hz')}, "
        f"wavelengths {span(*first.wavelength_m, 'm')}"
    )
    if first.uncertainty is not None:
        text += f", median uncertainty {first.uncertainty:.0%}"
    status = verdict_status(g3) if g3 is not None else measured_status(measures)
    said = [Sentence(mark="info" if status == "none" else status, text=text + ".")]
    deepest = f"its longest wavelength sets models about {number(first.wavelength_m[1] / 2)} m deep"
    said.append(
        Sentence(
            mark="info",
            text=(
                f"Shorter than {number(lambda_min)} m, wavelengths alias at this receiver "
                f"spacing; {deepest} at most."
                if lambda_min is not None
                else f"{deepest[:1].upper()}{deepest[1:]} at most."
            ),
        )
    )
    if others:
        said.append(
            Sentence(
                mark="info",
                text="Also picked: "
                + ", ".join(f"{one.label} ({one.n_points} points)" for one in others)
                + ".",
            )
        )
    failed = [
        metric.name for metric in measures if metric.threshold is not None and not metric.passed
    ]
    if failed and not paco:
        said.append(
            Sentence(
                mark="warn",
                text="Beyond the assistant's limits: "
                + ", ".join(flag_text(name) for name in failed)
                + ".",
            )
        )
    return said


def _again(attempts: tuple[Attempt, ...], done: str) -> list[Sentence]:
    """Each attempt after the first: what the check before it asked to change, and why."""
    said: list[Sentence] = []
    for before, attempt in pairwise(attempts):
        gate, _, name = attempt.triggered_by.partition(":")
        flag = next(
            (
                flag
                for result in before.results.values()
                for flag in result.flags
                if flag.name == name
            ),
            None,
        )
        changed = _overrides(flag.action.overrides or {}) if flag is not None else ""
        why = f"{gate}: {flag_text(name)}" if name else flag_text(gate)
        said.append(
            Sentence(
                mark="info",
                text=f"{done}" + (f" with {changed}" if changed else "") + f" ({why}).",
            )
        )
    return said


def _overrides(overrides: Mapping[str, Any]) -> str:
    """An override in words: {"dispersion": {"fmax": 49}} as "fmax 49 Hz"."""
    said: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            said.append(_overrides(cast(Mapping[str, Any], value)))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            said.append(f"{flag_text(key)} {number(value)} {UNITS.get(key, '')}".rstrip())
        else:
            said.append(f"{flag_text(key)} {value}")
    return ", ".join(part for part in said if part)


def _picking_settings(
    log: QCLog | None, origins: Counter[Origin], by_hand: set[str]
) -> tuple[Setting, ...]:
    """Who picked the windows, and what the curve check changed of the picks still the
    assistant's (`by_hand`: the windows picked by hand since)."""
    if log is None:
        return (
            Setting(
                key="picking",
                label="Picking",
                value=f"{sum(origins.values())} by hand",
                detail="windows picked",
                why="in the Dispersion picking page",
                origin="pac",
            ),
        )
    again = [
        attempt
        for attempt in log.attempts
        if attempt.stage == "picking"
        and attempt.unit.startswith("xmid_")
        and attempt.unit not in by_hand
        and attempt.attempt > 1
    ]
    asked = Counter(flag_text(attempt.triggered_by.partition(":")[2]) for attempt in again)
    why = "the assistant's picker tracked M0 along its ridge in every window"
    if again:
        windows = {attempt.unit for attempt in again}
        why += (
            f"; picked again in {plural(len(windows), 'window')} by the curve check (G3: "
            + ", ".join(f"{count} {name}" for name, count in asked.most_common())
            + ")"
        )
    if origins["hand"]:
        why += f"; {plural(origins['hand'], 'window')} picked again by hand afterwards"
    return (
        Setting(
            key="picking",
            label="Picking",
            value=f"{origins['auto']} automatic",
            detail=f"{origins['hand']} by hand" if origins["hand"] else "windows picked",
            why=why,
            origin="rule",
        ),
    )


def coherent_band(
    image: DispersionImage, thresholds: ImageThresholds
) -> tuple[tuple[float, float] | None, float]:
    """The band of the image's coherent columns, and their share of its columns."""
    coherent = coherent_columns(image, thresholds.coherent_level)
    share = round(float(coherent.mean()), 3) if coherent.size else 0.0
    if not coherent.any():
        return None, share
    fs = np.asarray(image.fs, dtype=float)[coherent]
    return (round(float(fs.min()), 2), round(float(fs.max()), 2)), share


def image_metrics(image: DispersionImage, thresholds: ImageThresholds) -> tuple[Metric, ...]:
    """sigpipe's measures of the image against G2's limits, as G2 names them."""
    coherent = coherent_columns(image, thresholds.coherent_level)
    n_coherent = int(coherent.sum())
    share = n_coherent / max(coherent.size, 1)
    metrics = [
        Metric(
            name="coherent_columns",
            value=round(share, 3),
            threshold=thresholds.min_coherent_columns,
            bound="min",
            passed=share >= thresholds.min_coherent_columns,
        )
    ]
    if n_coherent == 0:
        return tuple(metrics)
    low, high = edge_peaks(image, coherent, thresholds.edge_share, thresholds.vmin_floor)
    for name, count in (("ridge_at_vmin", low), ("ridge_at_vmax", high)):
        metrics.append(
            Metric(
                name=name,
                value=round(count / n_coherent, 3),
                threshold=thresholds.max_edge_columns,
                bound="max",
                passed=count / n_coherent <= thresholds.max_edge_columns,
            )
        )
    rows = np.flatnonzero(coherent)
    for name, touches in (
        ("band_at_fmin", rows[0] == 0),
        ("band_at_fmax", rows[-1] == coherent.size - 1),
    ):
        metrics.append(
            Metric(name=name, value=float(touches), threshold=0, bound="max", passed=not touches)
        )
    competing = competing_ridges(
        image,
        coherent,
        thresholds.competing_ratio,
        thresholds.competing_separation,
        thresholds.vmin_floor,
    )
    competing_share = int(competing.sum()) / n_coherent
    metrics.append(
        Metric(
            name="competing_ridges",
            value=round(competing_share, 3),
            threshold=thresholds.max_competing_columns,
            bound="max",
            passed=competing_share <= thresholds.max_competing_columns,
        )
    )
    # Reported only: which of the second ridges lie below the aliasing limit.
    alias = aliased(image, competing, thresholds.vmin_floor)
    metrics.append(
        Metric(
            name="aliased_ridges",
            value=round(int(alias.sum()) / max(int(competing.sum()), 1), 3),
            passed=True,
        )
    )
    return tuple(metrics)


def curve_stats(curve: DispersionCurve) -> CurveStats:
    fs, vs = np.asarray(curve.fs, dtype=float), np.asarray(curve.vs, dtype=float)
    lengths = wavelengths(curve)
    return CurveStats(
        label=curve.mode.label,
        n_points=int(fs.size),
        band_hz=(round(float(fs.min()), 2), round(float(fs.max()), 2)),
        wavelength_m=(round(float(lengths.min()), 2), round(float(lengths.max()), 2)),
        uncertainty=_uncertainty(curve, vs),
    )


def curve_metrics(
    curve: DispersionCurve, thresholds: CurveThresholds, lambda_min: float | None
) -> tuple[Metric, ...]:
    """What the curve's file gives against G3's limits, as G3 names them: its points, the
    points below the shortest wavelength the window resolves, the largest step between
    neighbours by wavelength (a jump onto another mode), its median uncertainty."""
    vs = np.asarray(curve.vs, dtype=float)
    lengths = wavelengths(curve)
    order = np.argsort(lengths)
    by_length = vs[order]
    jump = float(np.max(np.abs(np.diff(by_length)) / by_length[:-1])) if vs.size > 1 else 0.0
    metrics = [
        Metric(
            name="curve_points",
            value=int(vs.size),
            threshold=thresholds.min_points,
            bound="min",
            passed=vs.size >= thresholds.min_points,
        ),
        Metric(
            name="max_jump",
            value=round(jump, 3),
            threshold=thresholds.max_jump,
            bound="max",
            passed=jump <= thresholds.max_jump,
        ),
    ]
    if lambda_min is not None:
        below = float(np.mean(lengths < lambda_min))
        metrics.append(
            Metric(
                name="aliased_points",
                value=round(below, 3),
                threshold=0,
                bound="max",
                passed=below == 0,
            )
        )
    uncertainty = _uncertainty(curve, vs)
    if uncertainty is not None:
        metrics.append(
            Metric(
                name="uncertainty",
                value=uncertainty,
                threshold=thresholds.max_uncertainty,
                bound="max",
                passed=uncertainty <= thresholds.max_uncertainty,
            )
        )
    return tuple(metrics)


def _uncertainty(curve: DispersionCurve, vs: np.ndarray) -> float | None:
    if curve.vs_err is None:
        return None
    errors = np.asarray(curve.vs_err, dtype=float)
    known = np.isfinite(errors) & (vs > 0)
    return round(float(np.median(errors[known] / vs[known])), 3) if known.any() else None


def _metric(metrics: tuple[Metric, ...], name: str) -> float | None:
    return next((metric.value for metric in metrics if metric.name == name), None)
