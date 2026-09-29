"""The quality of a run's dispersion images and picks: sigpipe's measures of each window's image
(sigpipe.masw.quality.image: the coherent band, peaks on the grid's edges, competing ridges) and
what the picked curve's file gives (its points, band, wavelengths and uncertainties); for a run
the assistant made, its image and curve checks (G2 on the image, G3 on the M0 pick, G4 along the
line), with the attempts of the phase shift and of the picking, and who picked each window: the
assistant automatically, or a person by hand in PAC. Shown along the line at each window's
middle: the dispersion strip of Visualization, and the selected window's card."""

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from masw.io.paths import workspace
from masw.io.pick_origin import Origin, assistant_picked, pick_origin
from masw.io.quality.done import (
    most_common,
    record_ranges,
    record_settings,
    window_ranges,
    window_settings,
    with_spreads,
)
from masw.io.quality.files import folder_path, fundamental, read_manifest, wavelengths
from masw.io.quality.log import (
    Attempt,
    AttemptSummary,
    GateResult,
    Metric,
    QCLog,
    config_section,
    read_log,
)
from masw.io.quality.records import (
    MEASURES_FILE,
    RecordGather,
    SavedSpectra,
    SignalMeasures,
    before_muting,
    gather_of,
    image_band,
    saved_measures,
    saved_spectra,
    single_precision,
)
from masw.io.quality.records import thresholds_of as thresholds_of_signal
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
    covered,
    flag_text,
    gate_view,
    line_gate,
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
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.dataio.selection_plotting import SelectionScores, load_selection
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.masw.picks import load_curves
from sigpipe.masw.pipelines import PREPROCESSED, window_correlations
from sigpipe.masw.profiles import Profile, ProfileError, load_profile
from sigpipe.masw.quality.curve import CurveLimits, measure_curve, pick_of
from sigpipe.masw.quality.image import ImageLimits, coherent_columns, measure_image
from sigpipe.masw.quality.measures import SignalLimits, measure_signal, selection_measures
from sigpipe.masw.runs import RunManifest, load_image, window_folders, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE
from sigpipe.masw.windows import MASWWindow, nearest_offset

logger = logging.getLogger(__name__)

# The stacked correlations a passive or passive-active window's image is made of (the pipelines'
# saved stream).
WINDOW_STREAM = "Stream_0000.hdf5"

# The units the attempts' overrides are said in.
UNITS = {"fmin": "Hz", "fmax": "Hz", "vmin": "m/s", "vmax": "m/s", "wavelength_step": "m"}


class DispersionThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: ImageLimits  # G2's, sigpipe's: an image measured as G2 measures it
    curve: CurveLimits  # G3's, sigpipe's


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
    # Where G3's flags start: under, the aliasing zone; over, beyond the window's reach.
    wavelength_limits_m: tuple[float | None, float | None]


def thresholds_of(run_folder: Path) -> DispersionThresholds:
    return DispersionThresholds(
        image=ImageLimits.model_validate(config_section(run_folder, "image")),
        curve=CurveLimits.model_validate(config_section(run_folder, "curve")),
    )


def dispersion_overview(folder: str) -> Overview:
    """Each window at its middle: the checks' verdicts on its image and pick, or its pick's
    measures against their limits, and the longest wavelength it picked."""
    run_folder = folder_path(folder)
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    cells: list[Cell] = []
    origins: Counter[Origin] = Counter()
    computed = 0  # the windows with an image
    present: set[tuple[int, PartState]] = set()  # each part's states on the line
    for unit in window_folders(run_folder):
        window = run_folder / unit
        computed += (window / IMAGE_FILE).exists()
        curve = fundamental(window)
        saved = load_curves(window)
        modes = tuple(
            one.mode.label
            for one in sorted(
                saved.dispersion_curves if saved is not None else (),
                key=lambda one: (one.mode.number, one.mode.label),
            )
        )
        picked_by = pick_origin(window, log, curve is not None)
        if picked_by is not None:
            origins[picked_by] += 1
        g2, g3, g4 = _results(log, unit, picked_by)
        stats = curve_stats(curve) if curve is not None else None
        image, picks = _states(
            g2,
            g3,
            g4,
            curve,
            picked_by,
            partial(_saved_curve_measures, window, curve, thresholds),
        )
        # The image's state only when the assistant checked the images.
        parts = (image, picks) if log is not None else (picks,)
        present.update(enumerate(parts))
        # Its image and its curve apart, as its cell's parts, then its curve's figures.
        hover = [f"xmid {number(xmid_of(unit), 4)} m"]
        if log is not None:
            hover.append(_part_line("Image", image, IMAGE_STATES, g2))
        curve_states = CURVE_STATES if log is not None else MEASURED_CURVE_STATES
        hover.append(_part_line("Curve", picks, curve_states, g3, g4))
        if modes:
            hover.append(f"{plural(len(modes), 'mode')} picked: {', '.join(modes)}")
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
                modes=modes,
            )
        )
    picked = sum(origins.values())
    how = (
        "by hand"
        if not origins["auto"]
        else "automatically"
        if not origins["hand"]
        else f"({origins['auto']} automatically, {origins['hand']} by hand)"
    )
    summary = f"{computed} of {plural(len(cells), 'window')} computed"
    if picked and picked == computed:
        summary += f" and picked {how}"
    elif picked:
        summary += f" · {picked} picked {how}"
    if picked < computed:
        summary += f" · {computed - picked} without a curve"
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
        gates=line_gate(log, "picking", "G4"),
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
    measured: Callable[[], tuple[Metric, ...]],
) -> tuple[PartState, PartState]:
    """A window's image and curve apart: its image by G2's verdict (none: not checked); its curve
    by hand, none (no curve), by G3's and G4's verdicts, or by its measures against their
    limits (`measured`, when no check judged it)."""
    image: PartState = verdict_status(g2) if g2 is not None else "none"
    if curve is None:
        return image, "none"
    if picked_by == "hand":
        return image, "hand"
    if g3 is not None or g4 is not None:
        return image, verdict_status(g3, g4)
    return image, measured_status(measured())


def _saved_curve_measures(
    window: Path, curve: DispersionCurve | None, thresholds: DispersionThresholds
) -> tuple[Metric, ...]:
    """The measures of a window's saved curve against G3's limits, on its image; none without
    the image."""
    if curve is None or not (window / IMAGE_FILE).exists():
        return ()
    return curve_metrics(load_image(window), curve, thresholds.curve, _nearest(window))


def _nearest(window: Path) -> float | None:
    """The distance from the window's nearest shot to its receivers (sigpipe's nearest_offset,
    G3's)."""
    path = window / "window.json"
    if not path.exists():
        return None
    return nearest_offset(MASWWindow.model_validate_json(path.read_text()))


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
    manifest = read_manifest(run_folder)
    thresholds = thresholds_of(run_folder)
    image = load_image(window) if (window / IMAGE_FILE).exists() else None
    saved = load_curves(window)
    curves = saved.dispersion_curves if saved is not None else ()
    m0 = fundamental(window)
    picked_by = pick_origin(window, log, bool(curves))
    g2, g3, g4 = _results(log, unit, picked_by)
    lambda_min = min_resolvable_wavelength(image.acquisition) if image is not None else None
    lambda_max = longest_reached_wavelength(image.acquisition) if image is not None else None
    measured_image = (
        image_metrics(image, thresholds.image, records_band(run_folder, manifest, window))
        if image is not None
        else ()
    )
    band = (
        g2.kept.band_hz
        if g2 is not None
        else coherent_band(image, thresholds.image)[0]
        if image is not None
        else None
    )
    curve_measures = (
        g3.metrics
        if g3 is not None
        else curve_metrics(image, m0, thresholds.curve, _nearest(window))
        if m0 is not None and image is not None
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
        # Its measures (the share of its columns coherent) in its table.
        said.append(Sentence(mark="info", text=f"Its image is coherent from {span(*band, 'Hz')}."))
    selection = load_selection(window)
    stats = [curve_stats(curve) for curve in curves]
    if not by_hand:
        said += _picks(
            stats,
            picked_by,
            g3,
            curve_measures,
            lambda_min,
            judged,
            assistant_picked(window, log),
        )
    if log is not None:
        said += _again(log.of(unit, "phase_shift"), "Imaged again")
        if judged:
            said += _again(log.of(unit, "picking"), "Picked again")
    said += warnings(g2, g3, g4)
    # The image's measures; a passive or passive-active window's, its stacked correlations' and
    # its fk selection's too: G2's own when it judged the window, else as PAC's job saved them.
    gates = [
        gate_view(
            "G2",
            g2,
            measured_image
            + _correlation_metrics(window)
            + _selection_metrics(selection, thresholds_of_signal(run_folder)),
        )
    ]
    if by_hand:
        # The user's curve: passed as it is, and not checked along the line; measured as G3
        # measures a pick.
        gates += [
            GateView(
                gate="G3",
                verdict="pass",
                metrics=tuple(covered("G3", metric) for metric in curve_measures),
                by_hand=True,
            ),
            GateView(gate="G4", verdict=None, metrics=(), by_hand=True),
        ]
    else:
        gates.append(gate_view("G3", g3, curve_measures))
        if g4 is not None:
            gates.append(gate_view("G4", g4))
    # Its image and its curve apart, as its cell's parts: the image's only when the assistant
    # checked the images.
    image_state, curve_state = _states(g2, g3, g4, m0, picked_by, lambda: curve_measures)
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
        settings=_records_settings(window, manifest, log)
        + with_spreads(
            unit,
            _line_settings(run_folder, manifest, log),
            "window",
            ranges=window_ranges(manifest, log, window_folders(run_folder)),
        )
        if manifest is not None
        else (),
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
    assistant: bool,
) -> list[Sentence]:
    """What was picked, by whom (`assistant`: the assistant's automatic pick, else PAC's), and
    how deep it reaches."""
    if not stats:
        return [Sentence(mark="fail" if paco else "info", text="No curve picked.")]
    first, *others = stats
    by = (
        "by hand"
        if picked_by != "auto"
        else "automatically by the assistant"
        if assistant
        else "automatically"
    )
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


def _picking_setting(
    log: QCLog | None, unit: str, picked_by: Origin | None, assistant: bool
) -> Setting:
    """Who picked window `unit`'s curves, and why: by hand or PAC's automatic picking (the
    Dispersion picking page), or the assistant's picker along M0's ridge (`assistant`), picked
    again by the curve check (G3) when it was."""
    if picked_by is None:
        return Setting(
            key="picking", label="Picking", value="none", why="no curve picked", origin="default"
        )
    if picked_by == "hand" or log is None or not assistant:
        return Setting(
            key="picking",
            label="Picking",
            value="by hand" if picked_by == "hand" else "automatic",
            why="in the Dispersion picking page"
            if picked_by == "hand"
            else "PAC's automatic picking, in the Dispersion picking page",
            origin="pac",
        )
    again = [one for one in log.of(unit, "picking") if one.attempt > 1]
    asked = Counter(flag_text(one.triggered_by.partition(":")[2]) for one in again)
    why = "the assistant's picker tracked M0 along its ridge"
    if again:
        why += (
            "; picked again by the curve check (G3: "
            + ", ".join(f"{count} {name}" for name, count in asked.most_common())
            + ")"
        )
    return Setting(
        key="picking",
        label="Picking",
        value="automatic",
        detail=f"picked {len(again) + 1} times" if again else "",
        why=why,
        origin="rule",
    )


def _records_settings(
    window: Path, manifest: RunManifest, log: QCLog | None
) -> tuple[Setting, ...]:
    """How the records window folder `window` stacks were preprocessed, before its image was
    made of them: what most ran with, the others named (most_common); none when it says no
    records."""
    path = window / "window.json"
    if not path.exists():
        return ()
    names = [
        one.name
        for one in MASWWindow.model_validate_json(path.read_text()).selected_files
        if one.name not in manifest.exclusions.records
    ]
    return most_common(
        {name: record_settings(manifest, log, name) for name in names},
        "record",
        ranges=record_ranges(manifest, log, names),
    )


def _line_settings(
    run_folder: Path, manifest: RunManifest, log: QCLog | None
) -> dict[str, tuple[Setting, ...]]:
    """Each window's settings: how its image was made (window_settings), who picked its curves."""
    return {
        unit: (
            *window_settings(manifest, log, unit),
            _picking_setting(
                log,
                unit,
                pick_origin(run_folder / unit, log, fundamental(run_folder / unit) is not None),
                assistant_picked(run_folder / unit, log),
            ),
        )
        for unit in window_folders(run_folder)
    }


def coherent_band(
    image: DispersionImage, thresholds: ImageLimits
) -> tuple[tuple[float, float] | None, float]:
    """The band of the image's coherent columns, and their share of its columns."""
    coherent = coherent_columns(image, thresholds.coherent_level)
    share = round(float(coherent.mean()), 3) if coherent.size else 0.0
    if not coherent.any():
        return None, share
    fs = np.asarray(image.fs, dtype=float)[coherent]
    return (round(float(fs.min()), 2), round(float(fs.max()), 2)), share


def image_metrics(
    image: DispersionImage, limits: ImageLimits, usable_band: tuple[float, float] | None = None
) -> tuple[Metric, ...]:
    """The image's measures as G2 measures it (sigpipe's measure_image: one definition), each
    saying what it covers; with the records' `usable_band`, the coherent band's share of it."""
    return tuple(
        Metric(**one.model_dump()) for one in measure_image(image, limits, usable_band).measures
    )


def records_band(
    run_folder: Path, manifest: RunManifest | None, window: Path
) -> tuple[float, float] | None:
    """The band every record of `window` keeps usable, as PAC's job measured them (G2's
    shared_band): the highest low edge, the lowest high edge; None when none says one."""
    path = window / "window.json"
    if manifest is None or not path.exists():
        return None
    selected = MASWWindow.model_validate_json(path.read_text()).selected_files
    folders = {record.name: record.folder for record in manifest.records}
    bands = [
        band
        for one in selected
        if one.name in folders
        and (band := saved_measures(run_folder / folders[one.name] / PREPROCESSED)[1]) is not None
    ]
    if not bands:
        return None
    low, high = max(band[0] for band in bands), min(band[1] for band in bands)
    return (low, high) if low < high else None


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
    image: DispersionImage,
    curve: DispersionCurve,
    limits: CurveLimits,
    nearest_offset: float | None = None,
) -> tuple[Metric, ...]:
    """A saved curve's measures as G3 measures a pick (sigpipe's measure_curve: one definition),
    sampled as a pick is on its image (pick_of), each saying what it covers."""
    report = measure_curve(image, pick_of(curve, image), limits, nearest_offset)
    return tuple(Metric(**one.model_dump()) for one in report.measures)


def _uncertainty(curve: DispersionCurve, vs: np.ndarray) -> float | None:
    if curve.vs_err is None:
        return None
    errors = np.asarray(curve.vs_err, dtype=float)
    known = np.isfinite(errors) & (vs > 0)
    return round(float(np.median(errors[known] / vs[known])), 3) if known.any() else None


def _selection_metrics(
    selection: SelectionScores | None, limits: SignalLimits
) -> tuple[Metric, ...]:
    """A passive window's fk selection among its image's measures, as G2 judges it (sigpipe's
    selection_measures: the share of segments kept against its limit)."""
    if selection is None:
        return ()
    return tuple(Metric(**one.model_dump()) for one in selection_measures(selection, limits))


def _correlation_metrics(window: Path) -> tuple[Metric, ...]:
    """A passive or passive-active window's stacked correlations measured as a record is, as
    PAC's job saved them (measure_windows); none saved as new as the correlations: none."""
    metrics, _ = saved_measures(window / WINDOW_STREAM)
    return metrics


def measure_windows(run_folder: Path) -> None:
    """Save the measures of every passive or passive-active window's stacked correlations beside
    them (a run PAC made: the assistant's own are G2's, in its QC log): measured as a record is
    (sigpipe's measure_signal, from the virtual source), with the run's limits. A muted line's
    SNR and band on its correlations made again from its records before their muting (a
    muting zeroes their noise, and the correlations' after the slowest arrival), as G2's; not
    measured without them."""
    manifest = read_manifest(run_folder)
    if manifest is None or str(manifest.preset.mode) not in ("passive", "passive-active"):
        return
    limits = thresholds_of_signal(run_folder)
    values: dict[str, Any] = manifest.preset.model_dump()
    muting: dict[str, Any] = values.get("muting") or {}
    muted = muting.get("method", "none") != "none"
    profile: Profile | None = None
    if muted:
        try:
            profile = load_profile(manifest.profile.name, workspace())
        except ProfileError:
            profile = None
    unmuted: dict[str, Stream | None] = {}

    def record(name: str) -> Stream | None:
        if name not in unmuted:
            found = before_muting(manifest.preset, profile, name)
            unmuted[name] = None if found is None else single_precision(found)[0]
        return unmuted[name]

    for window in manifest.windows:
        path = run_folder / window.folder / WINDOW_STREAM
        if window.status != "succeeded" or not path.exists():
            continue
        before = (
            _correlations_before_muting(path.parent, manifest, record)
            if profile is not None
            else None
        )
        report = measure_signal(
            load_stream([path])[0],
            limits,
            source="virtual",
            spectra=True,
            records_muted=muted,
            before_muting=None if before is None else (before, 0.0),
            image_band=image_band(manifest.preset),
        )
        band = report.band
        measures = SignalMeasures(
            metrics=tuple(Metric(**one.model_dump()) for one in report.measures),
            band_hz=None if band is None else (round(band[0], 2), round(band[1], 2)),
        )
        (path.parent / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))


def _correlations_before_muting(
    folder: Path, manifest: RunManifest, record: Callable[[str], Stream | None]
) -> Stream | None:
    """A passive-active window's stacked correlations made again from its records before their
    muting (`record`, by name), as its pipeline made them; None without them all."""
    preset = manifest.preset
    if str(preset.mode) != "passive-active":
        return None
    window = MASWWindow.model_validate_json((folder / "window.json").read_text())
    streams: dict[str, Stream] = {}
    for path in window.selected_files:
        found = record(path.name)
        if found is None:
            return None
        streams[path.name] = found
    try:
        return window_correlations(preset, window, streams)
    except Exception:
        logger.exception("Could not correlate the records of %s before their muting", folder)
        return None


def window_selection(folder: str, xmid: float) -> SelectionScores:
    """Window `xmid`'s fk segment selection as its job saved it (a passive window with the fk
    selection on); a ValueError else."""
    found = load_selection(folder_path(folder) / f"xmid_{xmid:.2f}")
    if found is None:
        raise ValueError(f"No fk selection saved for folder={folder}, xmid={xmid}")
    return found


def window_spectra(folder: str, xmid: float) -> SavedSpectra:
    """The spectra of the stacked correlations window `xmid`'s image was made of (passive and
    passive-active), as its job saved them."""
    return saved_spectra(
        folder_path(folder) / f"xmid_{xmid:.2f}", f"window xmid={xmid} in folder={folder}"
    )


def window_gather(folder: str, xmid: float, norm: str = "trace") -> RecordGather:
    """The stacked correlations window `xmid`'s image was made of (passive and passive-active:
    its Stream_0000.hdf5), as the wiggle plot takes them, the whole of them; the virtual source
    its star. None in an active window: a ValueError."""
    path = folder_path(folder) / f"xmid_{xmid:.2f}" / WINDOW_STREAM
    if not path.exists():
        raise ValueError(f"No stacked correlations for folder={folder}, xmid={xmid}")
    stream = load_stream([path])[0]
    return gather_of(stream, f"xmid {xmid:g} m", norm)
