"""The quality of a run's dispersion images and picks: sigpipe's measures of each window's image
(sigpipe.masw.quality.image: the coherent band, peaks on the grid's edges, competing ridges) and
what the picked curve's file gives (its points, band, wavelengths and uncertainties); for a run
the agent made, its image and curve checks (G2 on the image, G3 on the M0 pick, G4 along the
line), with the attempts of the phase shift and of the picking, and who picked each window: the
agent automatically, or a person by hand in PAC. Shown along the line at each window's middle:
the dispersion strip of Visualization, and the selected window's card."""

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from masw.io.paths import workspace
from masw.io.pick_origin import assistant_picked, checks_current_in, pick_origin
from masw.io.quality.done import (
    Changes,
    attempt_steps,
    records_in_common,
    steps_changed,
    window_ranges,
    window_settings,
    with_spreads,
)
from masw.io.quality.files import folder_path, fundamental, read_manifest, wavelengths
from masw.io.quality.log import (
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
    BY_USER,
    Card,
    Cell,
    GateView,
    Mark,
    Overview,
    Part,
    PartLegend,
    PartState,
    Sentence,
    Setting,
    States,
    Status,
    Track,
    by_whom,
    covered,
    gate_view,
    line_gate,
    measured_status,
    measured_warnings,
    number,
    plural,
    span,
    state_lines,
    verdict_sentence,
    verdict_status,
    warnings,
    worst,
)
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurve, Mode
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
from sigpipe.masw.runs.origin import Origin
from sigpipe.masw.windows import MASWWindow, nearest_offset

logger = logging.getLogger(__name__)

# The stacked correlations a passive or passive-active window's image is made of (the pipelines'
# saved stream).
WINDOW_STREAM = "Stream_0000.hdf5"


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
    # Each part's states, as its checks judged them or as made by hand, window by window.
    images = States(none="not made")
    curves = States(none="not picked")
    windows = States(none="not picked")
    manifest = read_manifest(run_folder)
    pickers: Counter[str] = Counter()
    for unit in window_folders(run_folder):
        window = run_folder / unit
        curve = fundamental(window)
        saved = load_curves(window)
        modes = tuple(
            one.mode.label
            for one in sorted(
                saved.dispersion_curves if saved is not None else (),
                key=lambda one: (one.mode.number, one.mode.label),
            )
        )
        picked_by = pick_origin(
            window, log, [one.mode for one in saved.dispersion_curves] if saved is not None else ()
        )
        if picked_by is not None:
            pickers[_picker(log, picked_by, assistant_picked(window, log))] += 1
        g2, g3, g4 = _results(log, window, picked_by)
        stats = curve_stats(curve) if curve is not None else None
        # Its measures where no check judged it: against the agent's limits.
        image_measures = (
            _image_metrics(run_folder, manifest, window, thresholds) if g2 is None else None
        )
        curve_measures = (
            _saved_curve_measures(window, curve, thresholds)
            if g3 is None and g4 is None and picked_by != "hand"
            else ()
        )
        image, picks = _states(g2, g3, g4, curve, picked_by, curve_measures, image_measures)
        images.add(image, g2 is not None)
        curves.add(picks, g3 is not None or g4 is not None)
        state = _status(image, picks)
        status = windows.add(
            state,
            _judged(state, (image, g2 is not None), (picks, g3 is not None or g4 is not None)),
        )
        # Its image and its curve apart, as its cell's parts, then its curve's figures.
        hover = [f"xmid {number(xmid_of(unit), 4)} m"]
        hover += _part_lines("Image", image, g2, metrics=image_measures or ())
        hover += _part_lines("Curve", picks, g3, g4, metrics=curve_measures)
        if stats is not None:
            hover.append(f"{stats.label}: {_points(stats)}")
            hover.append(f"λmax / 2 = {number(stats.wavelength_m[1] / 2)} m")
        if len(modes) > 1:
            hover.append(f"Also picked: {', '.join(modes[1:])}")
        cells.append(
            Cell(
                key=unit,
                x=xmid_of(unit),
                status=status,
                hover=tuple(hover),
                value=stats.wavelength_m[1] if stats is not None else None,
                parts=(image, picks),
                modes=modes,
            )
        )
    paco = log is not None
    said = [
        plural(len(cells), "window"),
        *(f"{count} picked {by}" for by, count in pickers.items()),
        *windows.counts(),
    ]
    legends = (("Image", images), ("Curve", curves))
    return Overview(
        paco=paco,
        summary=" · ".join(said),
        legend={},
        cells=tuple(cells),
        parts=tuple(PartLegend(title=title, legend=states.legend()) for title, states in legends),
        track=Track(label="Longest wavelength picked (m)", short="λmax", kind="value"),
        gates=line_gate(log, "picking", "G4"),
    )


def _picker(log: QCLog | None, picked_by: Origin, assistant: bool) -> str:
    """Who picked a window's curve: the run's agent (its picker, automatically), or the user (by
    hand, or with PAC's automatic picking)."""
    if picked_by == "auto" and assistant:
        return by_whom(log, True)
    return f"{BY_USER}, automatically" if picked_by == "auto" else BY_USER


def _part_lines(
    part: str, state: PartState, *results: GateResult | None, metrics: Iterable[Metric] = ()
) -> list[str]:
    """One of a window's parts on its cell's hover, as its legend says it: its state, what its
    checks found when they flagged or rejected it; a warning they kept, or its measures' (made by
    hand, `metrics`), a line of its own."""
    if state == "none":
        return [f"{part}: {'not picked' if part == 'Curve' else 'not made'}"]
    if state == "hand":
        return [f"{part}: picked by the user"]
    head, *rest = state_lines(cast(Status, state), *results, metrics=metrics)
    return [f"{part}: {head[:1].lower()}{head[1:]}", *rest]


def _states(
    g2: GateResult | None,
    g3: GateResult | None,
    g4: GateResult | None,
    curve: DispersionCurve | None,
    picked_by: Origin | None,
    measured: tuple[Metric, ...],
    measured_image: tuple[Metric, ...] | None,
) -> tuple[PartState, PartState]:
    """A window's image and curve apart, each by its checks' verdicts or, when no check judged
    it, by its measures against their limits (`measured`, `measured_image`): its image none when
    not made; its curve by hand, or none (not picked)."""
    image: PartState = (
        verdict_status(g2)
        if g2 is not None
        else "none"
        if measured_image is None
        else measured_status(measured_image)
    )
    if curve is None:
        return image, "none"
    if picked_by == "hand":
        return image, "hand"
    if g3 is not None or g4 is not None:
        return image, verdict_status(g3, g4)
    return image, measured_status(measured)


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
    G3's); none on a passive line, whose records hold no shot (their source at the origin only by
    default)."""
    path = window / "window.json"
    manifest = read_manifest(window.parent)
    if not path.exists() or (manifest is not None and manifest.profile.kind == "passive"):
        return None
    return nearest_offset(MASWWindow.model_validate_json(path.read_text()))


def _image_metrics(
    run_folder: Path,
    manifest: RunManifest | None,
    window: Path,
    thresholds: DispersionThresholds,
    image: DispersionImage | None = None,
) -> tuple[Metric, ...] | None:
    """The measures of a window's image against G2's limits, as G2 measures it: a passive or
    passive-active window's stacked correlations and fk selection too; None without its image."""
    if image is None and not (window / IMAGE_FILE).exists():
        return None
    image = image if image is not None else load_image(window)
    return (
        image_metrics(image, thresholds.image, records_band(run_folder, manifest, window))
        + _correlation_metrics(window)
        + _selection_metrics(load_selection(window), thresholds_of_signal(run_folder))
    )


def _status(image: PartState, curve: PartState) -> Status:
    """A window as one state: a curve by hand, passed as it is, its image's state kept; without
    a curve, none, unless its image was rejected; else the worse of its image and its curve."""
    if curve == "hand":
        return "pass" if image in ("none", "hand") else cast(Status, image)
    if curve == "none":
        return "fail" if image == "fail" else "none"
    return worst(cast(Status, image), cast(Status, curve))


def _judged(status: Status, *parts: tuple[PartState, bool]) -> bool:
    """Whether checks made a window's state `status`: each of its `parts` (its state, whether a
    check judged it) in that state judged. A part measured against the agent's limits, as a
    curve picked in PAC on an image the agent's check passed, makes it a measure's state."""
    at = [judged for state, judged in parts if state == status]
    return bool(at) and all(at)


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
    picked_by = pick_origin(window, log, [one.mode for one in curves])
    g2, g3, g4 = _results(log, window, picked_by)
    lambda_min = min_resolvable_wavelength(image.acquisition) if image is not None else None
    lambda_max = longest_reached_wavelength(image.acquisition) if image is not None else None
    measured_image = (
        _image_metrics(run_folder, manifest, window, thresholds, image)
        if image is not None
        else None
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
    said = [Sentence(mark="info", text=f"Dispersion image made {_imaged_by(log, unit)}.")]
    # A pick changed by hand leaves the assistant's curve checks and picking attempts behind:
    # none of them is of the pick now in the window; nor after PAC picked it again itself, until
    # the assistant judges that pick.
    judged = log is not None and picked_by != "hand"
    current = judged and checks_current_in(window, log)
    by_hand = picked_by == "hand" and m0 is not None
    # What the checks said; of a curve picked by hand, the image's alone.
    verdict = verdict_sentence(g2, g3, g4) if log is not None else None
    if image is None:
        said.append(Sentence(mark="fail", text="No dispersion image."))
    elif band is None:
        said.append(Sentence(mark="warn", text="No coherent band in its dispersion image."))
    else:
        # Its measures (the share of its columns coherent) in its table.
        said.append(
            Sentence(
                mark="info",
                text=f"Its dispersion image is coherent from {number(band[0])} to "
                f"{number(band[1])} Hz.",
            )
        )
    stats = [curve_stats(curve) for curve in curves]
    said += _picks(
        stats,
        picked_by,
        g3,
        curve_measures,
        lambda_min,
        judged,
        _picker(log, picked_by, assistant_picked(window, log)) if picked_by is not None else "",
    )
    said += warnings(g2, g3, g4)
    # Made by hand, no check judged it: its measures' warnings, as the checks keep theirs.
    if g2 is None:
        said += measured_warnings(measured_image or ())
    if g3 is None and g4 is None and not by_hand:
        said += measured_warnings(curve_measures)
    # The image's measures; a passive or passive-active window's, its stacked correlations' and
    # its fk selection's too: G2's own when it judged the window, else as PAC's job saved them.
    gates = [gate_view("G2", g2, measured_image or ())]
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
    # Its image and its curve apart, as its cell's parts.
    image_state, curve_state = _states(g2, g3, g4, m0, picked_by, curve_measures, measured_image)
    parts = (
        Part(label="image", state=image_state, judged=g2 is not None),
        Part(label="curve", state=curve_state, judged=g3 is not None or g4 is not None),
    )
    return DispersionCard(
        key=unit,
        status=_status(image_state, curve_state),
        title=f"xmid {number(xmid, 4)} m",
        verdict=verdict,
        sentences=tuple(said),
        gates=tuple(gates),
        parts=parts,
        steps=attempt_steps(
            log,
            unit,
            ("phase_shift", "picking") if current else ("phase_shift",),
            spacing := manifest.profile.receiver_spacing_m if manifest is not None else 1.0,
            changes=_changes(spacing),
        ),
        picked_by=picked_by,
        curves=tuple(stats),
        settings=_settings(run_folder, window, manifest, log) if manifest is not None else (),
        band_hz=band,
        wavelength_limits_m=(lambda_min, lambda_max),
    )


# A pick's parameters as a step says them, by name: what it sets, its value in words.
_PICKING: dict[str, tuple[str, Callable[[Any], str]]] = {
    "wavelength_step": ("Resampling", lambda value: f"every {number(value)} m of wavelength"),
    "min_relative_coherence": (
        "Points kept",
        lambda value: f"down to {number(value)} of the ridge's median coherence",
    ),
    "mode_min_ratio": ("Mode rule", lambda value: f"{number(value)} times the noise floor"),
    "corridor": ("Tracking corridor", lambda value: f"{number(100 * value)} % of the velocity"),
    "max_wavelength": (
        "Longest wavelength searched",
        lambda value: "any" if value is None else f"{number(value)} window lengths",
    ),
    "guide": ("Guided", lambda value: f"along a curve of {len(value)} points" if value else "no"),
}


def _changes(spacing: float) -> Changes:
    """What a window's attempt changed, in words: the phase shift's steps as the run card says
    them; a pick's parameters, its band first."""

    def said(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
        if any(isinstance(value, Mapping) for value in after.values()):
            return steps_changed(before, after, spacing)
        changed = [key for key, value in after.items() if before.get(key) != value]
        lines: list[str] = []
        if {"fmin", "fmax"} & set(changed):
            low, high = after.get("fmin"), after.get("fmax")
            band = (
                span(low, high, "Hz", 4)
                if low is not None and high is not None
                else f"from {number(low, 4)} Hz"
                if low is not None
                else f"up to {number(high, 4)} Hz"
                if high is not None
                else "the image's"
            )
            lines.append(f"Picking band: {band}")
        for key in changed:
            if key in ("fmin", "fmax"):
                continue
            name, words = _PICKING.get(
                key, (key.replace("_", " ").capitalize(), lambda value: str(value))
            )
            lines.append(f"{name}: {words(after[key])}")
        return lines

    return said


def _results(
    log: QCLog | None, window: Path, picked_by: Origin | None
) -> tuple[GateResult | None, GateResult | None, GateResult | None]:
    """G2's, G3's and G4's latest results on the window; G3's and G4's none when they are not of
    its pick now: picked by hand, or picked again in PAC after them."""
    if log is None:
        return None, None, None
    unit = window.name
    g2 = log.result(unit, "phase_shift", "G2")
    if picked_by == "hand" or not checks_current_in(window, log):
        return g2, None, None
    return g2, log.result(unit, "picking", "G3"), log.result(unit, "picking", "G4")


def _modes(window: Path) -> list[Mode]:
    """The modes window folder `window` holds a curve of."""
    saved = load_curves(window)
    return [one.mode for one in saved.dispersion_curves] if saved is not None else []


def _imaged_by(log: QCLog | None, unit: str) -> str:
    """Who made window `unit`'s dispersion image: the run's agent when it logged it, else the
    user in PAC."""
    return by_whom(log, log is not None and bool(log.of(unit, "phase_shift")))


def _points(stats: CurveStats) -> str:
    """A curve's points and band: "53 points, 6-41.5 Hz"."""
    return f"{plural(stats.n_points, 'point')}, {span(*stats.band_hz, 'Hz')}"


def _picks(
    stats: list[CurveStats],
    picked_by: Origin | None,
    g3: GateResult | None,
    measures: tuple[Metric, ...],
    lambda_min: float | None,
    paco: bool,
    by: str,
) -> list[Sentence]:
    """What was picked and by whom (`by`: the run's agent, or the user), the wavelengths it
    spans, where they may alias."""
    # The pick's state as its sentence's mark: the curve check's flag, orange; beyond the agent's
    # limits when the user made it, a warning.
    marks: dict[str, Mark] = {"none": "info", "info": "info", "note": "warn", "warn": "flag"}
    if not stats:
        # None picked by the agent: as the curve check judged it (flagged, its retries spent).
        status = verdict_status(g3) if g3 is not None else "fail"
        mark = marks.get(status, cast(Mark, status)) if paco else "info"
        return [Sentence(mark=mark, text="Not picked.")]
    first, *others = stats
    status = (
        "info"
        if picked_by == "hand"
        else verdict_status(g3)
        if g3 is not None
        else measured_status(measures)
    )
    shortest, longest = first.wavelength_m
    mark = marks.get(status, cast(Mark, status))
    said = [
        Sentence(mark=mark, text=f"{first.label} picked {by}: {_points(first)}."),
        Sentence(
            mark="info",
            text=f"Its wavelengths from {number(shortest)} to {number(longest)} m: λmax / 2 = "
            f"{number(longest / 2)} m.",
        ),
    ]
    if lambda_min is not None:
        said.append(
            Sentence(
                mark="info",
                text=f"Under {number(lambda_min)} m, wavelengths may alias at this receiver "
                "spacing.",
            )
        )
    if others:
        said.append(
            Sentence(
                mark="info",
                # Higher modes: the user's alone, the agent picking M0.
                text=f"Also picked {BY_USER}: "
                + ", ".join(f"{one.label} ({plural(one.n_points, 'point')})" for one in others)
                + ".",
            )
        )
    return said


def _picking_setting(log: QCLog | None, picked_by: Origin | None, assistant: bool) -> Setting:
    """Who picked a window's curves: by hand or PAC's automatic picking (the Dispersion picking
    page), or the agent's picker along M0's ridge (`assistant`)."""
    if picked_by is None:
        return Setting(key="picking", label="Picking", value="none", why="", origin="default")
    if picked_by == "hand" or log is None or not assistant:
        return Setting(
            key="picking",
            label="Picking",
            value="by hand" if picked_by == "hand" else "automatic",
            detail="in the Dispersion picking page",
            why="",
            origin="you",
        )
    # Each time it was picked again, in its card's steps.
    return Setting(
        key="picking",
        label="Picking",
        value="automatic",
        detail="M0 tracked along its ridge",
        why="",
        origin=log.chooser,
    )


def _records_settings(
    window: Path, manifest: RunManifest, log: QCLog | None
) -> tuple[Setting, ...]:
    """How the records window folder `window` stacks were preprocessed, before its image was
    made of them: what most ran with, the others named (records_in_common); none when it says no
    records."""
    path = window / "window.json"
    if not path.exists():
        return ()
    names = [
        one.name
        for one in MASWWindow.model_validate_json(path.read_text()).selected_files
        if one.name not in manifest.exclusions.records
    ]
    return records_in_common(manifest, log, names)


def _settings(
    run_folder: Path, window: Path, manifest: RunManifest, log: QCLog | None
) -> tuple[Setting, ...]:
    """What window folder `window`'s display was made with, in the run card's order: its
    receivers, its records' preprocessing, its image's steps, its pick; each with how the line's
    other windows differ."""
    own = with_spreads(
        window.name,
        _line_settings(run_folder, manifest, log),
        "window",
        ranges=window_ranges(manifest, log, window_folders(run_folder)),
    )
    return (*own[:1], *_records_settings(window, manifest, log), *own[1:])


def _line_settings(
    run_folder: Path, manifest: RunManifest, log: QCLog | None
) -> dict[str, tuple[Setting, ...]]:
    """Each window's settings: how its image was made (window_settings), who picked its curves."""
    return {
        unit: (
            *window_settings(manifest, log, unit),
            _picking_setting(
                log,
                pick_origin(run_folder / unit, log, _modes(run_folder / unit)),
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
