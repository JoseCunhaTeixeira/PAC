"""The quality of a run's petrophysical inversions: which picked curves the Silex model covers
(sigpipe's catalog: its trained band and velocities), each window's soil column and the fit of
the curve it gives back to the pick by band of wavelength (sigpipe.masw.petro.measuring, as for
the seismic models), and for a run the assistant made, its checks (G7 on each window, G8 against
its neighbours along the line) and the attempts. Shown along the line at each window's middle:
the petrophysics strip of Visualization, and the selected window's card."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import folder_path, fundamental
from masw.io.quality.history import register_stage, unit_steps
from masw.io.quality.inversion import FitCurve, fit_curve
from masw.io.quality.log import (
    GateResult,
    Metric,
    QCLog,
    config_section,
    read_log,
)
from masw.io.quality.makers import PAC_PETRO_CONFIG, soil_by_agent
from masw.io.quality.view import (
    Card,
    Cell,
    Origin,
    Overview,
    Sentence,
    Setting,
    States,
    Track,
    by_whom,
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
)
from sigpipe.algorithms.inversion.rayleigh.petro.silex_catalog import (
    RangeGap,
    SilexCard,
    bundled_silex_model_dir,
    list_bundled_silex_models,
    load_silex_card,
    range_gaps,
)
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.masw.inversion.measuring import ModelFit
from sigpipe.masw.petro import load_modeled_curve
from sigpipe.masw.petro.measuring import PetroMeasures
from sigpipe.masw.petro.window import MODEL_FILE
from sigpipe.masw.quality.soil import SoilLimits, measure_soil
from sigpipe.masw.runs import window_folders, xmid_of

MEASURES_FILE = "PetroInversion_Measures_0000.json"  # the assistant's
STAGE = "petro_inversion"
# A soil column saved this long after the assistant's last attempt on the window was PAC's.
BAND_NAMES = {3: ("short", "middle", "long")}
# How a curve falls outside a model's range, in words.
GAPS: dict[RangeGap, str] = {
    "starts_late": "starts above {start:g} Hz",
    "ends_early": "ends below {end:g} Hz",
    "too_slow": "is slower than {slowest:.0f} m/s",
    "too_fast": "is faster than {fastest:.0f} m/s",
}


class PetroThresholds(BaseModel):
    """G7's and G8's limits: the assistant's defaults, or those the run's checks used."""

    model_config = ConfigDict(frozen=True)

    max_misfit: float = 2.0  # G7: in any band of wavelength
    n_bands: int = 3
    max_vs_misfit: float = 0.15  # G8: relative to a side's median Vs
    max_water_table_jump: float = 1.0  # G8: m, to a side's median


class SoilColumn(BaseModel):
    """A window's soils, top down, the last a half-space."""

    model_config = ConfigDict(frozen=True)

    soils: tuple[str, ...]
    thicknesses_m: tuple[float, ...]
    ns: tuple[int, ...]  # SPT N values
    water_table_m: float | None


class PetroCard(Card):
    """A window's card: its soil column, its fit, whether the model covers its curve."""

    model: str | None  # the Silex model
    column: SoilColumn | None
    curve: FitCurve | None
    fit: ModelFit | None
    gaps: tuple[RangeGap, ...]  # how its curve falls outside the model's range


def thresholds_of(run_folder: Path) -> PetroThresholds:
    g7, g8 = config_section(run_folder, "petro"), config_section(run_folder, "petro_line")
    defaults = PetroThresholds()
    return PetroThresholds(
        max_misfit=g7.get("max_misfit", defaults.max_misfit),
        n_bands=g7.get("n_bands", defaults.n_bands),
        max_vs_misfit=g8.get("max_misfit", defaults.max_vs_misfit),
        max_water_table_jump=g8.get("max_water_table_jump", defaults.max_water_table_jump),
    )


def fit_metrics(measures: PetroMeasures, thresholds: PetroThresholds) -> tuple[Metric, ...]:
    """A soil column measured as G7 measures it (sigpipe's measure_soil: one definition), each
    saying what it covers: its curve's fit by band, their residuals, its water table."""
    limits = SoilLimits(max_misfit=thresholds.max_misfit, n_bands=thresholds.n_bands)
    return tuple(
        Metric(**one.model_dump())
        for one in measure_soil(measures.fit, measures.water_table_m, limits)
    )


def petro_overview(folder: str) -> Overview:
    """Each window at its middle: the checks' verdicts on its soil column, or its fit against
    the limit, and its water table's depth."""
    run_folder = folder_path(folder)
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    model = model_of(run_folder, log)
    card = _card(model)
    cells: list[Cell] = []
    found: list[tuple[RangeGap, ...]] = []
    states = States(none="not inverted")
    # Who made each soil column: the run's agent, or the user (PAC's, after the agent's or alone).
    makers: Counter[str] = Counter()
    for unit in window_folders(run_folder):
        window = run_folder / unit
        x = xmid_of(unit)
        curve = fundamental(window)
        gaps = range_gaps(card, curve) if card is not None and curve is not None else ()
        if curve is not None:
            found.append(gaps)
        measures = _measures(window)
        hover = [f"xmid {number(x, 4)} m"]
        if measures is None:
            states.add("none", judged=False)
            hover.append(_unmeasured(window).capitalize())
            cells.append(Cell(key=unit, x=x, status="none", hover=tuple(hover)))
            continue
        makers[by_whom(log, soil_by_agent(log, window))] += 1
        g7, g8 = _results(log, window)
        measured = fit_metrics(measures, thresholds)
        status = states.add(
            verdict_status(g7, g8) if g7 is not None else measured_status(measured),
            g7 is not None,
        )
        hover += state_lines(status, g7, g8, metrics=measured)
        if gaps and card is not None:
            hover.append("Outside the model's range: " + _gaps_text(card, gaps))
        hover.append(" / ".join(measures.soils) + ", top down")
        hover.append(f"Water table at {number(measures.water_table_m)} m")
        misfits = [band.misfit for band in measures.fit.bands if band.misfit is not None]
        if misfits:
            hover.append(f"Misfit {max(misfits):.2f} at worst")
        cells.append(
            Cell(
                key=unit,
                x=x,
                status=status,
                hover=tuple(hover),
                value=measures.water_table_m,
                total=sum(measures.thicknesses_m) or None,
            )
        )
    paco = log is not None and any(attempt.stage == STAGE for attempt in log.attempts)
    said = [
        plural(len(cells), "window"),
        *(f"{count} inverted {by}" for by, count in makers.items()),
        *([f"soil model {model}"] if model is not None and makers else []),
        *states.counts(),
    ]
    return Overview(
        paco=paco,
        summary=" · ".join(said),
        legend=states.legend(),
        cells=tuple(cells),
        track=Track(label="Water table (m)", short="Water table", kind="depth"),
        gates=line_gate(log, "petro_inversion", "G8"),
    )


def petro_card(folder: str, xmid: float) -> PetroCard:
    run_folder = folder_path(folder)
    unit = f"xmid_{xmid:.2f}"
    window = run_folder / unit
    if not window.is_dir():
        raise ValueError(f"No window for folder={folder}, xmid={xmid}")
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    model = model_of(run_folder, log)
    card = _card(model)
    curve = fundamental(window)
    gaps = range_gaps(card, curve) if card is not None and curve is not None else ()
    measures = _measures(window)
    # A soil column PAC made again leaves the assistant's checks and attempts behind.
    ours = measures is not None and soil_by_agent(log, window)
    g7, g8 = _results(log, window) if measures is not None else (None, None)
    said: list[Sentence] = []
    verdict = verdict_sentence(g7, g8)
    if measures is not None:
        said.append(Sentence(mark="info", text=f"Inverted {by_whom(log, ours)}."))
    if card is not None and curve is not None:
        said.append(_coverage(card, curve, gaps))
    metrics: tuple[Metric, ...] = ()
    if measures is None:
        unmeasured = _unmeasured(window)
        said.append(
            Sentence(
                mark="info",
                text="Inverted before its measures were saved with it: invert it again to see them."
                if unmeasured != "not inverted"
                else "Not inverted.",
            )
        )
    else:
        metrics = fit_metrics(measures, thresholds)
        said += _column_sentences(measures)
        said += _fit_sentences(measures.fit, thresholds)
        said += warnings(g7, g8) if g7 is not None else measured_warnings(metrics)
    gates = (gate_view("G7", g7, metrics),) + ((gate_view("G8", g8),) if g8 is not None else ())
    return PetroCard(
        key=unit,
        settings=_model_setting(run_folder, log, model, card) if measures is not None else (),
        status=(
            "none"
            if measures is None
            else verdict_status(g7, g8)
            if g7 is not None
            else measured_status(metrics)
        ),
        title=f"xmid {number(xmid, 4)} m",
        verdict=verdict,
        sentences=tuple(said),
        gates=gates if measures is not None else (),
        steps=unit_steps(run_folder, unit, log, None, (STAGE,)),
        model=model,
        column=(
            SoilColumn(
                soils=measures.soils,
                thicknesses_m=measures.thicknesses_m,
                ns=measures.ns,
                water_table_m=measures.water_table_m,
            )
            if measures is not None
            else None
        ),
        curve=_curve(window, curve),
        fit=measures.fit if measures is not None else None,
        gaps=gaps,
    )


def model_of(run_folder: Path, log: QCLog | None) -> str | None:
    """The Silex model of the run's last petrophysical inversion, the assistant's (as its
    attempts name it) or PAC's (as its configuration does), whichever ran last; None when none
    ran."""
    attempt = next(
        (
            one
            for one in reversed(log.attempts if log is not None else ())
            if one.stage == STAGE and isinstance(one.parameters.get("model"), str)
        ),
        None,
    )
    path = run_folder / PAC_PETRO_CONFIG
    pac: object = json.loads(path.read_text()).get("model_name") if path.exists() else None
    if isinstance(pac, str) and (
        attempt is None or datetime.fromtimestamp(path.stat().st_mtime, UTC) > attempt.started_at
    ):
        return pac
    return str(attempt.parameters["model"]) if attempt is not None else None


def _results(log: QCLog | None, window: Path) -> tuple[GateResult | None, GateResult | None]:
    """G7's and G8's latest results on the window; none once PAC inverted it again after them."""
    if log is None or not soil_by_agent(log, window):
        return None, None
    return log.result(window.name, STAGE, "G7"), log.result(window.name, STAGE, "G8")


def _measures(window: Path) -> PetroMeasures | None:
    """The window's petrophysical inversion as measured when it ran (PAC's job or the assistant),
    none older than its model: Visualization never measures."""
    path = window / MEASURES_FILE
    model_file = window / MODEL_FILE
    if not path.exists() or (
        model_file.exists() and path.stat().st_mtime < model_file.stat().st_mtime
    ):
        return None
    return PetroMeasures.model_validate_json(path.read_text())


def _unmeasured(window: Path) -> str:
    """Why a window shows no soil column."""
    if (window / MODEL_FILE).exists():
        return "inverted before its measures were saved"
    return "not inverted"


def _card(model: str | None) -> SilexCard | None:
    if model is None or model not in list_bundled_silex_models():
        return None
    return load_silex_card(bundled_silex_model_dir(model))


def _model_setting(
    run_folder: Path, log: QCLog | None, model: str | None, card: SilexCard | None
) -> tuple[Setting, ...]:
    """The Silex model the run's windows were inverted with, the same for them all: what it was
    trained on, who chose it, and how many of the line's picked curves it covers."""
    found = [
        range_gaps(card, curve) if card is not None else ()
        for unit in window_folders(run_folder)
        if (curve := fundamental(run_folder / unit)) is not None
    ]
    paco = log is not None and any(attempt.stage == STAGE for attempt in log.attempts)
    return _settings(card, model, found, log.chooser if log is not None and paco else "you")


def _settings(
    card: SilexCard | None, model: str | None, found: list[tuple[RangeGap, ...]], origin: Origin
) -> tuple[Setting, ...]:
    if model is None:
        return ()
    if card is None:
        return (Setting(key="model", label="Silex model", value=model, why="", origin=origin),)
    covered = sum(1 for gaps in found if not gaps)
    counts: Counter[RangeGap] = Counter(gap for gaps in found for gap in gaps)
    start, end = card.band_needed
    slowest, fastest = card.velocities_allowed
    left_out = ", ".join(
        f"{count} {GAPS[gap].format(start=start, end=end, slowest=slowest, fastest=fastest)}"
        for gap, count in counts.most_common()
    )
    return (
        Setting(
            key="model",
            label="Silex model",
            value=model,
            detail=f"trained on {span(card.min_freq, card.max_freq, 'Hz')} · "
            f"{span(card.min_vel, card.max_vel, 'm/s')} · covers {covered} of the "
            f"{plural(len(found), 'picked curve')}",
            why=f"the others: {left_out}" if left_out else "",
            origin=origin,
        ),
    )


def _coverage(card: SilexCard, curve: DispersionCurve, gaps: tuple[RangeGap, ...]) -> Sentence:
    low, high = float(min(curve.fs)), float(max(curve.fs))
    if not gaps:
        return Sentence(
            mark="pass",
            text=f"The Silex model {card.name} covers its curve, {number(low)} to "
            f"{number(high)} Hz.",
        )
    return Sentence(
        mark="warn",
        text=f"Its curve {_gaps_text(card, gaps)}: outside the range {card.name} was trained on.",
    )


def _gaps_text(card: SilexCard, gaps: tuple[RangeGap, ...]) -> str:
    start, end = card.band_needed
    slowest, fastest = card.velocities_allowed
    return " and ".join(
        GAPS[gap].format(start=start, end=end, slowest=slowest, fastest=fastest) for gap in gaps
    )


def _column_sentences(measures: PetroMeasures) -> list[Sentence]:
    """The soil column top down, each layer with its thickness and N; its water table."""
    layers = [
        f"{soil} {number(thickness)} m (N {n})"
        for soil, thickness, n in zip(
            measures.soils, measures.thicknesses_m, measures.ns, strict=False
        )
    ]
    if len(measures.soils) > len(measures.thicknesses_m):
        layers.append(f"{measures.soils[-1]} below")
    return [
        Sentence(mark="info", text=f"Soil column, top down: {', '.join(layers)}."),
        Sentence(mark="info", text=f"Water table at {number(measures.water_table_m)} m."),
    ]


def _fit_sentences(fit: ModelFit, thresholds: PetroThresholds) -> list[Sentence]:
    """How the soil column's curve fits the pick: its worst band against the limit."""
    known = [(band.misfit, i) for i, band in enumerate(fit.bands) if band.misfit is not None]
    if not known:
        return [Sentence(mark="warn", text="Its curve could not be compared with the pick.")]
    names = BAND_NAMES.get(len(fit.bands))
    worst, at = max(known)
    limit = number(thresholds.max_misfit)
    if worst <= thresholds.max_misfit:
        return [
            Sentence(
                mark="pass", text=f"Fits its curve: misfit {worst:.2f} at worst, {limit} allowed."
            )
        ]
    where = f" at the {names[at]} wavelengths" if names else ""
    return [Sentence(mark="warn", text=f"Misfits its curve: {worst:.2f}{where}, {limit} allowed.")]


def _curve(window: Path, observed: DispersionCurve | None) -> FitCurve | None:
    return fit_curve(observed, load_modeled_curve(window)) if observed is not None else None


def _measured(run_folder: Path, unit: str) -> tuple[str, tuple[Metric, ...]] | None:
    """A soil column no check judged, measured against G7's limits as G7 measures one."""
    measures = _measures(run_folder / unit)
    if measures is None:
        return None
    return "G7", fit_metrics(measures, thresholds_of(run_folder))


register_stage(STAGE, measures=_measured)
