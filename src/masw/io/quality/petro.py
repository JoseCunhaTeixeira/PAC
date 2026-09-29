"""The quality of a run's petrophysical inversions: which picked curves the Silex model covers
(sigpipe's catalog: its trained band and velocities), each window's soil column and the fit of
the curve it gives back to the pick by band of wavelength (sigpipe.masw.petro.measuring, as for
the seismic models), and for a run the assistant made, its checks (G7 on each window, G8 against
its neighbours along the line) and the attempts. Shown along the line at each window's middle:
the petrophysics strip of Visualization, and the selected window's card."""

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import folder_path, fundamental
from masw.io.quality.inversion import FitCurve, fit_curve
from masw.io.quality.log import (
    AttemptSummary,
    GateResult,
    Metric,
    QCLog,
    config_section,
    metric_value,
    read_log,
)
from masw.io.quality.view import (
    Card,
    Cell,
    Overview,
    Sentence,
    Setting,
    Status,
    Track,
    gate_view,
    line_gate,
    measured_status,
    number,
    plural,
    span,
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
CONFIG_FILE = "petro_inversion_config.json"  # PAC's
STAGE = "petro_inversion"
# A soil column saved this long after the assistant's last attempt on the window was PAC's.
LATER = timedelta(minutes=1)
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

    attempts: tuple[AttemptSummary, ...] = ()
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
    statuses: list[Status] = []
    for unit in window_folders(run_folder):
        window = run_folder / unit
        x = xmid_of(unit)
        curve = fundamental(window)
        gaps = range_gaps(card, curve) if card is not None and curve is not None else ()
        if curve is not None:
            found.append(gaps)
        measures = _measures(window)
        hover = [f"xmid {number(x, 4)} m"]
        if gaps and card is not None:
            hover.append("outside the model's range: " + _gaps_text(card, gaps))
        if measures is None:
            hover.append(_unmeasured(window))
            cells.append(Cell(key=unit, x=x, status="none", hover=tuple(hover)))
            continue
        g7, g8 = _results(log, window)
        status = (
            verdict_status(g7, g8)
            if g7 is not None
            else measured_status(fit_metrics(measures, thresholds))
        )
        statuses.append(status)
        misfits = [band.misfit for band in measures.fit.bands if band.misfit is not None]
        hover.append(" / ".join(measures.soils) + " (top down)")
        hover.append(f"water table at {number(measures.water_table_m)} m")
        if misfits:
            hover.append(f"misfit {max(misfits):.2f} at worst")
        verdicts = [f"{one.gate} {one.verdict}" for one in (g7, g8) if one is not None]
        if verdicts:
            hover.append(" · ".join(verdicts))
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
    summary = f"{len(statuses)} of {plural(len(cells), 'window')} inverted"
    if model is not None and statuses:
        summary += f" with {model}"
        passed, flagged, rejected = (statuses.count(one) for one in ("pass", "warn", "fail"))
        summary += " · " + ", ".join(
            part
            for part in (
                f"{passed} pass" if passed else "",
                f"{flagged} flagged" if flagged else "",
                f"{rejected} rejected" if rejected else "",
            )
            if part
        )
    return Overview(
        paco=paco,
        summary=summary,
        legend={
            "pass": "passed the soil column checks" if paco else "fits its curve",
            "warn": "kept, still flagged" if paco else "misfits its curve",
            "fail": "rejected",
            "none": "not inverted",
        },
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
    ours = measures is not None and by_assistant(log, window)
    g7, g8 = _results(log, window) if measures is not None else (None, None)
    said: list[Sentence] = []
    verdict = verdict_sentence("soil column", g7, g8)
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
        said.append(_column_sentence(measures))
        said += _fit_sentences(measures.fit, thresholds)
        said += _neighbours(g8, thresholds)
        said += warnings(g7, g8)
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
        attempts=log.summaries(unit, STAGE) if log is not None and ours else (),
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
    path = run_folder / CONFIG_FILE
    pac: object = json.loads(path.read_text()).get("model_name") if path.exists() else None
    if isinstance(pac, str) and (
        attempt is None or datetime.fromtimestamp(path.stat().st_mtime, UTC) > attempt.started_at
    ):
        return pac
    return str(attempt.parameters["model"]) if attempt is not None else None


def by_assistant(log: QCLog | None, window: Path) -> bool:
    """Whether the soil column in window folder `window` is the assistant's: it logged one, and
    PAC did not invert the window again after it."""
    latest = log.latest(window.name, STAGE) if log is not None else None
    if latest is None:
        return False
    model = window / MODEL_FILE
    done = latest.finished_at or latest.started_at
    return not (model.exists() and model.stat().st_mtime > (done + LATER).timestamp())


def _results(log: QCLog | None, window: Path) -> tuple[GateResult | None, GateResult | None]:
    """G7's and G8's latest results on the window; none once PAC inverted it again after them."""
    if log is None or not by_assistant(log, window):
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
    return _settings(card, model, found, paco)


def _settings(
    card: SilexCard | None, model: str | None, found: list[tuple[RangeGap, ...]], paco: bool
) -> tuple[Setting, ...]:
    if model is None:
        return ()
    why = "chosen by the assistant" if paco else "set by hand"
    origin = "rule" if paco else "pac"
    if card is None:
        return (Setting(key="model", label="Silex model", value=model, why=why, origin=origin),)
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
            detail=f"trained on {span(card.min_freq, card.max_freq, 'Hz')}, "
            f"{span(card.min_vel, card.max_vel, 'm/s')}",
            why=f"{why}; it covers {covered} of the {plural(len(found), 'picked curve')}"
            + (f" ({left_out})" if left_out else ""),
            origin=origin,
        ),
    )


def _coverage(card: SilexCard, curve: DispersionCurve, gaps: tuple[RangeGap, ...]) -> Sentence:
    band = span(float(min(curve.fs)), float(max(curve.fs)), "Hz")
    if not gaps:
        return Sentence(
            mark="pass",
            text=f"The Silex model {card.name} covers its curve ({band}).",
        )
    return Sentence(
        mark="warn",
        text=f"Outside the range {card.name} was trained on: its curve ({band}) "
        f"{_gaps_text(card, gaps)}.",
    )


def _gaps_text(card: SilexCard, gaps: tuple[RangeGap, ...]) -> str:
    start, end = card.band_needed
    slowest, fastest = card.velocities_allowed
    return " and ".join(
        GAPS[gap].format(start=start, end=end, slowest=slowest, fastest=fastest) for gap in gaps
    )


def _column_sentence(measures: PetroMeasures) -> Sentence:
    layers = [
        f"{soil} {number(thickness)} m (N {n})"
        for soil, thickness, n in zip(
            measures.soils, measures.thicknesses_m, measures.ns, strict=False
        )
    ]
    if len(measures.soils) > len(measures.thicknesses_m):
        layers.append(f"{measures.soils[-1]} below")
    return Sentence(
        mark="info",
        text=f"Soil column, top down: {', '.join(layers)}; water table at "
        f"{number(measures.water_table_m)} m.",
    )


def _fit_sentences(fit: ModelFit, thresholds: PetroThresholds) -> list[Sentence]:
    misfits = [band.misfit for band in fit.bands]
    known = [value for value in misfits if value is not None]
    if not known:
        return [Sentence(mark="warn", text="Its curve could not be compared with the pick.")]
    names = BAND_NAMES.get(len(misfits))
    each = ", ".join(f"{value:.2f}" if value is not None else "no mode" for value in misfits)
    where = f" over the {', '.join(names[:-1])} and {names[-1]} wavelengths" if names else ""
    good = all(value is not None and value <= thresholds.max_misfit for value in misfits)
    return [
        Sentence(
            mark="pass" if good else "warn",
            text=f"{'Fits' if good else 'Misfits'} its curve: misfit {each}{where} (at most "
            f"{number(thresholds.max_misfit)}, in uncertainties).",
        )
    ]


def _neighbours(g8: GateResult | None, thresholds: PetroThresholds) -> list[Sentence]:
    """What G8 found against the neighbours' columns."""
    if g8 is None:
        return []
    vs = metric_value(g8, "misfit")
    jump = metric_value(g8, "water_table_jump")
    parts: list[str] = []
    if vs is not None:
        parts.append(f"Vs {vs:.0%} off their median (at most {thresholds.max_vs_misfit:.0%})")
    if jump is not None:
        parts.append(
            f"water table {number(jump)} m off theirs (at most "
            f"{number(thresholds.max_water_table_jump)} m)"
        )
    if not parts:
        return []
    return [
        Sentence(
            mark="pass" if g8.verdict == "pass" else "warn",
            text=f"Against its neighbours: {', '.join(parts)}.",
        )
    ]


def _curve(window: Path, observed: DispersionCurve | None) -> FitCurve | None:
    return fit_curve(observed, load_modeled_curve(window)) if observed is not None else None
