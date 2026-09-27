"""The quality of a run's seismic inversions: what sigpipe measures of each window's posterior
(sigpipe.masw.inversion.measuring: fits by band, R-hat, effective samples, autocorrelation,
acceptance, piling at the prior's bounds, the depth the data inform), the model itself and the
curve it gives back, the chains, and for a run the assistant made, its model checks (G5 on each
window, G6 along the line) with the attempts and what each changed. Shown along the line at each
window's middle: the inversion strip of Visualization, the selected window's card, and the
parameters the line was inverted with, each with why. The measures of a window are read from the
file the assistant writes next to its inversion, and written there when PAC inverted it."""

import logging
import statistics
from collections.abc import Iterable, Sequence
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import folder_path
from masw.io.quality.log import (
    Attempt,
    AttemptSummary,
    GateResult,
    Metric,
    QCLog,
    config_section,
    read_log,
    summarize,
)
from masw.io.quality.view import (
    Card,
    Cell,
    Overview,
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
)
from sigpipe.algorithms.inversion.rayleigh.seismic.data import NOISE_BOUNDS
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.masw.inversion import (
    PARAMETERS_FILE,
    InversionParameters,
    WindowParameters,
    load_parameters,
)
from sigpipe.masw.inversion.measuring import (
    BoundShare,
    InversionMeasures,
    ModelFit,
    measure_inversion,
)
from sigpipe.masw.inversion.section import ModelName, picked_curve, predicted_curve, window_model
from sigpipe.masw.inversion.window import SAMPLES_FILE, load_profiles, load_samples
from sigpipe.masw.runs import window_folders, xmid_of

logger = logging.getLogger(__name__)

MEASURES_FILE = "SeismicInversion_Measures_0000.json"  # the assistant's name, and its content
PAC_CONFIG_FILE = "seismic_inversion_config.json"  # what PAC's page inverted with
STAGE = "inversion"
GATE = "G5"
TRACE_POINTS = 400  # the most points a chain's trace keeps
BINS = 40  # a marginal's bins between the prior's bounds
PROFILE_POINTS = 400  # the most layers a profile keeps of a smooth model
# An inversion done this long after the assistant's last attempt on the window was PAC's.
LATER = timedelta(minutes=1)
BAND_NAMES = {3: ("short", "middle", "long")}
# The share of proposals the trial runs aimed at (%), for the runs saved before 2026-09-27.
TRIAL_BAND = (20.0, 30.0)
# How the assistant chooses its parameters, as it says it (PACo's inversion rules).
RULES = {
    "layers": "4 to start, never fewer than 3; one fewer where a layer was not resolved or two "
    "were alike, one more where the model misfit its curve (G5)",
    "vs": "wide at first (100 to 1,000 m/s, the half-space to 2,000), wider where a curve "
    "needed it or a posterior piled at a bound (G5)",
    "thickness": "a third of each curve's shortest wavelength: thinner is not resolved",
    "depth": "half each curve's longest wavelength at first, shrunk to the depth the data "
    "inform where that was shallower (G5)",
    "drop": "a fifth at most: under a layer much stiffer than the next, the forward model's "
    "fundamental mode is a wave trapped in the soft layer",
    "effort": "the default effort, longer where the chains did not agree (G5)",
}

type FigureName = Literal["marginals", "density_curves", "dispersion_image"]
FIGURES: dict[FigureName, str] = {
    "marginals": "SeismicInversion_Marginals_0000.png",
    "density_curves": "SeismicInversion_DensityCurves_0000.png",
    "dispersion_image": "SeismicInversion_DispersionImage_0000.png",
}


class InversionThresholds(BaseModel):
    """G5's limits: the assistant's defaults, or those the run's checks used."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_misfit: float = 2.0  # in any band of wavelength
    n_bands: int = 3
    max_rhat: float = 1.1
    min_ess: float = 200.0  # of the models' Vs at any depth watched
    min_samples_per_chain: int = 100
    bound_edge: float = 0.02  # of a prior's range, at each bound
    max_at_bound: float = 0.1  # share of a parameter's samples within that edge
    useful_std_ratio: float = 0.5
    min_useful_share: float = 0.8  # of the model's depth
    max_vs_drop: float = 0.5
    plausible_vs: tuple[float, float] = (50.0, 2_500.0)
    min_contrast: float = 0.05


class VsProfile(BaseModel):
    """A window's model against depth, for the profile plot: Vs and its spread by layer, the
    layered median, how deep the models go and how deep the data inform them."""

    model_config = ConfigDict(frozen=True)

    model: ModelName
    tops: tuple[float, ...]  # m, each layer's top, the half-space's last
    vs: tuple[float, ...]  # m/s
    std: tuple[float, ...]  # m/s
    layered_tops: tuple[float, ...]  # the layered median's
    layered_vs: tuple[float, ...]
    bottom: float  # m, the bottom of the models sigpipe builds
    informed: float | None  # m, how deep the data inform them; None: to the bottom
    deepest_top: float  # m, the deepest the half-space's top could be (the prior's)


class FitCurve(BaseModel):
    """The picked curve, and the one the model gives back."""

    model_config = ConfigDict(frozen=True)

    label: str
    observed_fs: tuple[float, ...]
    observed_vs: tuple[float, ...]
    observed_err: tuple[float, ...]  # empty without uncertainties
    predicted_fs: tuple[float, ...]
    predicted_vs: tuple[float, ...]


class Convergence(BaseModel):
    """One parameter of the model: its prior, what the inversion found of it (the posterior's
    median and its 10th and 90th percentiles, over every chain), and how well the chains
    sampled it."""

    model_config = ConfigDict(frozen=True)

    parameter: str  # vs1, ..., thick1, ... (layers from the top)
    prior: tuple[float, float] | None
    median: float | None
    low: float | None  # 10 % of the samples below
    high: float | None  # 10 % above
    rhat: float | None
    ess: float | None
    autocorrelation: float | None
    step: float | None  # the sampler's step, as it ran
    fixed: float | None = None  # the value, fixed: not sampled, so no prior, spread or measure


class InversionAttempt(AttemptSummary):
    n_layers: int | None
    depth_m: float | None  # every layer above the half-space at its thickest


class InversionCard(Card):
    """A window's card: its model, its fit, how well its chains sampled, and what it ran with."""

    attempts: tuple[InversionAttempt, ...] = ()  # with each attempt's layers and depth
    inverted: bool
    parameters: InversionParameters | None  # as run: each step as the trial runs tuned it
    tuning: tuple[tuple[float, float], ...]  # each trial run's step factor and acceptance (%)
    step_factor: float | None  # the factor the run kept; None: steps not tuned
    acceptance: tuple[float, ...]  # %, per chain
    samples_per_chain: int
    convergence: tuple[Convergence, ...]
    fits: tuple[ModelFit, ...]  # the smooth median's, then the layered median's
    at_bounds: tuple[BoundShare, ...]  # the most piled first
    profile: VsProfile | None
    curve: FitCurve | None
    figures: tuple[FigureName, ...]  # the figures the inversion saved


class ChainTraces(BaseModel):
    """A parameter's samples along each chain, every `step`-th kept."""

    model_config = ConfigDict(frozen=True)

    parameter: str
    step: int
    chains: tuple[tuple[float, ...], ...]


class Marginal(BaseModel):
    """A parameter's samples in BINS bins between its prior's bounds, per chain."""

    model_config = ConfigDict(frozen=True)

    parameter: str
    low: float
    high: float
    counts: tuple[tuple[int, ...], ...]


class Chains(BaseModel):
    """A window's chains, for the plots folded under its card."""

    model_config = ConfigDict(frozen=True)

    traces: tuple[ChainTraces, ...]
    marginals: tuple[Marginal, ...]


def thresholds_of(run_folder: Path) -> InversionThresholds:
    return InversionThresholds.model_validate(config_section(run_folder, "model"))


def window_measures(
    folder: Path, thresholds: InversionThresholds
) -> tuple[WindowParameters, InversionMeasures] | None:
    """The parameters window `folder` was inverted with and the measures of its posterior; None
    when it holds no inversion sigpipe saved its samples for. Measured when no file holds them
    yet, or an older one than the samples: then saved, as the assistant saves them. Raises
    ValueError when they cannot be measured (no fundamental mode among the curves inverted)."""
    samples = folder / SAMPLES_FILE
    if not (folder / PARAMETERS_FILE).exists() or not samples.exists():
        return None
    ran = load_parameters(folder / PARAMETERS_FILE)
    path = folder / MEASURES_FILE
    if path.exists() and path.stat().st_mtime >= samples.stat().st_mtime:
        return ran, InversionMeasures.model_validate_json(path.read_text())
    try:
        measures = measure_inversion(
            folder,
            ran.parameters,
            n_bands=thresholds.n_bands,
            bound_edge=thresholds.bound_edge,
            std_ratio=thresholds.useful_std_ratio,
        )
    except (StopIteration, OSError, ValueError) as exc:
        raise ValueError(f"The inversion of {folder.name} cannot be measured: {exc!r}") from exc
    path.write_text(measures.model_dump_json(indent=2))
    return ran, measures


def model_depth(parameters: InversionParameters) -> float:
    """The deepest the half-space's top may be: every layer above it at its thickest; the
    deepest interface allowed when the data chose the layers (resolved)."""
    if parameters.layering == "free":
        return round(parameters.free.depth_max or 0.0, 2)
    return round(sum(layer.bounds[1] for layer in parameters.thickness_layers), 2)


def step_factor(tuning: tuple[tuple[float, float], ...]) -> float | None:
    """The factor on every step the trial runs settled on: the last trial's, when its acceptance
    fell within the middle half of the target band, else the one closest to the band's middle."""
    if not tuning:
        return None
    low, high = TRIAL_BAND
    middle, margin = (low + high) / 2, (high - low) / 4
    factor, rate = tuning[-1]
    if low + margin <= rate <= high - margin:
        return factor
    return min(tuning, key=lambda trial: abs(trial[1] - middle))[0]


def model_metrics(
    parameters: InversionParameters, measures: InversionMeasures, thresholds: InversionThresholds
) -> tuple[Metric, ...]:
    """G5's measures of a window's model against G5's limits, as G5 names them: its fit by band,
    how its chains converged, the samples piled at a bound, the depth it is informed down to."""
    smooth = measures.fits[0]
    names = BAND_NAMES.get(
        len(smooth.bands), tuple(f"band{i + 1}" for i in range(len(smooth.bands)))
    )
    metrics = [
        Metric(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=thresholds.max_misfit,
            bound="max",
            passed=band.misfit is not None and band.misfit <= thresholds.max_misfit,
        )
        for name, band in zip(names, smooth.bands, strict=True)
    ]
    rhat, ess, _ = _convergence(measures)
    correlations = [
        value
        for name, value in measures.autocorrelation.items()
        if value is not None and name in _judged(measures)
    ]
    piled = measures.at_bounds[0] if measures.at_bounds else None
    depth = model_depth(parameters)
    useful = measures.useful_depth_m
    metrics += [
        Metric(
            name="rhat",
            value=rhat,
            threshold=thresholds.max_rhat,
            bound="max",
            passed=rhat is not None and rhat <= thresholds.max_rhat,
        ),
        Metric(
            name="ess",
            value=ess,
            threshold=thresholds.min_ess,
            bound="min",
            passed=ess is None or ess >= thresholds.min_ess,
        ),
        Metric(
            name="autocorrelation",
            value=max(correlations) if correlations else None,
            passed=True,
        ),
        Metric(
            name="at_bound",
            value=piled.share if piled else None,
            threshold=thresholds.max_at_bound,
            bound="max",
            passed=piled is None or piled.share <= thresholds.max_at_bound,
        ),
        Metric(
            name="useful_depth",
            value=useful,
            threshold=round(thresholds.min_useful_share * depth, 2),
            bound="min",
            passed=useful is None or useful >= thresholds.min_useful_share * depth,
            unit="m",
        ),
    ]
    return tuple(metrics)


def inversion_overview(folder: str) -> Overview:
    """Each window at its middle: the checks' verdicts on its model, or its measures against
    their limits, and how deep the data inform it of the depth modelled."""
    run_folder = folder_path(folder)
    log = read_log(run_folder)
    thresholds = thresholds_of(run_folder)
    cells: list[Cell] = []
    runs: list[InversionParameters] = []
    informed: list[float] = []
    statuses: list[Status] = []
    again = 0  # the assistant's inversions PAC made again after
    for unit in window_folders(run_folder):
        x = xmid_of(unit)
        try:
            measured = window_measures(run_folder / unit, thresholds)
        except ValueError:
            logger.warning("Inversion of %s left out of the overview", unit, exc_info=True)
            measured = None
        if measured is None:
            cells.append(
                Cell(key=unit, x=x, status="none", hover=(f"xmid {number(x, 4)} m", "not inverted"))
            )
            continue
        ran, measures = measured
        runs.append(ran.parameters)
        if log is not None and log.of(unit, STAGE) and not by_assistant(log, unit, run_folder):
            again += 1
        g5, g6 = _results(log, unit, run_folder)
        metrics = model_metrics(ran.parameters, measures, thresholds)
        status = verdict_status(g5, g6) if g5 is not None else measured_status(metrics)
        statuses.append(status)
        bottom = measures.depth_max_m
        useful = measures.useful_depth_m if measures.useful_depth_m is not None else bottom
        informed.append(useful)
        rhat, ess, acceptance = _convergence(measures)
        misfits = [band.misfit for band in measures.fits[0].bands if band.misfit is not None]
        hover = [
            f"xmid {number(x, 4)} m",
            f"{_layers(len(measures.vs_layers))}, modelled to {number(bottom)} m",
            f"informed to {number(useful)} m"
            + (" (its bottom)" if measures.useful_depth_m is None else ""),
        ]
        if misfits:
            hover.append(f"misfit {number(max(misfits), 2)} at worst")
        hover.append(
            " · ".join(
                part
                for part in (
                    f"R-hat {number(rhat, 3)}" if rhat is not None else "",
                    f"ESS {number(ess)}" if ess is not None else "",
                    f"acceptance {number(acceptance)} %" if acceptance is not None else "",
                )
                if part
            )
        )
        verdicts = [f"{one.gate} {one.verdict}" for one in (g5, g6) if one is not None]
        if verdicts:
            hover.append(" · ".join(verdicts))
        flags = [flag.name for one in (g5, g6) if one is not None for flag in one.flags]
        if flags:
            hover.append("flags: " + ", ".join(flag_text(name) for name in dict.fromkeys(flags)))
        cells.append(
            Cell(key=unit, x=x, status=status, hover=tuple(hover), value=useful, total=bottom)
        )
    paco = log is not None and any(attempt.stage == STAGE for attempt in log.attempts)
    summary = f"{len(runs)} of {plural(len(cells), 'window')} inverted" + (
        " by the assistant" if paco else " by hand" if runs else ""
    )
    if again:
        summary += f" ({again} of them again by hand)"
    if runs:
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
        summary += f" · informed {span(min(informed), max(informed), 'm')} deep"
    return Overview(
        paco=paco,
        summary=summary,
        legend={
            "pass": "passed the model checks" if paco else "within the limits",
            "warn": "kept, still flagged" if paco else "a measure beyond its limit",
            "fail": "rejected",
            "none": "not inverted",
        },
        cells=tuple(cells),
        track=Track(label="Depth informed, of the depth modelled (m)", short="Depth", kind="depth"),
        settings=inversion_settings(run_folder, runs, paco),
    )


def inversion_settings(
    run_folder: Path, runs: Sequence[InversionParameters], paco: bool
) -> tuple[Setting, ...]:
    """The parameters the windows were inverted with, as ranges over them, each with why: the
    assistant's rules, or PAC's form."""
    if not runs:
        return ()

    def why(rule: str) -> str:
        if paco:
            return RULES[rule]
        return "set by hand" + (
            "" if (run_folder / PAC_CONFIG_FILE).exists() else " (or by the assistant, earlier)"
        )

    origin = "rule" if paco else "pac"
    free = [run.free for run in runs if run.layering == "free"]
    given = [run for run in runs if run.layering == "fixed"]
    lows = [bound.vs_min or 0.0 for bound in free] + [
        layer.bounds[0] for run in given for layer in run.vs_layers
    ]
    highs = [bound.vs_max or 0.0 for bound in free] + [
        layer.bounds[1] for run in given for layer in run.vs_layers
    ]
    if free and not given:
        layers = ("chosen by the data", f"{_range(bound.max_layers for bound in free)} at most")
    elif given and not free:
        layers = (_range(run.n_layers - 1 for run in given), "over a half-space")
    else:
        layers = ("chosen by the data or given", "by window")
    thinnest = [bound.depth_min or 0.0 for bound in free] + [
        layer.bounds[0] for run in given for layer in run.thickness_layers
    ]
    return (
        Setting(
            key="layers",
            label="Layers",
            value=layers[0],
            detail=layers[1],
            why=why("layers"),
            origin=origin,
        ),
        Setting(
            key="vs",
            label="Vs bounds",
            value=span(min(lows), max(highs), "m/s", 4),
            detail="by window" if len(runs) > 1 else "",
            why=why("vs"),
            origin=origin,
        ),
        Setting(
            key="thickness",
            label="Shallowest interface" if free and not given else "Thinnest layer",
            value=f"{_range(thinnest)} m",
            detail="by window" if len(runs) > 1 else "",
            why=why("thickness"),
            origin=origin,
        ),
        Setting(
            key="depth",
            label="Deepest interface",
            value=f"{_range(model_depth(run) for run in runs)} m",
            detail="at most",
            why=why("depth"),
            origin=origin,
        ),
        Setting(
            key="drop",
            label="Vs drop",
            value=f"{_range(100 * run.max_vs_drop for run in runs)} %",
            detail="at most, from a layer to the next",
            why=why("drop"),
            origin=origin,
        ),
        Setting(
            key="effort",
            label="Sampling",
            value=f"{_range(run.n_iterations for run in runs)} iterations",
            detail=f"{_range(run.n_burnin_iterations for run in runs)} burn-in, "
            f"{plural(max(run.n_chains for run in runs), 'chain')}",
            why=why("effort"),
            origin=origin,
        ),
    )


def inversion_card(folder: str, xmid: float, model: ModelName = "smooth_median") -> InversionCard:
    run_folder = folder_path(folder)
    unit = f"xmid_{xmid:.2f}"
    window = run_folder / unit
    if not window.is_dir():
        raise ValueError(f"No window for folder={folder}, xmid={xmid}")
    thresholds = thresholds_of(run_folder)
    log = read_log(run_folder)
    measured = window_measures(window, thresholds)
    title = f"xmid {number(xmid, 4)} m"
    if measured is None:
        return _not_inverted(unit, title, log)
    ran, measures = measured
    parameters = ran.parameters
    # An inversion PAC made again leaves the assistant's checks and attempts behind: none of them
    # is of the model now in the window.
    ours = by_assistant(log, unit, run_folder)
    g5, g6 = _results(log, unit, run_folder)
    metrics = model_metrics(parameters, measures, thresholds)
    said: list[Sentence] = []
    verdict = verdict_sentence("model", g5, g6)
    said += _depth_sentences(parameters, measures, thresholds)
    said.append(_model_sentence(measures))
    said += _fit_sentences(measures, thresholds)
    said.append(_convergence_sentence(measures, thresholds))
    said.append(_run_sentence(parameters, ran.tuning, ours))
    if log is not None and ours:
        said += _history(log.of(unit, STAGE))
    said += warnings(g5, g6)
    return InversionCard(
        key=unit,
        status=verdict_status(g5, g6) if g5 is not None else measured_status(metrics),
        title=f"{title} · {_layers(len(measures.vs_layers))}",
        verdict=verdict,
        sentences=tuple(said),
        gates=(gate_view(GATE, g5, metrics),) + ((gate_view("G6", g6),) if g6 is not None else ()),
        attempts=_attempts(log, unit) if ours else (),
        inverted=True,
        parameters=parameters,
        tuning=ran.tuning,
        step_factor=step_factor(ran.tuning),
        acceptance=measures.acceptance,
        samples_per_chain=measures.samples_per_chain,
        convergence=_rows(window, parameters, measures),
        fits=measures.fits,
        at_bounds=measures.at_bounds,
        profile=_profile(window, model, parameters, measures),
        curve=_curve(window, model),
        figures=tuple(name for name, file in FIGURES.items() if (window / file).exists()),
    )


def inversion_chains(folder: str, xmid: float) -> Chains:
    window = folder_path(folder) / f"xmid_{xmid:.2f}"
    if not (window / SAMPLES_FILE).exists() or not (window / PARAMETERS_FILE).exists():
        raise ValueError(f"No inversion with its samples for folder={folder}, xmid={xmid}")
    parameters = load_parameters(window / PARAMETERS_FILE).parameters
    measures = _measures_of(window)
    series, n_chains = _series(window, parameters, measures.watched if measures else ())
    return Chains(
        traces=_traces(series, n_chains),
        marginals=_marginals(series, n_chains, parameters),
    )


def _measures_of(window: Path) -> InversionMeasures | None:
    path = window / MEASURES_FILE
    return InversionMeasures.model_validate_json(path.read_text()) if path.exists() else None


def _series(
    window: Path, parameters: InversionParameters, watched: Sequence[str]
) -> tuple[dict[str, np.ndarray], int]:
    """What the chains sampled, by name: Vs at the depths watched, the layers' own values when
    given (a value fixed stays the same: left out), the number of layers, the noise factor."""
    samples, n_chains = load_samples(window / SAMPLES_FILE)
    fixed = parameters.fixed()
    series: dict[str, np.ndarray] = {}
    if watched:
        depths = np.array([_depth_of(name) for name in watched])
        at = load_profiles(window / SAMPLES_FILE).at(depths)
        series = {name: at[:, j] for j, name in enumerate(watched)}
    series |= {name: values for name, values in samples.items() if name not in fixed}
    return series, n_chains


def _depth_of(name: str) -> float:
    """The depth of a series of Vs at a depth: "vs@2.5m" is 2.5."""
    return float(name.removeprefix("vs@").removesuffix("m"))


def figure_path(folder: str, xmid: float, name: FigureName) -> Path:
    path = folder_path(folder) / f"xmid_{xmid:.2f}" / FIGURES[name]
    if not path.exists():
        raise ValueError(f"No {name} figure for folder={folder}, xmid={xmid}")
    return path


def by_assistant(log: QCLog | None, unit: str, run_folder: Path) -> bool:
    """Whether the inversion in window `unit` is the assistant's: it logged one, and PAC did not
    invert the window again after it."""
    latest = log.latest(unit, STAGE) if log is not None else None
    if latest is None:
        return False
    samples = run_folder / unit / SAMPLES_FILE
    done = latest.finished_at or latest.started_at
    return not (samples.exists() and samples.stat().st_mtime > (done + LATER).timestamp())


def _results(
    log: QCLog | None, unit: str, run_folder: Path
) -> tuple[GateResult | None, GateResult | None]:
    """G5's and G6's latest results on the window; none once PAC inverted it again after
    them."""
    if log is None or not by_assistant(log, unit, run_folder):
        return None, None
    return log.result(unit, STAGE, GATE), log.result(unit, STAGE, "G6")


def _not_inverted(unit: str, title: str, log: QCLog | None) -> InversionCard:
    g3 = log.result(unit, "picking", "G3") if log is not None else None
    rejected = g3 is not None and g3.verdict == "reject"
    return InversionCard(
        key=unit,
        status="none",
        title=title,
        sentences=(
            Sentence(
                mark="info",
                text="Not inverted"
                + (": its curve was rejected by the curve check (G3)." if rejected else "."),
            ),
        ),
        inverted=False,
        parameters=None,
        tuning=(),
        step_factor=None,
        acceptance=(),
        samples_per_chain=0,
        convergence=(),
        fits=(),
        at_bounds=(),
        profile=None,
        curve=None,
        figures=(),
    )


def _depth_sentences(
    parameters: InversionParameters, measures: InversionMeasures, thresholds: InversionThresholds
) -> list[Sentence]:
    """How deep the model goes, and how much of it the data inform."""
    bottom = measures.depth_max_m
    useful = measures.useful_depth_m
    enough = thresholds.min_useful_share * model_depth(parameters)
    if useful is None:
        return [
            Sentence(
                mark="pass",
                text=f"The data inform the whole model, down to {number(bottom)} m: the samples "
                "stay tighter than the prior all the way down.",
            )
        ]
    if useful <= 0:
        return [
            Sentence(
                mark="warn",
                text=f"The data inform none of the {number(bottom)} m modelled: the samples "
                "spread as widely as the prior at every depth.",
            )
        ]
    return [
        Sentence(
            mark="pass" if useful >= enough else "warn",
            text=f"The data inform it down to {number(useful)} m of the {number(bottom)} m "
            "modelled: deeper, the samples spread as widely as the prior.",
        )
    ]


def _model_sentence(measures: InversionMeasures) -> Sentence:
    vs = ", ".join(number(value) for value in measures.vs_layers)
    interfaces = ", ".join(number(depth) for depth in measures.interfaces_m)
    return Sentence(
        mark="info",
        text=f"Layered median: Vs {vs} m/s top down"
        + (f", interfaces at {interfaces} m" if interfaces else "")
        + "; the last, a half-space.",
    )


def _fit_sentences(measures: InversionMeasures, thresholds: InversionThresholds) -> list[Sentence]:
    smooth = measures.fits[0]
    misfits = [band.misfit for band in smooth.bands]
    said: list[Sentence] = []
    known = [value for value in misfits if value is not None]
    if known:
        names = BAND_NAMES.get(len(misfits))
        each = ", ".join(f"{value:.2f}" if value is not None else "no mode" for value in misfits)
        where = f" over the {', '.join(names[:-1])} and {names[-1]} wavelengths" if names else ""
        good = all(value is not None and value <= thresholds.max_misfit for value in misfits)
        said.append(
            Sentence(
                mark="pass" if good else "warn",
                text=f"{'Fits' if good else 'Misfits'} its curve: misfit {each}{where} (at most "
                f"{number(thresholds.max_misfit)}, in uncertainties).",
            )
        )
    if smooth.n_missing:
        below = (
            f" below {number(smooth.lowest_missing_hz)} Hz"
            if smooth.lowest_missing_hz is not None
            else ""
        )
        said.append(
            Sentence(
                mark="warn",
                text=f"The model has no fundamental mode at {plural(smooth.n_missing, 'picked point')}"
                f"{below}.",
            )
        )
    return said


def _convergence_sentence(measures: InversionMeasures, thresholds: InversionThresholds) -> Sentence:
    rhat, ess, acceptance = _convergence(measures)
    agree = rhat is not None and rhat <= thresholds.max_rhat
    enough = ess is None or ess >= thresholds.min_ess
    depths = [f"{number(_depth_of(name))} m" for name in measures.watched]
    parts = [
        f"R-hat {number(rhat, 3)} (at most {number(thresholds.max_rhat)})"
        if rhat is not None
        else "",
        f"{number(ess)} effective samples (at least {number(thresholds.min_ess)})"
        if ess is not None
        else "",
        f"acceptance {number(acceptance)} %" if acceptance is not None else "",
    ]
    head = (
        "The chains disagree"
        if not agree
        else "The chains agree"
        if enough
        else "The chains agree, but hold few independent samples"
    )
    at = f" on Vs at {', '.join(depths[:-1])} and {depths[-1]}" if len(depths) > 1 else ""
    return Sentence(
        mark="pass" if agree and enough else "warn",
        text=f"{head}{at}: {', '.join(part for part in parts if part)}.",
    )


def _run_sentence(
    parameters: InversionParameters, tuning: tuple[tuple[float, float], ...], ours: bool
) -> Sentence:
    """What the inversion ran with, and who ran it: the assistant (`ours`), or PAC. The ranges
    are the values sampled; the values fixed are said apart, layer by layer."""
    if parameters.layering == "free":
        free = parameters.free
        parts = [
            f"the layers chosen by the data, {plural(free.max_layers, 'layer')} at most",
            f"Vs {span(free.vs_min or 0, free.vs_max or 0, 'm/s')}",
            f"interfaces {span(free.depth_min or 0, free.depth_max or 0, 'm')} deep",
        ]
    else:
        vs = [one for one in parameters.vs_layers[:-1] if one.vs_fixed is None]
        half_space = parameters.vs_layers[-1]
        thick = [one for one in parameters.thickness_layers if one.thickness_fixed is None]
        ranges = [
            f"Vs {span(min(one.vs_min for one in vs), max(one.vs_max for one in vs), 'm/s')}"
            if vs
            else "",
            f"the half-space's up to {number(half_space.vs_max)} m/s"
            if half_space.vs_fixed is None
            else "",
        ]
        parts = [
            f"{_layers(parameters.n_layers)} given",
            ", ".join(part for part in ranges if part),
            f"layers {span(min(one.thickness_min for one in thick), max(one.thickness_max for one in thick), 'm')} thick"
            if thick
            else "",
        ]
        fixed = _fixed(parameters)
        if fixed:
            parts.append(f"fixed: {fixed}")
    parts.append(
        f"Vs dropping by {number(100 * parameters.max_vs_drop)} % at most from a layer to the next"
    )
    text = (
        ("Ran " if ours else "Inverted by hand: ")
        + f"{plural(parameters.n_chains, 'chain')} of {parameters.n_iterations:,} iterations "
        f"({parameters.n_burnin_iterations:,} burn-in); "
        + "; ".join(part for part in parts if part)
    )
    if tuning:
        text += f"; steps tuned in {plural(len(tuning), 'trial run')}"
    return Sentence(mark="info", text=text + ".")


def _fixed(parameters: InversionParameters) -> str:
    """The values fixed, not sampled, top down: "layer 1's thickness 3 m, layer 2's Vs 450 m/s"."""
    said: list[str] = []
    if parameters.layering != "fixed":
        return ""
    last = len(parameters.vs_layers) - 1
    for i, vs in enumerate(parameters.vs_layers):
        name = "the half-space's" if i == last else f"layer {i + 1}'s"
        thickness = parameters.thickness_layers[i].thickness_fixed if i < last else None
        if thickness is not None:
            said.append(f"{name} thickness {number(thickness)} m")
        if vs.vs_fixed is not None:
            said.append(f"{name} Vs {number(vs.vs_fixed)} m/s")
    return ", ".join(said)


def _history(attempts: tuple[Attempt, ...]) -> list[Sentence]:
    """What each attempt after the first changed, and which flag asked for it."""
    said: list[Sentence] = []
    for before, attempt in pairwise(attempts):
        gate, _, name = attempt.triggered_by.partition(":")
        changed = _changed(before.parameters, attempt.parameters)
        why = f"{gate}: {flag_text(name)}" if name else flag_text(gate)
        said.append(
            Sentence(
                mark="info",
                text=f"Attempt {attempt.attempt}"
                + (f": {changed}" if changed else "")
                + f" ({why}).",
            )
        )
    return said


def _changed(before: dict[str, object], after: dict[str, object]) -> str:
    """What an attempt's parameters change on the one before, in words."""
    try:
        old = InversionParameters.model_validate(before)
        new = InversionParameters.model_validate(after)
    except ValueError:
        return ""
    said: list[str] = []
    if new.layering != old.layering:
        said.append(
            "the layers chosen by the data" if new.layering == "free" else "the layers given"
        )
    elif new.layering == "free" and new.free.max_layers != old.free.max_layers:
        said.append(f"{new.free.max_layers} layers at most (was {old.free.max_layers})")
    elif new.layering == "fixed" and new.n_layers != old.n_layers:
        said.append(f"{_layers(new.n_layers)} (was {old.n_layers - 1})")
    if model_depth(new) != model_depth(old):
        said.append(
            f"the half-space's top at most {number(model_depth(new))} m "
            f"(was {number(model_depth(old))} m)"
        )
    if new.n_iterations != old.n_iterations:
        said.append(f"{new.n_iterations:,} iterations (was {old.n_iterations:,})")
    old_vs, new_vs = _highest_vs(old), _highest_vs(new)
    if new_vs is not None and new_vs != old_vs:
        said.append(f"Vs up to {number(new_vs)} m/s (was {number(old_vs or 0)})")
    if new.layering == "fixed" and old.layering == "fixed":
        old_steps = [layer.vs_perturb_std for layer in old.vs_layers]
        new_steps = [layer.vs_perturb_std for layer in new.vs_layers]
        if len(old_steps) == len(new_steps) and old_steps != new_steps and not said:
            said.append(f"steps scaled by {number(new_steps[0] / old_steps[0], 2)}")
    return ", ".join(said)


def _highest_vs(parameters: InversionParameters) -> float | None:
    """The fastest Vs the priors allow; None when the data's picks set it (not resolved)."""
    if parameters.layering == "free":
        return parameters.free.vs_max
    return max(layer.bounds[1] for layer in parameters.vs_layers)


def _profile(
    window: Path, model: ModelName, parameters: InversionParameters, measures: InversionMeasures
) -> VsProfile | None:
    velocity = window_model(window, model)
    if velocity is None:
        return None
    thicknesses = np.asarray(velocity.thicknesses, dtype=float)
    tops = np.concatenate(([0.0], np.cumsum(thicknesses)[:-1]))
    vs = np.asarray(velocity.vs_s, dtype=float)
    std = np.asarray(velocity.vs_s_std, dtype=float)
    stride = max(1, -(-tops.size // PROFILE_POINTS))
    keep = np.unique(np.concatenate((np.arange(0, tops.size, stride), [tops.size - 1])))
    layered_tops = (0.0, *measures.interfaces_m)
    return VsProfile(
        model=model,
        tops=tuple(round(float(value), 3) for value in tops[keep]),
        vs=tuple(round(float(value), 1) for value in vs[keep]),
        std=tuple(round(float(value), 1) for value in std[keep]),
        layered_tops=tuple(round(value, 3) for value in layered_tops),
        layered_vs=measures.vs_layers,
        bottom=measures.depth_max_m,
        informed=measures.useful_depth_m,
        deepest_top=model_depth(parameters),
    )


def _curve(window: Path, model: ModelName) -> FitCurve | None:
    observed = picked_curve(window)
    if observed is None:
        return None
    return fit_curve(observed, predicted_curve(window, observed, model))


def fit_curve(observed: DispersionCurve, predicted: DispersionCurve | None) -> FitCurve:
    """The picked curve and the one a model gives back, for the fit plot."""
    errors = observed.vs_err
    return FitCurve(
        label=observed.mode.label,
        observed_fs=_rounded(observed.fs),
        observed_vs=_rounded(observed.vs),
        observed_err=_rounded(errors) if errors is not None else (),
        predicted_fs=_rounded(predicted.fs) if predicted is not None else (),
        predicted_vs=_rounded(predicted.vs) if predicted is not None else (),
    )


def _rounded(values: Iterable[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 3) for value in values)


def _convergence(measures: InversionMeasures) -> tuple[float | None, float | None, float | None]:
    """The chains' worst R-hat, fewest effective samples (on the series judged) and median
    acceptance."""
    judged = _judged(measures)
    rhats = [value for name, value in measures.rhat.items() if value is not None and name in judged]
    esses = [value for name, value in measures.ess.items() if value is not None and name in judged]
    acceptance = round(statistics.median(measures.acceptance), 2) if measures.acceptance else None
    return max(rhats, default=None), min(esses, default=None), acceptance


def _judged(measures: InversionMeasures) -> set[str]:
    """The series the chains are judged on: Vs at depths; every one for runs measured before."""
    return set(measures.watched) or set(measures.rhat)


def _layers(n_layers: int) -> str:
    """A model's layers in words: "3 layers over a half-space"."""
    return f"{plural(n_layers - 1, 'layer')} over a half-space"


def _range(values: Iterable[float]) -> str:
    """The values' range in words: "4", or "3-5"."""
    ordered = sorted(float(value) for value in values)
    return span(ordered[0], ordered[-1], digits=4) if ordered else "n/a"


def _attempts(log: QCLog | None, unit: str) -> tuple[InversionAttempt, ...]:
    if log is None:
        return ()
    found: list[InversionAttempt] = []
    for attempt in log.of(unit, STAGE):
        try:
            parameters = InversionParameters.model_validate(attempt.parameters)
        except ValueError:
            parameters = None
        found.append(
            InversionAttempt(
                **summarize(attempt).model_dump(),
                n_layers=(
                    parameters.n_layers
                    if parameters is not None and parameters.layering == "fixed"
                    else None
                ),
                depth_m=model_depth(parameters) if parameters is not None else None,
            )
        )
    return tuple(found)


def _chains(values: np.ndarray, n_chains: int) -> np.ndarray:
    """A parameter's samples, chain after chain, as chains x samples."""
    per_chain = values.size // n_chains
    return np.asarray(values[: per_chain * n_chains], dtype=float).reshape(n_chains, per_chain)


def _traces(samples: dict[str, np.ndarray], n_chains: int) -> tuple[ChainTraces, ...]:
    traces: list[ChainTraces] = []
    for name, values in samples.items():
        chains = _chains(values, n_chains)
        step = max(1, -(-chains.shape[1] // TRACE_POINTS))
        traces.append(
            ChainTraces(
                parameter=name,
                step=step,
                chains=tuple(tuple(round(float(v), 3) for v in chain[::step]) for chain in chains),
            )
        )
    return tuple(sorted(traces, key=lambda one: _order(one.parameter)))


def _rows(
    window: Path, parameters: InversionParameters, measures: InversionMeasures
) -> tuple[Convergence, ...]:
    """Each series' prior, posterior and sampling: Vs at the depths watched, then the layers'
    own values when given (Vs first, top down; a value fixed, as it was), the number of layers,
    the noise factor."""
    series, _ = _series(window, parameters, measures.watched)
    priors = _priors(parameters, series)
    fixed = parameters.fixed()
    rows: list[Convergence] = []
    for name in sorted(set(series) | set(fixed), key=_order):
        if name in fixed:
            rows.append(
                Convergence(
                    parameter=name,
                    prior=None,
                    median=fixed[name],
                    low=None,
                    high=None,
                    rhat=None,
                    ess=None,
                    autocorrelation=None,
                    step=None,
                    fixed=fixed[name],
                )
            )
            continue
        values = np.asarray(series.get(name, ()), dtype=float)
        low, median, high = (
            (float(value) for value in np.percentile(values, [10, 50, 90]))
            if values.size
            else (None, None, None)
        )
        rows.append(
            Convergence(
                parameter=name,
                prior=priors.get(name),
                median=_round(median),
                low=_round(low),
                high=_round(high),
                rhat=measures.rhat.get(name),
                ess=measures.ess.get(name),
                autocorrelation=measures.autocorrelation.get(name),
                step=measures.steps.get(name),
            )
        )
    return tuple(rows)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _priors(
    parameters: InversionParameters, series: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """Each series' prior bounds, by name: a layer's own; Vs at a depth, the widest any layer
    allows; the number of layers, 1 to the most allowed; the noise factor's. A value fixed has
    none."""
    if parameters.layering == "free":
        free = parameters.free
        vs_range = (free.vs_min or 0.0, free.vs_max or 0.0)
        layers = (1.0, float(free.max_layers))
    else:
        vs_range = (
            min(layer.vs_min for layer in parameters.vs_layers),
            max(layer.vs_max for layer in parameters.vs_layers),
        )
        layers = (float(parameters.n_layers), float(parameters.n_layers))
    priors = {
        f"vs{i + 1}": (layer.vs_min, layer.vs_max) for i, layer in enumerate(parameters.vs_layers)
    } | {
        f"thick{i + 1}": (layer.thickness_min, layer.thickness_max)
        for i, layer in enumerate(parameters.thickness_layers)
    }
    fixed = parameters.fixed()
    found: dict[str, tuple[float, float]] = {}
    for name in series:
        if name.startswith("vs@"):
            found[name] = vs_range
        elif name == "layers":
            found[name] = layers
        elif name == "noise":
            found[name] = NOISE_BOUNDS
        elif parameters.layering == "fixed" and name in priors and name not in fixed:
            found[name] = priors[name]
    return found


def _marginals(
    series: dict[str, np.ndarray], n_chains: int, parameters: InversionParameters
) -> tuple[Marginal, ...]:
    marginals: list[Marginal] = []
    for name, (low, high) in _priors(parameters, series).items():
        if high <= low:
            continue
        chains = _chains(series[name], n_chains)
        marginals.append(
            Marginal(
                parameter=name,
                low=low,
                high=high,
                counts=tuple(_histogram(chain, low, high) for chain in chains),
            )
        )
    return tuple(sorted(marginals, key=lambda one: _order(one.parameter)))


def _histogram(values: np.ndarray, low: float, high: float) -> tuple[int, ...]:
    counts: np.ndarray = np.histogram(values, bins=BINS, range=(low, high))[0]
    return tuple(int(count) for count in counts)


def _order(parameter: str) -> tuple[int, float]:
    """Vs at depths from the top, the layers' Vs then thicknesses from the top, the number of
    layers, the noise factor."""
    if parameter.startswith("vs@"):
        return 0, _depth_of(parameter)
    if parameter.startswith("vs"):
        return 1, float(parameter.removeprefix("vs") or 0)
    if parameter.startswith("thick"):
        return 2, float(parameter.removeprefix("thick") or 0)
    return (3, 0.0) if parameter == "layers" else (4, 0.0)
