"""What Visualization shows of every stage alike: one summary line and the settings the stage ran
with (each with why), one strip of colored cells along the line (a window or a shot each, its
hover and its stage's one telling measure), and the selected unit's card: its state, a few
sentences each with its badge, and folded under them each gate's metrics and the attempts. The
stage's modules fill them; the page draws them all alike."""

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from masw.io.quality.log import LINE, Flag, GateResult, Metric, QCLog, Verdict

# A unit's state on the strip: passed, warned or retried, failed or rejected, not there.
type Status = Literal["pass", "warn", "fail", "none"]
# A sentence's badge; "info" says what was done, without judging it.
type Mark = Literal["pass", "warn", "fail", "info"]
# Where a setting comes from: a rule on the data (the assistant's), the request, the preset's
# default, or PAC's form.
type Origin = Literal["rule", "given", "default", "pac"]

STATUS_OF: dict[Verdict, Status] = {"pass": "pass", "retry": "warn", "reject": "fail"}
DASH = "\u2013"  # an en dash, between a range's ends
RANK: dict[Status, int] = {"none": 0, "pass": 1, "warn": 2, "fail": 3}


class Setting(BaseModel):
    """A value a stage ran with, and why."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    value: str  # what to read first: "6 m"
    detail: str = ""  # what goes with it: "5 receivers"
    why: str
    origin: Origin
    # A unit's, how the line's other units differ in it: their values and on which, or its range
    # over them all; "" when they all ran with the same (quality.done's with_spreads).
    spread: str = ""


# A check said apart from the others of its unit (the dispersion's: its image's, its curve's):
# its state, or "hand" (made by hand: the user's, which no check judges).
type PartState = Literal["pass", "warn", "fail", "none", "hand"]


class Part(BaseModel):
    """One of a unit's checks said apart: what it checks, and its state."""

    model_config = ConfigDict(frozen=True)

    label: str  # "image", "curve"
    state: PartState


class PartLegend(BaseModel):
    """What one of the cells' parts says: what it checks, and each of its states present."""

    model_config = ConfigDict(frozen=True)

    title: str
    legend: dict[PartState, str]


class Cell(BaseModel):
    """One unit on the strip: a window at its middle, or a record at its shot."""

    model_config = ConfigDict(frozen=True)

    key: str  # the window's folder (xmid_<x>) or the record's file name
    x: float | None  # m along the line; None: a record without a source (passive)
    status: Status
    hover: tuple[str, ...]  # a few key numbers, in words
    value: float | None = None  # the stage's telling measure
    total: float | None = None  # what the measure is read against (the model's depth)
    # Its checks said apart, top down, in the order of the overview's parts; none: `status`.
    parts: tuple[PartState, ...] = ()
    # The modes picked in it (a window, the dispersion stage), by number: M0, M1...
    modes: tuple[str, ...] = ()


class Track(BaseModel):
    """The strip's one measure along the line, drawn under the cells."""

    model_config = ConfigDict(frozen=True)

    label: str
    short: str  # the label beside the strip: a word or two
    kind: Literal["value", "depth"]  # dots against a limit, or bars down from the surface
    limit: float | None = None
    bound: Literal["min", "max"] | None = None


class Sentence(BaseModel):
    """One line of a card."""

    model_config = ConfigDict(frozen=True)

    mark: Mark
    text: str
    detail: str = ""  # the whole of it, on hover, when the text is its short form


class GateView(BaseModel):
    """What one gate said of the unit, or for a run PAC made, the same measures against the
    gate's limits (verdict None). `by_hand`: the unit's curve was picked by hand, which the
    curve's check passes as it is (verdict pass) and the checks after it leave out (verdict
    None)."""

    model_config = ConfigDict(frozen=True)

    gate: str
    verdict: Verdict | None
    metrics: tuple[Metric, ...]
    by_hand: bool = False


class Overview(BaseModel):
    """A stage's units along the line."""

    model_config = ConfigDict(frozen=True)

    paco: bool  # a run the assistant judged: the cells are its gates' verdicts
    summary: str
    legend: dict[Status, str]  # what each color says here
    cells: tuple[Cell, ...]
    parts: tuple[PartLegend, ...] = ()  # the cells' parts, when they say their checks apart
    track: Track | None = None
    # The line check's measures of the whole line (G1's of its receivers, G4's of its curves,
    # G6's of its models, G8's of its soil columns), for a run the assistant judged.
    gates: tuple[GateView, ...] = ()


class Card(BaseModel):
    """The selected unit's card: its state, a title, a few sentences, and folded under them each
    gate's metrics; each stage's card adds its attempts."""

    model_config = ConfigDict(frozen=True)

    key: str
    status: Status
    title: str
    verdict: Sentence | None = None  # what the assistant's checks concluded; None: none judged
    sentences: tuple[Sentence, ...]
    gates: tuple[GateView, ...] = ()
    parts: tuple[Part, ...] = ()  # its checks said apart, as its cell's parts: for its badges
    # What was done to it, as its "Settings, and why": each step with the settings it ran with,
    # where they come from and their history (quality.done).
    settings: tuple[Setting, ...] = ()


def worst(*statuses: Status) -> Status:
    ranked = sorted(statuses, key=lambda status: RANK[status])
    return ranked[-1] if ranked else "none"


def verdict_status(*results: GateResult | None) -> Status:
    """The worst of the gates' verdicts; none when no gate judged."""
    statuses: list[Status] = [STATUS_OF[result.verdict] for result in results if result is not None]
    return worst(*statuses)


def measured_status(metrics: Iterable[Metric]) -> Status:
    """A unit no gate judged, by its measures against the gates' limits: pass, or warn when one
    is beyond its limit; none without a measure judged."""
    judged = [metric for metric in metrics if metric.threshold is not None]
    if not judged:
        return "none"
    return "pass" if all(metric.passed for metric in judged) else "warn"


def gate_view(gate: str, result: GateResult | None, measured: Iterable[Metric] = ()) -> GateView:
    """The gate's verdict and metrics; without a result, the unit's own measures; each saying
    what it describes and what it covers (said generally for a log from before they did)."""
    metrics = result.metrics if result is not None else tuple(measured)
    return GateView(
        gate=gate,
        verdict=result.verdict if result is not None else None,
        metrics=tuple(covered(gate, metric) for metric in metrics),
    )


def line_gate(log: QCLog | None, stage: str, gate: str) -> tuple[GateView, ...]:
    """What `gate` said of the whole line at `stage`, its latest check; none when the assistant
    did not check the line."""
    result = log.result(LINE, stage, gate) if log is not None else None
    return (gate_view(gate, result),) if result is not None else ()


# A line check's measures of the line, not of one unit against its neighbours.
_LINE = frozenset(
    {
        "curves",
        "without_curve",
        "depth_spread",
        "inverse_curves",
        "models",
        "without_model",
        "depth_informed_spread",
        "useful_depth_spread",
        "water_table_min",
        "water_table_max",
        "off_decay_receivers",
        "spectral_receivers",
    }
)
_CHAINS = frozenset(
    {"rhat", "ess", "autocorrelation", "acceptance", "samples_per_chain", "at_bound"}
)
_SPECTRUM = frozenset({"usable_band_hz", "spectral_outliers"})
_CORRELATIONS = frozenset(
    {"snr_db", "lateral_coherence", "dead_traces", "clipped_traces", "nan_traces",
     "virtual_shot_snr_db"}
)  # fmt: skip
# What each measure covers, said generally: a log from before the measures said it.
_COVERS = {
    "dead_traces": "the record's traces",
    "clipped_traces": "the record's traces",
    "nan_traces": "the record's traces",
    "rms_outliers": "its live traces: the amplitude's decay with offset",
    "snr_db": "its traces within the line's reach: the surface-wave window against the noise's",
    "usable_band_hz": "its traces within the line's reach: 6 dB over the noise",
    "lateral_coherence": "neighbouring pairs of traces, in the surface-wave window",
    "pulse_s": "the traces nearest the shot",
    "trigger_shift_s": "the first breaks",
    "trigger_error_s": "the first breaks nearest the shot",
    "trigger_scatter_s": "the first breaks",
    "energy_removed": "the record, before and after its mute",
    "spectral_outliers": "each trace against its neighbours' spectra",
    "coherent_columns": "the image's columns",
    "ridge_at_vmin": "the coherent columns: a peak at the grid's lowest velocity",
    "ridge_at_vmax": "the coherent columns: a peak at the grid's highest velocity",
    "band_at_fmin": "the coherent band against the image's lowest frequency",
    "band_at_fmax": "the coherent band against the image's highest frequency",
    "competing_ridges": "the coherent columns: a second ridge",
    "aliased_ridges": "the columns with a second ridge: under 2 dx f",
    "band_share_of_usable": "the coherent band over the records' usable band",
    "virtual_shot_snr_db": "the stacked correlations",
    "fk_segments": "the window's segments",
    "fk_kept": "the window's segments",
    "fk_flipped": "the window's segments",
    "sharpness": "the pick's points on its image",
    "prominence": "the pick's points on its image",
    "on_data": "the pick's points on its image",
    "constant_wavelength": "the pick's points on its image",
    "n_points": "the pick's points kept on its ridge",
    "aliased_points": "the curve's points: under twice the receiver spacing",
    "beyond_reach_points": "the curve's points: over three window lengths",
    "curve_points": "the pick resampled by wavelength",
    "wavelength_ratio": "the curve's wavelengths",
    "max_jump": "the curve's points: each step to the next by wavelength",
    "air_wave_share": "the curve's points: at the air wave's speed",
    "trend": "the curve's points: velocity against wavelength",
    "uncertainty": "the curve's points with one",
    "near_offset": "the window's nearest shot",
    "misfit": "its neighbours on each side",
    "neighbour_misfit": "its neighbours on each side",
    "sides_compared": "its sides",
    "water_table_jump": "its neighbours' water tables",
    "misfit_layered": "every picked point, against the layered median's curve",
    "rhat": "the models' Vs at the depths watched, over the chains",
    "ess": "the models' Vs at the depths watched, over the chains",
    "autocorrelation": "the models' Vs at the depths watched, over the chains",
    "acceptance": "the chains' moves",
    "samples_per_chain": "each chain, after its burn-in",
    "at_bound": "each parameter's samples at a prior's bound",
    "depth_informed": "the models' Vs spread, from the surface down",
    "useful_depth": "the models' Vs spread, from the surface down",
    "contrast": "the layered median's adjacent layers",
    "water_table": "the soil column",
    # The line checks'.
    "off_decay_receivers": "the line's receivers: off the amplitude decay in most of the "
    "records reaching them",
    "spectral_receivers": "the line's receivers: off their neighbours' spectra in most of the "
    "records reaching them",
    "curves": "the line's windows: a curve G3 passed",
    "without_curve": "the line's windows",
    "depth_spread": "the curves' longest wavelengths: MAD over median",
    "inverse_curves": "the curves: velocity falling with wavelength",
    "models": "the line's windows: a model G5 passed",
    "without_model": "the line's windows",
    "depth_informed_spread": "the models' depths informed: MAD over median",
    "useful_depth_spread": "the models' depths of investigation: MAD over median",
    "water_table_min": "the soil columns",
    "water_table_max": "the soil columns",
}
# ...where a measure's name is another gate's too.
_GATE_COVERS = {("G8", "models"): "the line's windows: a soil column G7 passed"}


def covered(gate: str, metric: Metric) -> Metric:
    """`metric` saying what it describes and what it covers: as measured, or, from a log older
    than that, the thing each gate measures and what the measure covers in general."""
    if metric.of and metric.over:
        return metric
    name = metric.name
    if name in _LINE:
        of = "line"
    elif gate in ("G4", "G6", "G8"):
        of = "neighbours"
    elif gate == "G5":
        of = (
            "chains"
            if name in _CHAINS
            else "fit"
            if name.startswith(("misfit", "residual"))
            else "model"
        )
    elif gate == "G7":
        of = "soil" if name == "water_table" else "fit"
    elif gate == "G3":
        of = "curve"
    elif gate == "G2":
        of = (
            "selection"
            if name.startswith("fk_")
            else "spectrum"
            if name in _SPECTRUM
            else "signal"
            if name in _CORRELATIONS
            else "image"
        )
    else:
        of = "spectrum" if name in _SPECTRUM else "signal"
    fit = (gate == "G7" and "soil column's") or "model's"
    over = (
        _GATE_COVERS.get((gate, name))
        or _COVERS.get(name)
        or (
            f"the band's picked points, against the {fit} curve"
            if name.startswith(("misfit_", "residual_"))
            else ""
        )
    )
    return metric.model_copy(update={"of": metric.of or of, "over": metric.over or over})


# The flags whose names do not say what they found.
FLAG_WORDS = {
    "rms_outliers": "amplitude off the offset decay",
    "low_snr": "low SNR",
    "no_usable_band": "no usable band",
    "budget_spent": "retries spent",
    "nothing_to_try": "nothing left to try",
    "redone_once": "redone once, still short",
    "uneven_depth_informed": "uneven depth informed",
}
# Why a gate refused a unit the retry it asked (PACo's budgets.budget_spent): said with the flags
# that asked it, never as the unit's reason.
REFUSALS = frozenset({"budget_spent", "nothing_to_try", "redone_once"})


def flag_text(name: str) -> str:
    return FLAG_WORDS.get(name, name.replace("_", " "))


def first_sentence(text: str) -> str:
    """The first sentence of a flag's message: what it found, without the advice."""
    head, dot, _ = text.partition(". ")
    return head + "." if dot else text


def finding(text: str) -> str:
    """What a flag's message found, in short: its first sentence up to its colon (or
    semicolon), where the assistant's messages go on with why it matters ("The nearest shot is
    0.75 m from the window, under half the longest wavelength kept (10.00 m): the long
    wavelengths...")."""
    head = first_sentence(text)
    for mark in (": ", "; "):
        found, cut, _ = head.partition(mark)
        if cut:
            return found.rstrip(".") + "."
    return head


def counted(names: Iterable[str], most: int = 4) -> str:
    """Names with their counts, the most frequent first: "40 competing ridges, 2 no ridge"."""
    counts = Counter(names).most_common()
    said = [
        f"{count} {flag_text(name)}" if count > 1 else flag_text(name)
        for name, count in counts[:most]
    ]
    return ", ".join(said) + (", …" if len(counts) > most else "")


def rejecting(flag: Flag) -> bool:
    """A flag that rejects its unit: nothing fixes it, or it says so."""
    return flag.action.kind == "reject" or not flag.fixable


def kept(flag: Flag) -> bool:
    """A flag that is information, kept with the result: a warning."""
    return flag.action.kind == "keep"


def warnings(*results: GateResult | None) -> tuple[Sentence, ...]:
    """The flags the gates kept with the result (reported, not acted on), as warnings."""
    return tuple(
        Sentence(mark="warn", text=finding(flag.message), detail=flag.message)
        for result in results
        if result is not None
        for flag in result.flags
        if kept(flag)
    )


def verdict_sentence(unit: str, *results: GateResult | None) -> Sentence | None:
    """What the gates concluded of the `unit` (window, record, model), in one sentence; None
    when none judged."""
    judged = [result for result in results if result is not None]
    if not judged:
        return None
    status = verdict_status(*judged)
    if status == "pass":
        gates = ", ".join(result.gate for result in judged)
        return Sentence(mark="pass", text=f"The assistant's checks ({gates}) passed this {unit}.")
    if status == "fail":
        rejected = [result for result in judged if result.verdict == "reject"]
        flags = [flag for result in rejected for flag in result.flags]
        reason = next((flag for flag in flags if flag.name not in REFUSALS), None)
        why = f": {finding(reason.message)}" if reason is not None else "."
        return Sentence(
            mark="fail",
            text=f"Rejected by {', '.join(one.gate for one in rejected)}{why}",
            detail=reason.message if reason is not None else "",
        )
    raised = [flag.name for result in judged for flag in result.flags]
    return Sentence(
        mark="warn",
        text=f"Kept, still flagged once the retries were spent: {counted(raised)}.",
    )


def number(value: float, digits: int = 3) -> str:
    """`value` with about `digits` significant digits, thousands grouped: 1,000; 0.52; 27.7."""
    if value == 0:
        return "0"
    if abs(value) >= 10 ** (digits - 1):
        return f"{round(value):,}"
    text = f"{value:.{digits}g}"
    return text if "e" not in text else f"{value:.2f}"


def span(low: float, high: float, unit: str = "", digits: int = 3) -> str:
    """A range in words, its ends joined by an en dash: "12-60 Hz"; one value when both ends
    agree."""
    said = number(low, digits)
    if number(high, digits) != said:
        said += f"{DASH}{number(high, digits)}"
    return f"{said} {unit}".rstrip()


def plural(count: int, word: str, many: str | None = None) -> str:
    """ "1 window", "3 windows"."""
    return f"{count:,} {word if count == 1 else many or word + 's'}"
