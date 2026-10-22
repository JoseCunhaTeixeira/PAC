"""What Visualization shows of every stage alike: one summary line and the settings the stage ran
with (each with why), one strip of colored cells along the line (a window or a shot each, its
hover and its stage's one telling measure), and the selected unit's card: its state, a few
sentences each with its badge, and folded under them each gate's metrics and the attempts. The
stage's modules fill them; the page draws them all alike."""

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from masw.io.quality.log import LINE, Flag, GateResult, Metric, QCLog, Verdict

# A unit's state on the strip: passed, warned or retried, failed or rejected, not there.
type Status = Literal["pass", "warn", "fail", "none"]
# A sentence's badge; "info" says what was done, without judging it.
type Mark = Literal["pass", "warn", "fail", "info"]
# Where a setting comes from: the user (their request, or PAC's form), PACo's workflow (its rules
# on the data), its autonomous agent (its choices), PACo's checks (what they left out or made
# again), or the preset's default.
type Origin = Literal["you", "workflow", "agent", "checks", "default"]

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


class Step(BaseModel):
    """A step of what was done, in order: a title and its facts, a line each; by whom. A run's,
    from the study's journal (the autonomous mode's) or from what the run's files say was done;
    a unit's, from its attempts."""

    model_config = ConfigDict(frozen=True)

    kind: str
    by: Origin
    at: datetime | None = None
    title: str
    lines: tuple[str, ...] = ()
    detail: str = ""  # more on hover


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
    None). `rejected_by`: the measures that made it reject the unit; the others beyond their
    limits only flag it."""

    model_config = ConfigDict(frozen=True)

    gate: str
    verdict: Verdict | None
    metrics: tuple[Metric, ...]
    by_hand: bool = False
    rejected_by: tuple[str, ...] = ()


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
    # What was done to it, as its "Settings": each step with the settings it ran with, where
    # they come from, and how the line's other units differ in them (quality.done).
    settings: tuple[Setting, ...] = ()
    # Its attempts, a step each, when one ran again (quality.done's attempt_steps); none when
    # each stage ran once.
    steps: tuple[Step, ...] = ()


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
    """The gate's verdict and metrics, and those that made it reject the unit; without a result,
    the unit's own measures; each saying what it describes (said for a log from before they
    did)."""
    metrics = result.metrics if result is not None else tuple(measured)
    return GateView(
        gate=gate,
        verdict=result.verdict if result is not None else None,
        metrics=tuple(covered(gate, metric) for metric in metrics),
        rejected_by=rejected_by(result),
    )


# The measures a flag judges, where they are named apart: a flag named as its measure judges it.
FLAG_METRICS: dict[str, tuple[str, ...]] = {
    "low_snr": ("snr_db",),
    "low_coherence": ("lateral_coherence",),
    "no_usable_band": ("usable_band_hz",),
    "inconsistent_trigger": ("trigger_scatter_s",),
    "shifted_trigger": ("trigger_error_s", "trigger_shift_s"),
    "no_coherent_energy": ("coherent_columns",),
    "weak_coherence": ("coherent_columns",),
    "noisy_correlations": ("snr_db", "usable_band_hz", "lateral_coherence"),
    "few_segments_kept": ("fk_kept",),
    "aliasing": ("competing_ridges", "aliased_ridges"),
    "narrower_than_usable": ("band_share_of_usable",),
    "air_wave": ("air_wave_share",),
    "too_few_points": ("curve_points", "n_points"),
    "no_ridge": ("n_points",),
    "mode_jump": ("max_jump",),
    "narrow_span": ("wavelength_ratio",),
    "aliasing_zone": ("aliased_points",),
    "beyond_reach": ("beyond_reach_points",),
    "inverse_dispersion": ("trend",),
    "near_field": ("near_offset",),
    "outlier": ("neighbour_misfit", "misfit"),
    "shared_change": ("neighbour_misfit", "misfit"),
    "model_change": ("neighbour_misfit", "misfit"),
    "non_unique": ("neighbour_misfit", "misfit"),
    "no_neighbours": ("sides_compared",),
    "petro_change": ("neighbour_misfit", "misfit", "water_table_jump"),
    "petro_outlier": ("neighbour_misfit", "misfit", "water_table_jump"),
    "not_converged": ("rhat", "ess", "samples_per_chain"),
    "few_samples": ("ess", "samples_per_chain"),
    "multimodal": ("rhat",),
    "uninformed": ("depth_informed", "useful_depth"),
    "too_deep": ("depth_informed", "useful_depth"),
    "underfit": ("misfit_short", "misfit_middle", "misfit_long", "misfit_layered"),
    "smoothing_misfit": ("misfit_short", "misfit_middle", "misfit_long"),
    "no_mode": ("misfit_layered",),
    "misfit": ("misfit_short", "misfit_middle", "misfit_long"),
}


def rejected_by(result: GateResult | None) -> tuple[str, ...]:
    """The measures beyond their limits that made `result`'s check reject its unit: those the
    flags it acted on judge (a flag kept is information, traces left out leave the unit, a
    refused retry is said with the flags that asked it); every one beyond its limit for a flag
    that judges none of them. None unless it rejected the unit."""
    if result is None or result.verdict != "reject":
        return ()
    beyond = [
        metric.name
        for metric in result.metrics
        if metric.threshold is not None and not metric.passed
    ]
    names: list[str] = []
    for flag in result.flags:
        if flag.action.kind in ("keep", "exclude_traces") or flag.name in REFUSALS:
            continue
        judged = FLAG_METRICS.get(flag.name, (flag.name,))
        names += [name for name in beyond if name in judged] or beyond
    return tuple(dict.fromkeys(names))


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
        "shallow_models",
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


def covered(gate: str, metric: Metric) -> Metric:
    """`metric` saying what it describes (its signal, its spectrum, the line): as measured, or,
    from a log older than that, the thing each gate measures."""
    if metric.of:
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
    return metric.model_copy(update={"of": of})


# The checks' flags in a few plain words: a unit's flags listed ("flagged: …"), a cell's hover.
FLAG_WORDS = {
    # A record's signal
    "dead_traces": "dead traces",
    "clipped_traces": "clipped traces",
    "nan_traces": "traces with NaN samples",
    "record_too_short": "record too short",
    "no_usable_band": "no usable band",
    "low_snr": "low SNR",
    "low_coherence": "neighbouring traces unalike",
    "inconsistent_trigger": "first breaks scattered",
    "shifted_trigger": "trigger off",
    "fast_arrivals": "energy faster than the surface waves",
    "rms_outliers": "amplitude off the decay with distance",
    "spectral_receivers": "receivers with an unusual spectrum",
    "off_decay_receivers": "receivers too weak or too loud",
    # A dispersion image
    "bad_correlations": "bad correlations",
    "no_coherent_energy": "no coherent energy",
    "weak_coherence": "weak coherence",
    "aliasing": "aliased ridge",
    "competing_ridges": "second ridge",
    "narrower_than_usable": "coherent band narrower than the usable band",
    "band_at_fmin": "coherent band at the lowest frequency",
    "band_at_fmax": "coherent band at the highest frequency",
    # A curve
    "aliasing_zone": "points in the aliasing zone",
    "beyond_reach": "points beyond the window's reach",
    "narrow_span": "narrow range of wavelengths",
    "air_wave": "points at the air wave's speed",
    "inverse_dispersion": "inverse dispersion",
    "near_field": "near field",
    "mode_jump": "jump to another mode",
    "too_few_points": "too few points",
    "no_ridge": "no ridge",
    "sharpness": "broad ridge",
    "prominence": "faint ridge",
    "on_data": "points off the ridge's top",
    "constant_wavelength": "points at constant wavelength",
    # Along the line
    "gaps": "gaps along the line",
    "uneven_depth": "uneven λmax along the line",
    "no_neighbours": "no neighbours to compare with",
    "outlier": "unlike its neighbours",
    "shared_change": "unlike its neighbours on one side",
    "inverse_trend": "inverse trends along the line",
    "model_change": "unlike its neighbours",
    "non_unique": "unlike its neighbours, fitting as well",
    "shallow": "informed too shallow to compare",
    "uneven_depth_informed": "uneven depth informed",
    "petro_change": "unlike its neighbours",
    "petro_outlier": "unlike its neighbours",
    # A model, a soil column
    "not_converged": "chains not converged",
    "few_samples": "few independent samples",
    "multimodal": "chains disagree",
    "uninformed": "no depth informed",
    "smoothing_misfit": "smooth model misfits",
    "underfit": "model misfits the curve",
    "acceptance": "moves accepted out of range",
    "no_mode": "points without a mode",
    "deep_interface": "interfaces at the deepest allowed",
    "most_layers": "most layers allowed",
    "too_deep": "depth beyond what the data inform",
    "strong_inversion": "strong velocity inversion",
    "implausible_vs": "implausible Vs",
    "at_bound": "samples at a bound",
    "misfit": "misfits the curve",
    "no_uncertainty": "no uncertainty",
    # Why a check refused a retry
    "budget_spent": "retries spent",
    "nothing_to_try": "nothing left to try",
    "redone_once": "redone once, still short",
}
# The checks, as a sentence names them.
CHECK_NAMES = {
    "G1": "the signal check",
    "G2": "the dispersion image check",
    "G3": "the curve check",
    "G4": "the check against the neighbouring curves",
    "G5": "the model check",
    "G6": "the check against the neighbouring models",
    "G7": "the soil column check",
    "G8": "the check against the neighbouring soil columns",
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


def kept(flag: Flag) -> bool:
    """A flag that is information, kept with the result: a warning."""
    return flag.action.kind == "keep"


def warnings(*results: GateResult | None) -> tuple[Sentence, ...]:
    """The flags the gates kept with the result (reported, not acted on), as warnings: what each
    found."""
    return tuple(
        Sentence(mark="warn", text=finding(flag.message))
        for result in results
        if result is not None
        for flag in result.flags
        if kept(flag)
    )


# A check's verdict in a word.
VERDICT_WORDS: dict[Verdict, str] = {"pass": "passed", "retry": "flagged", "reject": "rejected"}


def verdicts_said(*results: GateResult | None) -> str:
    """The checks' verdicts in words: "The model check passed, the check against the
    neighbouring models flagged"; "" when none judged."""
    said = ", ".join(
        f"{CHECK_NAMES.get(result.gate, result.gate)} {VERDICT_WORDS[result.verdict]}"
        for result in results
        if result is not None
    )
    return said[:1].upper() + said[1:]


def checks_named(results: Iterable[GateResult]) -> str:
    """The checks that gave `results`, in words: "the signal check", "the curve check and the
    check against the neighbouring curves"."""
    names = list(dict.fromkeys(CHECK_NAMES.get(result.gate, result.gate) for result in results))
    return " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]


def after_colon(text: str) -> str:
    """A sentence said after a colon: its first letter in lower case, unless its first word is
    an acronym (SNR)."""
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def verdict_sentence(*results: GateResult | None) -> Sentence | None:
    """What the checks concluded of a unit (a record, a window, a model), in one sentence; None
    when none judged."""
    judged = [result for result in results if result is not None]
    if not judged:
        return None
    status = verdict_status(*judged)
    if status == "pass":
        return Sentence(mark="pass", text=f"Passed {checks_named(judged)}.")
    if status == "fail":
        rejected = [result for result in judged if result.verdict == "reject"]
        flags = [flag for result in rejected for flag in result.flags]
        reason = next((flag for flag in flags if flag.name not in REFUSALS), None)
        why = f": {after_colon(finding(reason.message))}" if reason is not None else "."
        return Sentence(mark="fail", text=f"Rejected by {checks_named(rejected)}{why}")
    flagged = [result for result in judged if result.verdict == "retry"]
    raised = [flag.name for result in flagged for flag in result.flags]
    return Sentence(mark="warn", text=f"Flagged by {checks_named(flagged)}: {counted(raised)}.")


# The thousands apart as SI writes them: a narrow space from five digits on, none in four.
THOUSANDS = "\u202f"


def grouped(value: int) -> str:
    """A whole number as SI writes it: 1000, 12 500."""
    text = f"{value:,}"
    return text.replace(",", THOUSANDS if abs(value) >= 10_000 else "")


def number(value: float, digits: int = 3) -> str:
    """`value` with about `digits` significant digits, its thousands as SI writes them: 1000;
    12 500; 0.52; 27.7."""
    if value == 0:
        return "0"
    if abs(value) >= 10 ** (digits - 1):
        return grouped(round(value))
    text = f"{value:.{digits}g}"
    return text if "e" not in text else f"{value:.2f}"


def span(low: float, high: float, unit: str = "", digits: int = 3) -> str:
    """A range in words, its ends joined by an en dash: "12-60 Hz"; one value when both ends
    agree."""
    said = number(low, digits)
    if number(high, digits) != said:
        said += f"{DASH}{number(high, digits)}"
    return f"{said} {unit}".rstrip()


def receivers(count: int, spacing_m: float, length: bool = True) -> str:
    """Receivers with what they span: "21 receivers (30 m)" for a window's length (`length`: the
    first to the last), "1 receiver (1.5 m)" for a step."""
    metres = (count - 1) * spacing_m if length else count * spacing_m
    return f"{plural(count, 'receiver')} ({number(metres, 4)} m)"


def plural(count: int, word: str, many: str | None = None) -> str:
    """ "1 window", "3 windows"."""
    return f"{grouped(count)} {word if count == 1 else many or word + 's'}"
