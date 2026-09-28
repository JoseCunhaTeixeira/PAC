"""What Visualization shows of every stage alike: one summary line and the settings the stage ran
with (each with why), one strip of colored cells along the line (a window or a shot each, its
hover and its stage's one telling measure), and the selected unit's card: its state, a few
sentences each with its badge, and folded under them each gate's metrics and the attempts. The
stage's modules fill them; the page draws them all alike."""

from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from masw.io.quality.log import Flag, GateResult, Metric, Verdict

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


class Track(BaseModel):
    """The strip's one measure along the line, drawn under the cells."""

    model_config = ConfigDict(frozen=True)

    label: str
    short: str  # the label beside the strip: a word or two
    kind: Literal["value", "depth"]  # dots against a limit, or bars down from the surface
    limit: float | None = None
    bound: Literal["min", "max"] | None = None


class Overview(BaseModel):
    """A stage's units along the line."""

    model_config = ConfigDict(frozen=True)

    paco: bool  # a run the assistant judged: the cells are its gates' verdicts
    summary: str
    legend: dict[Status, str]  # what each color says here
    cells: tuple[Cell, ...]
    parts: tuple[PartLegend, ...] = ()  # the cells' parts, when they say their checks apart
    track: Track | None = None
    settings: tuple[Setting, ...] = ()  # what the stage ran with, and why


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
    """The gate's verdict and metrics; without a result, the unit's own measures."""
    if result is not None:
        return GateView(gate=gate, verdict=result.verdict, metrics=result.metrics)
    return GateView(gate=gate, verdict=None, metrics=tuple(measured))


# The flags whose names do not say what they found.
FLAG_WORDS = {
    "rms_outliers": "amplitude off the offset decay",
    "low_snr": "low SNR",
    "budget_spent": "retries spent",
}


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
        reason = next((flag for flag in flags if flag.name != "budget_spent"), None)
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
