"""PACo's records in a run folder, read as plain JSON: PACo is PAC's optional extra, and a run it
judged is read without it. The QC log (qc_log.jsonl: one line per attempt of a stage on a unit, a
record, a window or the line, the last line of an attempt its current state), the thresholds its
gates used (qc_config.json), the settings the user gave (the log's "given" events) and the window
length its coherence ladder chose (coherence.json). A run without a QC log is PAC's own: no gate
judged it. What the checks changed is PACo's workflow's, or its autonomous agent's when a study's
line is the run."""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from masw.io.quality.journal import study_of
from sigpipe.masw.runs.history import log_entries

LOG_FILE = "qc_log.jsonl"
CONFIG_FILE = "qc_config.json"
COHERENCE_FILE = "coherence.json"
MUTE_FILE = "mute.json"  # the mute trial's candidates and the one kept
SEGMENTS_FILE = "segments.json"  # a passive line's segments trial
LINE_LOOP_FILE = "line_loop.json"  # the changes of the line's settings tried, and kept
# The triggers of a record or window made again for a change of the line's settings (the mute
# trial's, the line loop's, a redo of the line's): the line's change, not the unit's own.
LINE_RULES = ("mute trial", "line change")
LINE = "line"  # the unit of a line-level result

type Verdict = Literal["pass", "retry", "reject"]
# Who chose what a run's checks changed: PACo's workflow, or its autonomous agent.
type Chooser = Literal["workflow", "agent"]


class Action(BaseModel):
    """What a flag advises (PACo's override, exclude_traces, exclude_record, reject or keep),
    with the fields of its kind."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    kind: str
    stage: str | None = None
    overrides: dict[str, Any] | None = None
    record: str | None = None
    traces: tuple[int, ...] | None = None
    reason: str | None = None
    note: str | None = None


class Metric(BaseModel):
    """One measurement against its threshold: the value must stay above it (min) or below it
    (max); without a threshold, reported only."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    value: float | None  # None: could not be measured
    threshold: float | None = None
    bound: Literal["min", "max"] | None = None
    passed: bool
    unit: str = ""
    of: str = ""  # the object it describes: signal, spectrum, image, ... ("": said by its name)
    over: str = ""  # what it covers: "52 of 96 traces, within 63.35 m of the source"


class Flag(BaseModel):
    """A problem a gate found, the stage most likely at fault, and what to do."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    message: str
    stage: str
    action: Action
    fixable: bool = True


class Kept(BaseModel):
    """What a result keeps of the data."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    band_hz: tuple[float, float] | None = None
    wavelength_m: tuple[float, float] | None = None
    n_points: int | None = None
    n_traces: int | None = None
    n_records: int | None = None


class GateResult(BaseModel):
    """A gate's verdict on one unit, its metrics and flags."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    gate: str
    unit: str
    verdict: Verdict
    metrics: tuple[Metric, ...] = ()
    flags: tuple[Flag, ...] = ()
    kept: Kept = Kept()


class Attempt(BaseModel):
    """One line of the QC log: a stage run once on one unit, with the parameters it changed on
    the run's preset, and what the gates said of it."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    unit: str
    stage: str
    attempt: int
    parameters: dict[str, Any] = {}
    triggered_by: str  # "initial", "<gate>:<flag>" of the attempt before, or "backtrack"
    started_at: datetime
    finished_at: datetime | None = None
    status: str
    error: str | None = None
    notes: tuple[str, ...] = ()
    results: dict[str, GateResult] = {}
    # How a pick was made (PACo's Attempt.how: tracker, neighbours, guide, lasso, delete, judge)
    # and what that way took (a lasso's polygon); None in a log from before attempts said it.
    how: str | None = None
    inputs: dict[str, Any] = {}


class AttemptSummary(BaseModel):
    """An attempt as the Quality sections list it: why it ran, what it changed, what the gates
    said."""

    model_config = ConfigDict(frozen=True)

    attempt: int
    stage: str
    triggered_by: str
    status: str
    error: str | None
    started_at: datetime
    parameters: dict[str, Any]
    notes: tuple[str, ...]
    verdicts: dict[str, Verdict]  # by gate
    flags: tuple[str, ...]  # every gate's, by name


class Reset(BaseModel):
    """A unit's stages started afresh, or their results set aside (sigpipe's runs.history
    `forget`): when, and by whom."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    unit: str
    stages: tuple[str, ...]
    at: datetime
    actor: str | None = None


class QCLog:
    """A run's attempts, each at its current state, in the order they were first logged; every
    state each went through (`history`), where a flag acted on (traces left out) is no longer
    raised by the attempt's last state; every run of a unit's stage at its last state (`runs`),
    those a stage done afresh set aside too (a redo of the agent's starts again at attempt 1);
    the runs and the resets in the order they were logged (`sequence`); and who chose what they
    changed (`chooser`)."""

    def __init__(
        self,
        attempts: Sequence[Attempt],
        history: Sequence[Attempt] = (),
        chooser: Chooser = "workflow",
        runs: Sequence[Attempt] | None = None,
        sequence: Sequence[Attempt | Reset] | None = None,
    ) -> None:
        self.attempts = tuple(attempts)
        self.history = tuple(history)
        self.chooser: Chooser = chooser
        self.runs = tuple(self.attempts if runs is None else runs)
        self.sequence = tuple(self.runs if sequence is None else sequence)

    def of(self, unit: str, stage: str) -> tuple[Attempt, ...]:
        return tuple(a for a in self.attempts if a.unit == unit and a.stage == stage)

    def runs_of(self, unit: str, stage: str) -> tuple[Attempt, ...]:
        """Every run of `unit`'s `stage`, in the order they ran, each at its last state."""
        return tuple(a for a in self.runs if a.unit == unit and a.stage == stage)

    def latest(self, unit: str, stage: str) -> Attempt | None:
        own = self.of(unit, stage)
        return own[-1] if own else None

    def result(self, unit: str, stage: str, gate: str) -> GateResult | None:
        """What `gate` said of the unit's latest attempt at `stage`; None when it has not said."""
        latest = self.latest(unit, stage)
        return latest.results.get(gate) if latest is not None else None

    def verdict(self, unit: str, stage: str, gate: str) -> Verdict | None:
        result = self.result(unit, stage, gate)
        return result.verdict if result is not None else None

    def summaries(self, unit: str, stage: str) -> tuple[AttemptSummary, ...]:
        return tuple(summarize(attempt) for attempt in self.of(unit, stage))

    def raised(self, unit: str, stage: str) -> tuple[Flag, ...]:
        """Every flag the unit's attempts at `stage` raised in any of their states, in order."""
        return tuple(
            flag
            for attempt in self.history
            if attempt.unit == unit and attempt.stage == stage
            for result in attempt.results.values()
            for flag in result.flags
        )


def summarize(attempt: Attempt) -> AttemptSummary:
    return AttemptSummary(
        attempt=attempt.attempt,
        stage=attempt.stage,
        triggered_by=attempt.triggered_by,
        status=attempt.status,
        error=attempt.error,
        started_at=attempt.started_at,
        parameters=attempt.parameters,
        notes=attempt.notes,
        verdicts={gate: result.verdict for gate, result in attempt.results.items()},
        flags=tuple(flag.name for result in attempt.results.values() for flag in result.flags),
    )


def read_log(run_folder: Path) -> QCLog | None:
    """The run's QC log; None for a run PACo never judged. The attempts as the run stands:
    those of a window's stage done afresh since (sigpipe's `log_entries`: a reset in the log)
    left out, as are the log's other events; every run, those too."""
    if not (run_folder / LOG_FILE).exists():
        return None
    current: dict[tuple[str, str, int], Attempt] = {}
    history: list[Attempt] = []
    for entry in log_entries(run_folder):
        if (attempt := _attempt(entry)) is None:
            continue
        current[attempt.unit, attempt.stage, attempt.attempt] = attempt
        history.append(attempt)
    chooser: Chooser = "agent" if is_autonomous(run_folder) else "workflow"
    runs, sequence = _runs(run_folder)
    return QCLog(list(current.values()), history, chooser, runs, sequence)


# What PACo's checks no longer judge or raise, read so from a log made before: the nearest shot,
# reported, its near-field note gone.
_REPORTED = frozenset({"near_offset"})
_RETIRED = frozenset({"near_field"})


def _attempt(entry: dict[str, Any]) -> Attempt | None:
    """A log line of an attempt's, read as PACo's checks say it now; None for another event, or
    a line that does not read."""
    if "attempt" not in entry:
        return None
    try:
        attempt = Attempt.model_validate(entry)
    except ValidationError:
        return None
    results = {gate: _as_now(result) for gate, result in attempt.results.items()}
    return attempt.model_copy(update={"results": results})


def _as_now(result: GateResult) -> GateResult:
    """A check's result as it reads now: a measure no longer judged reported, a flag no longer
    raised left out."""
    metrics = tuple(
        metric.model_copy(update={"threshold": None, "bound": None, "passed": True})
        if metric.name in _REPORTED
        else metric
        for metric in result.metrics
    )
    flags = tuple(flag for flag in result.flags if flag.name not in _RETIRED)
    return result.model_copy(update={"metrics": metrics, "flags": flags})


def _runs(run_folder: Path) -> tuple[list[Attempt], list[Attempt | Reset]]:
    """Every run of a unit's stage the log holds, each at its last state, in the order first
    logged: a run is its stage's start (its unit, stage, attempt and start), its gates' verdicts
    and notes the lines after it; the runs a reset set aside kept. And the runs with the resets,
    in the order logged."""
    runs: dict[tuple[str, str, int, datetime], Attempt] = {}
    order: list[tuple[str, str, int, datetime] | Reset] = []
    for line in (run_folder / LOG_FILE).read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        entry = cast(dict[str, Any], entry)
        if entry.get("event") == "reset":
            try:
                order.append(Reset.model_validate(entry))
            except ValidationError:
                continue
        elif attempt := _attempt(entry):
            key = (attempt.unit, attempt.stage, attempt.attempt, attempt.started_at)
            if key not in runs:
                order.append(key)
            runs[key] = attempt
    sequence = [one if isinstance(one, Reset) else runs[one] for one in order]
    return list(runs.values()), sequence


def is_autonomous(run_folder: Path) -> bool:
    """Whether PACo's autonomous mode made the run: a study's line is it."""
    return study_of(run_folder.parent, run_folder.name) is not None


class Given(BaseModel):
    """Settings the user gave for the run, as PACo locked them: one call's, of one family (the
    processing's overrides, the picking's changes, the inversion's parameters)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    family: str
    values: dict[str, Any]
    at: datetime


def given_settings(run_folder: Path) -> tuple[Given, ...]:
    """Every call's settings the user gave for the run, in order (none: a run PAC made, or one
    PACo judged before it logged them)."""
    if not (run_folder / LOG_FILE).exists():
        return ()
    given: list[Given] = []
    for entry in log_entries(run_folder):
        if entry.get("event") != "given":
            continue
        try:
            one = Given.model_validate(entry)
        except ValidationError:
            continue
        if one.values:
            given.append(one)
    return tuple(given)


def config_section(run_folder: Path, name: str) -> dict[str, Any]:
    """The thresholds of one gate (`name`, e.g. "model" for G5) in the configuration PACo ran
    the run with; empty when it recorded none: its defaults then."""
    path = run_folder / CONFIG_FILE
    if not path.exists():
        return {}
    config: object = json.loads(path.read_text())
    section = cast(dict[str, object], config).get(name) if isinstance(config, dict) else None
    return cast(dict[str, Any], section) if isinstance(section, dict) else {}


def line_receivers(log: QCLog | None) -> tuple[frozenset[int], str]:
    """The receivers G1 over the line left out of every window (a bad geophone: off the
    amplitude decay in most of the records that reach it), and what G1 said of them; none for
    a run the assistant did not judge, or judged before it judged the line's receivers."""
    result = log.result(LINE, "preprocessing", "G1") if log is not None else None
    for flag in result.flags if result is not None else ():
        if flag.action.kind == "exclude_traces" and flag.action.traces:
            return frozenset(flag.action.traces), flag.message
    return frozenset(), ""


def flag_names(*results: GateResult | None) -> tuple[str, ...]:
    return tuple(flag.name for result in results if result is not None for flag in result.flags)


def metric_value(result: GateResult | None, name: str) -> float | None:
    """The value of the result's metric `name`; None when it has none."""
    if result is None:
        return None
    return next((metric.value for metric in result.metrics if metric.name == name), None)
