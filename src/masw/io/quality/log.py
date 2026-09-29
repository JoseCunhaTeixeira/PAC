"""PACo's records in a run folder, read as plain JSON: PACo is PAC's optional extra, and a run it
judged is read without it. The QC log (qc_log.jsonl: one line per attempt of a stage on a unit, a
record, a window or the line, the last line of an attempt its current state), the thresholds its
gates used (qc_config.json) and the window length its coherence ladder chose (coherence.json).
A run without a QC log is PAC's own: no gate judged it."""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

LOG_FILE = "qc_log.jsonl"
CONFIG_FILE = "qc_config.json"
COHERENCE_FILE = "coherence.json"
LINE = "line"  # the unit of a line-level result

type Verdict = Literal["pass", "retry", "reject"]


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


class QCLog:
    """A run's attempts, each at its current state, in the order they were first logged; and
    every state each went through (`history`), where a flag acted on (traces left out) is no
    longer raised by the attempt's last state."""

    def __init__(self, attempts: Sequence[Attempt], history: Sequence[Attempt] = ()) -> None:
        self.attempts = tuple(attempts)
        self.history = tuple(history)

    def of(self, unit: str, stage: str) -> tuple[Attempt, ...]:
        return tuple(a for a in self.attempts if a.unit == unit and a.stage == stage)

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
    """The run's QC log; None for a run PACo never judged."""
    path = run_folder / LOG_FILE
    if not path.exists():
        return None
    current: dict[tuple[str, str, int], Attempt] = {}
    history: list[Attempt] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            attempt = Attempt.model_validate_json(line)
        except ValidationError:  # the last line, while PACo writes it
            continue
        current[attempt.unit, attempt.stage, attempt.attempt] = attempt
        history.append(attempt)
    return QCLog(list(current.values()), history)


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
