"""What was done, one way for every card and for the run card: in time order, what made the
results that stand now, each with who made it, when, why and how, then each check's verdict on
it, or the test of a result no check judged (its measures against the agent's limits). A stage
done again replaces its earlier runs; a result erased (set aside because what it was made of
changed, deleted, or failed) leaves with them.

Read from every maker's records: the workflow's and the autonomous agent's runs and resets
(PACo's QC log), what the user changed in PAC's pages (each window's history.jsonl, sigpipe's
runs.history `Change`), the records and windows of a run PAC processed (its run.json), and for a
run from before those records, what the window's files say (PAC's marks of its picks, its Vs
profile and soil column with the files' dates) and an autonomous run's study (how its agent
picked).

The kinds are declared once: `STAGES`, each one's words filled in by its card module
(`register_stage`), and `CHECKS`, PACo's gates (`register_check`, `unregister_check`). A kind
registered is said alike on every card and counted alike on the run card; one removed, nowhere."""

import json
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from masw.io.quality.files import read_manifest
from masw.io.quality.journal import PICKS_FILE, study_of
from masw.io.quality.log import LINE, Attempt, GateResult, Metric, QCLog, Reset
from masw.io.quality.makers import inversion_by_agent, soil_by_agent
from masw.io.quality.view import (
    BY_AGENT,
    CHECK_NAMES,
    FLAG_METRICS,
    JUDGED_WORDS,
    MEASURED_WORDS,
    REFUSALS,
    WARNED,
    Status,
    Step,
    after_colon,
    flag_text,
    measured_status,
    number,
    plural,
    verdict_status,
    warned_flags,
)
from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.inversion.window import SAMPLES_FILE
from sigpipe.masw.petro.window import MODEL_FILE
from sigpipe.masw.picks import CURVES_FILE, load_curves
from sigpipe.masw.runs import RunManifest, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE
from sigpipe.masw.runs.history import Change, read_changes
from sigpipe.masw.runs.origin import AUTO_FILE, EDITS_FILE, EVERY_MODE, JUDGED

# Who made a result: the user in PAC, the workflow agent, the autonomous agent.
type Maker = Literal["you", "workflow", "agent"]
# What a run's settings changed on the run before, in words, a line each (the receivers'
# spacing in metres, for a window's length).
type Changes = Callable[[Mapping[str, Any], Mapping[str, Any], float], list[str]]
# The measures of a result no check judged, in a run folder's unit: the gate whose limits they
# are taken against, and its metrics; None without a result to measure.
type Measurer = Callable[[Path, str], tuple[str, tuple[Metric, ...]] | None]


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class StageKind:
    """A stage whose results a unit holds: its unit (a record, a window), its entry's title on a
    card (a curve's with its mode) and on the run card, its place in the pipeline; what a run
    done again changed on the one before (`changes`; a first run, on `base`, the run's settings
    it started from) and the measures of a result no check judged (`measures`), filled in by
    its card module."""

    stage: str
    unit: Literal["record", "window"]
    made: str
    run: str
    order: int
    changes: Changes | None = None
    base: Callable[[Path], Mapping[str, Any]] | None = None
    measures: Measurer | None = None


@dataclass(frozen=True)
class CheckKind:
    """A check: its gate, the stage whose results it judges, its name."""

    gate: str
    stage: str
    name: str


STAGES: dict[str, StageKind] = {
    kind.stage: kind
    for kind in (
        StageKind("preprocessing", "record", "Preprocessed", "Read the records", 0),
        StageKind("phase_shift", "window", "Dispersion image made", "Processed the whole line", 1),
        StageKind("picking", "window", "{mode} picked", "Picked the curves", 2),
        StageKind("inversion", "window", "Inverted", "Inverted the curves into Vs profiles", 3),
        StageKind(
            "petro_inversion",
            "window",
            "Soil column made",
            "Inverted the curves into soil columns",
            4,
        ),
    )
}
CHECKS: dict[str, CheckKind] = {
    gate: CheckKind(gate, stage, _capitalized(CHECK_NAMES[gate].removeprefix("the ")))
    for gate, stage in (
        ("G1", "preprocessing"),
        ("G2", "phase_shift"),
        ("G3", "picking"),
        ("G4", "picking"),
        ("G5", "inversion"),
        ("G6", "inversion"),
        ("G7", "petro_inversion"),
        ("G8", "petro_inversion"),
    )
}


def register_stage(
    stage: str,
    *,
    changes: Changes | None = None,
    base: Callable[[Path], Mapping[str, Any]] | None = None,
    measures: Measurer | None = None,
) -> None:
    """A stage's words, from its card module: what a run done again changed, the run's settings
    a first run started from, the measures of a result no check judged; those not given kept."""
    kind = STAGES[stage]
    STAGES[stage] = replace(
        kind,
        changes=changes or kind.changes,
        base=base or kind.base,
        measures=measures or kind.measures,
    )


def register_check(kind: CheckKind) -> None:
    """A check said on every card and counted on the run card, or its name changed."""
    CHECKS[kind.gate] = kind


def unregister_check(gate: str) -> None:
    """A check left out everywhere."""
    CHECKS.pop(gate, None)


def _registered() -> None:
    """The stages' card modules imported, each filling in its stage's words."""
    for name in ("records", "dispersion", "inversion", "petro"):
        import_module(f"masw.io.quality.{name}")


# What made a result, as its entry says it and as the run card counts it, the same on every
# page: a curve, the way it was picked (the agent's, then PAC's, the user's); another result, the
# agent automatically, or the PAC page the user made it with.
_WAYS = {
    "tracker": ("Automatically", "automatically"),
    "neighbours": ("Automatically, along the neighbouring curves", "along the neighbouring curves"),
    "guide": ("Automatically, along points the agent gave", "along points the agent gave"),
    "lasso": ("Automatically, inside a lasso the agent drew", "inside a lasso the agent drew"),
    "auto": ("PAC's automatic picking", "PAC's automatic picking"),
    "hand lasso": ("By hand, with the lasso", "by hand, with the lasso"),
    "box": ("By hand, in a box", "by hand, in a box"),
    "hand": ("By hand", "by hand"),
}
_AUTOMATICALLY = "Automatically"
_PAC_TOOLS = {
    "preprocessing": "PAC's computing",
    "phase_shift": "PAC's computing",
    "inversion": "PAC's seismic inversion",
    "petro_inversion": "PAC's petrophysical inversion",
}
# The user's ways of picking: their curve, taken as it is, which no check judges.
_BY_HAND = frozenset({"hand lasso", "box", "hand"})
# Why a stage ran again, other than a check's flag or the agent's own redo, in words.
_CAUSES = {
    "mute trial": "with the mute trial's mute",
    "line change": "with a change of the line's settings",
    "asked": "asked by the user",
}
# Who made a result, in a sentence.
_WHO: dict[Maker, str] = {
    "you": "the user",
    "workflow": "the workflow agent",
    "agent": "the autonomous agent",
}
# How near a reset PAC logged its own change to the window: the same action.
SAME_ACTION = timedelta(seconds=10)
_EARLIEST = datetime.min.replace(tzinfo=UTC)


def cause(triggered_by: str, log: QCLog | None = None) -> str:
    """Why a stage ran, in words ("asked by the curve check: points off the ridge's top", "asked
    by the autonomous agent" for a redo `log`'s agent chose); "" for a first run."""
    gate, _, flag = triggered_by.partition(":")
    if flag:
        kind = CHECKS.get(gate)
        named = f"the {after_colon(kind.name)}" if kind is not None else gate
        return f"asked by {named}: {flag_text(flag)}"
    if triggered_by == "backtrack":
        return f"asked {BY_AGENT[log.chooser] if log is not None else 'by the agent'}"
    if triggered_by == "initial":
        return ""
    return _CAUSES.get(triggered_by, triggered_by.replace("_", " "))


def counted_cause(triggered_by: str, log: QCLog | None = None) -> str:
    """Why a stage ran again, as a count of units says it: a check's flag alone ("too few
    points"), another cause as `cause` says it."""
    _, _, flag = triggered_by.partition(":")
    return flag_text(flag) if flag else cause(triggered_by, log)


@dataclass(frozen=True)
class Verdict:
    """A check's verdict on a result, or a test of a result no check judged: its gate, its
    status, its title and what it found, as an entry says them."""

    gate: str
    status: Status
    test: bool
    title: str
    lines: tuple[str, ...]
    at: datetime | None


@dataclass(frozen=True)
class Done:
    """A result that stands: its stage (a curve's mode), who made it and when, why (a check or
    the user asked it), how (a pick's way), what its settings changed on the run it replaced, and
    the verdicts on it."""

    stage: str
    mode: str | None
    by: Maker
    at: datetime | None
    why: str
    way: str
    changes: tuple[str, ...]
    verdicts: tuple[Verdict, ...]

    @property
    def how(self) -> str:
        """What made it, in its entry's words: a curve, the way it was picked; another result,
        the agent automatically, or the PAC page the user made it with."""
        if self.stage == "picking":
            return _WAYS[self.way][0] if self.way in _WAYS else ""
        return _PAC_TOOLS[self.stage] if self.by == "you" else _AUTOMATICALLY


@dataclass
class _Run:
    """A run of a stage on a unit, as read: its result's key (a stage; a curve's, "picking:M0"),
    who made it, when, its order in the log, its words."""

    key: str
    by: Maker
    at: datetime | None
    seq: int
    why: str = ""
    way: str = ""
    changes: tuple[str, ...] = ()
    checks: list[Verdict] = field(default_factory=list[Verdict])
    deleted: bool = False
    failed: bool = False

    @property
    def stage(self) -> str:
        return self.key.partition(":")[0]


@dataclass(frozen=True)
class _Sources:
    """What a run's units' histories are read from, once: its folder, run.json and QC log, the
    log's runs and resets by unit, an autonomous run's study picks, the receivers' spacing."""

    run_folder: Path
    manifest: RunManifest | None
    log: QCLog | None
    sequence: Mapping[str, Sequence[Attempt | Reset]]
    picks: tuple[Mapping[str, Any], ...]
    spacing: float


def _sources(run_folder: Path, log: QCLog | None, manifest: RunManifest | None) -> _Sources:
    _registered()
    manifest = manifest if manifest is not None else read_manifest(run_folder)
    sequence: dict[str, list[Attempt | Reset]] = {}
    for one in log.sequence if log is not None else ():
        sequence.setdefault(one.unit, []).append(one)
    picks = tuple(_study_picks(run_folder)) if log is not None and log.chooser == "agent" else ()
    spacing = manifest.profile.receiver_spacing_m if manifest is not None else 1.0
    return _Sources(run_folder, manifest, log, sequence, picks, spacing)


def standing(
    run_folder: Path,
    unit: str,
    log: QCLog | None,
    manifest: RunManifest | None = None,
    tested: Collection[str] = tuple(STAGES),
) -> list[Done]:
    """`unit`'s results that stand now, in time order (`log`: the run's QC log; `manifest`: its
    run.json, read when not given); a result of a stage in `tested` no check judged, tested."""
    return _standing(_sources(run_folder, log, manifest), unit, tested)


def _standing(sources: _Sources, unit: str, tested: Collection[str]) -> list[Done]:
    """Of each of `unit`'s results (a stage's; each mode's curve), the run made last, unless a
    reset or a change set it aside after it, it deleted the result or failed, or its result is
    not in the unit's folder; with its checks' verdicts, else, for a stage in `tested`, its
    test."""
    manifest = sources.manifest
    record = manifest is not None and any(one.name == unit for one in manifest.records)
    window = sources.run_folder / unit
    changes = read_changes(window) if not record and window.is_dir() else ()
    runs, asides, judged = _from_log(sources, unit, changes)
    runs += _from_changes(changes, asides)
    runs += _from_pac(sources, unit, record, runs, changes)
    done: list[Done] = []
    for key in sorted({one.key for one in runs}, key=_order):
        own = sorted(
            (one for one in runs if one.key == key), key=lambda one: (one.at or _EARLIEST, one.seq)
        )
        last = own[-1]
        if last.deleted or last.failed or not _exists(window, key, record):
            continue
        if any(last.at is None or at >= last.at for at in asides.get(last.stage, ())):
            continue
        verdicts = list(last.checks)
        if key == "picking:M0":
            verdicts += [one for at, one in judged if last.at is None or at >= last.at]
        if not verdicts and last.stage in tested and last.way not in _BY_HAND:
            verdicts += _test(STAGES[last.stage], sources.run_folder, unit, key)
        done.append(
            Done(
                stage=last.stage,
                mode=key.partition(":")[2] or None,
                by=last.by,
                at=last.at,
                why=last.why,
                way=last.way,
                changes=last.changes,
                verdicts=tuple(verdicts),
            )
        )
    return sorted(done, key=lambda one: one.at or _EARLIEST)


def _from_log(
    sources: _Sources, unit: str, changes: Sequence[Change]
) -> tuple[list[_Run], dict[str, list[datetime]], list[tuple[datetime, Verdict]]]:
    """`unit`'s runs the agent made, in the order logged; when its resets set results aside, by
    stage (a reset followed by a run of one of its stages started that stage afresh: the run
    replaces the ones before, the other stages are set aside); the checks of a curve picked in
    PAC (a judgement, no run of its own)."""
    runs: list[_Run] = []
    asides: dict[str, list[datetime]] = {}
    judged: list[tuple[datetime, Verdict]] = []
    log = sources.log
    if log is None:
        return runs, asides, judged
    sequence = sources.sequence.get(unit, ())
    befores: dict[str, Attempt] = {}
    ways: dict[str, str] = {}
    for index, one in enumerate(sequence):
        if isinstance(one, Reset):
            if one.actor == "user" and _changed_then(changes, one.at):
                continue  # PAC's own change: its record says what it set aside
            nxt = sequence[index + 1] if index + 1 < len(sequence) else None
            redone = nxt.stage if isinstance(nxt, Attempt) and nxt.stage in one.stages else None
            for stage in one.stages:
                if stage != redone:
                    asides.setdefault(stage, []).append(one.at)
            continue
        if one.stage not in STAGES:
            continue
        at = one.finished_at or one.started_at
        if one.triggered_by == JUDGED or one.how == "judge":
            judged += [(at, verdict) for verdict in _checks(one.results, at)]
            continue
        key = "picking:M0" if one.stage == "picking" else one.stage
        before = befores.get(key)
        run = _Run(key, log.chooser, at, index)
        # Why it ran: a check or the user asked it, a change of the line; the agent's own redo
        # said by its tag alone.
        if one.triggered_by != "backtrack":
            run.why = _capitalized(cause(one.triggered_by, log))
        if one.stage == "picking":
            run.way = _way(one, before, ways.get(key), sources.picks)
        run.changes = tuple(_changed(sources, one, before))
        run.deleted = one.how == "delete" or any("deleted" in note for note in one.notes)
        run.failed = one.status == "failed"
        run.checks = _checks(one.results, at)
        runs.append(run)
        befores[key] = one
        ways[key] = run.way
    return runs, asides, judged


def _from_changes(changes: Sequence[Change], asides: dict[str, list[datetime]]) -> list[_Run]:
    """The runs the user made in PAC's pages, from the window's record; the later stages each
    set aside, made of what it replaced."""
    runs: list[_Run] = []
    for change in changes:
        if change.stage not in STAGES:
            continue
        run = _Run(_key(change), "you", change.at, 10**8)
        if change.stage == "picking":
            run.way = "hand lasso" if change.how == "lasso" else change.how
        run.deleted = change.how == "delete"
        runs.append(run)
        for stage in change.later:
            asides.setdefault(stage, []).append(change.at)
    return runs


def _key(change: Change) -> str:
    """The result a change made: a stage's; a curve's, by its mode."""
    return f"picking:{change.mode or 'M0'}" if change.stage == "picking" else change.stage


def _changed_then(changes: Sequence[Change], at: datetime) -> bool:
    """Whether PAC recorded a change of the window as it logged a reset at `at`."""
    return any(abs(change.at - at) <= SAME_ACTION for change in changes)


def _order(key: str) -> tuple[int, int, str]:
    """A result's place: its stage's, then a curve's mode's."""
    stage, _, label = key.partition(":")
    if not label:
        return (STAGES[stage].order, 0, "")
    mode = Mode.from_label(label)
    return (STAGES[stage].order, mode.number, mode.wave)


def _exists(window: Path, key: str, record: bool) -> bool:
    """Whether the unit's folder holds the result: a record's always does (its outcome says
    whether it failed)."""
    stage, _, label = key.partition(":")
    if record or stage == "preprocessing":
        return True
    if stage == "phase_shift":
        return (window / IMAGE_FILE).exists()
    if stage == "picking":
        return any(one.mode.label == label for one in load_curves(window) or ())
    if stage == "inversion":
        return (window / SAMPLES_FILE).exists()
    return (window / MODEL_FILE).exists()


def _way(
    run: Attempt, before: Attempt | None, way: str | None, picks: Sequence[Mapping[str, Any]]
) -> str:
    """How a pick was made: as its attempt logged it; else as the pick before when it kept its
    guide (a check's retry), else as the study's recipe or its parameters tell."""
    if run.how:
        return run.how
    guide = run.parameters.get("guide")
    if guide and before is not None and way and before.parameters.get("guide") == guide:
        return way
    recipe = _study_pick(run, picks)
    if recipe is not None:
        how = str(recipe.get("how"))
        if how == "guide":
            return "neighbours" if recipe.get("guide") == "neighbours" else "guide"
        return "tracker" if how in ("auto", "line") else how
    if guide:
        return "neighbours" if run.triggered_by.startswith("G4:") else "guide"
    return "tracker"


def _changed(sources: _Sources, run: Attempt, before: Attempt | None) -> list[str]:
    """What a run's settings changed on the run before it, in its card's words; a first run, on
    the settings its stage starts from (a pick, the run's picking settings); a guide left to
    the pick's way."""
    kind = STAGES[run.stage]
    if kind.changes is None:
        return []
    if before is not None:
        was = before.parameters
    elif kind.base is not None:
        was = kind.base(sources.run_folder)
    else:
        return []
    lines = kind.changes(was, run.parameters, sources.spacing)
    return [line for line in lines if not line.startswith("Guided")]


def _study_pick(run: Attempt, picks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The recipe of the study's pick of `run`'s window kept first after it ran."""
    ended = run.finished_at or run.started_at
    later = [
        (at, one)
        for one in picks
        if one.get("unit") == run.unit
        and (at := _parsed(one.get("at"))) is not None
        and at >= ended
    ]
    if not later:
        return None
    recipe = min(later, key=lambda pair: pair[0])[1].get("recipe")
    return cast(Mapping[str, Any], recipe) if isinstance(recipe, dict) else None


def _parsed(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        at = datetime.fromisoformat(value)
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def _study_picks(run_folder: Path) -> list[Mapping[str, Any]]:
    """The picks an autonomous run's study kept of the run (PACo's picks.jsonl), read as JSON."""
    study = study_of(run_folder.parent, run_folder.name)
    path = study / PICKS_FILE if study is not None else None
    if path is None or not path.exists():
        return []
    found: list[Mapping[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            one = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(one, dict) and cast(dict[str, Any], one).get("run_id") == run_folder.name:
            found.append(cast(dict[str, Any], one))
    return found


def _checks(results: Mapping[str, GateResult], at: datetime) -> list[Verdict]:
    """The registered checks' verdicts on a run, by gate: each its verdict, what it acted on,
    the traces it left out, what it kept as a note, a retry it could not make."""
    found: list[Verdict] = []
    for gate, result in sorted(results.items()):
        kind = CHECKS.get(gate)
        if kind is None or result.unit == LINE:
            continue
        status = verdict_status(result)
        flags = [flag for flag in result.flags if flag.name not in REFUSALS]
        acted = [
            flag_text(flag.name)
            for flag in flags
            if flag.action.kind not in ("keep", "exclude_traces")
        ]
        traces = [flag_text(flag.name) for flag in flags if flag.action.kind == "exclude_traces"]
        kept = [flag_text(flag.name) for flag in flags if flag.action.kind == "keep"]
        refused = [flag_text(flag.name) for flag in result.flags if flag.name in REFUSALS]
        lines = [
            *([_capitalized(", ".join(dict.fromkeys(acted)))] if acted else []),
            *([f"Traces left out: {', '.join(dict.fromkeys(traces))}"] if traces else []),
            *(f"Note: {one}" for one in dict.fromkeys(kept)),
            *([f"Not done again: {', '.join(dict.fromkeys(refused))}"] if refused else []),
        ]
        title = f"{kind.name}: {JUDGED_WORDS[status]}"
        found.append(Verdict(gate, status, False, title, tuple(lines), at))
    return found


def _test(kind: StageKind, run_folder: Path, unit: str, key: str) -> list[Verdict]:
    """A result no check judged, measured against the agent's limits as its check takes them:
    within them, or beyond them and what that check would flag, with its notes; none without its
    stage's measures or its check, nor for a curve but M0 (no check judges another mode)."""
    if kind.measures is None or (kind.stage == "picking" and key != "picking:M0"):
        return []
    found = kind.measures(run_folder, unit)
    if found is None or found[0] not in CHECKS:
        return []
    gate, metrics = found
    status = measured_status(metrics)
    if status == "none":
        return []
    beyond = dict.fromkeys(
        _flagged(gate, metric)
        for metric in metrics
        if metric.threshold is not None and not metric.passed and metric.name not in WARNED
    )
    lines = [_capitalized(", ".join(flag_text(one) for one in beyond))] if beyond else []
    lines += [f"Note: {flag_text(one)}" for one in warned_flags(metrics)]
    title = f"Measured: {MEASURED_WORDS[status]}"
    return [Verdict(gate, status, True, title, tuple(lines), None)]


def _flagged(gate: str, metric: Metric) -> str:
    """The flag `gate` raises for `metric` beyond its limit: a soil column's misfit, an image
    coherent nowhere or only in part; else the first flag that judges it, or one named as it."""
    if gate == "G7" and metric.name.startswith("misfit"):
        return "misfit"
    if metric.name == "coherent_columns":
        return "weak_coherence" if metric.value else "no_coherent_energy"
    return next(
        (flag for flag, judged in FLAG_METRICS.items() if metric.name in judged), metric.name
    )


def _from_pac(
    sources: _Sources,
    unit: str,
    record: bool,
    runs: Sequence[_Run],
    changes: Sequence[Change],
) -> list[_Run]:
    """What PAC made of `unit` with no record of its own: in a run PAC processed, its first
    result (the record's preprocessing, the window's image); for a window, what its files say
    PAC changed before it recorded its changes (its picks, its Vs profile, its soil column),
    dated by them, and a curve no one is known to have picked, the user's, by hand."""
    made: list[_Run] = []
    manifest = sources.manifest
    first = "preprocessing" if record else "phase_shift"
    if manifest is not None and not any(one.stage == first for one in runs):
        outcome = (
            next((one for one in manifest.records if one.name == unit), None)
            if record
            else next((one for one in manifest.windows if one.folder == unit), None)
        )
        if outcome is not None:
            at = manifest.finished_at or manifest.started_at
            run = _Run(first, "you", at if at.tzinfo else at.replace(tzinfo=UTC), -1)
            run.failed = outcome.status == "failed"
            made.append(run)
    if record:
        return made
    recorded: dict[str, datetime] = {}
    for change in changes:
        recorded[_key(change)] = max(change.at, recorded.get(_key(change), change.at))
    window = sources.run_folder / unit
    marked = (
        *_marked_picks(window),
        *_marked(sources, unit, "inversion"),
        *_marked(sources, unit, "petro_inversion"),
    )
    for run in marked:
        last = recorded.get(run.key)
        if last is None or (run.at is not None and run.at > last + SAME_ACTION):
            made.append(run)
    known = {one.key for one in (*runs, *made)}
    for curve in load_curves(window) or ():
        if (key := f"picking:{curve.mode.label}") not in known:
            made.append(_Run(key, "you", _dated(window / CURVES_FILE), 10**9, way="hand"))
    return made


def _marked_picks(window: Path) -> list[_Run]:
    """A window's picks PAC marked before it recorded them: each mode's last change by hand, its
    last automatic pick of M0."""
    made: list[_Run] = []
    path = window / EDITS_FILE
    if path.exists():
        data = cast(dict[str, Any], json.loads(path.read_text()))
        modes, edited = data.get("modes"), data.get("edited_at")
        times = (
            cast(dict[str, str], modes)
            if isinstance(modes, dict)
            else ({EVERY_MODE: edited} if isinstance(edited, str) else {})
        )
        for label, when in times.items():
            key = f"picking:{'M0' if label == EVERY_MODE else label}"
            made.append(_Run(key, "you", _parsed(when), 10**9, way="hand"))
    path = window / AUTO_FILE
    if path.exists():
        when = _parsed(json.loads(path.read_text()).get("picked_at"))
        made.append(_Run("picking:M0", "you", when, 10**9, way="auto"))
    return made


def _marked(sources: _Sources, unit: str, stage: str) -> list[_Run]:
    """A window's Vs profile or soil column PAC's page made before it recorded its changes, the
    agent's not: dated by its file."""
    window = sources.run_folder / unit
    path = window / (SAMPLES_FILE if stage == "inversion" else MODEL_FILE)
    if not path.exists():
        return []
    if stage == "inversion" and inversion_by_agent(sources.log, unit, sources.run_folder):
        return []
    if stage == "petro_inversion" and soil_by_agent(sources.log, window):
        return []
    return [_Run(stage, "you", _dated(path), 10**9)]


def _dated(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def unit_steps(
    run_folder: Path,
    unit: str,
    log: QCLog | None,
    manifest: RunManifest | None,
    stages: Collection[str],
) -> tuple[Step, ...]:
    """A card's "What was done": in time order, what made `unit`'s results of `stages` that
    stand, an entry each (who, when, what made it, why, what its settings changed), each followed
    by each verdict on it, an entry of its own."""
    steps: list[Step] = []
    for one in standing(run_folder, unit, log, manifest, tested=stages):
        if one.stage not in stages:
            continue
        lines = [*([one.how] if one.how else []), *([one.why] if one.why else []), *one.changes]
        steps.append(
            Step(
                kind=one.stage,
                by=one.by,
                at=one.at,
                title=STAGES[one.stage].made.format(mode=one.mode or ""),
                lines=tuple(lines),
            )
        )
        steps += [
            Step(
                kind="test" if verdict.test else "check",
                by="checks",
                at=verdict.at or one.at,
                title=verdict.title,
                lines=verdict.lines,
            )
            for verdict in one.verdicts
        ]
    return tuple(steps)


def run_steps(
    run_folder: Path,
    manifest: RunManifest,
    log: QCLog | None,
    stages: Sequence[str],
    lines: Mapping[str, Sequence[str]] | None = None,
    listed: int = 6,
) -> list[Step]:
    """The run card's entries of `stages`, the results that stand over the line counted: a
    stage's an entry by each who made some, first who made the first, each saying how many of
    its units, what made them, the curves of other modes, each check's verdicts, the user's units
    named when some of them, `listed` or fewer; a stage's facts of the whole run (`lines`) in its
    first entry. The tests of results no check judged stay on their cards."""
    sources = _sources(run_folder, log, manifest)
    units = {
        "record": [one.name for one in manifest.records],
        "window": [one.folder for one in manifest.windows],
    }
    found: dict[str, list[Done]] = {}
    steps: list[Step] = []
    for stage in stages:
        kind = STAGES[stage]
        own = units[kind.unit]
        for unit in own:
            if unit not in found:
                found[unit] = _standing(sources, unit, ())
        done = [(unit, one) for unit in own for one in found[unit] if one.stage == stage]
        makers: dict[Maker, list[tuple[str, Done]]] = {}
        for unit, one in sorted(done, key=lambda pair: pair[1].at or _EARLIEST):
            makers.setdefault(one.by, []).append((unit, one))
        for index, (maker, group) in enumerate(makers.items()):
            # A window's curve: its M0; the other modes, the user's, said apart.
            main = [one for _, one in group if one.mode in (None, "M0")]
            said = [f"{len(main)} of {plural(len(own), kind.unit)}"] if main else []
            said += _hows(main)
            said += (lines or {}).get(stage, ()) if index == 0 else ()
            said += _other_modes([one for _, one in group])
            said += _verdict_counts(main)
            named = sorted({xmid_of(unit) for unit, _ in group} if kind.unit == "window" else ())
            if maker == "you" and 0 < len(named) < len(own) and len(named) <= listed:
                said.append(", ".join(f"xmid {number(xmid, 4)} m" for xmid in named))
            steps.append(
                Step(kind=stage, by=maker, at=group[0][1].at, title=kind.run, lines=tuple(said))
            )
    return steps


def makers(
    run_folder: Path, manifest: RunManifest, log: QCLog | None, stages: Sequence[str]
) -> dict[str, dict[Maker, int]]:
    """By stage, who made its results that stand over the line, each with how many (a window's
    curve: its M0), the most first."""
    sources = _sources(run_folder, log, manifest)
    units = {
        "record": [one.name for one in manifest.records],
        "window": [one.folder for one in manifest.windows],
    }
    found: dict[str, list[Done]] = {}
    made: dict[str, dict[Maker, int]] = {}
    for stage in stages:
        counts: Counter[Maker] = Counter()
        for unit in units[STAGES[stage].unit]:
            if unit not in found:
                found[unit] = _standing(sources, unit, ())
            counts.update(
                one.by for one in found[unit] if one.stage == stage and one.mode in (None, "M0")
            )
        made[stage] = dict(counts.most_common())
    return made


def by_whom_counted(counts: Mapping[Maker, int]) -> str:
    """Who made results, as a sentence says it: "by the workflow agent", or each with how many
    it made, "by the workflow agent on 79, by the user on 5"."""
    if len(counts) == 1:
        return f"by {_WHO[next(iter(counts))]}"
    return ", ".join(f"by {_WHO[maker]} on {count}" for maker, count in counts.items())


def _hows(done: Sequence[Done]) -> list[str]:
    """What made the results, as their entries say it: one thing alone ("Automatically"); more,
    each with how many it made ("Automatically on 65, inside a lasso the agent drew on 16")."""
    hows = Counter(one.how for one in done if one.how)
    if len(hows) < 2:
        return list(hows)
    short = {one.how: _WAYS[one.way][1] if one.way in _WAYS else one.how for one in done}
    return [
        _capitalized(", ".join(f"{short[how]} on {count}" for how, count in hows.most_common()))
    ]


def _other_modes(done: Sequence[Done]) -> list[str]:
    """The curves of modes other than M0, a line each mode: how many windows hold one and what
    made them, "M1 on 3 windows, by hand, in a box"."""
    modes: dict[str, list[Done]] = {}
    for one in done:
        if one.mode is not None and one.mode != "M0":
            modes.setdefault(one.mode, []).append(one)
    said: list[str] = []
    for mode in sorted(modes, key=lambda label: _order(f"picking:{label}")):
        hows = _hows(modes[mode])
        made = f", {after_colon(hows[0])}" if hows else ""
        said.append(f"{mode} on {plural(len(modes[mode]), 'window')}{made}")
    return said


def _verdict_counts(done: Sequence[Done]) -> list[str]:
    """Each registered check's verdicts on the results, counted: "Curve check: 75 passed, 4
    flagged"."""
    counts: dict[str, Counter[Status]] = {}
    for one in done:
        for verdict in one.verdicts:
            if not verdict.test:
                counts.setdefault(verdict.gate, Counter())[verdict.status] += 1
    return [
        f"{CHECKS[gate].name}: "
        + ", ".join(
            f"{count[status]} {JUDGED_WORDS[status]}"
            for status in ("pass", "warn", "fail")
            if count[status]
        )
        for gate, count in sorted(counts.items())
        if gate in CHECKS
    ]


def picked(run_folder: Path, unit: str, log: QCLog | None) -> dict[str, str]:
    """Who picked each of `unit`'s curves that stand and how, by its mode's label, in a
    sentence: "Picked by the user, PAC's automatic picking"."""
    said: dict[str, str] = {}
    for one in _standing(_sources(run_folder, log, None), unit, ()):
        if one.stage == "picking" and one.mode is not None:
            how = f", {after_colon(one.how)}" if one.how else ""
            said[one.mode] = f"Picked by {_WHO[one.by]}{how}"
    return said
