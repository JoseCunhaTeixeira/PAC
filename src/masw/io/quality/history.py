"""Each unit's history: a record's, or a window's (its dispersion image, each mode's curve, its Vs
profile, its soil column), a step a result, in the pipeline's order. A step holds its version now
and the versions it replaced, latest first: who made each, when, why, how and what it made, the
checks that judged it (each check an element of its own) or, made by hand, its measures against
the agent's limits. A step done again replaces its versions before; a step whose result was made
of an older one is set aside until it is done again. Nothing is left out:

- the workflow's and the autonomous agent's attempts, from PACo's QC log: each run of a stage,
  those a redo set aside too, and the resets;
- what the user changed in PAC's pages, from each window's own record (sigpipe's runs.history
  `Change`);
- the records and windows of a run PAC processed, from its run.json;
- for a run from before those records, what the window's files say (PAC's marks of its picks,
  its inversion and soil configurations with the files' dates), and an autonomous run's study
  (how its agent picked).

The kinds a history holds are registered: its steps (`STEPS`, each stage's words filled in by
the stage's card module), its checks (`CHECKS`, PACo's gates) and the measures of a result no
check judged (`StepKind.measures`). A kind registered is said alike on every page; one removed,
on none."""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from masw.io.quality.files import read_manifest
from masw.io.quality.log import (
    LINE,
    Attempt,
    Chooser,
    GateResult,
    Metric,
    QCLog,
    Reset,
    config_section,
)
from masw.io.quality.view import (
    BY_AGENT,
    CHECK_NAMES,
    FLAG_METRICS,
    REFUSALS,
    WARNED,
    Element,
    GateView,
    HistoryKind,
    HistoryStep,
    Maker,
    Module,
    State,
    UnitHistory,
    Version,
    finding,
    flag_text,
    gate_view,
    measured_status,
    number,
    plural,
    span,
    verdict_status,
    warned_flags,
)
from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.picks import load_curves
from sigpipe.masw.runs import RunManifest
from sigpipe.masw.runs.history import Change, read_changes
from sigpipe.masw.runs.origin import AUTO_FILE, EDITS_FILE, EVERY_MODE, JUDGED

# What a version's settings change on the one before, in words, a line each (the receivers'
# spacing in metres, for a window's length).
type Changes = Callable[[Mapping[str, Any], Mapping[str, Any], float], list[str]]
# A version's settings in words, a line each, said when no version came before it.
type Settings = Callable[[Mapping[str, Any]], list[str]]
# The measures of a result no check judged, against the agent's limits, in a run folder's unit:
# the gate whose limits they are, and its metrics; None without a result to measure.
type Measurer = Callable[[Path, str], tuple[str, tuple[Metric, ...]] | None]


@dataclass(frozen=True)
class StepKind:
    """A kind of step a history holds, a stage's result: its unit (a record, a window), its title
    (a curve's with its mode), its kind's in the page's filter, its name in a sentence, its
    versions' titles (made,
    and made again), the page of PAC that makes it, what a version's settings change on the one
    before (`changes`; the first's, on `base`, the run's, or said whole: `settings`), and the
    measures of a result no check judged (`measures`). The stage's card module fills in its
    words (register_step)."""

    stage: str
    unit: Literal["record", "window"]
    title: str
    label: str
    noun: str
    made: str
    again: str
    page: str
    order: int
    changes: Changes | None = None
    base: Callable[[Path], Mapping[str, Any]] | None = None
    settings: Settings | None = None
    measures: Measurer | None = None


@dataclass(frozen=True)
class CheckKind:
    """A kind of check a history holds: a gate, the stage whose results it judges, its name."""

    gate: str
    stage: str
    name: str


STEPS: dict[str, StepKind] = {
    kind.stage: kind
    for kind in (
        StepKind("preprocessing", "record", "Preprocessed record", "Preprocessing",
                 "preprocessed record", "Preprocessed", "Preprocessed again", "processing", 0),
        StepKind("phase_shift", "window", "Dispersion image", "Dispersion images",
                 "dispersion image", "Made", "Made again", "processing", 1),
        StepKind("picking", "window", "{mode} curve", "Curves", "curve", "Picked", "Picked again",
                 "Dispersion picking", 2),
        StepKind("inversion", "window", "Vs profile", "Vs profiles", "Vs profile", "Inverted",
                 "Inverted again", "Seismic inversion", 3),
        StepKind("petro_inversion", "window", "Soil column", "Soil columns", "soil column", "Made",
                 "Made again", "Petrophysical inversion", 4),
    )
}  # fmt: skip
CHECKS: dict[str, CheckKind] = {
    gate: CheckKind(gate, stage, CHECK_NAMES[gate])
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


def register_step(
    stage: str,
    *,
    changes: Changes | None = None,
    base: Callable[[Path], Mapping[str, Any]] | None = None,
    settings: Settings | None = None,
    measures: Measurer | None = None,
) -> None:
    """A stage's words in its kind of step, from its card module: what a version changes on the
    one before, the run's settings a first one starts from, a version's settings said whole,
    the measures of a result no check judged; those not given left as they were."""
    kind = STEPS[stage]
    STEPS[stage] = replace(
        kind,
        changes=changes or kind.changes,
        base=base or kind.base,
        settings=settings or kind.settings,
        measures=measures or kind.measures,
    )


def register_check(kind: CheckKind) -> None:
    """A kind of check said in every history, or its words changed."""
    CHECKS[kind.gate] = kind


def unregister_check(gate: str) -> None:
    """A kind of check left out of every history."""
    CHECKS.pop(gate, None)


def _registered() -> None:
    """The stages' card modules imported, each filling in its stage's words."""
    for name in ("records", "dispersion", "inversion", "petro"):
        import_module(f"masw.io.quality.{name}")


# Each way of picking in words, as a version's how says it: the agent's, then PAC's, the user's.
_WAYS = {
    "tracker": "sigpipe's tracker",
    "neighbours": "sigpipe's tracker, along the neighbours' median curve",
    "guide": "sigpipe's tracker, along {count} points given",
    "lasso": "sigpipe's tracker, along the maxima inside a lasso of {count} vertices",
    "delete": "Deleted",
}
_PAC_WAYS = {
    "lasso": "By hand: the maxima inside a lasso of {count} vertices",
    "box": "By hand: the ridge inside a box, {box}",
    "auto": "PAC's automatic picking: sigpipe's tracker with its defaults",
    "delete": "Deleted by hand",
    "hand": "By hand",
    "inversion": "PAC's seismic inversion",
    "petro": "PAC's petrophysical inversion, Silex model {model}",
}
# What a step set aside was made of, by the stage redone before it.
_MADE_OF = {
    "preprocessing": "the older records",
    "phase_shift": "the older image",
    "picking": "the older curve",
}
# How near a reset PAC logged its own change to the window: the same action.
SAME_ACTION = timedelta(seconds=10)


@dataclass
class _Draft:
    """A version being read: its step, who made it and when, its order in the log, its words."""

    key: str
    by: Maker
    at: datetime | None
    seq: int
    why: str = ""
    how: list[str] = field(default_factory=list[str])
    result: list[str] = field(default_factory=list[str])
    later: list[str] = field(default_factory=list[str])
    elements: list[Element] = field(default_factory=list[Element])
    deleted: bool = False
    failed: bool = False
    hand: bool = False  # a curve picked by hand: the user's, taken as it is, not measured

    @property
    def stage(self) -> str:
        return self.key.split(":")[0]


@dataclass(frozen=True)
class _Aside:
    """A step's result set aside at `at` by `by`, because `why`."""

    at: datetime
    by: Maker
    why: str


def unit_history(
    run_folder: Path, unit: str, log: QCLog | None, manifest: RunManifest | None = None
) -> UnitHistory:
    """`unit`'s history in the run in `run_folder` (`log`: its QC log; `manifest`: its run.json,
    read when not given)."""
    _registered()
    manifest = manifest if manifest is not None else read_manifest(run_folder)
    record = manifest is not None and any(one.name == unit for one in manifest.records)
    window = run_folder / unit
    spacing = manifest.profile.receiver_spacing_m if manifest is not None else 1.0
    changes = read_changes(window) if not record and window.is_dir() else ()
    drafts, asides, judged = _from_log(run_folder, unit, log, changes, spacing)
    for change in changes:
        if change.stage not in STEPS:
            continue
        draft = _Draft(_key(change.stage, change.mode), "you", change.at, 10**8)
        draft.why = f"In PAC's {STEPS[change.stage].page} page"
        draft.how = _change_how(change)
        draft.result = _change_result(change)
        draft.deleted = change.how == "delete"
        draft.hand = change.how in ("lasso", "box")
        draft.later = _later_words(change.later)
        drafts.append(draft)
        for stage in change.later:
            for key in _keys_of(stage, drafts):
                asides.setdefault(key, []).append(
                    _Aside(change.at, "you", f"made of {_MADE_OF[change.stage]}")
                )
    drafts += _pac_made(run_folder, unit, record, manifest, log, drafts, changes)
    for at, element in judged:
        target = _standing([one for one in drafts if one.key == "picking:M0"], at)
        if target is not None:
            target.elements.append(element)
    steps = _steps(run_folder, unit, drafts, asides)
    return UnitHistory(unit=unit, steps=steps, kinds=_kinds(steps))


def _from_log(
    run_folder: Path, unit: str, log: QCLog | None, changes: Sequence[Change], spacing: float
) -> tuple[list[_Draft], dict[str, list[_Aside]], list[tuple[datetime, Element]]]:
    """`unit`'s versions the agent made, from the log's runs in the order logged; the steps its
    resets set aside (a reset followed by a run of its stage started that stage afresh: the run
    replaces its versions, and sets aside those of the stages after it); the checks of the
    user's curves (a judgement, no version)."""
    drafts: list[_Draft] = []
    asides: dict[str, list[_Aside]] = {}
    judged: list[tuple[datetime, Element]] = []
    if log is None:
        return drafts, asides, judged
    chooser: Maker = log.chooser
    picks = _study_picks(run_folder) if chooser == "agent" else []
    sequence = [one for one in log.sequence if one.unit == unit]
    befores: dict[str, Attempt] = {}
    ways: dict[str, str] = {}  # each step's latest way of picking
    for index, one in enumerate(sequence):
        if isinstance(one, Reset):
            nxt = sequence[index + 1] if index + 1 < len(sequence) else None
            redone = nxt.stage if isinstance(nxt, Attempt) and nxt.stage in one.stages else None
            if one.actor == "user" and _changed_then(changes, one.at):
                continue  # PAC's own change: its record says what it set aside
            for stage in one.stages:
                if stage == redone or stage not in STEPS:
                    continue
                by: Maker = "you" if one.actor == "user" else chooser
                why = (
                    f"made of {_MADE_OF[redone]}"
                    if redone in _MADE_OF
                    else "made of the older curve"
                    if one.actor == "user"
                    else "its stage started afresh"
                )
                for key in _keys_of(stage, drafts):
                    asides.setdefault(key, []).append(_Aside(one.at, by, why))
            continue
        if one.stage not in STEPS:
            continue
        at = one.finished_at or one.started_at
        if one.triggered_by == JUDGED or one.how == "judge":
            judged += [(at, element) for element in _checks(one.results, at)]
            continue
        key = _key(one.stage, "M0" if one.stage == "picking" else None)
        draft = _Draft(key, chooser, at, index)
        draft.why = _agent_why(one, chooser)
        way = _way(one, befores.get(key), ways.get(key), picks) if one.stage == "picking" else ""
        draft.how = _agent_how(one, befores.get(key), run_folder, picks, spacing, way)
        draft.result = _check_result(one)
        draft.deleted = one.how == "delete" or any("deleted" in note for note in one.notes)
        draft.failed = one.status == "failed"
        if draft.failed:
            draft.result = [f"Failed: {one.error}" if one.error else "Failed"]
        draft.elements = _checks(one.results, at)
        drafts.append(draft)
        befores[key] = one
        ways[key] = way
    return drafts, asides, judged


def _changed_then(changes: Sequence[Change], at: datetime) -> bool:
    """Whether PAC recorded a change of the window as it logged a reset at `at`."""
    return any(abs(change.at - at) <= SAME_ACTION for change in changes)


def _key(stage: str, mode: str | None) -> str:
    return f"{stage}:{mode or 'M0'}" if stage == "picking" else stage


def _keys_of(stage: str, drafts: Sequence[_Draft]) -> list[str]:
    """The steps a stage's results make: a stage's own; the curves', each mode's."""
    if stage != "picking":
        return [stage]
    return list(dict.fromkeys(one.key for one in drafts if one.stage == "picking")) or [
        "picking:M0"
    ]


def _standing(drafts: Sequence[_Draft], at: datetime) -> _Draft | None:
    """The version of a step standing at `at`: the latest made before it."""
    before = [one for one in drafts if one.at is not None and one.at <= at]
    return max(before, key=lambda one: (one.at or at, one.seq)) if before else None


def _agent_why(run: Attempt, chooser: Chooser) -> str:
    """Why the agent made a version: a check's retry and what it found, its own redo, the user's
    request, a change of the line's settings, the mute trial; nothing for a first run."""
    gate, _, flag = run.triggered_by.partition(":")
    if flag:
        name = CHECKS[gate].name if gate in CHECKS else CHECK_NAMES.get(gate, gate)
        return f"Asked by {name}: {flag_text(flag)}"
    return {
        "initial": "",
        "backtrack": f"Redone {BY_AGENT[chooser]}",
        "asked": "Asked by the user",
        "line change": "With a change of the line's settings",
        "mute trial": "With the mute the mute trial kept",
    }.get(run.triggered_by, run.triggered_by.replace("_", " ").capitalize())


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
    return _picked_how(run, picks)


def _agent_how(
    run: Attempt,
    before: Attempt | None,
    run_folder: Path,
    picks: Sequence[Mapping[str, Any]],
    spacing: float,
    how: str,
) -> list[str]:
    """How the agent made a version, a line each: a pick's way, then its settings; another
    stage's settings; what they change on the version before (the first's, on the run's, or
    said whole)."""
    kind = STEPS[run.stage]
    base: Mapping[str, Any] = (
        before.parameters if before is not None else (kind.base(run_folder) if kind.base else {})
    )
    changed = kind.changes(base, run.parameters, spacing) if kind.changes else []
    if run.stage != "picking":
        if before is None and kind.settings is not None:
            return kind.settings(run.parameters)
        return changed
    if how == "delete":
        return [_WAYS["delete"]]
    count = 0
    if how == "guide":
        count = len(run.parameters.get("guide") or ())
    elif how == "lasso":
        count = len(run.inputs.get("polygon") or ()) or _study_polygon(run, picks)
    way = _WAYS.get(how, how).format(count=count or "a few")
    return [way, *(one for one in changed if not one.startswith("Guided"))]


def _picked_how(run: Attempt, picks: Sequence[Mapping[str, Any]]) -> str:
    """How a pick logged before attempts said it was made: an autonomous run's study kept its
    recipe; else its parameters tell (along a guide: the neighbours' for G4's retry)."""
    recipe = _study_pick(run, picks)
    if recipe is not None:
        how = str(recipe.get("how"))
        if how == "guide":
            return "neighbours" if recipe.get("guide") == "neighbours" else "guide"
        return "tracker" if how in ("auto", "line") else how
    if run.parameters.get("guide"):
        return "neighbours" if run.triggered_by.startswith("G4:") else "guide"
    return "tracker"


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


def _study_polygon(run: Attempt, picks: Sequence[Mapping[str, Any]]) -> int:
    recipe = _study_pick(run, picks)
    polygon = recipe.get("polygon") if recipe is not None else None
    return len(cast(list[Any], polygon)) if isinstance(polygon, list) else 0


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
    from masw.io.quality.journal import study_of

    study = study_of(run_folder.parent, run_folder.name)
    path = study / "picks.jsonl" if study is not None else None
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


def _checks(results: Mapping[str, GateResult], at: datetime) -> list[Element]:
    """The registered checks' verdicts on a version, by gate, each an element of its own: its
    verdict in words, what its flags found, its measures."""
    found: list[Element] = []
    for gate, result in sorted(results.items()):
        kind = CHECKS.get(gate)
        if kind is None or result.unit == LINE:
            continue
        status = verdict_status(result)
        words = {"pass": "passed", "warn": "flagged", "fail": "rejected"}.get(status, status)
        lines = [
            f"{_FLAG_HEADS.get(flag.action.kind, 'Asked again: ')}{finding(flag.message)}"
            for flag in result.flags
            if flag.name not in REFUSALS
        ]
        if refused := [flag_text(flag.name) for flag in result.flags if flag.name in REFUSALS]:
            lines.append(f"No retry: {', '.join(refused)}")
        found.append(
            Element(
                kind="check",
                key=gate,
                title=f"{_named(kind)}: {words}",
                tone=status,
                at=at,
                gate=gate_view(gate, result),
                modules=(Module(key="found", label="Found", lines=tuple(lines)),) if lines else (),
            )
        )
    return found


# What a check did of each flag, as its element says it.
_FLAG_HEADS = {
    "keep": "Kept: ",
    "reject": "Rejected: ",
    "exclude_traces": "Left out: ",
    "exclude_record": "Left out: ",
}


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _named(kind: CheckKind) -> str:
    """A check's name as a title: "Curve check"."""
    return _capitalized(kind.name.removeprefix("the "))


def _check_result(run: Attempt) -> list[str]:
    """What a run made, as its checks kept it: a curve's points, band and wavelengths, an image's
    coherent band, a record's traces; a model's depth informed and fit."""
    g3, g2, g1, g5 = (run.results.get(gate) for gate in ("G3", "G2", "G1", "G5"))
    if g3 is not None and g3.kept.n_points is not None and g3.kept.band_hz is not None:
        lines = [f"{plural(g3.kept.n_points, 'point')}, {span(*g3.kept.band_hz, 'Hz')}"]
        if g3.kept.wavelength_m is not None:
            lines.append(f"Wavelengths {span(*g3.kept.wavelength_m, 'm')}")
        return lines
    if g2 is not None and g2.kept.band_hz is not None:
        low, high = g2.kept.band_hz
        return [f"Coherent from {number(low)} to {number(high)} Hz"]
    if g1 is not None and g1.kept.n_traces is not None:
        return [f"{plural(g1.kept.n_traces, 'trace')} kept"]
    if g5 is not None:
        return _fit_words(g5.metrics)
    return []


def _fit_words(metrics: Iterable[Metric]) -> list[str]:
    """A model's depth informed and worst misfit, as its check measured them."""
    misfits = [m.value for m in metrics if m.name.startswith("misfit") and m.value is not None]
    informed = next(
        (m.value for m in metrics if m.name in ("depth_informed", "useful_depth") and m.value),
        None,
    )
    lines = [f"Informed to {number(informed)} m"] if informed is not None else []
    if misfits:
        lines.append(f"Misfit {number(max(misfits), 2)} at worst")
    return lines


def _change_how(change: Change) -> list[str]:
    """How the user made a version in PAC: its way and what it took."""
    inputs = change.inputs
    if change.how == "box":
        box = _box(inputs)
        return [_PAC_WAYS["box"].format(box=box) if box else "By hand: the ridge inside a box"]
    if change.how == "lasso":
        return [_PAC_WAYS["lasso"].format(count=len(inputs.get("polygon") or ()))]
    if change.how == "inversion":
        lines = [_PAC_WAYS["inversion"]]
        if modes := inputs.get("modes"):
            lines.append(f"Modes: {', '.join(str(one) for one in cast(list[Any], modes))}")
        settings = STEPS["inversion"].settings
        if settings is not None and isinstance(parameters := inputs.get("parameters"), dict):
            lines += settings(cast(dict[str, Any], parameters))
        return lines
    if change.how == "petro":
        return [_PAC_WAYS["petro"].format(model=inputs.get("model", "?"))]
    return [_PAC_WAYS.get(change.how, change.how)]


def _box(inputs: Mapping[str, Any]) -> str:
    """A box's limits in words, each a range or a bound: its frequencies, its velocities, its
    wavelengths."""
    parts: list[str] = []
    for low, high, unit, words in (
        ("fmin", "fmax", "Hz", ""),
        ("vmin", "vmax", "m/s", ""),
        ("lbdmin", "lbdmax", "m", "wavelengths "),
    ):
        lo, hi = inputs.get(low), inputs.get(high)
        if lo is not None and hi is not None:
            parts.append(f"{words}{span(float(lo), float(hi), unit)}")
        elif lo is not None:
            parts.append(f"{words}from {number(float(lo))} {unit}")
        elif hi is not None:
            parts.append(f"{words}up to {number(float(hi))} {unit}")
    return ", ".join(parts)


def _change_result(change: Change) -> list[str]:
    """What a change in PAC made, as its record sums it up."""
    result = change.result
    if change.stage == "picking" and result:
        low, high = result["band_hz"]
        lines = [f"{plural(int(result['n_points']), 'point')}, {span(low, high, 'Hz')}"]
        if result.get("wavelength_m"):
            shortest, longest = result["wavelength_m"]
            lines.append(f"Wavelengths {span(shortest, longest, 'm')}")
        return lines
    if change.stage == "inversion":
        return _model_words(result)
    if change.stage == "petro_inversion" and result.get("soils"):
        soils = cast(list[str], result["soils"])
        lines = [f"{plural(len(soils), 'soil')}: {', '.join(soils)}"]
        if result.get("water_table_m") is not None:
            lines.append(f"Water table at {number(float(result['water_table_m']))} m")
        return lines
    return []


def _model_words(result: Mapping[str, Any]) -> list[str]:
    """A model's summary in words, as the inversion's overview says it."""
    lines: list[str] = []
    if result.get("layers") is not None and result.get("depth_m") is not None:
        layers = int(result["layers"]) - 1
        lines.append(
            f"{plural(layers, 'layer')} over a half-space, modelled to "
            f"{number(float(result['depth_m']))} m"
        )
    if result.get("informed_m") is not None:
        lines.append(f"Informed to {number(float(result['informed_m']))} m")
    if result.get("misfit") is not None:
        lines.append(f"Misfit {number(float(result['misfit']), 2)} at worst")
    return lines


def _later_words(later: Sequence[str]) -> list[str]:
    """The later steps a change set aside, made of what it replaced."""
    names = [STEPS[stage].noun for stage in later if stage in STEPS]
    if not names:
        return []
    return [f"Its {' and its '.join(names)}, made of the older result"]


def _pac_made(
    run_folder: Path,
    unit: str,
    record: bool,
    manifest: RunManifest | None,
    log: QCLog | None,
    drafts: Sequence[_Draft],
    changes: Sequence[Change],
) -> list[_Draft]:
    """What PAC made of `unit` with no record of its own: in a run PAC processed, its first result
    (the record's preprocessing, the window's image); in a run from before PAC recorded its
    changes, what the window's files say it changed (its picks, its inversion, its soil column),
    said by their dates."""
    made: list[_Draft] = []
    first = "preprocessing" if record else "phase_shift"
    if manifest is not None and not any(one.stage == first for one in drafts):
        outcome = (
            next((one for one in manifest.records if one.name == unit), None)
            if record
            else next((one for one in manifest.windows if one.folder == unit), None)
        )
        if outcome is not None:
            at = manifest.finished_at or manifest.started_at
            draft = _Draft(first, "you", at if at.tzinfo else at.replace(tzinfo=UTC), -1)
            draft.why = "PAC's processing"
            if outcome.status == "failed":
                draft.failed = True
                draft.result = [f"Failed: {outcome.error}" if outcome.error else "Failed"]
            made.append(draft)
    if record:
        return made
    # What the window's files say PAC changed, but what PAC recorded as it did it.
    recorded = {
        stage: max(change.at for change in changes if change.stage == stage)
        for stage in {change.stage for change in changes}
    }
    for draft in (
        *_marked_picks(run_folder / unit),
        *_marked_model(run_folder, unit, log),
        *_marked_soils(run_folder, unit, log),
    ):
        last = recorded.get(draft.stage)
        if last is None or (draft.at is not None and draft.at > last + SAME_ACTION):
            made.append(draft)
    return made


def _marked_picks(window: Path) -> list[_Draft]:
    """A window's picks PAC marked before it recorded them: each mode's last change by hand, its
    last automatic pick of M0."""
    made: list[_Draft] = []
    page = f"In PAC's {STEPS['picking'].page} page"
    path = window / EDITS_FILE
    if path.exists():
        data = cast(dict[str, Any], json.loads(path.read_text()))
        modes = data.get("modes")
        edited = data.get("edited_at")
        times = (
            cast(dict[str, str], modes)
            if isinstance(modes, dict)
            else ({EVERY_MODE: edited} if isinstance(edited, str) else {})
        )
        for label, when in times.items():
            key = _key("picking", None if label == EVERY_MODE else label)
            made.append(
                _Draft(
                    key, "you", _parsed(when), 10**9, why=page, how=[_PAC_WAYS["hand"]], hand=True
                )
            )
    path = window / AUTO_FILE
    if path.exists():
        when = _parsed(json.loads(path.read_text()).get("picked_at"))
        made.append(_Draft("picking:M0", "you", when, 10**9, why=page, how=[_PAC_WAYS["auto"]]))
    return made


def _marked_model(run_folder: Path, unit: str, log: QCLog | None) -> list[_Draft]:
    """A window's Vs profile PAC's page made before it recorded its changes: said by its file's
    date."""
    from masw.io.quality.makers import inversion_by_agent
    from sigpipe.masw.inversion.window import SAMPLES_FILE

    samples = run_folder / unit / SAMPLES_FILE
    if not samples.exists() or inversion_by_agent(log, unit, run_folder):
        return []
    page = f"In PAC's {STEPS['inversion'].page} page"
    return [_Draft("inversion", "you", _mtime(samples), 10**9, why=page,
                   how=[_PAC_WAYS["inversion"]])]  # fmt: skip


def _marked_soils(run_folder: Path, unit: str, log: QCLog | None) -> list[_Draft]:
    """A window's soil column PAC's page made before it recorded its changes."""
    from masw.io.quality.makers import soil_by_agent
    from sigpipe.masw.petro.window import MODEL_FILE

    window = run_folder / unit
    model = window / MODEL_FILE
    if not model.exists() or soil_by_agent(log, window):
        return []
    page = f"In PAC's {STEPS['petro_inversion'].page} page"
    how = [_PAC_WAYS["petro"].format(model=_petro_model(run_folder))]
    return [_Draft("petro_inversion", "you", _mtime(model), 10**9, why=page, how=how)]


def _petro_model(run_folder: Path) -> str:
    try:
        config = json.loads((run_folder / "petro_inversion_config.json").read_text())
    except OSError, ValueError:
        return "?"
    return str(cast(dict[str, Any], config).get("model_name", "?"))


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _steps(
    run_folder: Path, unit: str, drafts: Sequence[_Draft], asides: Mapping[str, Sequence[_Aside]]
) -> tuple[HistoryStep, ...]:
    """The unit's steps in the pipeline's order, each its versions in time: each replaced by the
    next; the latest set aside when something set it aside after it, else deleted, failed, or
    the one now, measured when no check judged it."""
    earliest = datetime.min.replace(tzinfo=UTC)
    keys = sorted({one.key for one in drafts}, key=_order)
    steps: list[HistoryStep] = []
    for key in keys:
        kind = STEPS[key.split(":")[0]]
        own = sorted(
            (one for one in drafts if one.key == key), key=lambda one: (one.at or earliest, one.seq)
        )
        versions: list[Version] = []
        for index, draft in enumerate(own):
            nxt = own[index + 1] if index + 1 < len(own) else None
            title = "Deleted" if draft.deleted else kind.made if index == 0 else kind.again
            ended: datetime | None = None
            ended_by: Maker | None = None
            ended_why = ""
            state: State
            if nxt is not None:
                state = "replaced"
                ended, ended_by = nxt.at, nxt.by
                ended_why = "deleted" if nxt.deleted else "made again"
            elif aside := next(
                (one for one in asides.get(key, ()) if draft.at is None or one.at >= draft.at),
                None,
            ):
                state = "set aside"
                ended, ended_by, ended_why = aside.at, aside.by, aside.why
            else:
                state = "deleted" if draft.deleted else "failed" if draft.failed else "current"
            elements = list(draft.elements)
            result = list(draft.result)
            if state == "current":
                result = _current_result(run_folder, unit, key) or result
                measurable = not elements and not draft.hand and key in (kind.stage, "picking:M0")
                if measurable and (measured := _measured(kind, run_folder, unit)):
                    elements.append(measured)
            modules = tuple(
                Module(key=name, label=label, lines=tuple(lines))
                for name, label, lines in (
                    ("why", "Why", [draft.why] if draft.why else []),
                    ("how", "How", draft.how),
                    ("result", "Result", result),
                    ("later", "Set aside", draft.later),
                )
                if lines
            )
            versions.append(
                Version(
                    title=f"{title}, failed" if draft.failed and not draft.deleted else title,
                    by=draft.by,
                    at=draft.at,
                    state=state,
                    modules=modules,
                    elements=tuple(elements),
                    ended=ended,
                    ended_by=ended_by,
                    ended_why=ended_why,
                )
            )
        latest = versions[-1]
        now = latest.state == "current"
        mode = key.split(":")[1] if ":" in key else ""
        steps.append(
            HistoryStep(
                key=key,
                kind=kind.stage,
                title=kind.title.format(mode=mode),
                state=latest.state,
                current=latest if now else None,
                replaced=tuple(reversed(versions[:-1] if now else versions)),
            )
        )
    return tuple(steps)


def _order(key: str) -> tuple[int, int, str]:
    """A step's place: its stage's, then a curve's mode's."""
    stage, _, mode = key.partition(":")
    if not mode:
        return (STEPS[stage].order, 0, "")
    one = Mode.from_label(mode)
    return (STEPS[stage].order, one.number, one.wave)


def _current_result(run_folder: Path, unit: str, key: str) -> list[str]:
    """What the unit's step holds now, read from its files: a curve's points, band and
    wavelengths; a model's layers, depth and fit."""
    window = run_folder / unit
    if key.startswith("picking:"):
        mode = Mode.from_label(key.split(":")[1])
        curve = next((one for one in load_curves(window) or () if one.mode == mode), None)
        if curve is None or not len(curve.fs):
            return []
        fs = np.asarray(curve.fs, dtype=float)
        lengths = np.asarray(curve.vs, dtype=float) / fs
        return [
            f"{plural(int(fs.size), 'point')}, {span(float(fs.min()), float(fs.max()), 'Hz')}",
            f"Wavelengths {span(float(lengths.min()), float(lengths.max()), 'm')}",
        ]
    if key == "inversion":
        from masw.io.changes import model_result

        return _model_words(model_result(window))
    return []


def _measured(kind: StepKind, run_folder: Path, unit: str) -> Element | None:
    """A result no check judged, measured against the agent's limits as its check would take
    it: an element of its own, within or beyond them; none for a check not registered."""
    found = kind.measures(run_folder, unit) if kind.measures is not None else None
    if found is None:
        return None
    gate, metrics = found
    status = measured_status(metrics)
    check = CHECKS.get(gate)
    if status == "none" or check is None:
        return None
    beyond = [
        one.name
        for one in metrics
        if one.threshold is not None and not one.passed and one.name not in WARNED
    ]
    # Each measure beyond its limit, by the flag its check would raise.
    flags = dict.fromkeys(
        next((flag for flag, judged in FLAG_METRICS.items() if name in judged), name)
        for name in beyond
    )
    lines = [f"Beyond the limits: {', '.join(flag_text(flag) for flag in flags)}"] if flags else []
    lines += [f"Warning: {flag_text(name)}" for name in warned_flags(metrics)]
    words = "within the agent's limits" if status == "pass" else "beyond the agent's limits"
    return Element(
        kind="measure",
        key=gate,
        title=f"Measured as {check.name} would: {words}",
        tone=status,
        gate=GateView(gate=gate, verdict=None, metrics=tuple(metrics)),
        modules=(Module(key="found", label="Found", lines=tuple(lines)),) if lines else (),
    )


def _kinds(steps: Sequence[HistoryStep]) -> tuple[HistoryKind, ...]:
    """The kinds of elements the steps hold, for the page's filter: each stage's, each check's,
    the measures'."""
    found: dict[str, HistoryKind] = {}
    for step in steps:
        kind = STEPS[step.kind]
        found.setdefault(kind.stage, HistoryKind(key=kind.stage, group="step", label=kind.label))
        for version in (*([step.current] if step.current else []), *step.replaced):
            for element in version.elements:
                if element.kind == "measure":
                    label = "Measures against the agent's limits"
                    found.setdefault("measure", HistoryKind(key="measure", group="measure",
                                                            label=label))  # fmt: skip
                elif element.key in CHECKS:
                    key = f"check:{element.key}"
                    label = _named(CHECKS[element.key])
                    found.setdefault(key, HistoryKind(key=key, group="check", label=label))
    return tuple(found.values())


def history_of(folder: str, unit: str) -> UnitHistory:
    """The history of `unit` (a window's folder, or a record's file name) in run `folder`;
    raises ValueError for a unit the run does not hold."""
    from masw.io.quality.files import folder_path
    from masw.io.quality.log import read_log

    run_folder = folder_path(folder)
    manifest = read_manifest(run_folder)
    known = (run_folder / unit).is_dir() and unit.startswith("xmid_")
    if manifest is not None:
        known = known or any(one.name == unit for one in manifest.records)
    if not known:
        raise ValueError(f"No unit {unit} in folder={folder}")
    return unit_history(run_folder, unit, read_log(run_folder), manifest)


def picking_base(run_folder: Path) -> Mapping[str, Any]:
    """The picking parameters a run's picks start from: its QC configuration's (PACo's)."""
    return config_section(run_folder, "picking")
