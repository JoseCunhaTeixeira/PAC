"""The runs Visualization offers, and each run's card: who made it (PAC's pages by hand, PACo's
workflow, or its autonomous mode), the settings its windows were built and imaged with, each with
where it comes from (the user, the workflow, the agent, or the preset's default), what was tried
to choose them, the run's history step by step, how far each stage went, and the line's geometry
for the profile plot: every receiver, every shot, every window's receivers."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from masw.io.folders import get_input_folders, get_output_folders
from masw.io.paths import INPUT_DIR
from masw.io.quality.files import (
    folder_path,
    line_geometry,
    preset_stage,
    read_manifest,
    shot_distances,
)
from masw.io.quality.journal import JournalEntry, SearchDone, journal_of, search_of, study_of
from masw.io.quality.log import (
    LINE,
    LOG_FILE,
    QCLog,
    config_section,
    given_settings,
    is_autonomous,
    line_receivers,
    read_log,
)
from masw.io.quality.sources import near_field
from masw.io.quality.tried import (
    MUTE_NAMES,
    LengthChoice,
    TriedTable,
    changes_text,
    length_choice,
    line_tried,
    mutes_tried,
    segments_tried,
    tried_tables,
)
from masw.io.quality.view import REFUSALS, Origin, Setting, flag_text, number, plural, span
from sigpipe.masw.inversion.section import is_inverted
from sigpipe.masw.petro.window import MODEL_FILE
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.presets.making import make_preset
from sigpipe.masw.runs import RunManifest, window_folders, xmid_of

# Who made a run: PAC's pages, PACo's workflow, or its autonomous mode.
type Maker = Literal["manual", "workflow", "autonomous"]
type StageKey = Literal["records", "dispersion", "inversion", "petro"]

GIVEN = "given in your request"
# A workflow run's request, which the assistant writes: the user's values, or its own choices.
REQUESTED = "given in the request: by you, or chosen by the assistant"
DEFAULT = "the preset's default"
PAC = "set by hand"
# How the autonomous agent's commits give a setting the user gave, in the line's notes.
USER_REASON = "given by the user"
ALL_RECORDS = "every record, none left out"
MIN_PASS_SHARE = 0.8  # the workflow's default: the trial windows a window length must pass
MAX_UNCERTAINTY = 0.2  # the workflow's default: picks precise enough to stop at a length
# The preset's entries the card says apart from the preprocessing's stages.
OWN_ENTRIES = ("mode", "masw", "dispersion")
# What asked for a stage to run again, other than a check's flag, in words.
_CAUSES = {
    "backtrack": "after an earlier stage",
    "mute trial": "the mute trial",
    "line change": "a change of the line's settings",
    "asked": "at the request",
}


class RunEntry(BaseModel):
    """A run in the selector."""

    model_config = ConfigDict(frozen=True)

    folder: str
    run_id: str | None  # None: a folder of the older layout
    started_at: datetime | None
    mode: str | None
    made_by: Maker | None
    windows: int
    window_length: int | None = None  # receivers per window; None: not said (the older layout)


class ProfileRuns(BaseModel):
    """A profile, and its runs, newest first."""

    model_config = ConfigDict(frozen=True)

    profile: str
    records: bool  # its records are in the input folder
    runs: tuple[RunEntry, ...]


class StageCount(BaseModel):
    """How far a run went at one stage: the records it used, the windows picked, inverted."""

    model_config = ConfigDict(frozen=True)

    key: StageKey
    done: int
    total: int


class LineWindow(BaseModel):
    """A window along the line: its middle, and its first and last receivers."""

    model_config = ConfigDict(frozen=True)

    key: str  # its folder, xmid_<x>
    xmid: float
    first: float
    last: float


class Step(BaseModel):
    """A step of how the run was made, in order: an entry of its study's journal (the autonomous
    agent's), its search, or what its files say was done (the settings the user gave, a stage,
    a trial), by whom."""

    model_config = ConfigDict(frozen=True)

    id: str = ""  # the journal's (O1, E3, L2, H1...), or the search's (S1)
    kind: str
    by: Origin
    at: datetime | None = None
    text: str
    detail: str = ""  # more on hover: a commit's reasons
    of: tuple[str, ...] = ()  # the entries it rests on


class RunCard(BaseModel):
    """What the page says of a run above its stages, and draws the profile from."""

    model_config = ConfigDict(frozen=True)

    folder: str
    profile: str
    run_id: str | None
    mode: str | None
    made_by: Maker | None
    started_at: datetime | None
    finished_at: datetime | None
    records: bool  # the profile's records are in the input folder: they can be shown
    settings: tuple[Setting, ...]
    tried: tuple[TriedTable, ...] = ()  # what was tried to choose them
    history: tuple[Step, ...] = ()
    stages: tuple[StageCount, ...]
    receivers: tuple[float, ...]  # every receiver's x
    sources: dict[str, float]  # each record's shot x, by file name; none on a passive line
    windows: tuple[LineWindow, ...]
    # The shots a window stacks: from, to this far from its middle (to None: any distance).
    reach: tuple[float, float | None] | None


def made_by(run_folder: Path) -> Maker:
    """Who made the run: PAC's pages, PACo's workflow, which logs its checks there, or its
    autonomous mode, whose study's line the run is."""
    if not (run_folder / LOG_FILE).exists():
        return "manual"
    return "autonomous" if is_autonomous(run_folder) else "workflow"


def list_runs() -> tuple[ProfileRuns, ...]:
    """Every profile with its runs, newest first: the output folder's runs, its folders of the
    older layout under their own name, and the input folder's profiles without a run (their
    records alone)."""
    found: dict[str, list[RunEntry]] = {}
    for folder in get_output_folders():
        try:
            path = folder_path(folder)
            manifest = read_manifest(path)
        except ValueError:  # a link leading out of the output folder, or a manifest unread
            continue
        profile = manifest.profile.name if manifest is not None else folder
        found.setdefault(profile, []).append(_entry(folder, path, manifest))
    inputs = set(get_input_folders())
    return tuple(
        ProfileRuns(profile=name, records=name in inputs, runs=tuple(found.get(name, ())))
        for name in sorted(inputs | set(found))
    )


def run_card(folder: str) -> RunCard:
    run_folder = folder_path(folder)
    manifest = read_manifest(run_folder)
    if manifest is None:
        return _older_card(folder, run_folder)
    line = line_geometry(run_folder, manifest)
    masw = preset_stage(manifest, "masw")
    log = read_log(run_folder)
    settings = processing_settings(run_folder, manifest, log, length_choice(run_folder))
    stages = _stages(run_folder, manifest)
    maker = made_by(run_folder)
    study = study_of(run_folder.parent, run_folder.name) if maker == "autonomous" else None
    journal = journal_of(study) if study is not None else ()
    search = search_of(study) if study is not None else None
    records = next(one for one in settings if one.key == "records")
    return RunCard(
        folder=folder,
        profile=manifest.profile.name,
        run_id=manifest.run_id,
        mode=manifest.preset.mode,
        made_by=maker,
        started_at=manifest.started_at,
        finished_at=manifest.finished_at,
        records=(INPUT_DIR / manifest.profile.name).is_dir(),
        settings=settings,
        tried=tried_tables(run_folder, journal, search),
        history=_study_history(journal, search)
        if study is not None
        else _run_history(run_folder, manifest, log, records, stages),
        stages=stages,
        receivers=line.receivers,
        sources=line.sources,
        windows=_windows(manifest, line.receivers),
        reach=shot_distances(masw) if line.sources else None,
    )


def processing_settings(
    run_folder: Path, manifest: RunManifest, log: QCLog | None, choice: LengthChoice | None
) -> tuple[Setting, ...]:
    """The run's windows and the settings of each stage, as run.json has them, each with where
    it comes from: the line's notes (the workflow's rules, the agent's reasons, the user's
    settings among them), the request, the preset's default, or PAC's form."""
    preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
    defaults: dict[str, Any] = make_preset(manifest.preset.mode).model_dump(mode="json")
    line = log.latest(LINE, "phase_shift") if log is not None else None
    changed: dict[str, Any] = line.parameters if line is not None else {}
    notes = line.notes if line is not None else ()
    chooser: Origin = log.chooser if log is not None else "you"

    def origin(stages: Iterable[str], keys: Iterable[str] | None = None) -> tuple[Origin, str]:
        """Where the values of `stages` (their `keys`, or all) come from, when the line's
        settings did not change them: the preset's default as the units' cards say it, else the
        request or PAC's form."""
        same = all(
            _same(preset[stage].get(key), defaults.get(stage, {}).get(key))
            for stage in stages
            for key in (keys if keys is not None else preset[stage])
        )
        if same:
            return "default", DEFAULT
        return "you", given_why(log)

    def rule(stage: str, key: str | None = None) -> tuple[Origin, str] | None:
        """Who set the value of `stage`'s `key` (or of any of its keys) the line's settings
        changed, and why, from the line's note; None when they did not change it."""
        values = changed.get(stage)
        if not isinstance(values, Mapping) or (key is not None and key not in values):
            return None
        who, reason = noted(notes, stage if key is None else f"{stage} {key}", chooser)
        return who, reason or "chosen by the checks on the line"

    spacing = manifest.profile.receiver_spacing_m
    masw: dict[str, Any] = preset["masw"]
    length, step = int(masw["length"]), int(masw["step"])
    if choice is not None and choice.length == length:
        where, why = chooser, _length_rule(choice, config_section(run_folder, "coherence"))
    else:
        where, why = rule("masw", "length") or origin(["masw"], ["length"])
    stepped, step_why = rule("masw", "step") or origin(["masw"], ["step"])
    settings = [
        Setting(
            key="length",
            label="Window length",
            value=f"{number((length - 1) * spacing, 4)} m",
            detail=plural(length, "receiver"),
            why=why,
            origin=where,
        ),
        Setting(
            key="step",
            label="Window step",
            value=f"{number(step * spacing, 4)} m",
            detail=plural(step, "receiver"),
            why=step_why,
            origin=stepped,
        ),
    ]
    xmids = [window.xmid for window in manifest.windows]
    missing = manifest.n_positions - len(xmids)
    settings.append(
        Setting(
            key="windows",
            label="Windows",
            value=f"{len(xmids)}",
            detail=f"xmid {span(min(xmids), max(xmids), 'm', digits=4)}" if xmids else "",
            why=f"one every {plural(step, 'receiver')} along the "
            f"{plural(manifest.profile.n_receivers, 'receiver')}"
            + (f"; {plural(missing, 'position')} without a shot to stack" if missing > 0 else ""),
            origin=stepped,
        )
    )
    if manifest.profile.kind == "active":
        ends = {
            "nearest": rule("near_field", "distance_m"),
            "farthest": rule("masw", "distance_max"),
        }
        set_by = {end: one for end, one in ends.items() if one is not None}
        if not set_by:
            where, why = origin(["masw"], ["distance_min", "distance_max"])
        else:
            where = _shared([who for who, _ in set_by.values()])
            why = "; ".join(f"{end}, {reason}" for end, (_, reason) in set_by.items())
        settings.append(
            Setting(
                key="reach",
                label="Shots stacked",
                value=_reach(masw),
                detail="from the window's middle"
                + (
                    f", out of its near field ({number(field, 4)} m)"
                    if (field := near_field(log)) is not None
                    else ""
                ),
                why=why,
                origin=where,
            )
        )
    settings.append(_records(manifest, log))
    dispersion: dict[str, Any] = preset["dispersion"]
    where, why = rule("dispersion", "fmax") or rule("dispersion", "fmin") or origin(["dispersion"])
    narrowed = _again(log, "phase_shift", "window")
    settings.append(
        Setting(
            key="band",
            label="Phase shift",
            value=span(dispersion["fmin"], dispersion["fmax"], "Hz"),
            detail=span(dispersion["vmin"], dispersion["vmax"], "m/s"),
            why=why + (f"; done again in {narrowed}" if narrowed else ""),
            origin=where,
        )
    )
    stages = [name for name in preset if name not in OWN_ENTRIES]
    # A record step chosen for the whole line (the mute trial's muting, logged with the line's
    # images; the agent's commits).
    chosen = {name: one for name in stages if (one := rule(name)) is not None}
    if chosen:
        where = _shared([who for who, _ in chosen.values()])
        why = "; ".join(f"its {name}, {reason}" for name, (_, reason) in chosen.items())
    else:
        where, why = origin(stages)
    redone = _again(log, "preprocessing", "record")
    settings.append(
        Setting(
            key="preprocessing",
            label="Preprocessing",
            value="; ".join(stage_text(name, preset[name]) for name in stages) or "none",
            why=why + (f"; done again for {redone}" if redone else ""),
            origin=where,
        )
    )
    return tuple(settings)


def given_why(log: QCLog | None) -> str:
    """Why a value of the run's request is the user's: PAC's form (no log), the user's settings
    as the autonomous mode locks them, or the workflow's request, which the assistant writes."""
    if log is None:
        return PAC
    return GIVEN if log.chooser == "agent" else REQUESTED


def noted(notes: Iterable[str], name: str, chooser: Origin) -> tuple[Origin, str | None]:
    """Who set setting `name` ('<stage> <key>', or a stage) on the line, and the reason the
    line's note on it gives (None: no note): the user, when the agent's commit gives it as
    theirs, else the checks' chooser."""
    note = next(
        (note for note in notes if note.startswith((f"{name}:", f"{name} "))),
        None,
    )
    if note is None:
        return chooser, None
    reason = note.split(": ", 1)[1].rstrip(".")
    return ("you" if reason.startswith(USER_REASON) else chooser), reason


def _shared(origins: Iterable[Origin]) -> Origin:
    """One origin for values set together: the user's when they set them all, else the
    other's."""
    for one in origins:
        if one != "you":
            return one
    return "you"


def _length_rule(choice: LengthChoice, rules: Mapping[str, Any]) -> str:
    """Why the workflow kept `choice.length`, in a line: the trials are in their table."""
    share = float(rules.get("min_pass_share", MIN_PASS_SHARE))
    precision = float(rules.get("max_uncertainty", MAX_UNCERTAINTY))
    kept = next(trial for trial in choice.trials if trial.length == choice.length)
    if kept.passed < math.ceil(share * len(kept.xmids)):
        how = f"no length passed {share:.0%} of its trials: the most passes"
    elif kept.uncertainty is None:
        how = f"the shortest passing {share:.0%} of its trials"
    elif kept.uncertainty <= precision:
        how = f"the first passing {share:.0%} of its trials with picks within {precision:.0%}"
    else:
        how = f"the most precise passing {share:.0%} of its trials"
    return f"the length trial's choice: {how} (each length tried below)"


def _reach(masw: Mapping[str, Any]) -> str:
    """The shots a window stacks, by their distance from its middle (both exclusive)."""
    near, far = shot_distances(masw)
    if near > 0:
        return f"beyond {number(near, 4)} m" if far is None else span(near, far, "m", 4)
    return "any distance" if far is None else f"within {number(far, 4)} m"


def _records(manifest: RunManifest, log: QCLog | None) -> Setting:
    """The records the windows stack, and those they leave out, with why."""
    total = len(manifest.records)
    failed = [record.name for record in manifest.records if record.status == "failed"]
    excluded = manifest.exclusions.records
    receivers, why_receivers = line_receivers(log)
    # Each record's own traces, the receivers left out of every window apart.
    traces = {
        name: own
        for name, left in manifest.exclusions.traces.items()
        if (own := tuple(trace for trace in left if trace not in receivers))
    }
    said: list[str] = []
    if excluded:
        said.append(
            f"{plural(len(excluded), 'record')} left out by the signal check (G1): "
            + ", ".join(_excluded(log, name) for name in excluded)
        )
    if failed:
        said.append(f"{plural(len(failed), 'record')} failed to preprocess: {', '.join(failed)}")
    if receivers:
        said.append(why_receivers.rstrip(".") + " (G1 over the line)")
    if traces:
        count = sum(len(one) for one in traces.values())
        why = _trace_reasons(log, traces)
        said.append(
            f"{plural(count, 'trace')} of {plural(len(traces), 'record')} left out"
            + (f" ({why})" if why else "")
        )
    used = total - len(set(failed) | set(excluded))
    return Setting(
        key="records",
        label="Records",
        value=f"{used} of {total}",
        detail="used",
        why="; ".join(said) or ALL_RECORDS,
        origin="you"
        if log is None
        else log.chooser
        if excluded or traces or receivers
        else "default",
    )


def _excluded(log: QCLog | None, name: str) -> str:
    """Record `name`, with the flags G1 left it out for (its latest, but the spent budget)."""
    result = log.result(name, "preprocessing", "G1") if log is not None else None
    flags = [flag.name for flag in result.flags if flag.name not in REFUSALS] if result else []
    return (
        f"{name} ({', '.join(flag_text(flag) for flag in dict.fromkeys(flags))})" if flags else name
    )


def _trace_reasons(log: QCLog | None, traces: Mapping[str, tuple[int, ...]]) -> str:
    """The flags that left traces out, with the records each did it in."""
    if log is None:
        return ""
    counts: Counter[str] = Counter()
    for record in traces:
        counts.update(
            {
                flag.name
                for flag in log.raised(record, "preprocessing")
                if flag.action.kind == "exclude_traces"
            }
        )
    return ", ".join(
        f"{flag_text(name)} in {plural(count, 'record')}" for name, count in counts.most_common()
    )


def _again(log: QCLog | None, stage: str, unit: str) -> str:
    """The units the assistant ran `stage` again on, and what asked for it, by check: "8 windows
    (G2: 6 aliasing, 2 band at fmax)". A redo's fresh first attempt counts, as every attempt
    after the first run."""
    again = [
        attempt
        for attempt in (log.attempts if log is not None else ())
        if attempt.stage == stage and attempt.unit != LINE and attempt.triggered_by != "initial"
    ]
    if not again:
        return ""
    units = {attempt.unit for attempt in again}
    causes: dict[str, Counter[str]] = {}
    for attempt in again:
        gate, _, flag = attempt.triggered_by.partition(":")
        if flag:
            causes.setdefault(gate, Counter())[flag_text(flag)] += 1
        else:
            causes.setdefault(_CAUSES.get(gate, gate), Counter())[""] += 1
    said = "; ".join(
        f"{cause}: " + ", ".join(f"{count} {what}".rstrip() for what, count in counts.most_common())
        for cause, counts in causes.items()
    )
    return f"{plural(len(units), unit)} ({said})"


def stage_text(name: str, values: Mapping[str, Any]) -> str:
    """A preprocessing stage in words: "no muting", "filtering iir (fmin 5, fmax 80)"."""
    method = values.get("method")
    if method == "none":
        return f"no {name}"
    t0 = values.get("t0")
    if name == "trigger" and t0 is None:
        return "trigger from each record's file"
    if name == "trigger" and isinstance(t0, int | float):
        return (
            f"trigger at {number(t0 * 1000)} ms"
            if 0 < abs(t0) < 1
            else f"trigger at {number(t0)} s"
        )
    said = [
        f"{key} {_value(value)}"
        for key, value in values.items()
        if key != "method" and value is not None
    ]
    head = f"{name} {method}" if method is not None else name
    return head + (f" ({', '.join(said)})" if said else "")


def _value(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return number(value)


def is_default(values: object, defaults: object) -> bool:
    """Whether a stage's `values` are its preset's `defaults`, field by field as `_same` says."""
    if isinstance(values, Mapping) and isinstance(defaults, Mapping):
        found = cast(Mapping[str, Any], values)
        expected = cast(Mapping[str, Any], defaults)
        return all(_same(value, expected.get(key)) for key, value in found.items())
    return _same(cast(object, values), defaults)


def _same(value: object, default: object) -> bool:
    """A value as the preset's default has it; a default derived from the profile (None) takes
    any value."""
    if default is None:
        return True
    if isinstance(value, int | float) and isinstance(default, int | float):
        return math.isclose(float(value), float(default))
    return value == default


def _counts(windows: list[Path]) -> tuple[StageCount, ...]:
    """The windows imaged, picked, inverted and inverted to a soil column."""
    total = len(windows)
    return (
        StageCount(
            key="dispersion", done=sum((one / CURVES_FILE).exists() for one in windows), total=total
        ),
        StageCount(key="inversion", done=sum(is_inverted(one) for one in windows), total=total),
        StageCount(
            key="petro", done=sum((one / MODEL_FILE).exists() for one in windows), total=total
        ),
    )


def _stages(run_folder: Path, manifest: RunManifest) -> tuple[StageCount, ...]:
    excluded = set(manifest.exclusions.records)
    used = sum(
        record.status == "succeeded" and record.name not in excluded for record in manifest.records
    )
    windows = [run_folder / unit for unit in window_folders(run_folder)]
    return (
        StageCount(key="records", done=used, total=len(manifest.records)),
        *_counts(windows),
    )


def _windows(manifest: RunManifest, receivers: tuple[float, ...]) -> tuple[LineWindow, ...]:
    """Each window's middle and its first and last receivers, as sigpipe builds them: `length`
    receivers every `step` receivers."""
    masw = preset_stage(manifest, "masw")
    length, step = int(masw["length"]), int(masw["step"])
    ends = [
        (receivers[start], receivers[start + length - 1])
        for start in range(0, len(receivers) - length + 1, step)
    ]
    tolerance = manifest.profile.receiver_spacing_m / 2
    found: list[LineWindow] = []
    for window in manifest.windows:
        near = min(ends, key=lambda pair: abs(sum(pair) / 2 - window.xmid), default=None)
        first, last = (
            near
            if near is not None and abs(sum(near) / 2 - window.xmid) <= tolerance
            else (window.xmid, window.xmid)
        )
        found.append(LineWindow(key=window.folder, xmid=window.xmid, first=first, last=last))
    return tuple(found)


def _entry(folder: str, path: Path, manifest: RunManifest | None) -> RunEntry:
    if manifest is None:
        return RunEntry(
            folder=folder,
            run_id=None,
            started_at=None,
            mode=None,
            made_by=None,
            windows=len(window_folders(path)),
        )
    length = preset_stage(manifest, "masw").get("length")
    return RunEntry(
        folder=folder,
        run_id=manifest.run_id,
        started_at=manifest.started_at,
        mode=manifest.preset.mode,
        made_by=made_by(path),
        windows=len(manifest.windows),
        window_length=int(length) if length is not None else None,
    )


def _older_card(folder: str, run_folder: Path) -> RunCard:
    """A folder of the older layout: its windows, without a manifest to say how they were made."""
    units = window_folders(run_folder)
    return RunCard(
        folder=folder,
        profile=folder,
        run_id=None,
        mode=None,
        made_by=None,
        started_at=None,
        finished_at=None,
        records=(INPUT_DIR / folder).is_dir(),
        settings=(),
        stages=_counts([run_folder / unit for unit in units]),
        receivers=(),
        sources={},
        windows=tuple(
            LineWindow(key=unit, xmid=xmid_of(unit), first=xmid_of(unit), last=xmid_of(unit))
            for unit in units
        ),
        reach=None,
    )


# The journal's entries the user made: what they asked, the settings they gave, a request after
# the report; the agent made the others (its reasoning, and code's record of what it did).
_USER_KINDS = frozenset({"objective", "settings", "reopen"})


def _study_history(journal: Iterable[JournalEntry], search: SearchDone | None) -> tuple[Step, ...]:
    """The history of a run the autonomous mode made: its study's journal, the search at its
    time."""
    steps = [
        Step(
            id=entry.id,
            kind=entry.kind,
            by="you" if entry.kind in _USER_KINDS else "agent",
            at=entry.at,
            text=text,
            detail=detail,
            of=entry.of,
        )
        for entry in journal
        for text, detail in [_entry_text(entry)]
    ]
    if search is not None:
        made = Step(id="S1", kind="search", by="agent", at=search.at, text=_search_text(search))
        later = next((n for n, one in enumerate(steps) if one.at and one.at > search.at), None)
        steps.insert(len(steps) if later is None else later, made)
    return tuple(steps)


def _entry_text(entry: JournalEntry) -> tuple[str, str]:
    """A journal entry in words, and more on hover: a commit's changes, their reasons on hover;
    an experiment's variants with their passes; else its text without its id."""
    data = entry.data
    changes, reasons = data.get("changes"), data.get("reasons")
    if entry.kind == "commit" and isinstance(changes, dict) and changes:
        how = "Made run" if data.get("new_run") else "Changed the line in run"
        said = changes_text(cast(dict[str, Any], changes))
        why = cast(dict[str, Any], reasons) if isinstance(reasons, dict) else {}
        return (
            f"{how} {data.get('run_id', '')}: {said}",
            "; ".join(f"{key}: {reason}" for key, reason in why.items()),
        )
    variants = data.get("variants")
    if entry.kind == "experiment" and isinstance(variants, list) and variants:
        tried = [cast(dict[str, Any], one) for one in cast(list[Any], variants)]
        count = max(int(one.get("windows", 0)) for one in tried)
        return (
            f"On {plural(count, 'trial window')}: "
            + "; ".join(_variant_text(one) for one in tried),
            "",
        )
    return entry.text.removeprefix(entry.id).strip(), ""


def _variant_text(variant: Mapping[str, Any]) -> str:
    """An experiment's variant in a line: "2, masw length 36: 9 of 12 passing"."""
    passing = f"{variant.get('passing', 0)} of {variant.get('windows', 0)} passing"
    if variant.get("label") == "current":
        return f"the line's settings, {passing}"
    changes = cast(Mapping[str, Any], variant.get("changes") or {})
    return f"{variant.get('label')}, {changes_text(changes)}: {passing}"


def _search_text(search: SearchDone) -> str:
    """The search in a line: the grid's variants, those too poor to show, how the eye judged
    the others, the finalists."""
    said = (
        f"{plural(search.variants, 'variant')} of the grid over "
        f"{plural(search.settings, 'setting')}"
    )
    if search.dropped:
        said += (
            f", {search.dropped} too poor to show (no image, or under a quarter of the windows "
            "passing)"
        )
    if search.sheets:
        said += (
            f"; the others judged by eye in {plural(search.sheets, 'sheet')} over "
            f"{plural(search.rounds, 'round')}"
        )
    if search.finalists and search.experiment:
        said += (
            f"; the finalists {', '.join(search.finalists)} tried side by side in "
            f"{search.experiment}"
        )
    return said


def _run_history(
    run_folder: Path,
    manifest: RunManifest,
    log: QCLog | None,
    records: Setting,
    stages: Iterable[StageCount],
) -> tuple[Step, ...]:
    """The history of a run PAC or the workflow made, from its files: the settings the user gave
    before each stage, the records used, the workflow's trials on the line, the windows imaged,
    picked, inverted and inverted to soil columns, each by who did it."""
    done = {count.key: count for count in stages}
    given = given_settings(run_folder)

    def gave(family: str) -> list[Step]:
        return [
            Step(
                kind="given",
                by="you",
                at=one.at,
                text=f"Your settings for the {family}: {changes_text(one.values)}",
            )
            for one in given
            if one.family == family
        ]

    def stage(kind: str, log_stage: str, text: str, unit: str = "window") -> Step:
        attempts = [one for one in log.attempts if one.stage == log_stage] if log else []
        again = _again(log, log_stage, unit)
        return Step(
            kind=kind,
            by=log.chooser if log is not None and attempts else "you",
            at=min((one.started_at for one in attempts), default=None),
            text=text + (f"; done again: {again}" if again else ""),
        )

    steps = gave("processing")
    used = records.value.split(" of ")
    left_out = f": {records.why}" if records.why != ALL_RECORDS else ""
    steps.append(
        stage(
            "records",
            "preprocessing",
            f"{used[0]} of {plural(int(used[-1]), 'record')} used{left_out}",
            "record",
        ).model_copy(update={"at": manifest.started_at})
    )
    steps += _trials(run_folder)
    imaged = len(manifest.windows)
    steps.append(stage("images", "phase_shift", f"{plural(imaged, 'window')} imaged"))
    steps += _line_tries(run_folder)
    steps += gave("picking")
    picked = done["dispersion"]
    steps.append(
        stage("picks", "picking", f"{picked.done} of {plural(picked.total, 'window')} picked")
    )
    steps += gave("inversion")
    for key, log_stage, what in (
        ("inversion", "inversion", "inverted"),
        ("petro", "petro_inversion", "inverted to a soil column"),
    ):
        count = done[key]
        if count.done:
            steps.append(
                stage(key, log_stage, f"{count.done} of {plural(count.total, 'window')} {what}")
            )
    return tuple(steps)


def _trials(run_folder: Path) -> list[Step]:
    """The workflow's trials before the line's windows: the window length, the mute, a passive
    line's segments, each with the one kept."""
    steps: list[Step] = []
    choice = length_choice(run_folder)
    if choice is not None and choice.trials:
        steps.append(
            Step(
                kind="length",
                by="workflow",
                text=f"{plural(len(choice.trials), 'length')} tried, "
                f"{plural(choice.length, 'receiver')} kept",
            )
        )
    mutes = mutes_tried(run_folder)
    if mutes:
        kept = next((one.candidate for one in mutes if one.kept), None)
        steps.append(
            Step(
                kind="mute",
                by="workflow",
                text=f"{plural(len(mutes), 'mute')} tried, {_mute_kept(kept)}",
            )
        )
    segments = segments_tried(run_folder)
    if segments:
        kept_segments = next((one for one in segments if one.kept), None)
        steps.append(
            Step(
                kind="segments",
                by="workflow",
                text=f"{plural(len(segments), 'segment length')} tried, "
                + (
                    f"{number(kept_segments.segment_s)} s kept"
                    if kept_segments is not None
                    else "none kept"
                ),
            )
        )
    return steps


def _mute_kept(candidate: str | None) -> str:
    """The mute a trial kept, in words."""
    if candidate is None:
        return "none kept"
    return "no mute kept" if candidate == "none" else f"{MUTE_NAMES.get(candidate, candidate)} kept"


def _line_tries(run_folder: Path) -> list[Step]:
    """The workflow's changes of the line's settings, asked by the windows whose curve failed."""
    tries = line_tried(run_folder)
    if not tries:
        return []
    kept = [one.change for one in tries if one.kept]
    return [
        Step(
            kind="line",
            by="workflow",
            text=f"{plural(len(tries), 'change')} of the line's settings tried, {len(kept)} kept"
            + (f" ({'; '.join(kept)})" if kept else ""),
        )
    ]
