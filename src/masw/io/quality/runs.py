"""The runs Visualization offers, and each run's card: who made it (PAC's pages by hand, PACo's
workflow, or its autonomous mode), the settings its windows were built and imaged with, each with
where it comes from (the user, the workflow, the agent, or the preset's default), what was tried
to choose them, the run's history step by step, how far each stage went, and the line's geometry
for the profile plot: every receiver, every shot, every window's receivers."""

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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
from masw.io.quality.journal import (
    JournalEntry,
    SearchDone,
    SurveyDone,
    corrected,
    journal_of,
    search_of,
    study_of,
    survey_of,
)
from masw.io.quality.log import (
    LINE,
    LOG_FILE,
    Given,
    QCLog,
    given_settings,
    is_autonomous,
    line_receivers,
    read_log,
)
from masw.io.quality.sources import near_field
from masw.io.quality.tried import (
    SEP,
    LengthChoice,
    TriedTable,
    length_choice,
    line_tried,
    mute_name,
    mutes_tried,
    plain,
    segments_tried,
    settings_said,
    shots_used,
    step_words,
    tried_tables,
)
from masw.io.quality.view import (
    REFUSALS,
    Origin,
    Setting,
    flag_text,
    number,
    plural,
    receivers,
)
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
# The keys of the card's settings, by their stage, where not the stage's name.
_KEYS = {"dispersion": "band", "muting": "mute"}
# The origins, the one to show first for values set together.
_RANKED: tuple[Origin, ...] = ("agent", "workflow", "checks", "you", "default")
# A journal entry's id the agent's reasons name in parentheses: "(E2)", "(L1, E2)", "(E2 current)".
_REFERENCES = re.compile(r"\s*\((?:[A-Z]\d+|current)(?:[\s,;]+(?:[A-Z]\d+|current))*\)")
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
    """A step of what was done, in order: a title and its facts, a line each; by whom. From the
    study's journal (the autonomous mode's), or from what the run's files say was done."""

    model_config = ConfigDict(frozen=True)

    kind: str
    by: Origin
    at: datetime | None = None
    title: str
    lines: tuple[str, ...] = ()
    detail: str = ""  # more on hover


class Reason(BaseModel):
    """Why a setting is what it is, in a line: the setting, the reason in words (the assistant's
    own, or the value the user set), and whose."""

    model_config = ConfigDict(frozen=True)

    setting: str
    text: str
    by: Origin


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
    reasons: tuple[Reason, ...] = ()  # the assistant's own reasons, and what the user set
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
    stages = _stages(run_folder, manifest)
    maker = made_by(run_folder)
    study = study_of(run_folder.parent, run_folder.name) if maker == "autonomous" else None
    journal = journal_of(study) if study is not None else ()
    search = search_of(study) if study is not None else None
    survey = survey_of(study) if study is not None else None
    spacing = manifest.profile.receiver_spacing_m
    defaults = make_preset(manifest.preset.mode).model_dump(mode="json")
    settings = (
        *processing_settings(manifest, log, length_choice(run_folder)),
        _curves(stages, journal, maker),
        _profiles(stages, journal, log, maker),
    )
    picked = corrected(study, run_folder.name) if study is not None else 0
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
        tried=tried_tables(
            run_folder,
            journal,
            search,
            study,
            spacing,
            defaults,
            survey.reach_m if survey is not None else None,
        ),
        reasons=_reasons(
            journal, settings, given_settings(run_folder), spacing, int(masw["length"])
        ),
        history=_study_history(journal, search, survey, manifest, log, picked)
        if study is not None
        else _run_history(run_folder, manifest, log, stages),
        stages=stages,
        receivers=line.receivers,
        sources=line.sources,
        windows=_windows(manifest, line.receivers),
        reach=shot_distances(masw) if line.sources else None,
    )


def processing_settings(
    manifest: RunManifest, log: QCLog | None, choice: LengthChoice | None
) -> tuple[Setting, ...]:
    """The run's settings in plain words, in the order its records went through them, each with
    where it comes from and why: its windows, the records used, the shots the windows stack, then
    each step of its method in its order (PAC's pages'); a step off by default and left off is not
    said."""
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
        sized: tuple[Origin, str] = (chooser, _length_rule(choice))
    else:
        sized = rule("masw", "length") or origin(["masw"], ["length"])
    stepped = rule("masw", "step") or origin(["masw"], ["step"])
    missing = manifest.n_positions - len(manifest.windows)
    reasons = [sized[1]]
    if stepped[0] != "default":
        reasons.append(f"one every {receivers(step, spacing, length=False)}: {stepped[1]}")
    if missing > 0:
        reasons.append(f"{plural(missing, 'position')} without a shot to stack")
    settings = [
        Setting(
            key="windows",
            label="Windows",
            value=receivers(length, spacing),
            detail=SEP.join(
                (
                    f"every {receivers(step, spacing, length=False)}",
                    plural(len(manifest.windows), "window"),
                )
            ),
            why="; ".join(reasons),
            origin=_strongest([sized[0], stepped[0]]),
        )
    ]
    settings.append(_records(manifest, log))
    if manifest.profile.kind == "active":
        field = near_field(log)
        nearest = (
            f"nearest, {number(field, 4)} m from a window's receivers"
            if field is not None
            else "nearest"
        )
        ends = {nearest: rule("near_field", "distance_m"), "farthest": rule("masw", "distance_max")}
        set_by = {end: one for end, one in ends.items() if one is not None}
        if set_by:
            where = _strongest([who for who, _ in set_by.values()])
            why = "; ".join(f"{end}: {reason}" for end, (_, reason) in set_by.items())
        else:
            where, why = origin(["masw"], ["distance_min", "distance_max"])
        settings.append(
            Setting(
                key="reach",
                label="Shots used",
                value=shots_used(masw, field, length, spacing),
                detail="from each window's middle",
                why=why,
                origin=where,
            )
        )
    for name in preset:
        found = step_words(name, preset, every=False) if name not in ("mode", "masw") else None
        if found is None:
            continue
        label, (value, *detail) = found
        if name == "dispersion":
            where, why = (
                rule("dispersion", "fmax") or rule("dispersion", "fmin") or origin(["dispersion"])
            )
            if again := _again(log, "phase_shift", "window"):
                why += f"; imaged again: {again}"
        else:
            where, why = rule(name) or origin([name])
        if value == "none" and where == "default":
            continue
        settings.append(
            Setting(
                key=_KEYS.get(name, name),
                label=label,
                value=value,
                detail=SEP.join(detail),
                why=why,
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
    line's note on it gives, its journal references left out (None: no note): the user, when the
    agent's commit gives it as theirs, else the checks' chooser."""
    note = next(
        (note for note in notes if note.startswith((f"{name}:", f"{name} "))),
        None,
    )
    if note is None:
        return chooser, None
    reason = plain_reason(note.split(": ", 1)[1])
    return ("you" if reason.startswith(USER_REASON) else chooser), reason


def plain_reason(reason: str) -> str:
    """A reason the agent wrote, without the journal's references it names: "(E2)", "(L1, E2)"."""
    return _REFERENCES.sub("", reason).strip().rstrip(".")


def _strongest(origins: Iterable[Origin]) -> Origin:
    """The origin to show for values set together: the agent's or the workflow's choice over
    the checks', over the user's, over the default."""
    found = set(origins)
    return next(one for one in _RANKED if one in found)


def _length_rule(choice: LengthChoice) -> str:
    """Why the workflow kept `choice.length`, in a line: the trials are in their table."""
    kept = next(trial for trial in choice.trials if trial.length == choice.length)
    return (
        f"the length trial kept it: {kept.passed} of its {len(kept.xmids)} test windows passed "
        f"({plural(len(choice.trials), 'length')} tried)"
    )


def _records(manifest: RunManifest, log: QCLog | None) -> Setting:
    """The records the windows stack, those left out and why, by what left them out."""
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
        said.append("left out by the signal check: " + "; ".join(_left_out(log, excluded)))
    if failed:
        said.append(f"failed to preprocess: {', '.join(failed)}")
    if receivers:
        said.append(why_receivers.rstrip("."))
    if traces:
        count = sum(len(one) for one in traces.values())
        why = _trace_reasons(log, traces)
        said.append(
            f"{plural(count, 'trace')} of {plural(len(traces), 'record')} left out"
            + (f" ({why})" if why else "")
        )
    if redone := _again(log, "preprocessing", "record"):
        said.append(f"preprocessed again: {redone}")
    used = total - len(set(failed) | set(excluded))
    left = total - used
    return Setting(
        key="records",
        label="Records",
        value=f"{used} of {total}",
        detail=f"{left} left out" if left else "all used",
        why="; ".join(said) or ALL_RECORDS,
        origin="checks" if excluded or traces or receivers or redone else "default",
    )


def _left_out(log: QCLog | None, names: Iterable[str]) -> list[str]:
    """The records G1 left out, by the flags it left them out for: "low SNR: 101.dat, 129.dat",
    "low coherence: 133.dat"."""
    by_flag: dict[str, list[str]] = {}
    for name in names:
        result = log.result(name, "preprocessing", "G1") if log is not None else None
        flags = [flag.name for flag in result.flags if flag.name not in REFUSALS] if result else []
        said = ", ".join(flag_text(flag) for flag in dict.fromkeys(flags)) or "no reason said"
        by_flag.setdefault(said, []).append(name)
    return [f"{flags}: {', '.join(found)}" for flags, found in by_flag.items()]


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
    """The units the checks ran `stage` again on, and what asked for it: "8 windows (6
    aliasing, 2 band at fmax)". A redo's fresh first attempt counts, as every attempt after the
    first run."""
    again = [
        attempt
        for attempt in (log.attempts if log is not None else ())
        if attempt.stage == stage and attempt.unit != LINE and attempt.triggered_by != "initial"
    ]
    if not again:
        return ""
    units = {attempt.unit for attempt in again}
    # A check's flag counted; another cause (a change of the line's settings) said once.
    flags: Counter[str] = Counter()
    others: dict[str, None] = {}
    for attempt in again:
        gate, _, flag = attempt.triggered_by.partition(":")
        if flag:
            flags[flag_text(flag)] += 1
        else:
            others[_CAUSES.get(gate, gate)] = None
    said = [f"{count} {flag}" for flag, count in flags.most_common()] + list(others)
    return f"{plural(len(units), unit)} ({', '.join(said)})"


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


# What the model's check said of a Vs profile, in the words PAC's pages use.
MODEL_CHECK = "passed, flagged or rejected by the model's check"
# How the freeze says its curves: "Frozen from L1: 68 curves accepted, 8 windows rejected".
_FROZEN = re.compile(r"(\d+) curves accepted, (\d+) windows rejected")
# Who chose what a run's maker chose, as the card tags it.
_MAKERS: dict[Maker, Origin] = {"autonomous": "agent", "workflow": "workflow", "manual": "you"}
# The settings as the commits' reasons name them ("<stage> <key>", or a stage), by the names the
# card gives them.
_SETTING_NAMES = {
    "masw length": "Windows",
    "masw step": "Windows",
    "masw distance_max": "Shots used",
    "masw distance_min": "Shots used",
    "near_field_m": "Shots used",
    "near_field": "Shots used",
    "trigger": "Mute",
    "muting": "Mute",
    "filtering": "Filter",
    "dispersion": "Dispersion image",
    "mode": "Method",
    "slicing": "Segments",
    "selection": "Segment selection",
    "whitening": "Whitening",
    "normalization": "Normalization",
    "stacking": "Stacking",
    "image_stacking": "Image stacking",
    "picking": "Picking",
}


def _curves(stages: Iterable[StageCount], journal: Sequence[JournalEntry], maker: Maker) -> Setting:
    """The curves tile: those the freeze kept and rejected, else those picked."""
    picked = next(count for count in stages if count.key == "dispersion")
    frozen = next(
        (
            found
            for entry in reversed(journal)
            if entry.kind == "freeze" and (found := _FROZEN.search(entry.text))
        ),
        None,
    )
    by = _MAKERS[maker]
    if frozen is not None:
        accepted, rejected = int(frozen.group(1)), int(frozen.group(2))
        return Setting(
            key="curves",
            label="Curves",
            value=f"{accepted} of {accepted + rejected} accepted",
            detail=f"{rejected} rejected" if rejected else "",
            why="the assistant accepted or rejected each window's curve",
            origin=by,
        )
    return Setting(
        key="curves",
        label="Curves",
        value=f"{picked.done} of {picked.total}",
        detail="windows picked",
        why="picked by the assistant" if maker != "manual" else "picked by you",
        origin=by if picked.done else "default",
    )


def _profiles(
    stages: Iterable[StageCount], journal: Sequence[JournalEntry], log: QCLog | None, maker: Maker
) -> Setting:
    """The Vs profiles tile: how many passed the model's check of those made, how many it
    flagged and rejected; else how many windows were inverted."""
    inverted = next(count for count in stages if count.key == "inversion")
    verdicts = _model_verdicts(journal, log)
    if verdicts:
        passed, flagged, rejected = _verdict_counts(verdicts)
        return Setting(
            key="profiles",
            label="Vs profiles",
            value=f"{passed} of {len(verdicts)} passed",
            detail=SEP.join(_counted((flagged, "flagged"), (rejected, "rejected"))),
            why=MODEL_CHECK,
            origin=_MAKERS[maker],
        )
    if inverted.done:
        return Setting(
            key="profiles",
            label="Vs profiles",
            value=f"{inverted.done} of {inverted.total}",
            detail="windows inverted",
            why="inverted by the assistant" if maker != "manual" else "inverted by you",
            origin=_MAKERS[maker],
        )
    return Setting(
        key="profiles",
        label="Vs profiles",
        value="none yet",
        why="no window inverted yet",
        origin="default",
    )


def _verdict_counts(verdicts: Mapping[str, str]) -> tuple[int, int, int]:
    """How many models passed the model's check, how many it flagged (still after its retries) and
    rejected."""
    counts = Counter(verdicts.values())
    return counts["pass"], counts["retry"], counts["reject"]


def _counted(*counts: tuple[int, str]) -> list[str]:
    """Counts in words, a line each, those of none left out: "3 flagged"."""
    return [f"{count} {word}" for count, word in counts if count]


def _model_verdicts(journal: Sequence[JournalEntry], log: QCLog | None) -> dict[str, str]:
    """Each inverted window's latest verdict on its model: the autonomous mode's inversions, from
    its journal; else the checks' in the run's log; none for a run no check judged."""
    found: dict[str, str] = {}
    for entry in journal:
        if entry.kind == "inversion":
            found |= {
                str(unit): str(verdict)
                for unit, verdict in cast(dict[str, Any], entry.data.get("g5") or {}).items()
            }
    if found or log is None:
        return found
    for attempt in log.attempts:
        if (
            attempt.stage == "inversion"
            and attempt.unit != LINE
            and (g5 := attempt.results.get("G5"))
        ):
            found[attempt.unit] = g5.verdict
    return found


def _reasons(
    journal: Sequence[JournalEntry],
    settings: Sequence[Setting],
    given: Sequence[Given],
    spacing: float,
    length: int,
) -> tuple[Reason, ...]:
    """Why the settings are what they are: the assistant's own reasons for those it chose (its
    commits', the latest of each setting; a commit's reasons for one setting `SEP` apart, each
    once), then
    the settings the user set, each its value in words (the windows' `length` where the shots
    used count from)."""
    found: dict[tuple[str, str], Reason] = {}
    for entry in journal:
        if entry.kind != "commit":
            continue
        changes = cast(dict[str, Any], entry.data.get("changes") or {})
        own: dict[str, list[str]] = {}
        theirs: dict[str, str] = {}
        for key, reason in cast(dict[str, Any], entry.data.get("reasons") or {}).items():
            name = _SETTING_NAMES.get(key) or _SETTING_NAMES.get(key.split(" ")[0]) or key
            said = plain_reason(str(reason))
            if said.startswith(USER_REASON):
                stage, _, field = key.partition(" ")
                value = changes.get(stage)
                values = (
                    {stage: {field: cast(dict[str, Any], value).get(field)}}
                    if field and isinstance(value, dict)
                    else {stage: value}
                )
                theirs[name] = settings_said(values, spacing, length).get(name) or said
            else:
                own.setdefault(name, []).append(said[:1].upper() + said[1:])
        for name, texts in own.items():
            found[name, "agent"] = Reason(
                setting=name, text=SEP.join(dict.fromkeys(texts)), by="agent"
            )
        for name, text in theirs.items():
            found[name, "you"] = Reason(setting=name, text=text, by="you")
    for one in given:
        if one.family == "processing":
            for name, value in settings_said(one.values, spacing, length).items():
                found[name, "you"] = Reason(setting=name, text=value, by="you")
    if not journal and not given:
        for setting in settings:
            # What the run gave, not a setting: its records used, its curves, its Vs profiles.
            if setting.origin == "you" and setting.key not in ("records", "curves", "profiles"):
                found[setting.label, "you"] = Reason(
                    setting=setting.label, text=setting.value, by="you"
                )
    return tuple(found.values())


def _study_history(
    journal: Sequence[JournalEntry],
    search: SearchDone | None,
    survey: SurveyDone | None,
    manifest: RunManifest,
    log: QCLog | None,
    picked: int,
) -> tuple[Step, ...]:
    """What the autonomous mode did, a step each, in order: what the user asked, the records, the
    choice of the settings, the line processed, the curves picked, the Vs profiles; the
    assistant's own reasoning left to its journal."""
    spacing = manifest.profile.receiver_spacing_m
    length = int(preset_stage(manifest, "masw")["length"])
    steps: list[Step] = []
    commits = 0
    for entry in journal:
        if entry.kind == "objective":
            said = entry.text.removeprefix(entry.id).strip()
            lines = tuple(part.strip().capitalize() for part in said.split(",") if part.strip())
            steps.append(
                Step(kind="objective", by="you", at=entry.at, title="You asked", lines=lines)
            )
        elif entry.kind == "settings":
            changed = tuple(str(one) for one in cast(list[Any], entry.data.get("changed") or []))
            steps.append(
                Step(kind="settings", by="you", at=entry.at, title="You changed", lines=changed)
            )
        elif entry.kind == "reopen":
            asked = str(entry.data.get("request") or "")
            steps.append(
                Step(kind="reopen", by="you", at=entry.at, title="You asked again", lines=(asked,))
            )
        elif entry.kind == "commit":
            commits += 1
            changes = cast(dict[str, Any], entry.data.get("changes") or {})
            title = "Processed the whole line" if commits == 1 else "Changed the line's settings"
            steps.append(
                Step(
                    kind="commit",
                    by="agent",
                    at=entry.at,
                    title=title,
                    lines=tuple(plain(changes, spacing, length)),
                )
            )
        elif entry.kind == "freeze":
            found = _FROZEN.search(entry.text)
            lines = [f"{plural(len(manifest.windows), 'window')} picked automatically"]
            if picked:
                lines.append(f"{picked} picked again by the assistant")
            if found:
                lines += [f"{found.group(1)} accepted", f"{found.group(2)} rejected"]
            steps.append(
                Step(
                    kind="freeze",
                    by="agent",
                    at=entry.at,
                    title="Picked the curves",
                    lines=tuple(lines),
                )
            )
    if survey is not None:
        steps.append(_survey_step(survey, journal, manifest, log))
    if (tried := _tried_step(search)) is not None:
        steps.append(tried)
    if (inverted := _inverted_step(journal)) is not None:
        steps.append(inverted)
    first = min((step.at for step in steps if step.at is not None), default=None)
    return tuple(sorted(steps, key=lambda step: step.at or first or datetime.min))


def _survey_step(
    survey: SurveyDone, journal: Sequence[JournalEntry], manifest: RunManifest, log: QCLog | None
) -> Step:
    """The records looked at, after the objective: how many, the band they share, those left
    out and why."""
    records = "noise records" if survey.kind == "passive" else "shots"
    lines = [f"{len(survey.shots)} {records} on {survey.receivers} receivers"]
    if survey.usable_spread_hz is not None:
        low, high = survey.usable_spread_hz
        lines.append(f"Signal usable from {number(low)} to {number(high)} Hz")
    excluded = manifest.exclusions.records
    if excluded:
        lines.append(f"{plural(len(excluded), 'record')} left out")
        lines += [one[:1].upper() + one[1:] for one in _left_out(log, excluded)]
    asked = next((entry.at for entry in journal if entry.kind == "objective"), None)
    return Step(
        kind="survey", by="agent", at=asked, title="Looked at your records", lines=tuple(lines)
    )


def _tried_step(search: SearchDone | None) -> Step | None:
    """The choice of the settings, in a few lines: what the search made, dropped and judged, and
    how it ended."""
    if search is None:
        return None
    ended = [f"The best of the last {len(search.finalists)} chosen"] if search.best else []
    lines = [
        f"{search.variants} combinations of settings made",
        *_counted((search.dropped, "dropped as obviously bad")),
        f"{search.variants - search.dropped} judged by eye in groups of 12",
        *ended,
    ]
    return Step(
        kind="search",
        by="agent",
        at=search.at,
        title="Chose the processing settings",
        lines=tuple(lines),
    )


def _inverted_step(journal: Sequence[JournalEntry]) -> Step | None:
    """The curves turned into Vs profiles: how many, and how many passed the model's check, were
    flagged and rejected."""
    runs = [entry for entry in journal if entry.kind == "inversion"]
    if not runs:
        return None
    latest = _model_verdicts(journal, None)
    passed, flagged, rejected = _verdict_counts(latest)
    return Step(
        kind="inversion",
        by="agent",
        at=runs[0].at,
        title="Turned the curves into Vs profiles",
        lines=(
            f"{plural(len(latest), 'curve')} inverted",
            *_counted((passed, "passed"), (flagged, "flagged"), (rejected, "rejected")),
        ),
        detail=MODEL_CHECK[:1].upper() + MODEL_CHECK[1:],
    )


def _run_history(
    run_folder: Path,
    manifest: RunManifest,
    log: QCLog | None,
    stages: Iterable[StageCount],
) -> tuple[Step, ...]:
    """What PAC or the workflow did, from the run's files, a step each: the settings the user
    gave, the records read, the workflow's choices on the line, the line processed, the curves
    picked, inverted and turned into soil columns, each by who did it."""
    spacing = manifest.profile.receiver_spacing_m
    done = {count.key: count for count in stages}
    given = given_settings(run_folder)

    def gave(family: str) -> list[Step]:
        title = "Your settings" if family == "processing" else f"Your settings for the {family}"
        return [
            Step(
                kind="given",
                by="you",
                at=one.at,
                title=title,
                lines=tuple(
                    plain(one.values, spacing, int(preset_stage(manifest, "masw")["length"]))
                ),
            )
            for one in given
            if one.family == family
        ]

    def stage(
        kind: str, log_stage: str, title: str, lines: list[str], unit: str = "window"
    ) -> Step:
        attempts = [one for one in log.attempts if one.stage == log_stage] if log else []
        if again := _again(log, log_stage, unit):
            lines.append(f"Done again: {again}")
        return Step(
            kind=kind,
            by=log.chooser if log is not None and attempts else "you",
            at=min((one.started_at for one in attempts), default=None),
            title=title,
            lines=tuple(lines),
        )

    steps = gave("processing")
    used = sum(record.status == "succeeded" for record in manifest.records) - len(
        manifest.exclusions.records
    )
    lines = [f"{used} of {plural(len(manifest.records), 'record')} used"]
    if manifest.exclusions.records:
        lines += [one[:1].upper() + one[1:] for one in _left_out(log, manifest.exclusions.records)]
    steps.append(
        stage("records", "preprocessing", "Read your records", lines, "record").model_copy(
            update={"at": manifest.started_at}
        )
    )
    steps += _trials(run_folder, spacing)
    masw = preset_stage(manifest, "masw")
    length, step = int(masw["length"]), int(masw["step"])
    steps.append(
        stage(
            "images",
            "phase_shift",
            "Processed the whole line",
            [
                f"{plural(len(manifest.windows), 'window')} of {receivers(length, spacing)}",
                f"One every {receivers(step, spacing, length=False)}",
            ],
        )
    )
    steps += _line_tries(run_folder)
    steps += gave("picking")
    picked = done["dispersion"]
    steps.append(
        stage(
            "picks",
            "picking",
            "Picked the curves",
            [f"{picked.done} of {plural(picked.total, 'window')}"],
        )
    )
    steps += gave("inversion")
    for key, log_stage, title in (
        ("inversion", "inversion", "Inverted the curves"),
        ("petro", "petro_inversion", "Made the soil columns"),
    ):
        count = done[key]
        if count.done:
            steps.append(
                stage(key, log_stage, title, [f"{count.done} of {plural(count.total, 'window')}"])
            )
    return tuple(steps)


def _trials(run_folder: Path, spacing: float) -> list[Step]:
    """The workflow's choices before the line's windows: the window length, the mute, a passive
    line's segments, each with the one kept."""
    steps: list[Step] = []
    choice = length_choice(run_folder)
    if choice is not None and choice.trials:
        count = max(len(trial.xmids) for trial in choice.trials)
        steps.append(
            Step(
                kind="length",
                by="workflow",
                title="Chose the window length",
                lines=(
                    f"{plural(len(choice.trials), 'length')} tried on {plural(count, 'sample window')}",
                    f"Kept {receivers(choice.length, spacing)}",
                ),
            )
        )
    mutes = mutes_tried(run_folder)
    if mutes:
        kept = next((one for one in mutes if one.kept), None)
        steps.append(
            Step(
                kind="mute",
                by="workflow",
                title="Chose the mute",
                lines=(
                    f"{plural(len(mutes), 'mute')} tried",
                    f"Kept: {mute_name(kept).lower() if kept is not None else 'none'}",
                ),
            )
        )
    segments = segments_tried(run_folder)
    if segments:
        kept_segments = next((one for one in segments if one.kept), None)
        steps.append(
            Step(
                kind="segments",
                by="workflow",
                title="Chose the segments",
                lines=(
                    f"{plural(len(segments), 'segment length')} tried",
                    f"Kept {number(kept_segments.segment_s)} s"
                    if kept_segments is not None
                    else "None kept",
                ),
            )
        )
    return steps


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
            title="Changed the line where curves failed",
            lines=(f"{plural(len(tries), 'change')} tried", *(f"Kept: {one}" for one in kept)),
        )
    ]
