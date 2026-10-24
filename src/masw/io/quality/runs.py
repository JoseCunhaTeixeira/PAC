"""The runs Visualization offers, and each run's card: who made it (PAC's pages by hand, PACo's
workflow, or its autonomous mode), the settings its windows were built and imaged with, each with
where it comes from (the user, the workflow, the agent, or the preset's default), what was tried
to choose them, the run's history step by step, how far each stage went, and the line's geometry
for the profile plot: every receiver, every shot, every window's receivers."""

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
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
    read_window,
    shot_distances,
)
from masw.io.quality.history import (
    ASKED,
    JUDGING,
    TRIALS,
    VISION,
    by_whom_counted,
    counted_cause,
    makers,
    run_steps,
)
from masw.io.quality.history import Maker as By
from masw.io.quality.journal import (
    JournalEntry,
    SearchDone,
    SurveyDone,
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
    line_change,
    line_tried,
    mute_name,
    mute_trigger,
    mutes_tried,
    plain,
    segments_tried,
    settings_said,
    shots_used,
    step_words,
    tried_tables,
    trigger_delay,
)
from masw.io.quality.view import (
    BY_AGENT,
    BY_USER,
    REFUSALS,
    How,
    Origin,
    Setting,
    Step,
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

GIVEN = "given in the user's request"
# A workflow run's request, which the assistant writes: the user's values, or its own choices.
REQUESTED = "given in the request: by the user, or chosen by the workflow agent"
DEFAULT = "the preset's default"
PAC = "set by the user"
# An autonomous run's settings its commit did not change: its search's, which sets every one.
SEARCHED = "chosen in the search"
SEARCH_SAME = "the same in every combination the search compared"
# How the autonomous agent's commits give a setting the user gave, in the line's notes.
USER_REASON = "given by the user"
# How the workflow's segments trial begins its note on the segments and the selection it kept.
SEGMENTS_NOTE = "Segments of "
ALL_RECORDS = "every record, none left out"
CHECKED_RECORDS = "the signal check kept every record"
MOST_LISTED = 6  # the windows a step names; more, counted only
# The keys of the card's settings, by their stage, where not the stage's name.
_KEYS = {"dispersion": "band", "muting": "mute"}
# The origins, the one to show first for values set together.
_RANKED: tuple[Origin, ...] = ("agent", "workflow", "checks", "you", "default")
# A journal entry's id the agent's reasons name in parentheses: "(E2)", "(L1, E2)", "(E2 current)".
_REFERENCES = re.compile(r"\s*\((?:[A-Z]\d+|current)(?:[\s,;]+(?:[A-Z]\d+|current))*\)")


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
    made = makers(run_folder, manifest, log, ("picking", "inversion"))
    settings = (
        *processing_settings(manifest, log, length_choice(run_folder), run_folder, search),
        _curves(stages, journal, maker, made["picking"]),
        _profiles(stages, journal, log, maker, made["inversion"]),
    )
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
            journal,
            settings,
            given_settings(run_folder),
            spacing,
            int(masw["length"]),
            idle=near_field(log) is not None and near_field_used(run_folder, log) is None,
        ),
        history=_study_history(journal, search, survey, manifest, log, run_folder)
        if study is not None
        else _run_history(run_folder, manifest, log),
        stages=stages,
        receivers=line.receivers,
        sources=line.sources,
        windows=_windows(manifest, line.receivers),
        reach=shot_distances(masw) if line.sources else None,
    )


def processing_settings(
    manifest: RunManifest,
    log: QCLog | None,
    choice: LengthChoice | None,
    run_folder: Path | None = None,
    search: SearchDone | None = None,
) -> tuple[Setting, ...]:
    """The run's settings in plain words, in the order its records went through them, each with
    where it comes from and why: its windows, the records used, the shots the windows stack, then
    each step of its method in its order (PAC's pages'); a step off by default and left off is not
    said. An autonomous run's `search` set every setting its commit did not change."""
    preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
    defaults: dict[str, Any] = make_preset(manifest.preset.mode).model_dump(mode="json")
    line = log.latest(LINE, "phase_shift") if log is not None else None
    changed: dict[str, Any] = line.parameters if line is not None else {}
    notes = line.notes if line is not None else ()
    chooser: Origin = log.chooser if log is not None else "you"

    def varied(stages: Sequence[str], keys: Sequence[str] | None = None) -> bool:
        """Whether the autonomous run's search compared other values of `stages` (their
        `keys`, or any)."""
        paths = stages if keys is None else [f"{stage} {key}" for stage in stages for key in keys]
        return search is not None and any(path in search.varied for path in paths)

    def origin(stages: Sequence[str], keys: Sequence[str] | None = None) -> tuple[Origin, str]:
        """Where the values of `stages` (their `keys`, or all) come from, when the line's
        settings did not change them: as the preset has them, an autonomous run's search, which
        sets every one, else the preset's default as the units' cards say it; else the request or
        PAC's form."""
        same = all(
            _same(preset[stage].get(key), defaults.get(stage, {}).get(key))
            for stage in stages
            for key in (keys if keys is not None else preset[stage])
        )
        if same and chooser == "agent":
            return "agent", SEARCHED if varied(stages, keys) else SEARCH_SAME
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
    if stepped[1] not in (DEFAULT, SEARCH_SAME):
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
        field = near_field(log) if run_folder is None else near_field_used(run_folder, log)
        nearest = (
            f"nearest, {number(field, 4)} m from a window's receivers"
            if field is not None
            else "nearest"
        )
        ends = {
            **({nearest: rule("near_field", "distance_m")} if field is not None else {}),
            "farthest": rule("masw", "distance_max"),
        }
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
        if name == "muting" and rule("trigger") is not None:
            # The trigger, the muting's, which the line set: said with it, its latest reason.
            detail = [*detail, *trigger_delay(preset.get("trigger") or {})]
            reason = noted(reversed(notes), "trigger", chooser)[1]
            why += f"; trigger: {reason or 'chosen by the checks on the line'}"
        # A step off and left so: the preset's, or every combination of the search's.
        if value == "none" and why in (DEFAULT, SEARCH_SAME):
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
    trial = next((note for note in notes if note.startswith(SEGMENTS_NOTE)), None)
    if note is None and trial is not None and name.split()[0] in ("slicing", "selection"):
        # The segments trial's, which sets both in one note.
        return chooser, f"the segments trial kept it: {plain_reason(trial.split(': ', 1)[1])}"
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
    # The records the signal check judged: those it used, as those it left out, its own.
    checked = log is not None and any(
        log.result(record.name, "preprocessing", "G1") is not None for record in manifest.records
    )
    return Setting(
        key="records",
        label="Records",
        value=f"{used} of {total}",
        detail=f"{left} left out" if left else "all used",
        why="; ".join(said) or (CHECKED_RECORDS if checked else ALL_RECORDS),
        origin="checks" if checked or excluded or traces or receivers or redone else "default",
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
    """The units `stage` ran again on, and what asked for it, as the run card's history counts
    it: "8 windows (6 too few points, 2 asked by the workflow agent)", each cause's units (one
    asked twice counted once). A redo's fresh first attempt counts, as every attempt after the
    first run."""
    again = [
        attempt
        for attempt in (log.attempts if log is not None else ())
        if attempt.stage == stage and attempt.unit != LINE and attempt.triggered_by != "initial"
    ]
    if not again:
        return ""
    units = {attempt.unit for attempt in again}
    asked: dict[str, set[str]] = {}
    for attempt in again:
        asked.setdefault(counted_cause(attempt.triggered_by, log), set()).add(attempt.unit)
    ranked = sorted(asked.items(), key=lambda item: -len(item[1]))
    said = [f"{len(found)} {words}" for words, found in ranked]
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
# Who made what a run's maker made, in the stages' words.
_BY: dict[Maker, str] = {
    "autonomous": BY_AGENT["agent"],
    "workflow": BY_AGENT["workflow"],
    "manual": BY_USER,
}
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


def _curves(
    stages: Iterable[StageCount],
    journal: Sequence[JournalEntry],
    maker: Maker,
    picked_by: Mapping[By, int] | None = None,
) -> Setting:
    """The curves tile: those the freeze kept and rejected, else those picked, and who picked
    the curves that stand (`picked_by`: each with how many, quality.history's makers)."""
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
            why="the autonomous agent accepted or rejected each window's curve",
            origin=by,
        )
    # Tagged by who picked the most of them.
    most: Origin | None = next(iter(picked_by)) if picked_by else None
    return Setting(
        key="curves",
        label="Curves",
        value=f"{picked.done} of {picked.total}",
        detail="windows picked",
        why=f"picked {by_whom_counted(picked_by) if picked_by else _BY[maker]}",
        origin=most or (by if picked.done else None),
    )


def _profiles(
    stages: Iterable[StageCount],
    journal: Sequence[JournalEntry],
    log: QCLog | None,
    maker: Maker,
    inverted_by: Mapping[By, int] | None = None,
) -> Setting:
    """The Vs profiles tile: how many passed the model's check of those made, how many it
    flagged and rejected; else how many windows were inverted, and by whom (`inverted_by`: each
    with how many of those that stand, quality.history's makers)."""
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
        # Tagged by who inverted the most of them.
        most: Origin | None = next(iter(inverted_by)) if inverted_by else None
        return Setting(
            key="profiles",
            label="Vs profiles",
            value=f"{inverted.done} of {inverted.total}",
            detail="windows inverted",
            why=f"inverted {by_whom_counted(inverted_by) if inverted_by else _BY[maker]}",
            origin=most or _MAKERS[maker],
        )
    return Setting(
        key="profiles",
        label="Vs profiles",
        value="none yet",
        why="no window inverted yet",
        origin=None,
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
    """Each inverted window's latest verdict on its model, as the checks' log holds it now, as
    the inversion's page shows it (a job stopped before its end leaves the journal behind); the
    autonomous mode's journal's for a run whose log holds none; none for a run no check judged."""
    found: dict[str, str] = {}
    for attempt in log.attempts if log is not None else ():
        if (
            attempt.stage == "inversion"
            and attempt.unit != LINE
            and (g5 := attempt.results.get("G5"))
        ):
            found[attempt.unit] = g5.verdict
    if found:
        return found
    for entry in journal:
        if entry.kind == "inversion":
            found |= {
                str(unit): str(verdict)
                for unit, verdict in cast(dict[str, Any], entry.data.get("g5") or {}).items()
            }
    return found


def _reasons(
    journal: Sequence[JournalEntry],
    settings: Sequence[Setting],
    given: Sequence[Given],
    spacing: float,
    length: int,
    idle: bool = False,
) -> tuple[Reason, ...]:
    """Why the settings are what they are: the assistant's own reasons for those it chose (its
    commits', the latest of each setting; a commit's reasons for one setting `SEP` apart, each
    once), then the settings the user set, each its value in words (the windows' `length` where
    the shots used count from); the near field's left out when it dropped no shot (`idle`)."""
    found: dict[tuple[str, str], Reason] = {}
    for entry in journal:
        if entry.kind != "commit":
            continue
        changes = cast(dict[str, Any], entry.data.get("changes") or {})
        own: dict[str, list[str]] = {}
        theirs: dict[str, str] = {}
        for key, reason in cast(dict[str, Any], entry.data.get("reasons") or {}).items():
            if idle and key in ("near_field_m", "near_field"):
                continue
            name = _SETTING_NAMES.get(key) or _SETTING_NAMES.get(key.split(" ")[0]) or key
            said = plain_reason(str(reason))
            # The search's one value, said by code as the setting's why: not the agent's words.
            if said == SEARCH_SAME:
                continue
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
    run_folder: Path,
) -> tuple[Step, ...]:
    """What the autonomous mode did, a step each, in order: what the user asked, the records, the
    choice of the settings, the line processed, who made the curves, the Vs profiles and the soil
    columns that stand and how (quality.history's run_steps), the agent's review of its curves
    (the freeze's verdicts); the assistant's own reasoning left to its journal."""
    spacing = manifest.profile.receiver_spacing_m
    length = int(preset_stage(manifest, "masw")["length"])
    steps: list[Step] = []
    commits = 0
    # The near field the run's windows show none of (no window kept a farther shot): not said.
    idle = near_field(log) if near_field_used(run_folder, log) is None else None
    for entry in journal:
        if entry.kind == "objective":
            said = entry.text.removeprefix(entry.id).strip()
            lines = [part.strip().capitalize() for part in said.split(",") if part.strip()]
            # The settings the user typed, as the run card says them.
            typed = cast(dict[str, Any], entry.data.get("given") or {})
            line = cast(dict[str, Any], typed.get("line") or {})
            lines += plain(
                {**({"mode": typed["mode"]} if typed.get("mode") else {}), **line}, spacing
            )
            lines += plain(cast(dict[str, Any], typed.get("inversion") or {}), spacing)
            steps.append(
                Step(
                    kind="objective",
                    by="you",
                    how=ASKED,
                    at=entry.at,
                    title="Asked",
                    lines=tuple(lines),
                )
            )
        elif entry.kind == "settings":
            changed = tuple(str(one) for one in cast(list[Any], entry.data.get("changed") or []))
            steps.append(
                Step(
                    kind="settings",
                    by="you",
                    how=ASKED,
                    at=entry.at,
                    title="Changed the settings",
                    lines=changed,
                )
            )
        elif entry.kind == "reopen":
            asked = str(entry.data.get("request") or "")
            steps.append(
                Step(
                    kind="reopen",
                    by="you",
                    how=ASKED,
                    at=entry.at,
                    title="Asked again",
                    lines=(asked,),
                )
            )
        elif entry.kind == "commit":
            commits += 1
            changes = cast(dict[str, Any], entry.data.get("changes") or {})
            if idle is not None:
                changes = {k: v for k, v in changes.items() if k != "near_field_m" or v != idle}
            title = "Processed the whole line" if commits == 1 else "Changed the line's settings"
            steps.append(
                Step(
                    kind="commit",
                    by="agent",
                    how=How(label=str(manifest.preset.mode)),
                    at=entry.at,
                    title=title,
                    lines=tuple(plain(changes, spacing, length)),
                )
            )
        elif entry.kind == "freeze" and (found := _FROZEN.search(entry.text)):
            # Each window's curve accepted or rejected, before the inversion.
            steps.append(
                Step(
                    kind="freeze",
                    by="agent",
                    how=JUDGING,
                    at=entry.at,
                    title="Reviewed the curves",
                    lines=tuple(
                        _counted(
                            (int(found.group(1)), "accepted"), (int(found.group(2)), "rejected")
                        )
                    ),
                )
            )
    if survey is not None:
        steps.append(_survey_step(survey, journal, manifest, log))
    if (tried := _tried_step(search)) is not None:
        steps.append(tried)
    stages = ("picking", "inversion", "petro_inversion")
    steps += run_steps(run_folder, manifest, log, stages, listed=MOST_LISTED)
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
        kind="survey",
        by="agent",
        how=How(label=str(manifest.preset.mode)),
        at=asked,
        title="Looked at the records",
        lines=tuple(lines),
    )


def _tried_step(search: SearchDone | None) -> Step | None:
    """The choice of the settings, in a few lines: how many combinations, their dispersion images
    compared by eye, the best of them chosen."""
    if search is None:
        return None
    lines = [
        f"{search.variants - search.dropped} combinations of settings",
        "Dispersion images compared by the agent's vision",
    ]
    if search.finalists:
        lines.append("The best of them chosen")
    return Step(
        kind="search",
        by="agent",
        how=VISION,
        at=search.at,
        title="Chose the processing settings",
        lines=tuple(lines),
    )


def near_field_used(run_folder: Path, log: QCLog | None) -> float | None:
    """The near field the line set (near_field) as the run's windows show it: where one of them
    stacks no shot nearer than it to its receivers; None when every window still stacks one (it
    dropped no shot: none kept a farther one), or none was set."""
    field = near_field(log)
    if field is None:
        return None
    for unit in window_folders(run_folder):
        window = read_window(run_folder / unit)
        if window is None or not window.acquisitions:
            continue
        xs = [receiver.x for receiver in window.acquisitions[0].receivers]
        first, last = min(xs), max(xs)
        if all(
            max(first - one.source.x, one.source.x - last) >= field for one in window.acquisitions
        ):
            return field
    return None


def _run_history(run_folder: Path, manifest: RunManifest, log: QCLog | None) -> tuple[Step, ...]:
    """What PAC or the workflow did, from the run's files, a step each: the settings the user
    gave, the records read, the workflow's choices on the line, the line processed, the curves
    picked, inverted into Vs profiles and into soil columns, each by who made what stands of it
    (quality.history's run_steps); what the user did in PAC after the workflow, last."""
    spacing = manifest.profile.receiver_spacing_m
    masw = preset_stage(manifest, "masw")
    length, step = int(masw["length"]), int(masw["step"])
    given = given_settings(run_folder)
    excluded = manifest.exclusions.records
    left = [f"{plural(len(excluded), 'record')} left out"] if excluded else []
    left += [one[:1].upper() + one[1:] for one in _left_out(log, excluded)]
    windows = f"Windows of {receivers(length, spacing)}, one every "
    windows += receivers(step, spacing, length=False)
    found = run_steps(
        run_folder,
        manifest,
        log,
        ("preprocessing", "phase_shift", "picking", "inversion", "petro_inversion"),
        {"preprocessing": left, "phase_shift": [windows]},
        MOST_LISTED,
    )
    # The run's maker's steps in the pipeline's order; the user's after the workflow's, last.
    maker = log.chooser if log is not None else "you"

    def stage(kind: str) -> list[Step]:
        return [one for one in found if one.kind == kind and one.by == maker]

    def gave(family: str) -> list[Step]:
        title = "Gave the settings" if family == "processing" else f"Gave the {family} settings"
        return [
            Step(
                kind="given",
                by="you",
                how=ASKED,
                at=one.at,
                title=title,
                lines=tuple(plain(one.values, spacing, length)),
            )
            for one in given
            if one.family == family
        ]

    steps = [
        *gave("processing"),
        *stage("preprocessing"),
        *_trials(run_folder, spacing),
        *stage("phase_shift"),
        *_line_tries(run_folder, spacing),
        *gave("picking"),
        *stage("picking"),
        *gave("inversion"),
        *stage("inversion"),
        *stage("petro_inversion"),
    ]
    later = sorted(
        (one for one in found if one.by != maker),
        key=lambda one: one.at or datetime.min.replace(tzinfo=UTC),
    )
    return (*steps, *later)


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
                how=TRIALS,
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
        trigger = mute_trigger(run_folder)
        steps.append(
            Step(
                kind="mute",
                by="workflow",
                how=TRIALS,
                title="Chose the mute",
                lines=(
                    *(
                        [f"{trigger.capitalize()}, where the shots' first breaks put them"]
                        if trigger
                        else []
                    ),
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
                how=TRIALS,
                title="Chose the segments",
                lines=(
                    f"{plural(len(segments), 'combination')} of segments and selection tried",
                    f"Kept {number(kept_segments.segment_s)} s"
                    if kept_segments is not None
                    else "None kept",
                ),
            )
        )
    return steps


def _line_tries(run_folder: Path, spacing: float) -> list[Step]:
    """The workflow's changes of the line's settings, asked by the windows whose curve failed."""
    tries = line_tried(run_folder)
    if not tries:
        return []
    kept = [f"Kept: {line_change(one, spacing)}" for one in tries if one.kept] or ["None kept"]
    return [
        Step(
            kind="line",
            by="workflow",
            how=TRIALS,
            title="Changed the line where curves failed",
            lines=(f"{plural(len(tries), 'change')} tried", *kept),
        )
    ]
