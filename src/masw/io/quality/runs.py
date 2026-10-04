"""The runs Visualization offers, and each run's card: who made it (the assistant, or PAC), the
settings its windows were built and imaged with, each with where it comes from (a rule the
assistant applied to the data, the request, the preset's default, or PAC's form), how far each
stage went, and the line's geometry for the profile plot: every receiver, every shot, every
window's receivers."""

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
from masw.io.quality.log import (
    COHERENCE_FILE,
    LINE,
    LOG_FILE,
    QCLog,
    config_section,
    line_receivers,
    read_log,
)
from masw.io.quality.sources import near_field
from masw.io.quality.view import REFUSALS, Origin, Setting, flag_text, number, plural, span
from sigpipe.masw.inversion.section import is_inverted
from sigpipe.masw.petro.window import MODEL_FILE
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.presets.making import make_preset
from sigpipe.masw.runs import RunManifest, window_folders, xmid_of

type Maker = Literal["assistant", "pac"]
type StageKey = Literal["records", "dispersion", "inversion", "petro"]

GIVEN = "given in the request (by you, or chosen by the assistant)"
DEFAULT = "the preset's default"
PAC = "set by hand"
MIN_PASS_SHARE = 0.8  # the assistant's default: the trial windows a window length must pass
MAX_UNCERTAINTY = 0.2  # the assistant's default: picks precise enough to stop at a length
# The preset's entries the card says apart from the preprocessing's stages.
OWN_ENTRIES = ("mode", "masw", "dispersion")
# What asked for a stage to run again, other than a check's flag, in words.
_CAUSES = {
    "backtrack": "after an earlier stage",
    "mute trial": "the mute trial",
    "asked": "at the request",
}


class LengthTrial(BaseModel):
    """One window length the assistant tried on a few trial windows along the line."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    length: int  # receivers
    metres: float = 0.0
    xmids: tuple[float, ...] = ()
    passed: int  # trial windows G3 passed
    windows: int = 0  # windows the line gets at this length
    wavelengths_m: tuple[float, float] | None = None  # median shortest and longest passed
    flags: tuple[str, ...] = ()
    uncertainty: float | None = None  # the passed curves' median velocity uncertainty
    compared: bool = False  # tried past the length kept, to compare


class LengthChoice(BaseModel):
    """The window length the assistant kept for the line (coherence.json), and its trials."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    length: int
    trials: tuple[LengthTrial, ...]
    notes: tuple[str, ...] = ()


class RunEntry(BaseModel):
    """A run in the selector."""

    model_config = ConfigDict(frozen=True)

    folder: str
    run_id: str | None  # None: a folder of the older layout
    started_at: datetime | None
    mode: str | None
    by: Maker | None
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


class RunCard(BaseModel):
    """What the page says of a run above its stages, and draws the profile from."""

    model_config = ConfigDict(frozen=True)

    folder: str
    profile: str
    run_id: str | None
    mode: str | None
    by: Maker | None
    started_at: datetime | None
    finished_at: datetime | None
    records: bool  # the profile's records are in the input folder: they can be shown
    settings: tuple[Setting, ...]
    trials: tuple[LengthTrial, ...]  # the window lengths the assistant tried, when it chose one
    stages: tuple[StageCount, ...]
    receivers: tuple[float, ...]  # every receiver's x
    sources: dict[str, float]  # each record's shot x, by file name; none on a passive line
    windows: tuple[LineWindow, ...]
    # The shots a window stacks: from, to this far from its middle (to None: any distance).
    reach: tuple[float, float | None] | None


def maker(run_folder: Path) -> Maker:
    """Who made the run: the assistant, which logs its checks there, or PAC."""
    return "assistant" if (run_folder / LOG_FILE).exists() else "pac"


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
    choice = length_choice(run_folder)
    return RunCard(
        folder=folder,
        profile=manifest.profile.name,
        run_id=manifest.run_id,
        mode=manifest.preset.mode,
        by=maker(run_folder),
        started_at=manifest.started_at,
        finished_at=manifest.finished_at,
        records=(INPUT_DIR / manifest.profile.name).is_dir(),
        settings=processing_settings(run_folder, manifest, read_log(run_folder), choice),
        trials=choice.trials if choice is not None else (),
        stages=_stages(run_folder, manifest),
        receivers=line.receivers,
        sources=line.sources,
        windows=_windows(manifest, line.receivers),
        reach=shot_distances(masw) if line.sources else None,
    )


def length_choice(run_folder: Path) -> LengthChoice | None:
    path = run_folder / COHERENCE_FILE
    return LengthChoice.model_validate_json(path.read_text()) if path.exists() else None


def processing_settings(
    run_folder: Path, manifest: RunManifest, log: QCLog | None, choice: LengthChoice | None
) -> tuple[Setting, ...]:
    """The run's windows and the settings of each stage, as run.json has them, each with where
    it comes from: the assistant's rules (its notes on the line), the request, the preset's
    default, or PAC's form."""
    preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
    defaults: dict[str, Any] = make_preset(manifest.preset.mode).model_dump(mode="json")
    line = log.latest(LINE, "phase_shift") if log is not None else None
    changed: dict[str, Any] = line.parameters if line is not None else {}
    notes = line.notes if line is not None else ()

    def origin(stages: Iterable[str], keys: Iterable[str] | None = None) -> tuple[Origin, str]:
        """Where the values of `stages` (their `keys`, or all) come from, when no rule set them:
        the preset's default as the units' cards say it, else the request or PAC's form."""
        same = all(
            _same(preset[stage].get(key), defaults.get(stage, {}).get(key))
            for stage in stages
            for key in (keys if keys is not None else preset[stage])
        )
        if same:
            return "default", DEFAULT
        return ("given", GIVEN) if log is not None else ("pac", PAC)

    def rule(stage: str, key: str) -> str | None:
        """The assistant's reason for the value it set on the line, from its note."""
        if key not in cast(Mapping[str, Any], changed.get(stage, {})):
            return None
        note = next((note for note in notes if note.startswith(f"{stage} {key}")), None)
        return note.split(": ", 1)[1].rstrip(".") if note is not None else "set by its rules"

    spacing = manifest.profile.receiver_spacing_m
    masw: dict[str, Any] = preset["masw"]
    length, step = int(masw["length"]), int(masw["step"])
    if choice is not None and choice.length == length:
        where, why = "rule", _length_rule(choice, config_section(run_folder, "coherence"))
    else:
        where, why = origin(["masw"], ["length"])
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
            why=origin(["masw"], ["step"])[1],
            origin=origin(["masw"], ["step"])[0],
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
            origin="rule",
        )
    )
    if manifest.profile.kind == "active":
        near, far = rule("near_field", "distance_m"), rule("masw", "distance_max")
        if near is None and far is None:
            where, why = origin(["masw"], ["distance_min", "distance_max"])
        else:
            where = "rule"
            why = "; ".join(
                part
                for part in (
                    f"nearest, {near}" if near else "",
                    f"farthest, {far}" if far else "",
                )
                if part
            )
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
    band = rule("dispersion", "fmax") or rule("dispersion", "fmin")
    where, why = ("rule", band) if band is not None else origin(["dispersion"])
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
    # A record step the checks chose for the whole line (the mute trial's muting, logged with
    # the line's images).
    chosen = [name for name in stages if name in changed]
    where, why = (
        ("rule", f"the checks chose its {', '.join(chosen)} on the line")
        if chosen
        else origin(stages)
    )
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


def _length_rule(choice: LengthChoice, rules: Mapping[str, Any]) -> str:
    """Why the assistant kept `choice.length`, in a line: the trials are in their table."""
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
    return f"the assistant's choice: {how} (every trial under Every setting)"


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
        why="; ".join(said) or "every record, none left out",
        origin="pac" if log is None else "rule" if excluded or traces or receivers else "default",
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
            by=None,
            windows=len(window_folders(path)),
        )
    length = preset_stage(manifest, "masw").get("length")
    return RunEntry(
        folder=folder,
        run_id=manifest.run_id,
        started_at=manifest.started_at,
        mode=manifest.preset.mode,
        by=maker(path),
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
        by=None,
        started_at=None,
        finished_at=None,
        records=(INPUT_DIR / folder).is_dir(),
        settings=(),
        trials=(),
        stages=_counts([run_folder / unit for unit in units]),
        receivers=(),
        sources={},
        windows=tuple(
            LineWindow(key=unit, xmid=xmid_of(unit), first=xmid_of(unit), last=xmid_of(unit))
            for unit in units
        ),
        reach=None,
    )
