"""What was done to a unit, as its stage's "Settings" say it: each step it went through with the
settings it ran with (the run's preset with the unit's own changes on top: the agent's retries,
from its QC log; a window's, the line's too), in the run card's words, where each comes from,
and how the line's other units differ in it; and its attempts, a step each, when one ran again.
Read, never measured."""

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from masw.io.quality.files import merged, preset_stage
from masw.io.quality.log import LINE, LINE_RULES, Attempt, GateResult, QCLog
from masw.io.quality.runs import is_default, noted
from masw.io.quality.tried import SEP, settings_said, step_words
from masw.io.quality.view import (
    CHECK_NAMES,
    REFUSALS,
    Origin,
    Setting,
    Step,
    by_whom,
    counted,
    flag_text,
    number,
    plural,
    receivers,
    span,
)
from sigpipe.masw.presets import make_preset
from sigpipe.masw.runs import RunManifest, xmid_of

# The stages as the run card names them, where its steps say none (a step off).
_NAMES = {
    "muting": "Mute",
    "filtering": "Filter",
    "slicing": "Segments",
    "selection": "Segment selection",
    "whitening": "Whitening",
    "normalization": "Normalization",
    "stacking": "Stacking",
    "dispersion": "Dispersion image",
    "image_stacking": "Image stacking",
}
# A stage's fields, as a range over units names them, and their units.
_FIELDS = {
    "t0": "trigger delay",
    "vmin": "slowest velocity",
    "vmax": "fastest velocity",
    "tmin": "start",
    "tmax": "end",
    "width": "signal width",
    "taper": "taper",
    "fmin": "lowest frequency",
    "fmax": "highest frequency",
    "nv": "velocity steps",
    "order": "order",
    "length": "receivers",
    "step": "step",
    "segment_duration": "segment length",
    "segment_step": "segment step",
    "threshold": "threshold",
    "taper_width_Hz": "taper",
}
_UNITS = {
    "t0": "s",
    "tmin": "s",
    "tmax": "s",
    "width": "s",
    "taper": "samples",
    "segment_duration": "s",
    "segment_step": "s",
    "fmin": "Hz",
    "fmax": "Hz",
    "taper_width_Hz": "Hz",
    "vmin": "m/s",
    "vmax": "m/s",
}
# The stages a record's preprocessing and a window's image went through, by mode.
_RECORD_STAGES = ("trigger", "muting", "filtering")
_WINDOW_STAGES = {
    "active": ("dispersion", "image_stacking"),
    "passive-active": ("stacking", "dispersion"),
    "passive": ("slicing", "selection", "whitening", "normalization", "stacking", "dispersion"),
}
# A spread names the other values, the most common first, and with each the units that ran with
# it, when so few; more are counted.
_SPREAD_VALUES = 3
_SPREAD_UNITS = 3
# A setting's kind, whatever numbers a unit had of its own (a trigger's delay): its value
# without its numbers.
_NUMBERS = re.compile(r"\d[\d.,]*")
# From sigpipe c17b3de on (its commit's time, in UTC), a shot's trigger is corrected with its
# muting only; the runs before corrected it whatever the muting.
_TRIGGER_WITH_MUTING = datetime(2026, 9, 28, 14, 46, tzinfo=UTC)
# What asked a stage to run again, other than a check's flag, in words: a redo of the agent's
# (backtrack) said by its kind, the run's.
_CAUSES = {
    "initial": "",
    "mute trial": "with the mute trial's mute",
    "line change": "with a change of the line's settings",
    "asked": "asked by the user",
}
# What each stage did, as a step's title.
_DONE = {
    "preprocessing": "Preprocessed",
    "phase_shift": "Dispersion image made",
    "picking": "Curve picked",
    "inversion": "Inverted",
    "petro_inversion": "Soil column made",
}
_VERDICTS = {"pass": "passed", "retry": "flagged", "reject": "rejected"}


def cause(triggered_by: str, log: QCLog | None = None) -> str:
    """Why an attempt ran, in words ("asked by the curve check: points off the ridge's top",
    "asked by the autonomous agent" for a redo `log`'s agent chose); "" for a first run."""
    gate, _, flag = triggered_by.partition(":")
    if flag:
        return f"asked by {CHECK_NAMES.get(gate, gate)}: {flag_text(flag)}"
    if triggered_by == "backtrack":
        return f"asked {by_whom(log, True) if log is not None else 'by the agent'}"
    return _CAUSES.get(triggered_by, triggered_by.replace("_", " "))


def trigger_text(attempt: Attempt, log: QCLog | None = None) -> str:
    """Why an attempt ran, in words: its first run said so."""
    return cause(attempt.triggered_by, log) or "first run"


def trigger_words(t0: object, own: str = "its") -> str:
    """A record's trigger as its card says it: "delay 7 ms", or from `own` file."""
    if not isinstance(t0, int | float):
        return f"from {own} file"
    return f"delay {number(t0 * 1000)} ms" if 0 < abs(t0) < 1 else f"delay {number(t0)} s"


def _field(key: str, value: object) -> str:
    """A field's value with its unit; a trigger's delay in ms."""
    if key == "t0" and isinstance(value, int | float) and 0 < abs(value) < 1:
        return f"{number(value * 1000)} ms"
    unit = _UNITS.get(key)
    shown = number(value, 4) if isinstance(value, int | float) else str(value)
    return f"{shown} {unit}" if unit else shown


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


class _Origins:
    """Where a unit's settings of one log stage come from: its own retries, the line's (a
    window's; a record's, the line's mute trial; the agent's commits), else the run's (an
    autonomous run's search, which sets every one; the preset's default, the request, PAC's
    form). Why, only for its own retries: the line's are the run card's to say."""

    def __init__(
        self, manifest: RunManifest, log: QCLog | None, unit: str, log_stage: str, noun: str
    ) -> None:
        self.noun = noun
        self.log = log
        self.chooser: Origin = log.chooser if log is not None else "you"
        # Each of its attempts with the stages it changed on the one before (the first, its
        # overrides of the run's); a change of the line's that made the unit again says what it
        # changed, the line's.
        attempts = log.of(unit, log_stage) if log is not None else ()
        self.own = [
            (one, _changed(before.parameters if before is not None else {}, one.parameters))
            for before, one in zip((None, *attempts), attempts, strict=False)
            if one.parameters and one.triggered_by not in LINE_RULES
        ]
        self.line = log.latest(LINE, log_stage) if log is not None else None
        # The muting the line's mute trial kept, logged with the line's images.
        self.trial = (
            log.latest(LINE, "phase_shift")
            if log is not None and log_stage == "preprocessing"
            else None
        )
        self.preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
        self.defaults: dict[str, Any] = make_preset(manifest.preset.mode).model_dump(mode="json")

    def of(self, stage: str) -> tuple[Origin, str]:
        changed = [one for one, stages in self.own if stage in stages]
        if changed:
            asked = list(dict.fromkeys(trigger_text(one, self.log) for one in changed))
            return "checks", f"changed for this {self.noun}, {'; '.join(asked)}"
        line = next(
            (one for one in (self.line, self.trial) if one is not None and stage in one.parameters),
            None,
        )
        if line is not None:
            return noted(line.notes, stage, self.chooser)[0], ""
        if is_default(self.preset.get(stage), self.defaults.get(stage)):
            return ("agent" if self.chooser == "agent" else "default"), ""
        return "you", ""

    def untouched(self, stage: str) -> bool:
        """Whether `stage` ran as the preset has it: neither the unit's retries nor the line
        changed it, nor the request."""
        changed = any(stage in stages for _, stages in self.own)
        line = any(one is not None and stage in one.parameters for one in (self.line, self.trial))
        same = is_default(self.preset.get(stage), self.defaults.get(stage))
        return not changed and not line and same

    def causes(self, stage: str) -> list[str]:
        """What asked the unit's own retries that changed `stage`, in words, in order."""
        return [trigger_text(one, self.log) for one, stages in self.own if stage in stages]


def _stages(
    manifest: RunManifest, log: QCLog | None, unit: str, log_stage: str, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """The settings `names` ran with for `unit` at `log_stage`: the run's preset, the line's
    changes (a window's), the unit's own last."""
    own = log.latest(unit, log_stage) if log is not None else None
    line = log.latest(LINE, log_stage) if log is not None and log_stage == "phase_shift" else None
    changes = merged(line.parameters if line else {}, own.parameters if own else {})
    return {
        name: merged(preset_stage(manifest, name), changes.get(name, {}) or {}) for name in names
    }


def _said(stage: str, stages: Mapping[str, Mapping[str, Any]], origins: _Origins) -> Setting:
    """Step `stage` as the run card says it: its name, its value, its other parameters."""
    origin, why = origins.of(stage)
    found = step_words(stage, {stage: dict(stages[stage])})
    label, parts = found if found is not None else (_NAMES.get(stage, stage), ("none",))
    return Setting(
        key=stage,
        label=label,
        value=parts[0],
        detail=SEP.join(parts[1:]),
        why=why,
        origin=origin,
    )


def record_settings(manifest: RunManifest, log: QCLog | None, name: str) -> tuple[Setting, ...]:
    """What was done to record `name` before any window used it, in the run card's words: its
    trigger corrected (a shot's, when muted), its mute and its filter (each when on, or not the
    preset's)."""
    stages = _stages(manifest, log, name, "preprocessing", _RECORD_STAGES)
    origins = _Origins(manifest, log, name, "preprocessing", "record")
    fields = type(manifest.preset).model_fields
    settings: list[Setting] = []
    muted = stages["muting"].get("method", "none") != "none"
    started = manifest.started_at
    before = (started if started.tzinfo else started.replace(tzinfo=UTC)) < _TRIGGER_WITH_MUTING
    if "trigger" in fields:
        origin, why = origins.of("trigger")
        shifted = muted or before
        # Not muted, the record is left as recorded: its trigger said only when not the preset's.
        if shifted or not origins.untouched("trigger"):
            settings.append(
                Setting(
                    key="trigger",
                    label="Trigger",
                    value=trigger_words(stages["trigger"].get("t0")) if shifted else "none",
                    detail="" if shifted else "not muted: left as recorded",
                    why=why,
                    origin=origin,
                )
            )
    # A step off by default and left off is not said, as the run card says none.
    for name in ("muting", "filtering"):
        if name not in fields:
            continue
        said = _said(name, stages, origins)
        if not (said.value == "none" and origins.untouched(name)):
            settings.append(said)
    return tuple(settings)


def window_settings(manifest: RunManifest, log: QCLog | None, unit: str) -> tuple[Setting, ...]:
    """What was done to window `unit` to make its dispersion image, in the run card's words and
    its steps' order: its receivers, then per mode, the shots' images stacked (active), their
    correlations stacked (passive-active), or the noise's segments selected, whitened,
    normalized and stacked (passive); its dispersion image. A step off by default and left off
    is not said, as the run card says none."""
    mode = str(manifest.preset.mode)
    names = _WINDOW_STAGES[mode]
    stages = _stages(manifest, log, unit, "phase_shift", ("masw", *names))
    origins = _Origins(manifest, log, unit, "phase_shift", "window")
    length = int(stages["masw"].get("length", 0))
    origin, why = origins.of("masw")
    settings = [
        Setting(
            key="window",
            label="Window",
            value=receivers(length, manifest.profile.receiver_spacing_m),
            why=why,
            origin=origin,
        )
    ]
    for name in names:
        said = _said(name, stages, origins)
        if not (said.value == "none" and origins.untouched(name)):
            settings.append(said)
    return tuple(settings)


def with_spreads(
    unit: str,
    per_unit: Mapping[str, Sequence[Setting]],
    noun: str,
    ranges: Mapping[str, str] | None = None,
) -> tuple[Setting, ...]:
    """`unit`'s settings (`per_unit`: each unit of the line's), each with how the line's other
    units differ in it (`spread`, "other records: …"; "" when they all ran with the same): the
    other values, the most common first, each with the units that ran with it (named when a
    few, else counted, a `noun` each), or its range over them all (`ranges`) when too many to
    name."""
    others = [(other, settings) for other, settings in per_unit.items() if other != unit]
    spread: list[Setting] = []
    for setting in per_unit.get(unit, ()):
        values: dict[str, list[str]] = {}
        for other, settings in others:
            match = next((one for one in settings if one.key == setting.key), None)
            if match is not None and _shown(match) != _shown(setting):
                values.setdefault(_shown(match), []).append(other)
        said = ""
        if len(values) > _SPREAD_VALUES:
            said = (
                f"{ranges[setting.key]} over the {plural(len(per_unit), noun)}"
                if ranges is not None and ranges.get(setting.key)
                else f"{plural(len(values), 'other value')} on "
                f"{plural(sum(len(units) for units in values.values()), noun)}"
            )
        elif values:
            ranked = sorted(values.items(), key=lambda item: -len(item[1]))
            said = "; ".join(
                f"{shown} on {_named(units, noun, len(others))}" for shown, units in ranked
            )
        update = {"spread": f"other {noun}s: {said}" if said else ""}
        spread.append(setting.model_copy(update=update))
    return tuple(spread)


def records_in_common(
    manifest: RunManifest, log: QCLog | None, names: Sequence[str]
) -> tuple[Setting, ...]:
    """How the records `names` (a window's) were preprocessed before their image was made, step
    by step, as most of them were (in_common)."""
    origins = {name: _Origins(manifest, log, name, "preprocessing", "record") for name in names}
    return in_common(
        {name: record_settings(manifest, log, name) for name in names},
        {name: _stages(manifest, log, name, "preprocessing", _RECORD_STAGES) for name in names},
        {name: {stage: origins[name].causes(stage) for stage in _RECORD_STAGES} for name in names},
        "record",
    )


def in_common(
    per_unit: Mapping[str, Sequence[Setting]],
    values: Mapping[str, Mapping[str, Mapping[str, Any]]],
    causes: Mapping[str, Mapping[str, Sequence[str]]],
    noun: str,
) -> tuple[Setting, ...]:
    """The settings most of `per_unit`'s units ran with (a window's records), by kind: a method
    (a mute, none), a trigger's delay (or their files'), whatever numbers each unit had of its
    own (`values`: each unit's steps) given as their range over those units, and why (`causes`:
    what changed each unit's step); with the other kinds and the units that ran with them
    (`spread`, "other records: …"; named when a few, else counted)."""
    first = next(iter(per_unit.values()), ())
    common: list[Setting] = []
    for setting in first:
        groups: dict[str, list[tuple[str, Setting]]] = {}
        for unit, settings in per_unit.items():
            match = next((one for one in settings if one.key == setting.key), None)
            if match is not None:
                groups.setdefault(_NUMBERS.sub("#", match.value), []).append((unit, match))
        ranked = sorted(groups.values(), key=len, reverse=True)
        shown = [_together(group, values, causes, noun) for group in ranked]
        if len(ranked) - 1 > _SPREAD_VALUES:
            others = sum(len(group) for group in ranked[1:])
            said = f"{plural(len(ranked) - 1, 'other value')} on {plural(others, noun)}"
        else:
            said = "; ".join(
                f"{_shown(one)} on {_named([unit for unit, _ in group], noun, -1)}"
                for one, group in zip(shown[1:], ranked[1:], strict=True)
            )
        update = {"spread": f"other {noun}s: {said}" if said else ""}
        common.append(shown[0].model_copy(update=update))
    return tuple(common)


def _together(
    group: Sequence[tuple[str, Setting]],
    values: Mapping[str, Mapping[str, Mapping[str, Any]]],
    causes: Mapping[str, Mapping[str, Sequence[str]]],
    noun: str,
) -> Setting:
    """One setting for the units of `group`, which ran with the same kind: the first's when they
    all said the same; else what they share, each number that differs as its range, and why."""
    first = group[0][1]
    if len({(_shown(one), one.why) for _, one in group}) == 1:
        return first
    key = first.key
    steps = [values[unit].get(key, {}) for unit, _ in group]
    value, detail = first.value, first.detail
    if key == "trigger":
        delays = [step.get("t0") for step in steps]
        if all(isinstance(one, int | float) for one in delays) and len(set(delays)) > 1:
            value = f"delay {_field_together('t0', delays)}"
    else:
        # The parameters that differ as their ranges, in place of the first's own.
        differ = [
            name
            for name in steps[0]
            if name != "method" and len({str(step.get(name)) for step in steps}) > 1
        ]
        words = [_FIELDS.get(name, name.replace("_", " ")) for name in differ]
        ranged = [
            f"{word} {said}"
            for name, word in zip(differ, words, strict=True)
            if (said := _field_together(name, [step.get(name) for step in steps]))
        ]
        if ranged:
            kept = [
                part
                for part in first.detail.split(SEP)
                if part and not part.startswith(tuple(words))
            ]
            detail = SEP.join([*kept, *ranged])
    origin = Counter(one.origin for _, one in group).most_common(1)[0][0]
    why = first.why
    # Each unit's own change, said for them all.
    if (
        len(group) > 1
        and origin in ("workflow", "agent", "checks")
        and (origin == "checks" or len({one.why for _, one in group}) > 1)
    ):
        asked = sorted({cause for unit, _ in group for cause in causes[unit].get(key, ())})
        why = f"changed for {plural(len(group), noun)}" + (f", {'; '.join(asked)}" if asked else "")
    return first.model_copy(update={"value": value, "detail": detail, "origin": origin, "why": why})


def _field_together(name: str, found: Sequence[object]) -> str:
    """A field's values over units, in words: the one they share, or the range of its numbers
    ("0-22.8 ms"); "" when none has it."""
    present = [one for one in found if one is not None]
    if not present:
        return ""
    numbers = [
        float(one) for one in present if isinstance(one, int | float) and not isinstance(one, bool)
    ]
    if len({str(one) for one in present}) == 1:
        return _field(name, present[0])
    if len(numbers) == len(present):
        if name == "t0" and max(abs(one) for one in numbers) < 1:
            return span(1000 * min(numbers), 1000 * max(numbers), "ms", 4)
        return span(min(numbers), max(numbers), _UNITS.get(name, ""), 4)
    return "varies"


def record_ranges(
    manifest: RunManifest, log: QCLog | None, names: Sequence[str] | None = None
) -> dict[str, str]:
    """The numbers the records' preprocessing ran with (`names`, else every record's) that vary
    over them, by setting, as their ranges ("trigger delay 0-22.8 ms")."""
    return _ranges(
        {
            name: _stages(manifest, log, name, "preprocessing", _RECORD_STAGES)
            for name in (names if names is not None else [one.name for one in manifest.records])
        }
    )


def window_ranges(manifest: RunManifest, log: QCLog | None, units: Sequence[str]) -> dict[str, str]:
    """The numbers the windows' images were made with that vary over them, by setting, as
    their ranges ("highest frequency 42-100 Hz")."""
    names = ("masw", *_WINDOW_STAGES[str(manifest.preset.mode)])
    ranges = _ranges({unit: _stages(manifest, log, unit, "phase_shift", names) for unit in units})
    if "masw" in ranges:
        ranges["window"] = ranges.pop("masw")
    return ranges


def _ranges(per_unit: Mapping[str, dict[str, dict[str, Any]]]) -> dict[str, str]:
    """By stage, the numeric fields of its values (`per_unit`: each unit's) that vary over the
    units, each as its range."""
    numbers: dict[str, dict[str, list[float]]] = {}
    for stages in per_unit.values():
        for stage, values in stages.items():
            for field, value in values.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numbers.setdefault(stage, {}).setdefault(field, []).append(float(value))
    return {
        stage: SEP.join(
            f"{_FIELDS.get(field, field.replace('_', ' '))} {_field_together(field, found)}"
            for field, found in fields.items()
            if min(found) != max(found)
        )
        for stage, fields in numbers.items()
    }


# What an attempt changed of the one before, in words, a line each.
type Changes = Callable[[Mapping[str, Any], Mapping[str, Any]], list[str]]


def attempt_steps(
    log: QCLog | None,
    unit: str,
    stages: Sequence[str],
    spacing: float,
    changes: Changes | None = None,
) -> tuple[Step, ...]:
    """What was done to `unit` at the log's `stages`, a run a step in the order they ran, those a
    redo set aside too (the agent's picks and inversions again): what ran, why it ran again,
    what it changed (`changes`; by default its processing's steps in the run card's words), what
    the checks said of it. None when each stage ran once: the card's badges and settings say it
    all then."""
    if log is None:
        return ()
    attempts = sorted(
        (one for stage in stages for one in log.runs_of(unit, stage)),
        key=lambda one: one.started_at,
    )
    again = any(len(log.runs_of(unit, stage)) > 1 for stage in stages)
    if not again and not any(one.status == "failed" for one in attempts):
        return ()
    steps: list[Step] = []
    last: dict[str, Attempt] = {}
    for one in attempts:
        before = last.get(one.stage)
        lines: list[str] = []
        if why := cause(one.triggered_by, log):
            lines.append(_capitalized(why))
        if before is not None:
            lines += (changes or partial(steps_changed, spacing=spacing))(
                before.parameters, one.parameters
            )
        if one.status == "failed":
            lines.append(f"Failed: {one.error}" if one.error else "Failed")
        lines += [_verdict_line(gate, result) for gate, result in one.results.items()]
        steps.append(
            Step(
                kind=one.stage,
                by="checks" if ":" in one.triggered_by else log.chooser,
                at=one.started_at,
                title=_DONE.get(one.stage, _capitalized(one.stage.replace("_", " ")))
                + (" again" if before is not None else ""),
                lines=tuple(lines),
            )
        )
        last[one.stage] = one
    return tuple(steps)


def _verdict_line(gate: str, result: GateResult) -> str:
    """A check's verdict on a run, in a line, with what it flagged when it did not pass:
    "Curve check: flagged (near field)"."""
    said = f"{_capitalized(CHECK_NAMES.get(gate, gate).removeprefix('the '))}: "
    said += _VERDICTS[result.verdict]
    flags = [flag.name for flag in result.flags if flag.name not in REFUSALS]
    return said + (f" ({counted(flags)})" if result.verdict != "pass" and flags else "")


def steps_changed(before: Mapping[str, Any], after: Mapping[str, Any], spacing: float) -> list[str]:
    """What an attempt changed of its processing's steps, in the run card's words: "Mute:
    80-1500 m/s"; a record's trigger alone, as its card says it: "Trigger: delay 12 ms"."""
    changed = _changed(before, after)
    said: list[str] = []
    trigger = changed.get("trigger")
    if isinstance(trigger, Mapping) and "muting" not in changed:
        del changed["trigger"]
        said.append(f"Trigger: {trigger_words(cast(Mapping[str, Any], trigger).get('t0'))}")
    return said + [f"{name}: {value}" for name, value in settings_said(changed, spacing).items()]


def _changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """What `after` changed of `before`: its values that differ, a stage's fields apart."""
    changed: dict[str, Any] = {}
    for key, value in after.items():
        was = before.get(key)
        if isinstance(value, Mapping) and isinstance(was, Mapping):
            inner = _changed(cast(Mapping[str, Any], was), cast(Mapping[str, Any], value))
            if inner:
                changed[key] = inner
        elif value != was:
            changed[key] = value
    return changed


def _shown(setting: Setting) -> str:
    """A setting in a line, as a spread names it: "80-1500 m/s (signal width 0.05 s)"."""
    return f"{setting.value} ({setting.detail})" if setting.detail else setting.value


def _named(units: Sequence[str], noun: str, others: int) -> str:
    """The units a value was run with: all the others, a few by name, or counted."""
    if len(units) == others and others > 1:
        return f"the {plural(others, 'other ' + noun)}"
    if len(units) <= _SPREAD_UNITS:
        return ", ".join(unit_name(one) for one in units)
    return plural(len(units), noun)


def unit_name(unit: str) -> str:
    """A unit as the pages name it: a record by its file, a window by its middle."""
    return f"xmid {number(xmid_of(unit), 4)} m" if unit.startswith("xmid_") else unit
