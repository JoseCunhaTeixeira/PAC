"""What was done to a unit, as its stage's "Settings, and why" says it: each step it went through
with the settings it ran with (the run's preset with the unit's own changes on top: the
assistant's retries, from its QC log; a window's, the line's too), where each comes from, its
history when the retries changed it, and how the line's other units differ in it. Read, never
measured."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from masw.io.quality.files import merged, preset_stage
from masw.io.quality.log import LINE, Attempt, QCLog
from masw.io.quality.runs import DEFAULT, GIVEN, PAC, is_default
from masw.io.quality.view import Origin, Setting, number, plural, span
from sigpipe.masw.presets import make_preset
from sigpipe.masw.runs import RunManifest, xmid_of

# The stages as their steps say them, and their values' units.
_NAMES = {
    "trigger": "Trigger",
    "muting": "Muting",
    "filtering": "Filter",
    "slicing": "Segments",
    "selection": "Selection",
    "whitening": "Whitening",
    "normalization": "Normalization",
    "stacking": "Stacking",
    "dispersion": "Phase shift",
    "image_stacking": "Images stacked",
}
_UNITS = {
    "t0": "s",
    "tmin": "s",
    "tmax": "s",
    "width": "s",
    "length_s": "s",
    "overlap_s": "s",
    "fmin": "Hz",
    "fmax": "Hz",
    "taper_width_Hz": "Hz",
    "vmin": "m/s",
    "vmax": "m/s",
}
# A step every run does the same way.
_ALWAYS = "every run does it"
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
# A setting's kind, whatever numbers a unit had of its own (a trigger's shift): its value
# without its numbers.
_NUMBERS = re.compile(r"\d[\d.,]*")
# The field a setting's value says, when a number of each unit's own (a trigger's shift).
_VALUE_FIELD = {"trigger": "t0"}
# From sigpipe c17b3de on (its commit's time, in UTC), a shot's trigger is corrected with its
# muting only; the runs before corrected it whatever the muting.
_TRIGGER_WITH_MUTING = datetime(2026, 9, 28, 14, 46, tzinfo=UTC)


def _value(key: str, value: object) -> str:
    unit = _UNITS.get(key)
    shown = number(value, 4) if isinstance(value, int | float) else str(value)
    return f"{shown} {unit}" if unit else shown


def _values(values: dict[str, Any]) -> str:
    """A stage's values but its method, in words: "fmin 5 Hz, fmax 220 Hz"."""
    return ", ".join(
        f"{key.replace('_', ' ')} {_value(key, value)}"
        for key, value in values.items()
        if key != "method" and value is not None
    )


def trigger_text(attempt: Attempt) -> str:
    """Why an attempt ran, in words."""
    if attempt.triggered_by == "initial":
        return "first run"
    if attempt.triggered_by == "backtrack":
        return "an earlier stage ran again"
    return attempt.triggered_by.replace(":", " ").replace("_", " ")


class _Origins:
    """Where a unit's settings of one log stage come from: its own retries (their history), the
    line's (a window's; a record's, the line's mute trial), else the run's (the preset's
    default, the request, PAC's form)."""

    def __init__(self, manifest: RunManifest, log: QCLog | None, unit: str, log_stage: str) -> None:
        self.log = log
        self.own = [one for one in log.of(unit, log_stage) if one.parameters] if log else []
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
        changed = [one for one in self.own if stage in one.parameters]
        if changed:
            history = "; ".join(
                f"attempt {one.attempt} ({trigger_text(one)}): "
                + (_values(one.parameters[stage]) or str(one.parameters[stage].get("method")))
                for one in changed
            )
            return "rule", f"the checks changed it on this unit: {history}"
        if any(line is not None and stage in line.parameters for line in (self.line, self.trial)):
            return "rule", "the checks changed it on the line"
        if is_default(self.preset.get(stage), self.defaults.get(stage)):
            return "default", DEFAULT
        return ("given", GIVEN) if self.log is not None else ("pac", PAC)

    def causes(self, stage: str) -> list[str]:
        """What asked the unit's own retries that changed `stage`, in words, in order."""
        return [trigger_text(one) for one in self.own if stage in one.parameters]


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


def _setting(stage: str, values: dict[str, Any], origins: _Origins) -> Setting:
    origin, why = origins.of(stage)
    method = values.get("method")
    return Setting(
        key=stage,
        label=_NAMES.get(stage, stage.replace("_", " ").capitalize()),
        value=str(method) if method is not None else _values(values) or "-",
        detail=_values(values) if method not in (None, "none") else "",
        why=why,
        origin=origin,
    )


def _fixed(key: str, label: str, value: str) -> Setting:
    return Setting(key=key, label=label, value=value, why=_ALWAYS, origin="default")


def record_settings(manifest: RunManifest, log: QCLog | None, name: str) -> tuple[Setting, ...]:
    """What was done to record `name` before any window used it: its trigger corrected (a shot's,
    when muted), detrended, muted, filtered."""
    stages = _stages(manifest, log, name, "preprocessing", _RECORD_STAGES)
    origins = _Origins(manifest, log, name, "preprocessing")
    fields = type(manifest.preset).model_fields
    settings: list[Setting] = []
    muted = stages["muting"].get("method", "none") != "none"
    started = manifest.started_at
    before = (started if started.tzinfo else started.replace(tzinfo=UTC)) < _TRIGGER_WITH_MUTING
    if "trigger" in fields:
        t0 = stages["trigger"].get("t0")
        origin, why = origins.of("trigger")
        shifted = muted or before
        settings.append(
            Setting(
                key="trigger",
                label="Trigger",
                value=(_value("t0", t0) if t0 is not None else "its file's") if shifted else "none",
                detail="" if shifted else "not muted: left as recorded",
                why=why,
                origin=origin,
            )
        )
    settings.append(_fixed("detrend", "Detrend", "mean, then linear"))
    if "muting" in fields:
        settings.append(_setting("muting", stages["muting"], origins))
    settings.append(_setting("filtering", stages["filtering"], origins))
    return tuple(settings)


def window_settings(manifest: RunManifest, log: QCLog | None, unit: str) -> tuple[Setting, ...]:
    """What was done to window `unit` to make its dispersion image, in its steps' order: its
    receivers, then per mode, the shots' images stacked (active), their correlations stacked
    (passive-active), or the noise's segments selected, whitened, normalized, correlated and
    stacked (passive); the phase shift."""
    mode = str(manifest.preset.mode)
    names = _WINDOW_STAGES[mode]
    stages = _stages(manifest, log, unit, "phase_shift", ("masw", *names))
    origins = _Origins(manifest, log, unit, "phase_shift")
    length = int(stages["masw"].get("length", 0))
    spacing = manifest.profile.receiver_spacing_m
    origin, why = origins.of("masw")
    settings = [
        Setting(
            key="window",
            label="Window",
            value=f"{number((length - 1) * spacing, 4)} m",
            detail=f"{length} receivers",
            why=why,
            origin=origin,
        )
    ]
    apodized = _fixed("apodize", "Apodized", "hanning, 10 %")
    if mode == "active":
        settings += [
            _setting("dispersion", stages["dispersion"], origins),
            _setting("image_stacking", stages["image_stacking"], origins),
        ]
    elif mode == "passive-active":
        settings += [
            apodized,
            _fixed("correlate", "Correlated", "each shot with the receiver nearest it"),
            _setting("stacking", stages["stacking"], origins),
            _setting("dispersion", stages["dispersion"], origins),
        ]
    else:
        settings += [_setting(name, stages[name], origins) for name in names[:4]]
        settings += [
            apodized,
            _fixed("correlate", "Correlated", "with the first receiver, causal"),
            _setting("stacking", stages["stacking"], origins),
            _setting("dispersion", stages["dispersion"], origins),
        ]
    return tuple(settings)


def with_spreads(
    unit: str,
    per_unit: Mapping[str, Sequence[Setting]],
    noun: str,
    ranges: Mapping[str, str] | None = None,
) -> tuple[Setting, ...]:
    """`unit`'s settings (`per_unit`: each unit of the line's), each with how the line's other
    units differ in it (`spread`, "along the line: …"; "" when they all ran with the same): the
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
        update = {"spread": f"along the line: {said}" if said else ""}
        spread.append(setting.model_copy(update=update))
    return tuple(spread)


def records_in_common(
    manifest: RunManifest, log: QCLog | None, names: Sequence[str]
) -> tuple[Setting, ...]:
    """How the records `names` (a window's) were preprocessed before their image was made, step
    by step, as most of them were (in_common)."""
    origins = {name: _Origins(manifest, log, name, "preprocessing") for name in names}
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
    (mute, none), a trigger's shift (a t0, its file's, none), whatever numbers each unit had of
    its own (`values`: each unit's steps) given as their range over those units, and why
    (`causes`: what changed each unit's step); with the other kinds and the units that ran
    with them (`spread`, "among its records: …"; named when a few, else counted)."""
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
        update = {"spread": f"among its {noun}s: {said}" if said else ""}
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
    field = _VALUE_FIELD.get(key)
    if field is not None:
        value = _field_together(field, [step.get(field) for step in steps])
        detail = first.detail
    else:
        value = first.value
        fields = [name for name in steps[0] if name != "method"]
        detail = ", ".join(
            f"{name.replace('_', ' ')} {said}"
            for name in fields
            if (said := _field_together(name, [step.get(name) for step in steps]))
        )
    origin = Counter(one.origin for _, one in group).most_common(1)[0][0]
    why = first.why
    if len({one.why for _, one in group}) > 1 and origin == "rule":
        asked = sorted({cause for unit, _ in group for cause in causes[unit].get(key, ())})
        why = f"the checks changed it on {plural(len(group), noun)}" + (
            f" ({', '.join(asked)})" if asked else ""
        )
    return first.model_copy(
        update={"value": value or first.value, "detail": detail, "origin": origin, "why": why}
    )


def _field_together(name: str, found: Sequence[object]) -> str:
    """A field's values over units, in words: the one they share, or the range of its numbers
    ("0.035-0.094 s"); "" when none has it."""
    present = [one for one in found if one is not None]
    if not present:
        return ""
    numbers = [
        float(one) for one in present if isinstance(one, int | float) and not isinstance(one, bool)
    ]
    if len({str(one) for one in present}) == 1:
        return _value(name, present[0])
    if len(numbers) == len(present):
        return span(min(numbers), max(numbers), _UNITS.get(name, ""), 4)
    return "varies"


def record_ranges(
    manifest: RunManifest, log: QCLog | None, names: Sequence[str] | None = None
) -> dict[str, str]:
    """The numbers the records' preprocessing ran with (`names`, else every record's) that vary
    over them, by setting, as their ranges ("t0 0-0.0228 s")."""
    return _ranges(
        {
            name: _stages(manifest, log, name, "preprocessing", _RECORD_STAGES)
            for name in (names if names is not None else [one.name for one in manifest.records])
        }
    )


def window_ranges(manifest: RunManifest, log: QCLog | None, units: Sequence[str]) -> dict[str, str]:
    """The numbers the windows' images were made with that vary over them, by setting, as
    their ranges ("fmax 42-100 Hz")."""
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
        stage: ", ".join(
            f"{field.replace('_', ' ')} {span(min(found), max(found), _UNITS.get(field, ''), 4)}"
            for field, found in fields.items()
            if min(found) != max(found)
        )
        for stage, fields in numbers.items()
    }


def _shown(setting: Setting) -> str:
    """A setting in a line, as a spread names it: "mute (vmin 80 m/s, width 0.05 s)"."""
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
