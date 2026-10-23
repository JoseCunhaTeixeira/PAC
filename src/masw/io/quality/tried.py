"""What was tried to choose a run's settings, as comparisons the run's card shows: the workflow's
trials on the line (the window lengths, coherence.json; the mutes, mute.json; a passive line's
segments, segments.json; the changes made where curves failed, line_loop.json), and the
autonomous mode's search and trials (its study's journal and experiments/). Each comparison: what
was done, in plain lines; its options, each with every setting of its processing; how many sample
windows gave a curve the checks passed with each, how deep its curves reach; the option kept. The
settings are said here in the card's words, its tiles' too, in the order the records go through
them."""

from collections.abc import Callable, Mapping, Sequence
from graphlib import TopologicalSorter
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from masw.io.quality.files import merged
from masw.io.quality.journal import (
    EXPERIMENT_IMAGES,
    SEARCH_IMAGES,
    ExperimentDone,
    JournalEntry,
    SearchDone,
    experiment_of,
)
from masw.io.quality.log import COHERENCE_FILE, LINE_LOOP_FILE, MUTE_FILE, SEGMENTS_FILE, Chooser
from masw.io.quality.view import number, plural, receivers, span
from sigpipe.masw.presets.stages import ACTIVE_STAGES, PASSIVE_ACTIVE_STAGES, PASSIVE_STAGES

MUTE_NAMES = {
    "none": "none",
    "standard": "standard",
    "wider": "wider cone",
    "cone": "cone",
    "tight": "tight cone",
}
# A setting's parameters apart: a tile's lines, a comparison's cell's.
SEP = " · "
# The segments' options shown beside the one kept: the next by the picks' span.
SEGMENTS_NEXT = 2


def _pipeline_order() -> tuple[str, ...]:
    """Every method's steps in one order, each method's own: sigpipe's pipelines, merged."""
    before: dict[str, set[str]] = {}
    for stages in (ACTIVE_STAGES, PASSIVE_STAGES, PASSIVE_ACTIVE_STAGES):
        for first, then in pairwise(stages):
            before.setdefault(then, set()).add(first)
    return tuple(TopologicalSorter(before).static_order())


STEP_ORDER = _pipeline_order()


class LengthTrial(BaseModel):
    """One window length the workflow tried on a few trial windows along the line."""

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
    segment_s: float | None = None  # a passive line's segments for this length, s


class LengthChoice(BaseModel):
    """The window length the workflow kept for the line (coherence.json), and its trials."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    length: int
    trials: tuple[LengthTrial, ...]
    notes: tuple[str, ...] = ()


class MuteTried(BaseModel):
    """One mute the workflow's mute trial tried on trial windows along the line (mute.json):
    none, the standard one, or the gathers' cone, widened or tight."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    candidate: str
    muting: dict[str, Any] | None = None  # None: no mute
    passed: int  # trial windows G3 passed
    verdicts: tuple[str, ...] = ()  # one a trial window
    wavelengths_m: tuple[float, float] | None = None  # median shortest and longest passed
    kept: bool = False


class SegmentsTried(BaseModel):
    """One segment length and FK selection the workflow tried on a passive line's trial windows
    (segments.json): the line's own first."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    segment_s: float
    threshold: float | None = None  # the FK selection's; None: no selection
    vmin: float | None = None  # the FK selection's band
    vmax: float | None = None
    wavelength_span: float | None = None  # the picks' longest wavelength over their shortest
    coherence: float | None = None  # the images' along the picks
    kept: bool = False
    own: bool = False  # the line's own settings, which the others are judged against


class LineTried(BaseModel):
    """One change of the line's settings the workflow's line loop tried (line_loop.json): the
    change, the check that asked it and how many windows asked, the sample windows passing G3
    with the line's settings before it and with it, those asking it alike, and whether it was
    kept for the line."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    change: str = ""  # an older line_loop.json says it in the note alone
    overrides: dict[str, Any] = Field(default_factory=dict[str, Any])  # the change, as set
    flag: str
    asked_by: int
    xmids: tuple[float, ...] = ()
    before: int
    after: int
    asked_xmids: tuple[float, ...] = ()  # an older line_loop.json tried the change on one sample
    asked_before: int | None = None
    asked_after: int | None = None
    kept: bool
    note: str = ""


class Tried(BaseModel):
    """One option compared: its name; its settings, one a column of its table; how many sample
    windows gave a curve the checks passed with it; how deep its curves reach; whether it was
    kept. An option of the autonomous mode: its trial, the windows it was tried on, and whether
    its dispersion images can be shown (kept by its study, or the run's own)."""

    model_config = ConfigDict(frozen=True)

    name: str
    values: tuple[str, ...] = ()
    clear: str = ""  # "10 of 12"; "" when none was judged on its own
    depth: str = ""  # "23.3 m"; "" when not measured
    kept: bool = False
    note: str = ""
    trial: str = ""
    windows: tuple[float, ...] = ()
    images: bool = False


class TriedTable(BaseModel):
    """A comparison, in plain lines: what was done, then its settings' columns and its options."""

    model_config = ConfigDict(frozen=True)

    title: str
    intro: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[Tried, ...]
    by: Chooser


def tried_tables(
    run_folder: Path,
    journal: Sequence[JournalEntry],
    search: SearchDone | None,
    study: Path | None,
    spacing: float,
    defaults: Mapping[str, Any],
    reach: float | None,
) -> tuple[TriedTable, ...]:
    """Every comparison made to choose the run's settings: its line's trials, then its study's
    search, its finalists side by side; each read against the receivers' `spacing`, the preset's
    `defaults` and the line's `reach` (the farthest shot when the settings give none)."""
    tables = [
        table
        for table in (
            _lengths(length_choice(run_folder), spacing),
            _mutes(mutes_tried(run_folder), spacing),
            _segments(segments_tried(run_folder), segment_windows(run_folder)),
            _line(line_tried(run_folder), spacing),
        )
        if table is not None
    ]
    if search is None or not search.finalists:
        return tuple(tables)
    commits = [entry for entry in journal if entry.kind == "commit"]
    # The commits that made this run; a journal older than their runs' ids, its last.
    made = [one for one in commits if one.data.get("run_id") == run_folder.name] or commits[-1:]
    # A search of an older PACo tried its finalists side by side, an experiment.
    entry = next(
        (one for one in journal if one.kind == "experiment" and one.id == search.experiment), None
    )
    found = experiment_of(study, entry.id) if study is not None and entry is not None else None
    base = _base(defaults, found.current if found is not None else {}, reach)
    if entry is not None:
        tables.append(_tried_finalists(entry, found, base, spacing, made, search, study))
    else:
        tables.append(_searched(search, base, spacing, made, study))
    return tuple(tables)


def plain(changes: Mapping[str, Any], spacing: float, length: int | None = None) -> list[str]:
    """Settings in the card's words, a line each, with the parameters a tile says: "Windows: 21
    receivers (30 m)", "Shots used: up to 31.6 m", "Mute: 102.5-1,391 m/s"."""
    said = settings_said(changes, spacing, length, every=False)
    return [f"{column}: {value}" for column, value in said.items()]


def settings_said(
    settings: Mapping[str, Any], spacing: float, length: int | None = None, every: bool = True
) -> dict[str, str]:
    """The settings `settings` sets, by the name the card gives each, in the order the records go
    through them, each its value in the card's words, its parameters `SEP` apart (`every`: all of
    them, else those a tile says); those the card names no other way as their keys say them.
    `length`: the windows' receivers when the settings give none, where the shots used count
    from."""
    said: dict[str, str] = {}
    if isinstance(mode := settings.get("mode"), str):
        said["Method"] = mode
    masw = _stage(settings, "masw")
    windows: list[str] = []
    if isinstance(size := masw.get("length"), int | float):
        windows.append(receivers(int(size), spacing))
    if isinstance(step := masw.get("step"), int | float):
        windows.append(f"every {receivers(int(step), spacing, length=False)}")
    if windows:
        said["Windows"] = SEP.join(windows)
    if {"distance_min", "distance_max"} & masw.keys() or "near_field_m" in settings:
        size = masw.get("length", length)
        said["Shots used"] = shots_used(masw, settings.get("near_field_m"), size, spacing)
    for stage in sorted((one for one in settings if one in STEP_ORDER), key=STEP_ORDER.index):
        if (found := step_words(stage, settings, every)) is not None:
            said[found[0]] = SEP.join(found[1])
    known = {"mode", "masw", "near_field_m", *STEP_ORDER}
    for stage, values in settings.items():
        if stage not in known:
            said[stage.replace("_", " ").capitalize()] = _leaves(values)
    return said


def step_words(
    stage: str, settings: Mapping[str, Any], every: bool = True
) -> tuple[str, tuple[str, ...]] | None:
    """Step `stage` of `settings` in the card's words: its name, and its parameters' values, its
    value first (`every`: all of them, else those a tile says); None for a step said with another
    (the trigger, with the mute it goes with) or without a value."""
    if stage == "trigger" and "muting" in settings:
        return None
    if stage == "trigger":
        # Changed alone (the line loop's): a delay of its own, no mute with it.
        delay = [one.removeprefix("trigger ") for one in _trigger(_stage(settings, "trigger"))]
        return ("Trigger", tuple(delay)) if delay else None
    if stage == "muting":
        muting = _stage(settings, "muting")
        if muting.get("method") == "none":
            return "Mute", ("none",)
        main, more = _mute(muting) if muting else ([], [])
        parts = [*main, *more, *_trigger(_stage(settings, "trigger"))] if every else main
        return ("Mute", tuple(parts)) if parts else None
    if stage not in _STEPS:
        return None
    name, words = _STEPS[stage]
    main, more = words(_stage(settings, stage))
    parts = [*main, *more] if every else main
    return (name, tuple(parts)) if parts else None


def shots_used(masw: Mapping[str, Any], near_field: object, length: object, spacing: float) -> str:
    """The shots a window stacks, by their distance from its middle: from the settings' nearest,
    or from the near field's end (a distance from the window's receivers, its half length
    further), whichever is farther, to their farthest. A near field ending past the farthest
    leaves the windows their near shots, as it does a window with no farther one."""
    near, far = masw.get("distance_min"), masw.get("distance_max")
    low = float(near) if isinstance(near, int | float) and near > 0 else None
    high = float(far) if isinstance(far, int | float) else None
    if isinstance(near_field, int | float) and near_field > 0 and isinstance(length, int | float):
        end = near_field + (length - 1) * spacing / 2
        if high is None or end < high:
            low = max(low or 0.0, end)
    if low is None:
        return "any distance" if high is None else f"up to {number(high, 4)} m"
    return f"beyond {number(low, 4)} m" if high is None else span(low, high, "m", 4)


def _range(low: object, high: object, unit: str) -> str:
    """A range of which the settings may give one end: "0-100 Hz", "up to 60 Hz", "from 5 Hz";
    "" with neither."""
    low = low if isinstance(low, int | float) else None
    high = high if isinstance(high, int | float) else None
    if low is not None and high is not None:
        return span(low, high, unit, 4)
    if high is not None:
        return f"up to {number(high, 4)} {unit}"
    return f"from {number(low, 4)} {unit}" if low is not None else ""


def _stage(settings: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = settings.get(name)
    return dict(cast(Mapping[str, Any], value)) if isinstance(value, Mapping) else {}


# The methods by the names PAC's pages give them.
_METHODS = {
    "none": "none",
    "iir": "IIR",
    "fk": "FK",
    "mute": "mute",
    "onebit": "one-bit",
    "onebit_apod": "one-bit, tapered",
    "linear": "linear",
    "phase_weighted": "phase-weighted",
    "root": "root",
}

type _Words = tuple[list[str], list[str]]  # a step's values a tile says, then the others


def _method(values: Mapping[str, Any]) -> list[str]:
    method = values.get("method")
    return [] if method is None else [_METHODS.get(str(method), str(method).replace("_", " "))]


def _given(values: Mapping[str, Any], key: str, said: str) -> list[str]:
    """A parameter given a value, in words: `said` with its value."""
    value = values.get(key)
    return [said.format(number(value, 4))] if isinstance(value, int | float) else []


def _mute(values: Mapping[str, Any]) -> _Words:
    main = [
        one
        for one in (
            _range(values.get("vmin"), values.get("vmax"), "m/s"),
            _range(values.get("tmin"), values.get("tmax"), "s"),
        )
        if one
    ]
    more = [
        *_given(values, "width", "signal width {} s"),
        *_given(values, "taper", "taper {} samples"),
    ]
    return (main or _method(values)), more


def _trigger(values: Mapping[str, Any]) -> list[str]:
    t0 = values.get("t0")
    if not isinstance(t0, int | float):
        return []
    return [
        f"trigger delay {number(t0 * 1000)} ms" if 0 < t0 < 1 else f"trigger delay {number(t0)} s"
    ]


def _filter(values: Mapping[str, Any]) -> _Words:
    if values.get("method") == "none":
        return ["none"], []
    band = _range(values.get("fmin"), values.get("fmax"), "Hz")
    return ([band] if band else _method(values)), _given(values, "order", "order {}")


def _slicing(values: Mapping[str, Any]) -> _Words:
    return [
        *_given(values, "segment_duration", "{} s"),
        *_given(values, "segment_step", "every {} s"),
    ], []


def _selection(values: Mapping[str, Any]) -> _Words:
    if values.get("method") == "none":
        return ["none"], []
    velocities = _range(values.get("vmin"), values.get("vmax"), "m/s")
    return [
        *(_given(values, "threshold", "threshold {}") or _method(values)),
        *([velocities] if velocities else []),
    ], []


def _whitening(values: Mapping[str, Any]) -> _Words:
    band = _range(values.get("fmin"), values.get("fmax"), "Hz")
    return [*_method(values), *([band] if band else [])], _given(
        values, "taper_width_Hz", "taper {} Hz"
    )


def _stacking(values: Mapping[str, Any]) -> _Words:
    return _method(values), [*_given(values, "nu", "power {}"), *_given(values, "n", "n {}")]


def _dispersion(values: Mapping[str, Any]) -> _Words:
    main = [
        one
        for one in (
            _range(values.get("fmin"), values.get("fmax"), "Hz"),
            _range(values.get("vmin"), values.get("vmax"), "m/s"),
        )
        if one
    ]
    return main, _given(values, "nv", "{} steps")


# The steps of the records and the dispersion images, by the names the card gives them, each in
# its words.
_STEPS: dict[str, tuple[str, Callable[[Mapping[str, Any]], _Words]]] = {
    "filtering": ("Filter", _filter),
    "slicing": ("Segments", _slicing),
    "selection": ("Segment selection", _selection),
    "whitening": ("Whitening", _whitening),
    "normalization": ("Normalization", lambda values: (_method(values), [])),
    "stacking": ("Stacking", _stacking),
    "dispersion": ("Dispersion image", _dispersion),
    "image_stacking": ("Image stacking", _stacking),
}


def _leaves(values: object) -> str:
    """A setting the card names no other way, as its keys say it."""
    if not isinstance(values, Mapping):
        return _shown(values)
    inner = cast(Mapping[str, Any], values)
    return SEP.join(f"{key.replace('_', ' ')} {_shown(one)}" for key, one in inner.items())


def _shown(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none" if value is None else str(value)
    return number(value, 4)


def length_choice(run_folder: Path) -> LengthChoice | None:
    path = run_folder / COHERENCE_FILE
    return LengthChoice.model_validate_json(path.read_text()) if path.exists() else None


class _MuteFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    chosen: str | None = None
    trials: tuple[MuteTried, ...] = ()


class _SegmentsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    windows: tuple[float, ...] = ()
    baseline: SegmentsTried | None = None
    candidates: tuple[SegmentsTried, ...] = ()
    chosen: SegmentsTried | None = None


class _LineLoopFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    trials: tuple[LineTried, ...] = ()


def mutes_tried(run_folder: Path) -> tuple[MuteTried, ...]:
    """The mutes the run's mute trial tried, the one kept said; none without a trial."""
    path = run_folder / MUTE_FILE
    if not path.exists():
        return ()
    found = _MuteFile.model_validate_json(path.read_text())
    return tuple(
        one.model_copy(update={"kept": one.candidate == found.chosen}) for one in found.trials
    )


def segments_tried(run_folder: Path) -> tuple[SegmentsTried, ...]:
    """The segments a passive run's trial tried, the line's own first, the one kept said."""
    path = run_folder / SEGMENTS_FILE
    if not path.exists():
        return ()
    found = _SegmentsFile.model_validate_json(path.read_text())
    own, chosen = found.baseline, found.chosen
    tried = ([own] if own is not None else []) + [one for one in found.candidates if one != own]
    return tuple(
        one.model_copy(
            update={"own": one is own, "kept": one == chosen if chosen is not None else one is own}
        )
        for one in tried
    )


def segment_windows(run_folder: Path) -> int:
    """How many sample windows a passive run's segments trial tried each option on."""
    path = run_folder / SEGMENTS_FILE
    return len(_SegmentsFile.model_validate_json(path.read_text()).windows) if path.exists() else 0


def line_tried(run_folder: Path) -> tuple[LineTried, ...]:
    """The changes of the line's settings the run's line loop tried, in order."""
    path = run_folder / LINE_LOOP_FILE
    if not path.exists():
        return ()
    trials = _LineLoopFile.model_validate_json(path.read_text()).trials
    return tuple(
        one if one.change else one.model_copy(update={"change": _change_of(one.note)})
        for one in trials
    )


# How the line loop words a change it did not keep; one it kept, the change then ": the line loop".
_NOT_KEPT = "Tried for the line, not kept: "


def _change_of(note: str) -> str:
    """The change a try's note words, for a line_loop.json older than its `change`."""
    if note.startswith(_NOT_KEPT):
        return note.removeprefix(_NOT_KEPT).split(", asked by ")[0]
    return note.split(": the line loop")[0]


def _depth(wavelengths_m: tuple[float, float] | None) -> str:
    """The depth curves reach, half their longest wavelength."""
    return f"{number(wavelengths_m[1] / 2)} m" if wavelengths_m is not None else ""


def _options(rows: Sequence[Tried]) -> tuple[Tried, ...]:
    """A comparison's options as every one shows them, the autonomous search's too: those kept
    first, each named by its place ("Option 1") unless it has a name of its own (the line's
    starting settings)."""
    named: list[Tried] = []
    for row in sorted(rows, key=lambda row: not row.kept):
        place = sum(1 for one in named if one.name.startswith("Option ")) + 1
        named.append(row if row.name else row.model_copy(update={"name": f"Option {place}"}))
    return tuple(named)


def _compared(
    options: Sequence[Mapping[str, Any]], spacing: float
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """The options' settings in the card's words: their columns, and each option's values in them."""
    said = [settings_said(one, spacing) for one in options]
    columns = _columns(said)
    return columns, [tuple(one.get(column, "-") for column in columns) for one in said]


def _lengths(choice: LengthChoice | None, spacing: float) -> TriedTable | None:
    if choice is None or not choice.trials:
        return None
    count = max(len(trial.xmids) for trial in choice.trials)
    # A passive line's lengths each with its own segments: the segments the line keeps with it.
    columns, values = _compared(
        [
            {"masw": {"length": trial.length}}
            | (
                {"slicing": {"segment_duration": trial.segment_s, "segment_step": trial.segment_s}}
                if trial.segment_s is not None
                else {}
            )
            for trial in choice.trials
        ],
        spacing,
    )
    rows = [
        Tried(
            name="",
            values=said,
            clear=f"{trial.passed} of {len(trial.xmids)}",
            depth=_depth(trial.wavelengths_m),
            kept=trial.length == choice.length,
        )
        for trial, said in zip(choice.trials, values, strict=True)
    ]
    return TriedTable(
        title="Window lengths tried",
        intro=(f"Each on the same {plural(count, 'sample window')}",),
        columns=columns,
        rows=_options(rows),
        by="workflow",
    )


def _mutes(mutes: Sequence[MuteTried], spacing: float) -> TriedTable | None:
    if not mutes:
        return None
    count = max(len(mute.verdicts) for mute in mutes)
    columns, values = _compared(
        [{"muting": mute.muting or {"method": "none"}} for mute in mutes], spacing
    )
    rows = [
        Tried(
            name="",
            values=tuple(
                _kind(mute, column, value) for column, value in zip(columns, said, strict=True)
            ),
            clear=f"{mute.passed} of {len(mute.verdicts)}",
            depth=_depth(mute.wavelengths_m),
            kept=mute.kept,
        )
        for mute, said in zip(mutes, values, strict=True)
    ]
    return TriedTable(
        title="Mutes tried",
        intro=(f"Each on the same {plural(count, 'sample window')}",),
        columns=columns,
        rows=_options(rows),
        by="workflow",
    )


def _kind(mute: MuteTried, column: str, value: str) -> str:
    """A mute's cell, its kind under its velocities: "80-1500 m/s · standard mute · ..."."""
    if column != "Mute" or mute.muting is None:
        return value
    first, *more = value.split(SEP)
    return SEP.join((first, f"{MUTE_NAMES.get(mute.candidate, mute.candidate)} mute", *more))


def mute_name(mute: MuteTried) -> str:
    """A mute the trial tried, by its kind and the velocities it keeps."""
    if mute.candidate == "none" or mute.muting is None:
        return "No mute"
    low, high = mute.muting.get("vmin"), mute.muting.get("vmax")
    kind = MUTE_NAMES.get(mute.candidate, mute.candidate).capitalize()
    if isinstance(low, int | float) and isinstance(high, int | float):
        return f"{kind} mute ({span(low, high, 'm/s', 4)})"
    return f"{kind} mute"


def _segments(segments: Sequence[SegmentsTried], windows: int) -> TriedTable | None:
    """The segments a passive run's trial tried: the one kept, the next by the picks' span (the
    trial's measure), the line's starting settings last."""
    if not segments:
        return None
    others = sorted(
        (one for one in segments if not one.kept and not one.own),
        key=lambda one: (one.wavelength_span or 0.0, one.coherence or 0.0),
        reverse=True,
    )
    shown = [
        *(one for one in segments if one.kept),
        *others[:SEGMENTS_NEXT],
        *(one for one in segments if one.own and not one.kept),
    ]
    columns, values = _compared(
        [
            {
                "slicing": {"segment_duration": one.segment_s, "segment_step": one.segment_s},
                "selection": {"method": "none"}
                if one.threshold is None
                else {
                    "method": "fk",
                    "threshold": one.threshold,
                    "vmin": one.vmin,
                    "vmax": one.vmax,
                },
            }
            for one in shown
        ],
        0.0,
    )
    tried = plural(len(segments), "combination") + " of segments and selection tried"
    on = f", each on the same {plural(windows, 'sample window')}" if windows else ""
    rows = [
        Tried(
            name="Starting settings" if one.own else "",
            values=(
                *said,
                number(one.wavelength_span) if one.wavelength_span is not None else "-",
                number(one.coherence) if one.coherence is not None else "-",
            ),
            kept=one.kept,
        )
        for one, said in zip(shown, values, strict=True)
    ]
    return TriedTable(
        title="Segments tried",
        intro=(
            tried + on,
            *(
                ["Shown: the one kept, the next by the picks' span, the starting settings"]
                if len(shown) < len(segments)
                else []
            ),
        ),
        columns=(*columns, "Picks' span", "Coherence"),
        rows=_options(rows),
        by="workflow",
    )


def _line(tries: Sequence[LineTried], spacing: float) -> TriedTable | None:
    if not tries:
        return None
    columns, values = _compared([one.overrides or {"change": one.change} for one in tries], spacing)
    rows = [
        Tried(
            name="",
            values=(*said, plural(one.asked_by, "window")),
            clear=f"{one.before} → {one.after} of {len(one.xmids)}"
            + (
                f"{SEP}{one.asked_before} → {one.asked_after} of the {len(one.asked_xmids)} asking it"
                if one.asked_xmids and set(one.asked_xmids) != set(one.xmids)
                else ""
            ),
            kept=one.kept,
        )
        for one, said in zip(tries, values, strict=True)
    ]
    # An older line_loop.json names no sample window.
    count = max(len(one.xmids) for one in tries)
    return TriedTable(
        title="Changes tried where curves failed",
        intro=(
            f"Each on the same {plural(count, 'sample window')}"
            if count
            else "Asked by the windows whose curve failed",
        ),
        columns=(*columns, "Asked by"),
        rows=_options(rows),
        by="workflow",
    )


def line_change(one: LineTried, spacing: float) -> str:
    """A change the line loop tried, in the card's words: "mute 80-1500 m/s"; an older
    line_loop.json's, as its note says it."""
    if not one.overrides:
        return one.change
    said = settings_said(one.overrides, spacing, every=False)
    return "; ".join(f"{column.lower()} {value}" for column, value in said.items())


def _base(
    defaults: Mapping[str, Any], current: Mapping[str, Any], reach: float | None
) -> dict[str, Any]:
    """The line's settings an experiment started from: the preset's defaults with the line's own
    over them; the farthest shot the line's reach when they give none, as the trials take it."""
    base = merged(dict(defaults), dict(current))
    base.setdefault("near_field_m", None)
    masw = dict(_stage(base, "masw"))
    if masw.get("distance_max") is None and reach is not None and base.get("mode") != "passive":
        masw["distance_max"] = reach
    base["masw"] = masw
    return base


def _search_intro(search: SearchDone) -> tuple[str, ...]:
    """What the search did, in short: how many combinations the eye compared and on what, then its
    best."""
    seen = (
        f", each on {search.places} dispersion images spread along the line"
        if search.places
        else " on their dispersion images"
    )
    return (
        f"{search.variants - search.dropped} combinations of settings compared by the agent's "
        f"vision{seen}",
        f"The {len(search.finalists)} best of them:",
    )


def _kept(
    settings: Sequence[Mapping[str, Any]], base: Mapping[str, Any], made: Sequence[JournalEntry]
) -> list[bool]:
    """Whether each of `settings` is the line's: as a commit that `made` the run set them, the
    user's as given."""
    lines = [(merged(dict(base), _changes(one)), _theirs(one)) for one in made]
    return [any(merged(dict(one), theirs) == line for line, theirs in lines) for one in settings]


def _searched(
    search: SearchDone,
    base: Mapping[str, Any],
    spacing: float,
    made: Sequence[JournalEntry],
    study: Path | None,
) -> TriedTable:
    """The search's finalists side by side, the one it chose first, each with every setting, how
    many of its windows a curve passed, how deep its curves reach, and its dispersion images (the
    search's thumbnails, which its study keeps)."""
    settings = [merged(dict(base), dict(one.changes)) for one in search.finalists]
    said = [settings_said(one, spacing) for one in settings]
    columns = _columns(said)
    kept = _kept(settings, base, made)
    images = study / SEARCH_IMAGES if study is not None else None
    rows: list[Tried] = []
    for index, (one, values) in enumerate(zip(search.finalists, said, strict=True)):
        depth = one.card.get("depth_m")
        rows.append(
            Tried(
                name=f"Option {index + 1}",
                values=tuple(values.get(column, "-") for column in columns),
                clear=f"{one.card.get('covered', 0)} of {one.card.get('windows', 0)}",
                depth=f"{number(depth)} m" if isinstance(depth, int | float) else "",
                kept=kept[index],
                trial=one.label,
                images=images is not None and (images / f"{one.label}.npz").exists(),
            )
        )
    return TriedTable(
        title="Search", intro=_search_intro(search), columns=columns, rows=tuple(rows), by="agent"
    )


def _tried_finalists(
    entry: JournalEntry,
    found: ExperimentDone | None,
    base: Mapping[str, Any],
    spacing: float,
    made: Sequence[JournalEntry],
    search: SearchDone,
    study: Path | None,
) -> TriedTable:
    """The finalists of an older PACo's search, tried side by side at full size (an experiment),
    each with every setting, the one the run was made with first; the line's settings they were
    tried against last, when the line had its own."""
    variants = [
        cast(dict[str, Any], one)
        for one in cast(list[Any], entry.data.get("variants", []))
        if isinstance(one, dict)
    ]
    current = [one for one in variants if one.get("label") == "current"]
    tried = [one for one in variants if one.get("label") != "current"]
    settings = [merged(dict(base), dict(one.get("changes") or {})) for one in tried]
    kept = _kept(settings, base, made)
    said = [settings_said(one, spacing) for one in settings]
    line = found is not None and bool(found.current)
    start = settings_said(base, spacing)
    columns = _columns([*said, *([start] if line else [])])
    trials = {one.label: one for one in found.tried} if found is not None else {}
    images = study / EXPERIMENT_IMAGES if study is not None else None

    def row(name: str, values: Mapping[str, str], variant: Mapping[str, Any], keep: bool) -> Tried:
        done = trials.get(str(variant.get("label")))
        trial = done.trial if done is not None else ""
        shown = bool(trial) and (
            keep or (images is not None and (images / f"{trial}.npz").exists())
        )
        return _row(name, values, columns, variant, keep).model_copy(
            update={
                "trial": trial,
                "windows": done.scorecard.windows if done is not None else (),
                "images": shown,
            }
        )

    order = sorted(range(len(tried)), key=lambda index: not kept[index])
    rows = [
        row(f"Option {at}", said[index], tried[index], kept[index])
        for at, index in enumerate(order, start=1)
    ]
    rows += [row("Line's settings", start, one, False) for one in current if line]
    return TriedTable(
        title="Search", intro=_search_intro(search), columns=columns, rows=tuple(rows), by="agent"
    )


def _columns(options: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    """Every setting the options have, in the order the records go through them; the method only
    when they differ in it (the card says it otherwise)."""
    every = dict.fromkeys(column for one in options for column in one)
    return tuple(
        column
        for column in every
        if column != "Method" or len({one.get(column) for one in options}) > 1
    )


def _row(
    name: str,
    said: Mapping[str, str],
    columns: Sequence[str],
    variant: Mapping[str, Any],
    kept: bool = False,
) -> Tried:
    depth = variant.get("depth_m")
    return Tried(
        name=name,
        values=tuple(said.get(column, "-") for column in columns),
        clear=f"{variant.get('passing') or 0} of {variant.get('windows', 0)}",
        depth=f"{number(depth)} m" if isinstance(depth, int | float) else "",
        kept=kept,
    )


def _changes(commit: JournalEntry) -> dict[str, Any]:
    return dict(cast(dict[str, Any], commit.data.get("changes") or {}))


def _theirs(commit: JournalEntry) -> dict[str, Any]:
    """The settings a commit gave as the user's (its reasons say so), as it set them: a trial ran
    with them whatever its own changes."""
    changes = _changes(commit)
    reasons = cast(dict[str, Any], commit.data.get("reasons") or {})
    theirs: dict[str, Any] = {}
    for key, reason in reasons.items():
        if not str(reason).startswith("given by the user"):
            continue
        stage, _, field = key.partition(" ")
        values = changes.get(stage)
        if not field:
            theirs[stage] = values
        elif isinstance(values, Mapping) and field in values:
            theirs.setdefault(stage, {})[field] = cast(Mapping[str, Any], values)[field]
    return theirs
