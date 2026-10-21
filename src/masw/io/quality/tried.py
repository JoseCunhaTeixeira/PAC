"""What was tried to choose a run's settings, in tables alike for every run that tried something:
PACo's workflow's trials on the line (the window lengths, coherence.json; the mutes, mute.json; a
passive line's segments, segments.json; the changes of the line's settings, line_loop.json), and
its autonomous agent's experiments (its study's journal). Each row: what was tried, how many
trial windows passed with it, its measures, and whether it was kept."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from masw.io.quality.journal import JournalEntry, SearchDone
from masw.io.quality.log import COHERENCE_FILE, LINE_LOOP_FILE, MUTE_FILE, SEGMENTS_FILE, Chooser
from masw.io.quality.view import flag_text, number, plural, span

MUTE_NAMES = {
    "none": "none",
    "standard": "standard",
    "wider": "wider cone",
    "cone": "cone",
    "tight": "tight cone",
}
# A setting's unit, by the end of its name: "near_field_m" is a distance in metres.
_UNITS = {"_m": "m", "_s": "s", "_Hz": "Hz"}
_PASSED = "passed: the trial windows whose curve passed the curve's check (G3)"


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
    wavelength_span: float | None = None  # the picks' longest wavelength over their shortest
    coherence: float | None = None  # the images' along the picks
    kept: bool = False
    own: bool = False  # the line's own settings, which the others are judged against


class LineTried(BaseModel):
    """One change of the line's settings the workflow's line loop tried (line_loop.json): the
    change, the check that asked it and how many windows asked, the trial windows passing G3
    with the line's settings before it and with it, and whether it was kept for the line."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    change: str = ""  # an older line_loop.json says it in the note alone
    flag: str
    asked_by: int
    xmids: tuple[float, ...] = ()
    before: int
    after: int
    kept: bool
    note: str = ""


class Tried(BaseModel):
    """One thing tried: what, how many trial windows passed with it, its measures, whether it
    was kept, and more on hover."""

    model_config = ConfigDict(frozen=True)

    label: str
    passed: str = ""  # "5 of 6"; "" when none was judged on its own
    measures: str = ""
    kept: bool = False
    note: str = ""


class TriedTable(BaseModel):
    """What was tried for one choice, by whom, and on what."""

    model_config = ConfigDict(frozen=True)

    title: str
    about: str
    by: Chooser
    rows: tuple[Tried, ...]


def tried_tables(
    run_folder: Path, journal: Sequence[JournalEntry], search: SearchDone | None
) -> tuple[TriedTable, ...]:
    """Every table of what was tried for the run: its line's trials, then its study's
    experiments, in order."""
    tables = [
        table
        for table in (
            _lengths(length_choice(run_folder)),
            _mutes(mutes_tried(run_folder)),
            _segments(segments_tried(run_folder)),
            _line(line_tried(run_folder)),
        )
        if table is not None
    ]
    commits = [entry for entry in journal if entry.kind == "commit"]
    finalists = search.experiment if search is not None else None
    tables += [
        _experiment(entry, journal, commits, entry.id == finalists)
        for entry in journal
        if entry.kind == "experiment"
    ]
    return tuple(tables)


def changes_text(changes: Mapping[str, Any]) -> str:
    """Settings in words, a stage each: "masw length 24; muting mute, vmin 80, width 0.05 s"."""
    said: list[str] = []
    for stage, value in changes.items():
        if isinstance(value, Mapping):
            inner = cast(Mapping[str, Any], value)
            parts = [
                str(one) if key == "method" else _setting(key, one) for key, one in inner.items()
            ]
            said.append(f"{stage.replace('_', ' ')} {', '.join(parts)}")
        else:
            said.append(_setting(stage, value))
    return "; ".join(said)


def _setting(key: str, value: object) -> str:
    """A setting and its value: "distance max 30", "near field 8.25 m"."""
    unit = next((unit for end, unit in _UNITS.items() if key.endswith(end)), "")
    name = key.removesuffix(f"_{unit}") if unit else key
    if isinstance(value, bool) or not isinstance(value, int | float):
        shown = "none" if value is None else str(value)
    else:
        shown = number(value, 4)
    return f"{name.replace('_', ' ')} {shown} {unit}".rstrip()


def length_choice(run_folder: Path) -> LengthChoice | None:
    path = run_folder / COHERENCE_FILE
    return LengthChoice.model_validate_json(path.read_text()) if path.exists() else None


class _MuteFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    chosen: str | None = None
    trials: tuple[MuteTried, ...] = ()


class _SegmentsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

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


def _measures(*parts: str) -> str:
    return ", ".join(part for part in parts if part)


def _wavelengths(found: tuple[float, float] | None) -> str:
    return f"wavelengths {span(*found, 'm')}" if found is not None else ""


def _lengths(choice: LengthChoice | None) -> TriedTable | None:
    if choice is None or not choice.trials:
        return None
    return TriedTable(
        title="Window lengths tried",
        about=f"Each on the same trial windows along the line; {_PASSED}.",
        by="workflow",
        rows=tuple(
            Tried(
                label=f"{plural(trial.length, 'receiver')} ({number(trial.metres, 4)} m)",
                passed=f"{trial.passed} of {len(trial.xmids)}",
                measures=_measures(
                    f"picks ±{number(100 * trial.uncertainty, 2)} %"
                    if trial.uncertainty is not None
                    else "",
                    _wavelengths(trial.wavelengths_m),
                    f"{plural(trial.windows, 'window')} on the line" if trial.windows else "",
                ),
                kept=trial.length == choice.length,
                note="tried past the length kept, to compare" if trial.compared else "",
            )
            for trial in choice.trials
        ),
    )


def _mutes(mutes: Sequence[MuteTried]) -> TriedTable | None:
    if not mutes:
        return None
    return TriedTable(
        title="Mutes tried",
        about=f"Each on the same trial windows, at the line's length; {_PASSED}.",
        by="workflow",
        rows=tuple(
            Tried(
                label=MUTE_NAMES.get(mute.candidate, mute.candidate) + _bounds(mute.muting),
                passed=f"{mute.passed} of {len(mute.verdicts)}",
                measures=_wavelengths(mute.wavelengths_m),
                kept=mute.kept,
            )
            for mute in mutes
        ),
    )


def _bounds(muting: Mapping[str, Any] | None) -> str:
    """A mute's velocities: ", 80-1,500 m/s"."""
    low, high = (muting or {}).get("vmin"), (muting or {}).get("vmax")
    if isinstance(low, int | float) and isinstance(high, int | float):
        return f", {span(low, high, 'm/s', 4)}"
    return ""


def _segments(segments: Sequence[SegmentsTried]) -> TriedTable | None:
    if not segments:
        return None
    return TriedTable(
        title="Segments tried",
        about="Each on the same trial windows, judged against the line's own by the picks' span "
        "(their longest wavelength over their shortest) and the images' coherence along them.",
        by="workflow",
        rows=tuple(
            Tried(
                label=f"{number(one.segment_s)} s, "
                + ("no selection" if one.threshold is None else f"FK {number(one.threshold)}")
                + (" (the line's)" if one.own else ""),
                measures=_measures(
                    f"span {number(one.wavelength_span)}"
                    if one.wavelength_span is not None
                    else "",
                    f"coherence {number(one.coherence)}" if one.coherence is not None else "",
                ),
                kept=one.kept,
            )
            for one in segments
        ),
    )


def _line(tries: Sequence[LineTried]) -> TriedTable | None:
    if not tries:
        return None
    return TriedTable(
        title="Changes of the line tried",
        about="Each on the whole line, asked by the checks of the windows whose curve failed; "
        "passed: the trial windows passing G3 with the line's settings, then with the change.",
        by="workflow",
        rows=tuple(
            Tried(
                label=one.change,
                passed=f"{one.before} → {one.after} of {len(one.xmids)}",
                measures=f"asked by {plural(one.asked_by, 'window')} ({flag_text(_flag(one))})",
                kept=one.kept,
                note=one.note,
            )
            for one in tries
        ),
    )


def _flag(tried: LineTried) -> str:
    """The flag that asked a change most: "G2:weak_coherence" as "weak_coherence"."""
    return tried.flag.partition(":")[2] or tried.flag


def _experiment(
    entry: JournalEntry,
    journal: Sequence[JournalEntry],
    commits: Sequence[JournalEntry],
    finalists: bool,
) -> TriedTable:
    """An experiment of the agent: each variant against the line's settings then, those a commit
    of the experiment took kept."""
    variants = [
        cast(dict[str, Any], one)
        for one in cast(list[Any], entry.data.get("variants", []))
        if isinstance(one, dict)
    ]
    taken = [
        cast(dict[str, Any], commit.data.get("changes", {}))
        for commit in commits
        if entry.id in commit.of
    ]
    tested = [one for one in journal if one.id in entry.of]
    count = max((int(one.get("windows", 0)) for one in variants), default=0)
    about = " ".join(
        [
            *(f"{one.id}: {one.text.rstrip('.')}." for one in tested),
            f"On {plural(count, 'trial window')}; {_PASSED}; gained and lost: against the line's "
            "settings then.",
        ]
    )
    return TriedTable(
        title=f"{entry.id}: the search's finalists, side by side"
        if finalists
        else f"{entry.id}: {plural(len(variants) - 1, 'variant')} against the line's settings",
        about=about,
        by="agent",
        rows=tuple(_variant(one, taken) for one in variants),
    )


def _variant(variant: Mapping[str, Any], taken: Sequence[Mapping[str, Any]]) -> Tried:
    changes = cast(Mapping[str, Any], variant.get("changes") or {})
    current = variant.get("label") == "current"
    depth = variant.get("depth_m")
    gained = len(cast(list[Any], variant.get("gained") or []))
    lost = len(cast(list[Any], variant.get("lost") or []))
    beaten = variant.get("dominated_by")
    return Tried(
        label=_named("current") if current else f"{variant.get('label')}: {changes_text(changes)}",
        passed=f"{variant.get('passing', 0)} of {variant.get('windows', 0)}",
        measures=_measures(
            "" if current else f"{gained} gained, {lost} lost",
            f"depth {number(depth)} m" if isinstance(depth, int | float) else "",
        ),
        kept=not current and any(_within(changes, one) for one in taken),
        note=f"dominated: {_named(str(beaten))} as good or better on images, coverage and depth"
        if beaten
        else "",
    )


def _named(label: str) -> str:
    return "the line's settings" if label == "current" else f"variant {label}"


def _within(changes: Mapping[str, Any], taken: Mapping[str, Any]) -> bool:
    """Whether `taken` sets every setting `changes` sets, to the same value."""
    for key, value in changes.items():
        other = taken.get(key)
        if isinstance(value, Mapping) and isinstance(other, Mapping):
            if not _within(cast(Mapping[str, Any], value), cast(Mapping[str, Any], other)):
                return False
        elif other != value:
            return False
    return True
