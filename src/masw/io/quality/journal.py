"""The study of a run PACo's autonomous mode made: the study whose line the run is
(<output>/<profile>/studies/<id>/: its line, line.json, and its journal, journal.jsonl), read as
plain JSON, PACo not needed. The journal says what the agent set out to do, each experiment and
commit with its numbers, its hypotheses, conclusions and decisions, the freeze, the inversions and
the report; the search beside it (searches/), the grid's variants and how the eye judged them."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

STUDIES = "studies"  # in a profile's output folder
LINE_FILE = "line.json"  # in a study's folder
JOURNAL_FILE = "journal.jsonl"
SEARCH_FILE = "searches/S1.json"
# The search's thumbnails, each variant's (its finalists' alone once a session ends): <label>.npz.
SEARCH_IMAGES = "searches/S1_images"
KNOCKOUT_FILE = "searches/S1_knockout.json"
JUDGED_FILE = "searches/S1_judged.jsonl"
SURVEY_FILE = "survey.json"
EXPERIMENTS = "experiments"  # each experiment, <id>.json
# Each experiment's variants' dispersion images, as PACo's tidy keeps them: <trial>.npz.
EXPERIMENT_IMAGES = "experiments/images"
PICKS_FILE = "picks.jsonl"


class JournalEntry(BaseModel):
    """One entry: its id (E3, L2, H1...), its kind, when, what it says, the entries it rests
    on, and code's numbers (an experiment's variants, a commit's changes)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    kind: str
    at: datetime
    text: str
    of: tuple[str, ...] = ()
    data: dict[str, Any] = Field(default_factory=dict[str, Any])


class SearchRow(BaseModel):
    """A variant of the search: its label, the changes of the line's settings it makes, and its
    scores (how many of its windows a curve passed, how deep its curves reach)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    changes: dict[str, Any] = Field(default_factory=dict[str, Any])
    card: dict[str, Any] = Field(default_factory=dict[str, Any])


class SearchDone(BaseModel):
    """The study's search: how many variants of the grid it made, over how many settings, how
    many no sheet showed (the really bad), the places along the line each variant was shown at,
    the sheets the eye judged over how many rounds; its finalists, best first, with the one the
    eye chose of them (or, a search of an older PACo, the experiment that tried them)."""

    model_config = ConfigDict(frozen=True)

    at: datetime
    variants: int
    settings: int
    dropped: int
    places: int = 0
    sheets: int
    rounds: int
    finalists: tuple[SearchRow, ...] = ()
    best: str | None = None
    experiment: str | None = None
    # The line's settings its combinations did not all share, a stage ("selection") and each of
    # its keys ("selection threshold"): the others were the same in every one.
    varied: frozenset[str] = frozenset()


class SurveyDone(BaseModel):
    """The study's survey, as the run's history says it: the line's receivers and records, the
    band the records share."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    receivers: int
    shots: tuple[Any, ...] = ()  # each record
    kind: str = "active"
    usable_spread_hz: tuple[float, float] | None = None
    reach_m: float | None = None  # the farthest shot a window takes unless the settings say


class Scored(BaseModel):
    """What a variant's scorecard says here: the windows it was tried on."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    windows: tuple[float, ...] = ()


class TryDone(BaseModel):
    """A variant of an experiment: its label, its trial's key, its scorecard."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    trial: str = ""
    scorecard: Scored = Scored()


class ExperimentDone(BaseModel):
    """An experiment's file: the line's settings it started from, the windows it ran on (as the
    agent asked them, and their middles), and each variant's trial."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    current: dict[str, Any] = Field(default_factory=dict[str, Any])
    windows: str = ""
    xmids: tuple[float, ...] = ()
    tried: tuple[TryDone, ...] = ()


def study_of(profile_folder: Path, run_id: str) -> Path | None:
    """The study whose line is run `run_id` (now, or at one of its commits), the newest such;
    None when no study made it."""
    studies = profile_folder / STUDIES
    if not studies.is_dir():
        return None
    for study in sorted((path for path in studies.iterdir() if path.is_dir()), reverse=True):
        if run_id in _runs(study / LINE_FILE):
            return study
    return None


def journal_of(study: Path) -> tuple[JournalEntry, ...]:
    """The study's journal, in order; a line unread left out."""
    try:
        lines = (study / JOURNAL_FILE).read_text().splitlines()
    except OSError:
        return ()
    entries: list[JournalEntry] = []
    for line in lines:
        try:
            entries.append(JournalEntry.model_validate_json(line))
        except ValidationError:
            continue
    return tuple(entries)


def search_of(study: Path) -> SearchDone | None:
    """The study's search, when it ran; None when its files cannot be read."""
    try:
        made: Any = json.loads((study / SEARCH_FILE).read_text())
        at = datetime.fromisoformat(made["made_at"])
        knockout: Any = (
            json.loads((study / KNOCKOUT_FILE).read_text())
            if (study / KNOCKOUT_FILE).exists()
            else {}
        )
        judged = _judged(study, at)
        rows = {str(row["label"]): row for row in made["rows"]}
        best = knockout.get("best")
        labels = sorted(knockout.get("finalists", ()), key=lambda label: label != best)
        return SearchDone(
            at=at,
            variants=len(rows),
            settings=len(made.get("axes", {})),
            dropped=int(made.get("dropped", 0)),
            places=len(made.get("places", ())),
            sheets=len(judged),
            rounds=len({sheet.get("round") for sheet in judged}),
            finalists=tuple(
                SearchRow.model_validate(rows.get(label, {"label": label})) for label in labels
            ),
            best=best,
            experiment=knockout.get("experiment"),
            varied=_varied([dict(row.get("changes") or {}) for row in rows.values()]),
        )
    except OSError, ValueError, KeyError, TypeError:
        return None


def _varied(changes: list[dict[str, Any]]) -> frozenset[str]:
    """The stages, and their keys ("<stage> <key>"), whose values not every one of `changes`
    shares (a value left out, as one)."""

    def shown(value: object) -> str:
        return json.dumps(value, sort_keys=True)

    varied: set[str] = set()
    for stage in {name for one in changes for name in one}:
        values = [one.get(stage) for one in changes]
        if len({shown(value) for value in values}) > 1:
            varied.add(stage)
        stages = [
            cast(dict[str, Any], value) if isinstance(value, dict) else {} for value in values
        ]
        for key in {key for one in stages for key in one}:
            if len({shown(one.get(key)) for one in stages}) > 1:
                varied.add(f"{stage} {key}")
    return frozenset(varied)


def survey_of(study: Path) -> SurveyDone | None:
    """The study's survey, when it ran and its file read."""
    try:
        return SurveyDone.model_validate_json((study / SURVEY_FILE).read_text())
    except OSError, ValueError:
        return None


def experiment_of(study: Path, name: str) -> ExperimentDone | None:
    """Experiment `name` of the study (E<n>), when its file is read."""
    try:
        return ExperimentDone.model_validate_json(
            (study / EXPERIMENTS / f"{name}.json").read_text()
        )
    except OSError, ValueError:
        return None


def _judged(study: Path, since: datetime) -> list[dict[str, Any]]:
    """The sheets the eye judged since `since`, the search's making (a search made again is
    judged afresh)."""
    path = study / JUDGED_FILE
    lines = path.read_text().splitlines() if path.exists() else []
    sheets = [cast(dict[str, Any], json.loads(line)) for line in lines if line.strip()]
    return [sheet for sheet in sheets if datetime.fromisoformat(sheet["at"]) >= since]


def _runs(path: Path) -> set[str]:
    """The runs a study's line was, now and at each commit."""
    try:
        line: Any = json.loads(path.read_text())
    except OSError, ValueError:
        return set()
    if not isinstance(line, dict):
        return set()
    found = cast(dict[str, Any], line)
    commits = found.get("commits")
    runs = {found.get("run_id")}
    if isinstance(commits, list):
        runs |= {
            cast(dict[str, Any], commit).get("run_id")
            for commit in cast(list[Any], commits)
            if isinstance(commit, dict)
        }
    return {run for run in runs if isinstance(run, str)}
