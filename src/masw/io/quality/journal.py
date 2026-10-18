"""The geophysicist's journal of a run: the study of PACo's autonomous agent whose line the run is
(<output>/<profile>/studies/<id>/: its line, line.json, and its journal, journal.jsonl), read as
plain JSON, PACo not needed. The journal says what the agent set out to do, each experiment and
commit with its numbers, its hypotheses, conclusions and decisions, the freeze, the inversions and
the report."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

STUDIES = "studies"  # in a profile's output folder
LINE_FILE = "line.json"  # in a study's folder
JOURNAL_FILE = "journal.jsonl"


class JournalEntry(BaseModel):
    """One entry: its id (E3, L2, H1...), its kind, when, what it says, and the entries it rests
    on."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    kind: str
    at: datetime
    text: str
    of: tuple[str, ...] = ()


def run_journal(profile_folder: Path, run_id: str) -> tuple[JournalEntry, ...]:
    """The journal of the study whose line is run `run_id` (now, or at one of its commits), the
    newest such study's; none when no study made it."""
    studies = profile_folder / STUDIES
    if not studies.is_dir():
        return ()
    for study in sorted((path for path in studies.iterdir() if path.is_dir()), reverse=True):
        if run_id in _runs(study / LINE_FILE):
            return _entries(study / JOURNAL_FILE)
    return ()


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


def _entries(path: Path) -> tuple[JournalEntry, ...]:
    """The journal's entries, in order; a line unread left out."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ()
    entries: list[JournalEntry] = []
    for line in lines:
        try:
            entries.append(JournalEntry.model_validate_json(line))
        except ValidationError:
            continue
    return tuple(entries)
