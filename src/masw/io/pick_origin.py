"""Who picked a window's curves: PACo's picker, automatically, or a person in PAC, by hand. PAC
records every edit of a window's picks (picks_edited.json in its folder, with the time); a window
is automatic when PACo's QC log holds a picking attempt for it that no edit came after. In a run
PAC made alone, every pick is by hand."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from masw.io.quality.log import QCLog

EDITS_FILE = "picks_edited.json"

type Origin = Literal["auto", "hand"]


def mark_edited(window: Path) -> None:
    """Record that PAC changed the picks of window folder `window`, now."""
    (window / EDITS_FILE).write_text(json.dumps({"edited_at": datetime.now(UTC).isoformat()}))


def edited_at(window: Path) -> datetime | None:
    """When PAC last changed the window's picks; None when it never did."""
    path = window / EDITS_FILE
    if not path.exists():
        return None
    value: object = json.loads(path.read_text()).get("edited_at")
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def pick_origin(window: Path, log: QCLog | None, picked: bool) -> Origin | None:
    """Who picked the curves of window folder `window` (`picked`: it holds some); None when it
    holds none."""
    if not picked:
        return None
    attempts = log.of(window.name, "picking") if log is not None else ()
    done = [attempt for attempt in attempts if attempt.status == "succeeded"]
    if not done:
        return "hand"
    last = done[-1]
    edited = edited_at(window)
    return (
        "hand" if edited is not None and edited > (last.finished_at or last.started_at) else "auto"
    )
