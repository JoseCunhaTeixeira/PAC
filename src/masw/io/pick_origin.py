"""Who picked a window's curves: a picker, automatically, or a person in PAC, by hand. PAC records
every edit of a window's picks (picks_edited.json in its folder, with the time), and each pick of
its own automatic picking (picks_auto.json); a window is automatic when PACo's QC log holds a
picking attempt for it, or PAC picked it automatically, and no edit came after."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from masw.io.quality.log import QCLog

EDITS_FILE = "picks_edited.json"
AUTO_FILE = "picks_auto.json"

type Origin = Literal["auto", "hand"]


def mark_edited(window: Path) -> None:
    """Record that PAC changed the picks of window folder `window`, now."""
    (window / EDITS_FILE).write_text(json.dumps({"edited_at": datetime.now(UTC).isoformat()}))


def mark_auto(window: Path) -> None:
    """Record that PAC's automatic picking picked window folder `window`, now."""
    (window / AUTO_FILE).write_text(json.dumps({"picked_at": datetime.now(UTC).isoformat()}))


def _time(window: Path, name: str, key: str) -> datetime | None:
    path = window / name
    if not path.exists():
        return None
    value: object = json.loads(path.read_text()).get(key)
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def edited_at(window: Path) -> datetime | None:
    """When PAC last changed the window's picks; None when it never did."""
    return _time(window, EDITS_FILE, "edited_at")


def auto_at(window: Path) -> datetime | None:
    """When PAC's automatic picking last picked the window; None when it never did."""
    return _time(window, AUTO_FILE, "picked_at")


def pick_origin(window: Path, log: QCLog | None, picked: bool) -> Origin | None:
    """Who picked the curves of window folder `window` (`picked`: it holds some); None when it
    holds none."""
    if not picked:
        return None
    attempts = log.of(window.name, "picking") if log is not None else ()
    done = [attempt for attempt in attempts if attempt.status == "succeeded"]
    # The last automatic pick: PACo's, or PAC's own.
    picks = [done[-1].finished_at or done[-1].started_at] if done else []
    if (own := auto_at(window)) is not None:
        picks.append(own)
    if not picks:
        return "hand"
    edited = edited_at(window)
    return "hand" if edited is not None and edited > max(picks) else "auto"
