"""Who picked a window's curves, by sigpipe's rule (masw.runs.origin): a picker, automatically
(the assistant, from its QC log, or PAC's own automatic picking), or a person in PAC, by hand.
PAC records each change a person makes to a mode's curve and each of its own automatic picks;
the assistant's picks and checks are its QC log's picking attempts."""

from collections.abc import Collection
from datetime import datetime
from pathlib import Path

from masw.io.quality.log import Attempt, QCLog
from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.runs.origin import JUDGED, M0, Origin, auto_at, checks_current, curve_origin


def _done(window: Path, log: QCLog | None) -> list[Attempt]:
    attempts = log.of(window.name, "picking") if log is not None else ()
    return [attempt for attempt in attempts if attempt.status == "succeeded"]


def _ended(attempt: Attempt) -> datetime:
    return attempt.finished_at or attempt.started_at


def picked_at(window: Path, log: QCLog | None) -> datetime | None:
    """When the assistant last picked window folder `window`'s M0 (not a judgement of a curve
    it did not pick); None when it never did."""
    picks = [attempt for attempt in _done(window, log) if attempt.triggered_by != JUDGED]
    return _ended(picks[-1]) if picks else None


def checks_current_in(window: Path, log: QCLog | None) -> bool:
    """Whether the assistant's latest checks of window folder `window`'s M0 (G3, G4) are of the
    curve it holds now: made after the curve's last change in PAC."""
    done = _done(window, log)
    return checks_current(window, _ended(done[-1]) if done else None)


def assistant_picked(window: Path, log: QCLog | None) -> bool:
    """Whether window folder `window`'s last automatic pick is the assistant's, not PAC's own
    automatic picking."""
    at = picked_at(window, log)
    if at is None:
        return False
    own = auto_at(window)
    return own is None or at >= own


def pick_origin(window: Path, log: QCLog | None, modes: Collection[Mode]) -> Origin | None:
    """Who made the curves of window folder `window`, holding `modes` (none: None): its M0's
    maker, whose curve the checks judge; a person's when it holds higher modes alone."""
    if not modes:
        return None
    if M0 not in modes:
        return "hand"
    return curve_origin(window, M0, picked_at(window, log))
