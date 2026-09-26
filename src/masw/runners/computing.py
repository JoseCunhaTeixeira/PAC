"""A profile processed, as PAC's run job does it: sigpipe's run (sigpipe.masw.runs), and its
failed windows as the job's errors, each with its traceback."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from masw.io.paths import OUTPUT_DIR, PACKAGES, workspace
from masw.models.processing import ProcessingRequest
from sigpipe.masw.runs import WindowOutcome, run_processing

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


@dataclass
class WindowError:
    xmid: float
    error_type: str
    message: str
    traceback: str


def run_compute(
    request: ProcessingRequest,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, list[WindowError]]:
    """The run's output folder (<profile>/<run_id>), and its failed windows."""
    logger.info(
        "Starting %s processing of %s, %d workers",
        request.mode.value,
        request.profile,
        request.workers,
    )
    manifest = run_processing(
        request.profile,
        request.mode,
        request.overrides,
        workspace(request.workers),
        on_progress=None
        if on_progress is None
        else lambda done, total: on_progress(done, total, None),
        packages=PACKAGES,
    )
    folder = f"{manifest.profile.name}/{manifest.run_id}"
    errors = [
        window_error(OUTPUT_DIR / folder, outcome)
        for outcome in manifest.windows
        if outcome.status == "failed"
    ]
    logger.info(
        "%d/%d succeeded, %d failed",
        len(manifest.windows) - len(errors),
        len(manifest.windows),
        len(errors),
    )
    return folder, errors


def window_error(run_folder: Path, outcome: WindowOutcome) -> WindowError:
    """A failed window as the job reports it: the error's type and message, and its traceback
    from the window's error.log."""
    error_type, _, message = (outcome.error or "").partition(": ")
    log = run_folder / outcome.folder / "error.log"
    return WindowError(
        xmid=outcome.xmid,
        error_type=error_type,
        message=message,
        traceback=log.read_text() if log.exists() else "",
    )
