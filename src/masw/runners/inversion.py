"""Seismic inversions of a run's positions, as PAC's inversion job does them: every position's
chains share the job's workers, each on whichever is free, the costliest first (sigpipe's
Scheduler); once a position's chains are in, its files and measures are made in a worker too,
into a staging folder whose files replace the window's only once it succeeded; then the line's
section and comparison figures. Stoppable: see sigpipe.masw.runs.stopping."""

import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import cast

from masw.io import inversion as io
from masw.io.changes import record_inversion
from masw.io.dispersion_images import xmid_folder
from masw.io.history import redone, writing
from masw.io.paths import output_folder
from masw.logging_config import setup_logging
from masw.models.inversion import InversionRunConfig
from masw.runners.computing import WindowError
from sigpipe.algorithms.inversion.rayleigh.seismic.mcmc import Ran
from sigpipe.masw.inversion import InversionParameters, WindowSampling, finish_window
from sigpipe.masw.runs.history import STAGE_FILES
from sigpipe.masw.runs.scheduling import Scheduler, Split
from sigpipe.masw.runs.stopping import Stopped, commit, staging, undo
from sigpipe.workers import one_thread_each

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


def _timed(part: Callable[[], Ran]) -> tuple[Ran, float]:
    """Runs in a worker: a part of a window's inversion (one of its chains, or the layers
    given's chains together), and the time it took."""
    start = time.perf_counter()
    ran = part()
    return ran, time.perf_counter() - start


def _finish_position(
    window: WindowSampling,
    folder: str,
    xmid: float,
    parameters: InversionParameters,
    output: Path,
    ran: Sequence[tuple[Ran, float]],
) -> float:
    """Runs in a worker: the window's inversion made of its chains' outputs, then its measures
    (how deep its data inform it among them), saved with it: Visualization only reads them. The
    time the window took in the workers, its chains' and this."""
    start = time.perf_counter()
    finish_window(window, [chain for chain, _ in ran], output)
    try:
        io.measure_position(folder, xmid, parameters, output)
    except StopIteration, OSError, ValueError:
        # Its model stands, shown without its measures (no fundamental mode picked, say).
        logger.warning("The inversion of xmid=%.2f was not measured", xmid, exc_info=True)
    return sum(seconds for _, seconds in ran) + time.perf_counter() - start


def run_inversion(
    config: InversionRunConfig,
    on_progress: ProgressCallback | None = None,
    stop: threading.Event | None = None,
) -> list[WindowError]:
    """The failed positions; Stopped once `stop` is set: the positions that finished kept, the
    others as they were, the line's figures left as they were. The job holds the
    run while it writes it, shared with PAC's pages (the assistant kept out meanwhile); one the
    assistant writes fails the job, saying so."""
    with writing(output_folder(config.folder)):
        return _run_inversion(config, on_progress, stop)


def _run_inversion(
    config: InversionRunConfig,
    on_progress: ProgressCallback | None,
    stop: threading.Event | None,
) -> list[WindowError]:
    total = len(config.positions)

    logger.info(
        "Starting inversion: %d positions, %d workers",
        total,
        config.n_workers,
    )

    out_dir = output_folder(config.folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seismic_inversion_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    errors: list[WindowError] = []
    results: list[dict[str, object]] = []
    completed = 0
    if on_progress is not None:
        on_progress(completed, total, None)

    # Each position's curves read and its parameters checked here, before any worker starts: a
    # position refused fails alone, the others run.
    windows: dict[float, WindowSampling] = {}
    for xmid in config.positions:
        try:
            windows[xmid] = io.sample_position(
                config.folder, xmid, config.labels, config.parameters
            )
        except Exception as exc:
            pos_err = _failed(config.folder, xmid, exc)
            errors.append(pos_err)
            results.append(
                {"xmid": xmid, "status": "failed", "duration_s": None, **asdict(pos_err)}
            )
            completed += 1
            if on_progress is not None:
                on_progress(completed, total, pos_err)

    parts = sum(len(window.sampling.parts) for window in windows.values())
    one_thread_each()
    with Scheduler(
        max(1, min(config.n_workers, parts)),
        initializer=setup_logging,
    ) as scheduler:
        futures: dict[Future[float], float] = {}
        for xmid, window in windows.items():
            output = staging(xmid_folder(config.folder, xmid))
            split = Split(
                parts=[partial(_timed, part) for part in window.sampling.parts],
                finish=partial(
                    _finish_position, window, config.folder, xmid, config.parameters, output
                ),
                cost=window.sampling.cost,
            )
            futures[scheduler.submit(split)] = xmid
        try:
            for future in scheduler.finished(futures, stop):
                xmid = futures.pop(future)
                pos_err = None
                try:
                    duration_s = future.result()
                    # Its new results replace all its old ones, set aside, and its earlier
                    # attempts' (the assistant's) are forgotten; the change in its history.
                    folder = xmid_folder(config.folder, xmid)
                    aside = commit(folder, replacing=STAGE_FILES["inversion"], aside="inversion")
                    redone(folder, "inversion", aside)
                    record_inversion(folder, config.labels, config.parameters, aside)
                    logger.info("Finished xmid=%.2f", xmid)
                    results.append({"xmid": xmid, "status": "success", "duration_s": duration_s})
                except Exception as exc:
                    pos_err = _failed(config.folder, xmid, exc)
                    errors.append(pos_err)
                    results.append(
                        {"xmid": xmid, "status": "failed", "duration_s": None, **asdict(pos_err)}
                    )
                finally:
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed, total, pos_err)
        except Stopped:
            for xmid in futures.values():
                undo(xmid_folder(config.folder, xmid), created=False)
            _write_outcome(out_dir, results)
            logger.info("Inversion stopped: %d of %d positions finished", completed, total)
            raise

    _write_outcome(out_dir, results)

    n_failed = len(errors)
    logger.info("%d/%d succeeded, %d failed", total - n_failed, total, n_failed)

    try:
        io.save_velocity_section_plot(config.folder)
    except Exception:
        logger.exception("Failed to save velocity section plot for folder=%s", config.folder)

    try:
        io.save_velocity_xzv(config.folder)
    except Exception:
        logger.exception("Failed to save velocity XZV file for folder=%s", config.folder)

    for label in config.labels:
        try:
            io.save_pseudo_section_comparison_plot(config.folder, label)
        except Exception:
            logger.exception(
                "Failed to save pseudo-section comparison for folder=%s, label=%s",
                config.folder,
                label,
            )

    try:
        io.save_line_summary_plot(config.folder)
    except Exception:
        logger.exception("Failed to save the line summary for folder=%s", config.folder)

    return errors


def _failed(folder: str, xmid: float, exc: Exception) -> WindowError:
    """A position whose inversion failed, said so: the window keeps the files it had (a failed
    inversion's are not kept)."""
    undo(xmid_folder(folder, xmid), created=False)
    logger.error("Inversion failed for xmid=%.2f", xmid, exc_info=exc)
    return WindowError(
        xmid=xmid,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(exc)),
    )


def _write_outcome(out_dir: Path, results: list[dict[str, object]]) -> None:
    """The positions that finished, by xmid, with their status and duration: the time each one's
    chains, files and measures took in the workers."""
    results.sort(key=lambda r: cast(float, r["xmid"]))
    (out_dir / "seismic_inversion_outcome.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
