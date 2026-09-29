"""Seismic inversions of a run's positions, as PAC's inversion job does them: each position in a
worker process, into a staging folder whose files replace the window's only once it succeeded,
then the line's section and comparison figures. Stoppable: see sigpipe.masw.runs.stopping."""

import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import cast

from masw.io import inversion as io
from masw.io.dispersion_images import xmid_folder
from masw.io.history import redone
from masw.io.paths import output_folder
from masw.logging_config import setup_logging
from masw.models.inversion import InversionRunConfig
from masw.runners.computing import WindowError
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.runs.history import STAGE_FILES
from sigpipe.masw.runs.stopping import Stopped, commit, finished, staging, undo
from sigpipe.workers import one_thread_each

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


def _invert_position_timed(
    folder: str,
    xmid: float,
    labels: Sequence[str],
    parameters: InversionParameters,
    output: Path,
    chain_jobs: int,
) -> float:
    """Runs in a worker: the window's inversion, then its measures (how deep its data inform
    it among them), saved with it: Visualization only reads them."""
    start = time.perf_counter()
    io.invert_position(folder, xmid, labels, parameters, output, chain_jobs)
    try:
        io.measure_position(folder, xmid, parameters, output)
    except StopIteration, OSError, ValueError:
        # Its model stands, shown without its measures (no fundamental mode picked, say).
        logger.warning("The inversion of xmid=%.2f was not measured", xmid, exc_info=True)
    return time.perf_counter() - start


def chain_jobs(workers: int, windows: int, chains: int) -> int:
    """The processes each window's chains run in: the workers the windows running at once leave
    idle, shared between them, never more than its chains (PACo's rule,
    paco.qc.inverting.chain_jobs). The workers asked are all the cores a job takes (the user,
    2026-09-29: 6 workers on 12 cores took them all, each window's chains sharing the
    machine's cores)."""
    return max(1, min(chains, workers // max(1, windows)))


def run_inversion(
    config: InversionRunConfig,
    on_progress: ProgressCallback | None = None,
    stop: threading.Event | None = None,
) -> list[WindowError]:
    """The failed positions; Stopped once `stop` is set: the positions that finished kept, the
    others as they were, the line's figures left as they were."""
    total = len(config.positions)

    logger.info(
        "Starting inversion: %d positions, %d workers",
        total,
        config.n_workers,
    )

    out_dir = output_folder(config.folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seismic_inversion_config.json").write_text(config.model_dump_json(indent=2))

    errors: list[WindowError] = []
    results: list[dict[str, object]] = []
    completed = 0
    windows = max(1, min(config.n_workers, total))
    jobs = chain_jobs(config.n_workers, windows, config.parameters.n_chains)
    if on_progress is not None:
        on_progress(completed, total, None)

    one_thread_each()
    with ProcessPoolExecutor(
        max_workers=windows,
        initializer=setup_logging,
    ) as executor:
        futures: dict[Future[float], float] = {}
        for xmid in config.positions:
            output = staging(xmid_folder(config.folder, xmid))
            future = executor.submit(
                _invert_position_timed,
                config.folder,
                xmid,
                config.labels,
                config.parameters,
                output,
                jobs,
            )
            futures[future] = xmid
        try:
            for future in finished(executor, futures, stop):
                xmid = futures.pop(future)
                pos_err = None
                try:
                    duration_s = future.result()
                    # Its new results replace all its old ones, and its earlier attempts' (the
                    # assistant's) are forgotten.
                    window = xmid_folder(config.folder, xmid)
                    commit(window, replacing=STAGE_FILES["inversion"])
                    redone(window, "inversion")
                    logger.info("Finished xmid=%.2f", xmid)
                    results.append({"xmid": xmid, "status": "success", "duration_s": duration_s})
                except Exception as exc:
                    # The window keeps the files it had: a failed inversion's are not kept.
                    undo(xmid_folder(config.folder, xmid), created=False)
                    logger.exception("Inversion failed for xmid=%.2f", xmid)
                    pos_err = WindowError(
                        xmid=xmid,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                    )
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


def _write_outcome(out_dir: Path, results: list[dict[str, object]]) -> None:
    """The positions that finished, by xmid, with their status and duration."""
    results.sort(key=lambda r: cast(float, r["xmid"]))
    (out_dir / "seismic_inversion_outcome.json").write_text(json.dumps(results, indent=2))
