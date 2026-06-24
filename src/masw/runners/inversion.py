import json
import logging
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import cast

from masw.io import inversion as io
from masw.io.paths import OUTPUT_DIR
from masw.logging_config import setup_logging
from masw.models.inversion import InversionParameters, InversionRunConfig
from masw.runners.computing import WindowError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


def _invert_position_timed(
    folder: str, xmid: float, labels: Sequence[str], parameters: InversionParameters
) -> float:
    start = time.perf_counter()
    io.invert_position(folder, xmid, labels, parameters)
    return time.perf_counter() - start


def run_inversion(
    config: InversionRunConfig,
    on_progress: ProgressCallback | None = None,
) -> list[WindowError]:
    total = len(config.positions)

    logger.info(
        "Starting inversion: %d positions, %d workers",
        total,
        config.n_workers,
    )

    out_dir = OUTPUT_DIR / config.folder
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inversion_params.json").write_text(config.model_dump_json(indent=2))

    errors: list[WindowError] = []
    results: list[dict[str, object]] = []
    completed = 0
    if on_progress is not None:
        on_progress(completed, total, None)

    with ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=setup_logging,
    ) as executor:
        futures = {
            executor.submit(
                _invert_position_timed, config.folder, xmid, config.labels, config.parameters
            ): xmid
            for xmid in config.positions
        }
        for future in as_completed(futures):
            xmid = futures[future]
            pos_err = None
            try:
                duration_s = future.result()
                logger.info("Finished xmid=%.2f", xmid)
                results.append({"xmid": xmid, "status": "success", "duration_s": duration_s})
            except Exception as exc:
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

    results.sort(key=lambda r: cast(float, r["xmid"]))
    (out_dir / "inversion_outcome.json").write_text(json.dumps(results, indent=2))

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

    return errors
