import json
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import cast

from masw.io import petro_inversion as io
from masw.io.paths import OUTPUT_DIR
from masw.logging_config import setup_logging
from masw.models.petro_inversion import PetroInversionRunConfig
from masw.runners.computing import WindowError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


def _invert_position_timed(folder: str, xmid: float, model_name: str) -> float:
    start = time.perf_counter()
    io.invert_position(folder, xmid, model_name)
    return time.perf_counter() - start


def run_petro_inversion(
    config: PetroInversionRunConfig,
    on_progress: ProgressCallback | None = None,
) -> list[WindowError]:
    total = len(config.positions)

    logger.info(
        "Starting petro inversion: %d positions, model=%s, %d workers",
        total,
        config.model_name,
        config.n_workers,
    )

    out_dir = OUTPUT_DIR / config.folder
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "petro_inversion_config.json").write_text(config.model_dump_json(indent=2))

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
            executor.submit(_invert_position_timed, config.folder, xmid, config.model_name): xmid
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
                logger.exception("Petro inversion failed for xmid=%.2f", xmid)
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
    (out_dir / "petro_inversion_outcome.json").write_text(json.dumps(results, indent=2))

    io.clear_petro_inversion_cache()

    n_failed = len(errors)
    logger.info("%d/%d succeeded, %d failed", total - n_failed, total, n_failed)

    try:
        io.save_petro_section_plot(config.folder)
    except Exception:
        logger.exception("Failed to save petro section plot for folder=%s", config.folder)

    try:
        io.save_petro_section_hdf5(config.folder)
    except Exception:
        logger.exception("Failed to save petro section HDF5 for folder=%s", config.folder)

    try:
        io.save_shear_modulus_section_plot(config.folder)
    except Exception:
        logger.exception("Failed to save shear modulus section plot for folder=%s", config.folder)

    try:
        io.save_shear_modulus_section_hdf5(config.folder)
    except Exception:
        logger.exception("Failed to save shear modulus section HDF5 for folder=%s", config.folder)

    try:
        io.save_vs_section_plot(config.folder)
    except Exception:
        logger.exception("Failed to save Vs section plot for folder=%s", config.folder)

    try:
        io.save_vs_section_hdf5(config.folder)
    except Exception:
        logger.exception("Failed to save Vs section HDF5 for folder=%s", config.folder)

    return errors
