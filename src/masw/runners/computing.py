import json
import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from masw.adapters.registry import PIPELINE_BUILDERS
from masw.adapters.windows import MASWWindow, build_windows
from masw.logging_config import setup_logging
from masw.models.computing import AnyComputingConfig

logger = logging.getLogger(__name__)

# now also receives optional error info
ProgressCallback = Callable[[int, int, "WindowError | None"], None]


@dataclass
class WindowError:
    xmid: float
    error_type: str
    message: str
    traceback: str


def run_compute(
    config: AnyComputingConfig,
    on_progress: ProgressCallback | None = None,
) -> list[WindowError]:
    windows = build_windows(config.acquisition_params, config.masw_params)
    total = len(windows)
    builder = PIPELINE_BUILDERS[config.mode]

    logger.info(
        "Starting %s processing: %d windows, %d workers",
        config.mode.value,
        total,
        config.execution_params.n_workers,
    )

    errors: list[WindowError] = []
    results: list[dict] = []
    completed = 0
    if on_progress is not None:
        on_progress(completed, total, None)

    with ProcessPoolExecutor(
        max_workers=config.execution_params.n_workers,
        initializer=setup_logging,
    ) as executor:
        futures = {
            executor.submit(process_window, config, window, builder): window for window in windows
        }
        for future in as_completed(futures):
            window = futures[future]
            win_err = None
            try:
                future.result()
                logger.info("Finished xmid=%.2f", window.xmid)
                results.append({"xmid": window.xmid, "status": "success"})
            except Exception as exc:
                logger.exception("Processing failed for xmid=%.2f", window.xmid)
                win_err = WindowError(
                    xmid=window.xmid,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
                errors.append(win_err)
                results.append({"xmid": window.xmid, "status": "failed", **asdict(win_err)})
            finally:
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total, win_err)

    profile = config.acquisition_params.folder_path.name
    out_dir = config.execution_params.output_folder / profile
    out_dir.mkdir(parents=True, exist_ok=True)

    results.sort(key=lambda r: r["xmid"])
    (out_dir / "outcome.json").write_text(json.dumps(results, indent=2))

    n_failed = len(errors)
    logger.info("%d/%d succeeded, %d failed", total - n_failed, total, n_failed)

    return errors


def process_window(
    config: AnyComputingConfig,
    window: MASWWindow,
    build_pipeline: Callable,
) -> None:
    profile = config.acquisition_params.folder_path.name
    base = config.execution_params.output_folder / profile
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text(config.model_dump_json(indent=2))

    output_folder = base / f"xmid_{window.xmid:.2f}"
    output_folder.mkdir(parents=True, exist_ok=True)  # <-- was missing
    (output_folder / "window.json").write_text(window.model_dump_json(indent=2))

    logger.info("Processing xmid=%.2f -> %s", window.xmid, output_folder)
    try:
        pipeline = build_pipeline(config=config, window=window, output_folder=output_folder)
        pipeline.run(show_log=False)
    except Exception:
        (output_folder / "error.log").write_text(traceback.format_exc())
        raise
