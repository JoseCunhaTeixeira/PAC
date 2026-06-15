import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from masw.adapters.registry import PIPELINE_BUILDERS
from masw.adapters.windows import MASWWindow, build_windows
from masw.logging_config import setup_logging
from masw.models.computing import AnyComputingConfig

logger = logging.getLogger(__name__)


def run_compute(config: AnyComputingConfig) -> None:
    windows = build_windows(
        config.acquisition_params,
        config.masw_params,
    )

    builder = PIPELINE_BUILDERS[config.mode]

    logger.info(
        "Starting %s processing: %d windows, %d workers",
        config.mode.value,
        len(windows),
        config.execution_params.n_workers,
    )

    with ProcessPoolExecutor(
        max_workers=config.execution_params.n_workers,
        initializer=setup_logging,
    ) as executor:
        futures = {
            executor.submit(process_window, config, window, builder): window
            for window in windows
        }

        for future in as_completed(futures):
            window = futures[future]
            try:
                future.result()
                logger.info("Finished xmid=%.2f", window.xmid)
            except Exception:
                logger.exception("Processing failed for xmid=%.2f", window.xmid)


def process_window(
    config: AnyComputingConfig,
    window: MASWWindow,
    build_pipeline: Callable,
) -> None:
    output_folder = config.execution_params.output_folder / f"xmid_{window.xmid:.2f}"
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Processing xmid=%.2f -> %s", window.xmid, output_folder)

    pipeline = build_pipeline(
        config=config,
        window=window,
        output_folder=output_folder,
    )

    pipeline.run()
