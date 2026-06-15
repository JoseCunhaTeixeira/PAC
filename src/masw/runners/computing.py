from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable
from venv import logger

from masw.adapters.registry import PIPELINE_BUILDERS, ProcessingMode
from masw.adapters.windows import MASWWindow, build_windows
from masw.models.computing import ComputingConfig


def run_compute(config: ComputingConfig, mode: ProcessingMode):

    windows = build_windows(
        config.acquisition_params,
        config.masw_params,
    )

    builder = PIPELINE_BUILDERS[mode]

    with ProcessPoolExecutor(max_workers=config.execution_params.n_workers) as executor:
        futures = [
            executor.submit(
                process_window,
                config,
                window,
                builder,
            )
            for window in windows
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.exception(f"Window processing failed: {exc}")


def process_window(
    config: ComputingConfig,
    window: MASWWindow,
    build_pipeline: Callable,
):

    output_folder = config.execution_params.output_folder / f"xmid_{window.xmid:.2f}"

    pipeline = build_pipeline(
        config=config,
        window=window,
        output_folder=output_folder,
    )

    pipeline.run()
