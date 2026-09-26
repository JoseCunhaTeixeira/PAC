import json
import logging
from collections.abc import Callable
from dataclasses import asdict

from masw.io.dispersion_images import xmid_folder
from masw.io.folders import get_xmid_folders
from masw.io.paths import output_folder
from masw.models.petro_inversion import PetroInversionRunConfig
from masw.runners.computing import WindowError
from sigpipe.masw.petro import PetroOutcome, invert_line_petro, save_line_sections

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, "WindowError | None"], None]


def run_petro_inversion(
    config: PetroInversionRunConfig,
    on_progress: ProgressCallback | None = None,
) -> list[WindowError]:
    """sigpipe's petrophysical inversion of the selected positions (sigpipe.masw.petro), with
    PAC's config and outcome files, then the line's sections."""
    total = len(config.positions)
    logger.info(
        "Starting petro inversion: %d positions, model=%s, %d workers",
        total,
        config.model_name,
        config.n_workers,
    )

    out_dir = output_folder(config.folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "petro_inversion_config.json").write_text(config.model_dump_json(indent=2))

    xmids = {xmid_folder(config.folder, xmid).name: xmid for xmid in config.positions}
    errors: dict[str, WindowError] = {}
    if on_progress is not None:
        on_progress(0, total, None)

    def on_window(done: int, total: int, outcome: PetroOutcome) -> None:
        error = None
        if outcome.model is None:
            error = WindowError(
                xmid=xmids[outcome.unit],
                error_type=outcome.error_type or "Error",
                message=outcome.message or "",
                traceback=outcome.traceback or "",
            )
            errors[outcome.unit] = error
            logger.error("Petro inversion failed for xmid=%.2f: %s", error.xmid, error.message)
        else:
            logger.info("Finished xmid=%.2f", xmids[outcome.unit])
        if on_progress is not None:
            on_progress(done, total, error)

    outcomes = invert_line_petro(
        out_dir, list(xmids), config.model_name, config.n_workers, on_window
    )

    results: list[dict[str, object]] = [
        {"xmid": xmids[outcome.unit], "status": "success", "duration_s": outcome.duration_s}
        if outcome.model is not None
        else {
            "xmid": xmids[outcome.unit],
            "status": "failed",
            "duration_s": None,
            **asdict(errors[outcome.unit]),
        }
        for outcome in sorted(outcomes, key=lambda one: xmids[one.unit])
    ]
    (out_dir / "petro_inversion_outcome.json").write_text(json.dumps(results, indent=2))
    logger.info("%d/%d succeeded, %d failed", total - len(errors), total, len(errors))

    # Over every window of the folder holding a model, those inverted before included.
    units = [xmid_folder(config.folder, xmid).name for xmid in get_xmid_folders(config.folder)]
    save_line_sections(out_dir, units)

    return list(errors.values())
