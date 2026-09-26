"""The profiles in the input folder, and the results in the output folder: runs
(<profile>/<run_id>, newest first), then the folders of the older layout (a profile's windows
straight in <profile>/)."""

import logging

from masw.io.paths import INPUT_DIR, OUTPUT_DIR, output_folder, workspace
from sigpipe.masw.profiles import list_profiles
from sigpipe.masw.runs import list_runs, window_folders, xmid_of

logger = logging.getLogger(__name__)


def get_input_folders() -> list[str]:
    if not INPUT_DIR.exists():
        logger.warning(f"INPUT_DIR {INPUT_DIR} does not exist")
        return []
    return list_profiles(workspace())


def get_output_folders() -> list[str]:
    if not OUTPUT_DIR.exists():
        logger.warning(f"OUTPUT_DIR {OUTPUT_DIR} does not exist")
        return []
    older = sorted(
        folder.name for folder in OUTPUT_DIR.iterdir() if folder.is_dir() and window_folders(folder)
    )
    return [*list_runs(workspace()), *older]


def get_xmid_folders(folder: str) -> list[float]:
    """The window positions of output folder `folder`: a run, or a folder of the older layout."""
    folder_path = output_folder(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Output folder not found: {folder}")
    return [xmid_of(name) for name in window_folders(folder_path)]
