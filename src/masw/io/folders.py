import logging

from masw.io.paths import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def get_input_folders() -> list[str]:
    if not INPUT_DIR.exists():
        logger.warning(f"INPUT_DIR {INPUT_DIR} does not exist")
        return []
    folders = sorted(folder.name for folder in INPUT_DIR.iterdir() if folder.is_dir())
    logger.info(f"Folders found: {folders}")
    return folders


def get_output_folders() -> list[str]:
    if not OUTPUT_DIR.exists():
        logger.warning(f"OUTPUT_DIR {OUTPUT_DIR} does not exist")
        return []
    folders = sorted(folder.name for folder in OUTPUT_DIR.iterdir() if folder.is_dir())
    logger.info(f"Folders found: {folders}")
    return folders


def get_xmid_folders(folder: str) -> list[float]:
    folder_path = OUTPUT_DIR / folder
    if not folder_path.exists():
        raise ValueError(f"Output folder not found: {folder}")
    return sorted(
        float(sub.name.removeprefix("xmid_"))
        for sub in folder_path.iterdir()
        if sub.is_dir() and sub.name.startswith("xmid_")
    )
