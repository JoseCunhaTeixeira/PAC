import logging

from masw.io.paths import INPUT_DIR

logger = logging.getLogger(__name__)


def get_input_folders() -> list[str]:
    if not INPUT_DIR.exists():
        logger.warning(f"INPUT_DIR {INPUT_DIR} does not exist")
        return []
    folders = sorted(folder.name for folder in INPUT_DIR.iterdir() if folder.is_dir())
    logger.info(f"Folders found: {folders}")
    return folders
