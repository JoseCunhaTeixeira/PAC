"""Where PAC reads profiles and writes runs: the project's data/input and data/output, or the
folders MASW_INPUT_DIR and MASW_OUTPUT_DIR name."""

import os
from pathlib import Path

from sigpipe.masw.workspace import Folders

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = Path(os.environ.get("MASW_INPUT_DIR", PROJECT_ROOT / "data/input"))

OUTPUT_DIR = Path(os.environ.get("MASW_OUTPUT_DIR", PROJECT_ROOT / "data/output"))

# Recorded in a run's manifest with sigpipe's version.
PACKAGES = ("PAC",)


def workspace(workers: int = 1) -> Folders:
    """PAC's folders, as sigpipe's MASW layer reads them."""
    return Folders(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, workers=workers)


def output_folder(folder: str) -> Path:
    """Output folder `folder` (a run, <profile>/<run_id>, or a folder of the older layout);
    raises ValueError for a path leading out of the output directory."""
    path = OUTPUT_DIR / folder
    if not path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise ValueError(f"Output folder not found: {folder}")
    return path
