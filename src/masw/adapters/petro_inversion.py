from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.transformers import Invert, Save


def build_petro_inversion_pipeline(
    model_dir: Path,
    output_folder: Path,
) -> Pipeline:
    return Invert(method="silex", model_dir=model_dir) >> Save(
        folder_path=output_folder, file_name="PetroInversion_Model"
    )
