from pathlib import Path

from pydantic import BaseModel, Field

from masw.io.paths import OUTPUT_DIR


class ExecutionParameters(BaseModel):
    output_folder: Path = OUTPUT_DIR
    n_workers: int = Field(gt=0)
