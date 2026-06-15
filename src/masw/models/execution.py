from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ExecutionParameters(BaseModel):
    output_folder: Path = Path("./results")
    n_workers: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_output_folder(self):
        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )
        return self
