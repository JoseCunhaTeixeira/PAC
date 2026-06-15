from pathlib import Path

from pydantic import BaseModel, computed_field, model_validator


class AcquisitionParameters(BaseModel):
    folder_path: Path
    files: list[str]
    positions: list[float]
    source_positions: list[float]

    @computed_field
    @property
    def file_paths(self) -> list[Path]:
        return [self.folder_path / file for file in self.files]

    @model_validator(mode="after")
    def validate_config(self):
        for file in self.files:
            if not (self.folder_path / file).exists():
                raise ValueError(f"File not found: {file}")

        if sorted(self.positions) != self.positions:
            raise ValueError("Sensor positions must be sorted")

        if len(self.positions) < 2:
            raise ValueError("At least two sensor positions are required")

        if len(self.files) != len(self.source_positions):
            raise ValueError("files and source_positions must have the same length")

        return self


class AcquisitionInfo(BaseModel):
    folder_path: Path
    files: list[str]
    durations: list[float]
    source_positions: list[float]
    sensor_positions: list[float]
    n_receivers: int
