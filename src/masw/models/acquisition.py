from pathlib import Path

from pydantic import BaseModel, computed_field, model_validator


class AcquisitionParameters(BaseModel):
    folder_path: Path
    files: list[str]
    durations: list[float]
    sampling_frequencies: list[float]
    source_positions: list[float]
    receiver_positions: list[float]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_receivers(self) -> int:
        return len(self.receiver_positions)

    @model_validator(mode="after")
    def validate_config(self):
        for file in self.files:
            if not (self.folder_path / file).exists():
                raise ValueError(f"File not found: {file}")

        if sorted(self.receiver_positions) != self.receiver_positions:
            raise ValueError("Sensor positions must be sorted")

        if len(self.receiver_positions) < 2:
            raise ValueError("At least two receiver positions are required")

        if len(self.files) != len(self.source_positions):
            raise ValueError("files and source_positions must have the same length")

        if len(self.files) != len(self.durations):
            raise ValueError("files and durations must have the same length")

        if not all(
            self.sampling_frequencies[0] == sampling_freq
            for sampling_freq in self.sampling_frequencies
        ):
            raise ValueError("all sampling frequencies must be the same")

        return self
