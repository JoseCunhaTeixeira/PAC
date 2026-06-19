from pathlib import Path
from typing import Self

from pydantic import BaseModel, computed_field, model_validator

type PositionXZ = tuple[float, float]


class AcquisitionParameters(BaseModel):
    folder_path: Path
    files: list[str]
    durations: list[float]
    sampling_frequencies: list[float]
    source_positions: list[PositionXZ]
    receiver_positions: list[PositionXZ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_receivers(self) -> int:
        return len(self.receiver_positions)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        for file in self.files:
            if not (self.folder_path / file).exists():
                raise ValueError(f"File not found: {file}")

        receiver_x = [position[0] for position in self.receiver_positions]
        if sorted(receiver_x) != receiver_x:
            raise ValueError("Receiver positions must be sorted by x")

        if len(self.receiver_positions) < 2:
            raise ValueError("At least two receiver positions are required")

        # an empty list means the survey has no real source (e.g. passive
        # acquisitions); otherwise there must be one position per file
        if self.source_positions and len(self.files) != len(self.source_positions):
            raise ValueError("files and source_positions must have the same length")

        if len(self.files) != len(self.durations):
            raise ValueError("files and durations must have the same length")

        if not all(
            self.sampling_frequencies[0] == sampling_freq
            for sampling_freq in self.sampling_frequencies
        ):
            raise ValueError("all sampling frequencies must be the same")

        return self
