from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ActiveMASWConfig(BaseModel):
    # Input data
    folder_path: Path
    files: list[str]

    # Geometry
    positions: list[float]
    source_positions: list[float]

    # MASW window parameters
    masw_length: int = Field(gt=1)
    masw_step: int = Field(gt=0)

    distance_min: float = Field(ge=0)
    distance_max: float = Field(gt=0)

    # Dispersion parameters
    fmin: float = Field(gt=0)
    fmax: float = Field(gt=0)

    vmin: float = Field(gt=0)
    vmax: float = Field(gt=0)

    dv: float = Field(gt=0)

    # Parallelization
    n_workers: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        if self.dv >= self.vmax - self.vmin:
            raise ValueError("dv must be smaller than vmax - vmin")

        if self.distance_max <= self.distance_min:
            raise ValueError("distance_max must be greater than distance_min")

        if self.masw_length > len(self.positions):
            raise ValueError("masw_length cannot exceed the number of sensors")

        if self.masw_step > self.masw_length:
            raise ValueError("masw_step cannot exceed masw_length")

        if len(self.files) != len(self.source_positions):
            raise ValueError("files and source_positions must have the same length")

        if len(self.positions) < 2:
            raise ValueError("at least two sensor positions are required")

        return self
