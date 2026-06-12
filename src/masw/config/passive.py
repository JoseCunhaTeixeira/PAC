from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class PassiveMASWConfig(BaseModel):
    # Input data
    folder_path: Path
    files: list[str]

    # Geometry
    positions: list[float]

    # Slicing
    segment_duration: float = Field(gt=0)
    segment_step: float = Field(ge=0)

    # Selection
    fk_threshold: float = Field(ge=0)

    # Stack
    pws_nu: float = Field(ge=0)

    # MASW window parameters
    masw_length: int = Field(ge=3)
    masw_step: int = Field(gt=0)

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

        if self.masw_length > len(self.positions):
            raise ValueError("masw_length cannot exceed the number of sensors")

        if self.masw_step > self.masw_length:
            raise ValueError("masw_step cannot exceed masw_length")

        if len(self.positions) < 2:
            raise ValueError("at least two sensor positions are required")

        return self
