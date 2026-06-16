from enum import Enum

from pydantic import BaseModel, Field, model_validator


class WhiteningMethod(str, Enum):
    NONE = "none"
    ONEBIT_APOD = "onebit_apod"


class WhiteningParameters(BaseModel):
    method: WhiteningMethod = WhiteningMethod.ONEBIT_APOD
    fmin: float = Field(ge=0)
    fmax: float = Field(gt=0)
    taper_width_Hz: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        return self
