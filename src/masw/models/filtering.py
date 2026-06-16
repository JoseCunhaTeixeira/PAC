from enum import Enum

from pydantic import BaseModel, Field, model_validator


class FilteringMethod(str, Enum):
    NONE = "none"
    IIR = "iir"


class FilteringParameters(BaseModel):
    method: FilteringMethod = FilteringMethod.NONE
    fmin: float = Field(ge=0)
    fmax: float = Field(gt=0)
    order: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        return self
