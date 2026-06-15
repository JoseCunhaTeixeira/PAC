from enum import Enum

from pydantic import BaseModel, Field, model_validator


class StackingMethod(str, Enum):
    LINEAR = "linear"
    PWS = "pws"
    ROOT = "root"


class StackingParameters(BaseModel):
    method: StackingMethod = StackingMethod.LINEAR
    power: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_pws(self):
        if (
            self.method in [StackingMethod.PWS, StackingMethod.ROOT]
            and self.power is None
        ):
            raise ValueError("power is required when method='pws' or 'root'")
        return self
