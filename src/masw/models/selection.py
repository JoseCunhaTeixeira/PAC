from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SelectionMethod(str, Enum):
    NONE = "none"
    FK = "fk"


class SelectionParameters(BaseModel):
    method: SelectionMethod = SelectionMethod.NONE
    threshold: float = Field(gt=0)
    vmin: float = Field(gt=0)
    vmax: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.threshold > 1:
            raise ValueError("threshold cannot exceed 1")

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        return self
