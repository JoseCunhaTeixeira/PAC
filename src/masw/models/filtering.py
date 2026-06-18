from typing import Annotated, Literal, TypeAlias, Union

from pydantic import BaseModel, Field, model_validator


class NoneFiltering(BaseModel):
    method: Literal["none"] = "none"


class IIRFiltering(BaseModel):
    method: Literal["iir"] = "iir"
    fmin: float = Field(ge=0)
    fmax: float = Field(gt=0)
    order: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        return self


FilteringParameters: TypeAlias = Annotated[
    Union[NoneFiltering, IIRFiltering],
    Field(discriminator="method"),
]
