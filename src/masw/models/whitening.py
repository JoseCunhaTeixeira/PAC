from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


class NoneWhitening(BaseModel):
    method: Literal["none"] = "none"


class OnebitWhitening(BaseModel):
    method: Literal["onebit"] = "onebit"


class OnebitApodWhitening(BaseModel):
    method: Literal["onebit_apod"] = "onebit_apod"
    fmin: float = Field(ge=0)
    fmax: float = Field(gt=0)
    taper_width_Hz: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        return self


type WhiteningParameters = Annotated[
    NoneWhitening | OnebitWhitening | OnebitApodWhitening,
    Field(discriminator="method"),
]
