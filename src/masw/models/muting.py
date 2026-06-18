from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


class NoneFiltering(BaseModel):
    method: Literal["none"] = "none"


class Muting(BaseModel):
    method: Literal["mute"] = "mute"
    tmin: float = Field(ge=0)
    tmax: float = Field(ge=0)
    vmin: float = Field(ge=0)
    vmax: float = Field(ge=0)
    taper: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin")
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        return self


type MutingParameters = Annotated[
    NoneFiltering | Muting,
    Field(discriminator="method"),
]
