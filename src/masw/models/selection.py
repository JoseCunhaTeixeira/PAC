from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


class NoneSelection(BaseModel):
    method: Literal["none"] = "none"


class FKSelection(BaseModel):
    method: Literal["fk"] = "fk"
    threshold: float = Field(ge=0)
    vmin: float = Field(ge=0)
    vmax: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:

        if self.threshold > 1:
            raise ValueError("threshold cannot exceed 1")

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        return self


type SelectionParameters = Annotated[
    NoneSelection | FKSelection,
    Field(discriminator="method"),
]
