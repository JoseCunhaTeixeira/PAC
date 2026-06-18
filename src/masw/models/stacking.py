from typing import Annotated, Literal, TypeAlias, Union

from pydantic import BaseModel, Field, model_validator


class LinearStacking(BaseModel):
    method: Literal["linear"] = "linear"


class PhaseWeightedStacking(BaseModel):
    method: Literal["phase_weighted"] = "phase_weighted"
    nu: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_root(self):
        if self.nu is None:
            raise ValueError("nu is required when method='phase_weighted'")
        return self


class RootStacking(BaseModel):
    method: Literal["root"] = "root"
    n: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_root(self):
        if self.n is None:
            raise ValueError("n is required when method='root'")
        return self


StackingParameters: TypeAlias = Annotated[
    Union[LinearStacking, PhaseWeightedStacking, RootStacking],
    Field(discriminator="method"),
]
