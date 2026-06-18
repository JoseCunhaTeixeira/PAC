from typing import Annotated, Literal, TypeAlias, Union

from pydantic import BaseModel, Field


class NoneNormalization(BaseModel):
    method: Literal["none"] = "none"


class OneBitNormalization(BaseModel):
    method: Literal["onebit"] = "onebit"


NormalizationParameters: TypeAlias = Annotated[
    Union[NoneNormalization, OneBitNormalization],
    Field(discriminator="method"),
]
