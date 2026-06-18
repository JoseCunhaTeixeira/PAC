from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NoneNormalization(BaseModel):
    method: Literal["none"] = "none"


class OneBitNormalization(BaseModel):
    method: Literal["onebit"] = "onebit"


type NormalizationParameters = Annotated[
    NoneNormalization | OneBitNormalization,
    Field(discriminator="method"),
]
