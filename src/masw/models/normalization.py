from enum import Enum

from pydantic import BaseModel


class NormalizationMethod(str, Enum):
    NONE = "none"
    ONEBIT = "onebit"


class NormalizationParameters(BaseModel):
    method: NormalizationMethod = NormalizationMethod.NONE
