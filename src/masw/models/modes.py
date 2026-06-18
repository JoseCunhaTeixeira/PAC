from enum import Enum


class ProcessingMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    PASSIVE_ACTIVE = "passive-active"
