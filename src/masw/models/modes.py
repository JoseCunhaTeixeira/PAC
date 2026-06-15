from enum import Enum


class ProcessingMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    ACTIVE_PASSIVE = "active_passive"
