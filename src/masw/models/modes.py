from enum import StrEnum


class ProcessingMode(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"
    PASSIVE_ACTIVE = "passive-active"
