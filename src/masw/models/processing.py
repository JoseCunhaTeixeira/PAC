from typing import Any

from pydantic import BaseModel, Field

from sigpipe.masw.profiles import ProcessingMode


class ProcessingRequest(BaseModel):
    """A profile processed in one of its modes, with the settings changed from the preset's
    (sigpipe.masw.presets: the one schema of PAC's forms and PACo's agent)."""

    profile: str
    mode: ProcessingMode
    overrides: dict[str, Any] = {}
    workers: int = Field(default=1, gt=0)
