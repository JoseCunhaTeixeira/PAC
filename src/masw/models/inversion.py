from pydantic import BaseModel, Field

from sigpipe.masw.inversion import InversionParameters


class InversionRunConfig(BaseModel):
    folder: str
    positions: list[float] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    parameters: InversionParameters  # sigpipe's: the one inversion schema of PAC and PACo
    n_workers: int = Field(gt=0)
