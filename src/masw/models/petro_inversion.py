from pydantic import BaseModel, Field


class PetroInversionRunConfig(BaseModel):
    folder: str
    positions: list[float] = Field(min_length=1)
    model_name: str
    n_workers: int = Field(gt=0)
