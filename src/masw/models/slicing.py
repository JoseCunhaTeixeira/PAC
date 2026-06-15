from pydantic import BaseModel, Field


class SlicingParameters(BaseModel):
    segment_duration: float = Field(gt=0)
    segment_step: float = Field(gt=0)
