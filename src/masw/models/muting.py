from pydantic import BaseModel, Field, model_validator


class MutingParameters(BaseModel):
    vmin: float = Field(gt=0)
    vmax: float = Field(gt=0)
    taper: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        return self
