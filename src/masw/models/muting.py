from pydantic import BaseModel, Field, model_validator


class MutingParameters(BaseModel):
    vmin: float = Field(ge=0)
    vmax: float = Field(ge=0)
    taper: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        return self
