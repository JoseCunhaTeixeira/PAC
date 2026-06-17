from pydantic import BaseModel, Field, model_validator


class DispersionParameters(BaseModel):
    fmin: float = Field(ge=0)
    fmax: float = Field(gt=0)
    vmin: float = Field(gt=0)
    vmax: float = Field(gt=0)
    nv: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")

        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")

        return self
