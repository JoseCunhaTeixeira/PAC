from pydantic import BaseModel, Field, model_validator


class MASWParameters(BaseModel):
    length: int = Field(ge=3)
    step: int = Field(gt=0)
    distance_min: float = Field(ge=0)
    distance_max: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.distance_max <= self.distance_min:
            raise ValueError("distance_max must be greater than distance_min")

        return self
