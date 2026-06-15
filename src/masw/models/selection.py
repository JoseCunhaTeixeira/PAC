from pydantic import BaseModel, Field, model_validator


class SelectionParameters(BaseModel):
    fk_threshold: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self):

        if self.fk_threshold > 1:
            raise ValueError("fk_threshold cannot exceed 1")

        return self
