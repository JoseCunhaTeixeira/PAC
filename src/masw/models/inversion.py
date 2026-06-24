from typing import Self

from pydantic import BaseModel, Field, model_validator


class VsLayerParameters(BaseModel):
    vs_min: float = Field(gt=0)
    vs_max: float = Field(gt=0)
    vs_perturb_std: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.vs_max <= self.vs_min:
            raise ValueError("vs_max must be greater than vs_min")
        return self


class ThicknessLayerParameters(BaseModel):
    thickness_min: float = Field(gt=0)
    thickness_max: float = Field(gt=0)
    thickness_perturb_std: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.thickness_max <= self.thickness_min:
            raise ValueError("thickness_max must be greater than thickness_min")
        return self


class InversionParameters(BaseModel):
    n_layers: int = Field(ge=2)
    vs_layers: list[VsLayerParameters]
    thickness_layers: list[ThicknessLayerParameters]
    n_iterations: int = Field(gt=0)
    n_burnin_iterations: int = Field(gt=0)
    n_chains: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if len(self.vs_layers) != self.n_layers:
            raise ValueError(f"vs_layers must have length n_layers ({self.n_layers})")
        if len(self.thickness_layers) != self.n_layers - 1:
            raise ValueError(
                f"thickness_layers must have length n_layers - 1 ({self.n_layers - 1})"
            )
        return self


class InversionRunConfig(BaseModel):
    folder: str
    positions: list[float] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    parameters: InversionParameters
    n_workers: int = Field(gt=0)
