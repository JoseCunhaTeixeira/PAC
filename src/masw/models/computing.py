from pydantic import BaseModel, model_validator

from masw.models.acquisition import AcquisitionParameters
from masw.models.execution import ExecutionParameters
from masw.models.muting import MutingParameters
from masw.models.stacking import StackingParameters

from .dispersion import DispersionParameters
from .masw import MASWParameters


class ComputingConfig(BaseModel):
    acquisition_params: AcquisitionParameters
    masw_params: MASWParameters
    execution_params: ExecutionParameters

    @model_validator(mode="after")
    def validate_config(self):

        n_receivers = len(self.acquisition_params.positions)
        if self.masw_params.length > n_receivers:
            raise ValueError(
                f"length ({self.masw_params.length}) exceeds "
                f"number of receivers ({n_receivers})"
            )

        return self


class ActiveComputingConfig(ComputingConfig):
    muting_params: MutingParameters
    dispersion_params: DispersionParameters
    stacking_params: StackingParameters


class PassiveComputingConfig(ComputingConfig): ...


class ActivePassiveComputingConfig(ComputingConfig): ...
