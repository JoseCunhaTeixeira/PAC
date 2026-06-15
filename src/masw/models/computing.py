from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from masw.models.acquisition import AcquisitionParameters
from masw.models.dispersion import DispersionParameters
from masw.models.execution import ExecutionParameters
from masw.models.masw import MASWParameters
from masw.models.modes import ProcessingMode
from masw.models.muting import MutingParameters
from masw.models.stacking import StackingParameters


class ComputingConfig(BaseModel):
    acquisition_params: AcquisitionParameters
    masw_params: MASWParameters
    execution_params: ExecutionParameters

    @model_validator(mode="after")
    def validate_config(self):
        n_receivers = len(self.acquisition_params.receiver_positions)
        if self.masw_params.length > n_receivers:
            raise ValueError(
                f"length ({self.masw_params.length}) exceeds "
                f"number of receivers ({n_receivers})"
            )
        return self


class ActiveComputingConfig(ComputingConfig):
    mode: Literal[ProcessingMode.ACTIVE] = ProcessingMode.ACTIVE
    muting_params: MutingParameters
    dispersion_params: DispersionParameters
    stacking_params: StackingParameters


class PassiveComputingConfig(ComputingConfig):
    mode: Literal[ProcessingMode.PASSIVE] = ProcessingMode.PASSIVE


class ActivePassiveComputingConfig(ComputingConfig):
    mode: Literal[ProcessingMode.ACTIVE_PASSIVE] = ProcessingMode.ACTIVE_PASSIVE


AnyComputingConfig = Annotated[
    ActiveComputingConfig | PassiveComputingConfig | ActivePassiveComputingConfig,
    Field(discriminator="mode"),
]
