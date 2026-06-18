from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from masw.models.acquisition import AcquisitionParameters
from masw.models.dispersion import DispersionParameters
from masw.models.execution import ExecutionParameters
from masw.models.filtering import FilteringParameters
from masw.models.masw import MASWParameters
from masw.models.modes import ProcessingMode
from masw.models.muting import MutingParameters
from masw.models.normalization import NormalizationParameters
from masw.models.selection import SelectionParameters
from masw.models.slicing import SlicingParameters
from masw.models.stacking import StackingParameters
from masw.models.whitening import WhiteningParameters


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
    filtering_params: FilteringParameters
    dispersion_params: DispersionParameters


class PassiveComputingConfig(ComputingConfig):
    mode: Literal[ProcessingMode.PASSIVE] = ProcessingMode.PASSIVE
    muting_params: MutingParameters
    filtering_params: FilteringParameters
    slicing_params: SlicingParameters
    selection_params: SelectionParameters
    whitening_params: WhiteningParameters
    normalization_params: NormalizationParameters
    stacking_params: StackingParameters
    dispersion_params: DispersionParameters


class PassiveActiveComputingConfig(ComputingConfig):
    mode: Literal[ProcessingMode.PASSIVE_ACTIVE] = ProcessingMode.PASSIVE_ACTIVE
    muting_params: MutingParameters
    filtering_params: FilteringParameters
    stacking_params: StackingParameters
    dispersion_params: DispersionParameters


AnyComputingConfig = Annotated[
    ActiveComputingConfig | PassiveComputingConfig | PassiveActiveComputingConfig,
    Field(discriminator="mode"),
]
