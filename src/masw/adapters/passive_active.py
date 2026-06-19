from pathlib import Path

from sigproc.base import Pipeline
from sigproc.transformers import (
    ActiveShotCorrelation,
    Apodize,
    Detrend,
    Dispersion,
    Filter,
    Load,
    Mute,
    Pad,
    Plot,
    Save,
    Stack,
)

from masw.adapters.windows import MASWWindow
from masw.models.computing import PassiveActiveComputingConfig


def build_passive_active_pipeline(
    config: PassiveActiveComputingConfig,
    window: MASWWindow,
    output_folder: Path,
) -> Pipeline:

    load_kwargs = {
        "file_paths": window.selected_files,
        "acquisitions": window.acquisitions,
        "data_type": "segd",
        "receivers_to_load": window.receiver_indices,
    }

    mute_kwargs = config.muting_params.model_dump(exclude_none=True)
    filter_kwargs = config.filtering_params.model_dump(exclude_none=True)
    stacking_kwargs = config.stacking_params.model_dump(exclude_none=True)
    dispersion_kwargs = config.dispersion_params.model_dump(exclude_none=True)

    return (
        Load(**load_kwargs)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**mute_kwargs)
        >> Filter(**filter_kwargs)
        >> Apodize(method="hanning", frac=0.1)
        >> ActiveShotCorrelation(method="cross")
        >> Stack(**stacking_kwargs)
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Pad(n=1_000, taper=25)
        >> Dispersion(method="phase", **dispersion_kwargs)
        >> Plot(
            folder_path=output_folder,
            normalize=True,
        )
        >> Save(folder_path=output_folder)
    )
