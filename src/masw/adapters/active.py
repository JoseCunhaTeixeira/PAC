from pathlib import Path

from sigproc.transformers import (
    Detrend,
    Dispersion,
    Load,
    Mute,
    Plot,
    Save,
    Stack,
)

from masw.adapters.windows import MASWWindow
from masw.models.computing import ActiveComputingConfig


def build_active_pipeline(
    config: ActiveComputingConfig,
    window: MASWWindow,
    output_folder: Path,
):

    load_kwargs = {
        "file_paths": window.selected_files,
        "data_type": "segd",
        "receivers_to_load": window.receiver_indices,
    }

    dispersion_kwargs = config.dispersion_params.model_dump()

    mute_kwargs = config.muting_params.model_dump()

    return (
        Load(**load_kwargs)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**mute_kwargs)
        >> Dispersion(**dispersion_kwargs)
        >> Stack(method="linear")
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
    )
