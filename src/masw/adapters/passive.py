from pathlib import Path
from typing import Any

from sigproc.base import Pipeline
from sigproc.transformers import (
    Apodize,
    BidirectionalCorrelate,
    Detrend,
    Dispersion,
    Filter,
    Load,
    Mute,
    Normalize,
    Pad,
    Plot,
    Save,
    Selection,
    Slice,
    Stack,
    Whiten,
)

from masw.adapters.windows import MASWWindow
from masw.models.computing import PassiveComputingConfig


def build_passive_pipeline(
    config: PassiveComputingConfig,
    window: MASWWindow,
    output_folder: Path,
) -> Pipeline:

    load_kwargs: dict[str, Any] = {
        "file_paths": window.selected_files,
        "acquisitions": window.acquisitions,
        "data_type": "segd",
        "receivers_to_load": window.receiver_indices,
    }

    mute_kwargs = config.muting_params.model_dump(exclude_none=True)
    filter_kwargs = config.filtering_params.model_dump(exclude_none=True)
    slicing_kwargs = config.slicing_params.model_dump(exclude_none=True)
    selection_kwargs = config.selection_params.model_dump(exclude_none=True)
    whitening_kwargs = config.whitening_params.model_dump(exclude_none=True)
    normalization_kwargs = config.normalization_params.model_dump(exclude_none=True)
    stacking_kwargs = config.stacking_params.model_dump(exclude_none=True)
    dispersion_kwargs = config.dispersion_params.model_dump(exclude_none=True)

    return (
        Load(**load_kwargs)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**mute_kwargs)
        >> Filter(**filter_kwargs)
        >> Slice(**slicing_kwargs)
        >> Selection(**selection_kwargs, flip_negatives=True)
        >> Whiten(**whitening_kwargs)
        >> Normalize(**normalization_kwargs)
        >> Apodize(method="hanning", frac=0.1)
        >> BidirectionalCorrelate(method="cross")
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
