from pathlib import Path

from sigproc.transformers import (
    Detrend,
    Dispersion,
    Filter,
    Load,
    Plot,
    Save,
    Stack,
)

from masw.adapters.windows import MASWWindow
from masw.models.computing import PassiveComputingConfig


def build_passive_pipeline(
    config: PassiveComputingConfig,
    window: MASWWindow,
    output_folder: Path,
):

    load_kwargs = {
        "file_paths": window.selected_files,
        "data_type": "segd",
        "receivers_to_load": window.receiver_indices,
    }

    dispersion_kwargs = config.dispersion_params.model_dump()
    filter_kwargs = config.filtering_params.model_dump()

    return (
        Load(**load_kwargs)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Filter(**filter_kwargs)
        >> Dispersion(**dispersion_kwargs)
        >> Stack(method="linear")
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
    )


# pipeline = (
#     Load(
#         file_paths=file_paths,
#         data_type="gero_passive",
#         acquisition=acquisition,
#         sort=True,
#         receivers_to_load=receivers_to_load,
#     )
#     >> Detrend(method="constant")
#     >> Detrend(method="linear")
#     >> Filter(method="iir", fmin=10_000, fmax=20_000, order=4)
#     >> Slice(segment_duration=0.002, segment_step=0.002)
#     >> Selection
#     >> Whiten(method="onebit_apod", fmin=10_000, fmax=20_000, taper_width_Hz=1_000)
#     >> Normalize(method="onebit")
#     >> Apodize(method="hanning", frac=0.1)
#     >> Correlate(method="cross", virtual_source_index=0)
#     >> Stack(method="phase_weighted", nu=2)
#     >> Plot(folder_path=saving_dir, normalize=True)
# )
