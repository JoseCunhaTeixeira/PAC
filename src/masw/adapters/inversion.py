from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.transformers import Invert, Save

from masw.models.inversion import InversionParameters

DZ = 0.01
VP_VS_RATIO = 1.77


def build_inversion_pipeline(
    parameters: InversionParameters,
    output_folder: Path,
) -> Pipeline:

    invert_kwargs = {
        "n_layers": parameters.n_layers,
        "thicknesses_min": tuple(layer.thickness_min for layer in parameters.thickness_layers),
        "thicknesses_max": tuple(layer.thickness_max for layer in parameters.thickness_layers),
        "thickness_perturbations": tuple(
            layer.thickness_perturb_std for layer in parameters.thickness_layers
        ),
        "Vs_mins": tuple(layer.vs_min for layer in parameters.vs_layers),
        "Vs_maxs": tuple(layer.vs_max for layer in parameters.vs_layers),
        "Vs_perturbations": tuple(layer.vs_perturb_std for layer in parameters.vs_layers),
        "n_iterations": parameters.n_iterations,
        "n_burnin": parameters.n_burnin_iterations,
        "n_chains": parameters.n_chains,
        "Vp_Vs_ratio": VP_VS_RATIO,
        "dz": DZ,
    }

    return Invert(method="mcmc", **invert_kwargs) >> Save(folder_path=output_folder)
