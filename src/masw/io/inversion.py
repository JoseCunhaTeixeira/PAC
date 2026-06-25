import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
from disba import DispersionError
from sigpipe.algorithms.inversion.dispersion_curve.rayleigh.forward import (
    fwd_rayleigh_all_modes,
    fwd_rayleigh_phase,
)
from sigpipe.algorithms.picking.dispersion.curve import min_resolvable_wavelength
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurves, DispersionCurvesSection
from sigpipe.base.inversion import InversionResult
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.dispersion.section import (
    plot_pseudo_section_comparison,
    pseudo_section_comparison_grids,
)
from sigpipe.dataio.inversion.forward import MODEL_NAMES, forward_model_all
from sigpipe.dataio.inversion.plotting import plot_density_curves, plot_posterior_marginals
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.dataio.velocity_model.section import (
    plot_velocity_and_std_section,
    save_velocity_models_sections,
    smooth_laterally,
)
from sigpipe.transformers import Plot

from masw.adapters.inversion import DZ, VP_VS_RATIO, build_inversion_pipeline
from masw.algorithms.dispersion_picking import label_to_mode
from masw.io.dispersion_images import load_dispersion_image, xmid_folder
from masw.io.folders import get_xmid_folders
from masw.io.paths import OUTPUT_DIR
from masw.models.inversion import InversionParameters

logger = logging.getLogger(__name__)

# Mirrors sigpipe's MODEL_NAMES tuple -- spelled out as a Literal so FastAPI can
# validate/document it as an enum at the API boundary.
ModelName = Literal["best", "smooth_best", "median", "smooth_median", "ensemble"]


def _inversion_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "InversionResult_0000.csv"


def _result_paths(folder: str, xmid: float) -> tuple[Path, Path, Path, Path, Path]:
    base = _inversion_path(folder, xmid)
    return (
        base.with_name(f"{base.stem}_best{base.suffix}"),
        base.with_name(f"{base.stem}_smooth_best{base.suffix}"),
        base.with_name(f"{base.stem}_median{base.suffix}"),
        base.with_name(f"{base.stem}_smooth_median{base.suffix}"),
        base.with_name(f"{base.stem}_ensemble{base.suffix}"),
    )


def _modeled_curves_path(folder: str, xmid: float, model_name: str) -> Path:
    return xmid_folder(folder, xmid) / f"InversionDispersionCurves_0000_{model_name}.csv"


def curves_for_labels(folder: str, xmid: float, labels: Sequence[str]) -> DispersionCurves:
    image = load_dispersion_image(folder, xmid)
    if not image.dispersion_curves:
        raise ValueError(f"No picked curves for folder={folder}, xmid={xmid}")
    modes = {label_to_mode(label) for label in labels}
    matched = tuple(c for c in image.dispersion_curves if c.mode in modes)
    if not matched:
        raise ValueError(
            f"No curve matching labels {list(labels)} for folder={folder}, xmid={xmid}"
        )
    return DispersionCurves(dispersion_curves=matched)


def invert_position(
    folder: str,
    xmid: float,
    labels: Sequence[str],
    parameters: InversionParameters,
) -> InversionResult:
    image = load_dispersion_image(folder, xmid)
    curves = curves_for_labels(folder, xmid, labels)
    output_folder = xmid_folder(folder, xmid)

    pipeline = build_inversion_pipeline(parameters, output_folder=output_folder)
    result: InversionResult = pipeline.run(data=[curves], show_log=False)[0]

    (output_folder / "InversionLog_0000.log").write_text(result.log)

    # Single mode at each picked curve's own frequencies, forward-modeled from the
    # (blocky) median model -- matches the old Streamlit app's `pred_modes`.
    modeled_curves = DispersionCurves(
        dispersion_curves=tuple(
            fwd_rayleigh_phase(
                thickness_per_layer=list(result.median.thicknesses),
                Vs_per_layer=list(result.median.vs_s),
                mode=curve.mode.number,
                fs=curve.fs,
                Vp_Vs_ratio=VP_VS_RATIO,
            )
            for curve in curves
        )
    )
    # Every superior mode the median model supports, across the image's full
    # frequency axis -- matches the old Streamlit app's `full_pred_modes`.
    full_modeled_curves = fwd_rayleigh_all_modes(
        thickness_per_layer=list(result.median.thicknesses),
        Vs_per_layer=list(result.median.vs_s),
        fs=image.fs,
        Vp_Vs_ratio=VP_VS_RATIO,
    )
    dispersion_fig = plot_dispersion_image(
        image,
        picked_curves=curves,
        modeled_curves=modeled_curves,
        full_modeled_curves=full_modeled_curves,
        lbmin=min_resolvable_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=output_folder / "InversionDispersion_0000.png", figure=dispersion_fig)
    plt.close(dispersion_fig)

    forward_modeled = forward_model_all(result, curves, VP_VS_RATIO)
    for model_name in MODEL_NAMES:
        modeled = forward_modeled[model_name]
        if modeled is None:
            logger.warning(
                "Could not forward-model '%s' for folder=%s, xmid=%.2f (disba root finder"
                " failed for every mode); skipping its dispersion-curve CSV.",
                model_name,
                folder,
                xmid,
            )
            continue
        save_dispersion_curves(modeled, path=_modeled_curves_path(folder, xmid, model_name))

    density_fig = plot_density_curves(result, curves, VP_VS_RATIO)
    Plot.savefig(path=output_folder / "InversionDensityCurves_0000.png", figure=density_fig)
    plt.close(density_fig)

    try:
        samples = {f"Vs{i + 1} [m/s]": result.samples[f"vs{i + 1}"] for i in range(result.n_layers)}
        samples.update(
            {f"H{i + 1} [m]": result.samples[f"thick{i + 1}"] for i in range(result.n_layers - 1)}
        )
        marginals_fig = plot_posterior_marginals(samples)
        Plot.savefig(path=output_folder / "InversionMarginals_0000.png", figure=marginals_fig)
        plt.close(marginals_fig)
    except Exception:
        logger.exception(
            "Failed to plot posterior marginals for folder=%s, xmid=%.2f", folder, xmid
        )

    return result


@dataclass(slots=True, frozen=True)
class InversionModels:
    best: VelocityModel
    smooth_best: VelocityModel
    median: VelocityModel
    smooth_median: VelocityModel
    ensemble: VelocityModel


def load_inversion_result(folder: str, xmid: float) -> InversionModels | None:
    paths = _result_paths(folder, xmid)
    if not all(path.exists() for path in paths):
        return None
    best_path, smooth_best_path, median_path, smooth_median_path, ensemble_path = paths
    best = load_velocity_models([best_path])[0][0]
    smooth_best = load_velocity_models([smooth_best_path])[0][0]
    median = load_velocity_models([median_path])[0][0]
    smooth_median = load_velocity_models([smooth_median_path])[0][0]
    ensemble = load_velocity_models([ensemble_path])[0][0]
    return InversionModels(
        best=best,
        smooth_best=smooth_best,
        median=median,
        smooth_median=smooth_median,
        ensemble=ensemble,
    )


def list_inversion_status(folder: str) -> list[tuple[float, bool]]:
    xmids = get_xmid_folders(folder)
    return [(xmid, load_inversion_result(folder, xmid) is not None) for xmid in xmids]


@dataclass(slots=True, frozen=True)
class VelocitySection:
    positions: np.ndarray
    elevations: np.ndarray
    vs_grid: np.ndarray
    vs_std_grid: np.ndarray


def _model_section(folder: str, model: ModelName = "smooth_median") -> VelocityModelsSection:
    xmids = get_xmid_folders(folder)
    models_by_position = [
        getattr(models, model)
        for xmid in xmids
        if (models := load_inversion_result(folder, xmid)) is not None
    ]
    if len(models_by_position) < 2:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return VelocityModelsSection(velocity_models=tuple(models_by_position))


def _section_suffix(model: ModelName, lateral_smoothing: bool) -> str:
    """Filename suffix disambiguating non-default model/smoothing choices.

    Empty for the default (smooth_median, no lateral smoothing) so existing
    saved filenames -- and the runner's end-of-run auto-save -- don't change.
    """
    parts: list[str] = []
    if model != "smooth_median":
        parts.append(model)
    if lateral_smoothing:
        parts.append("lateralsmooth")
    return ("_" + "_".join(parts)) if parts else ""


def get_velocity_section(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> VelocitySection:
    section = _model_section(folder, model)
    xs, zs, vs_s_grid, _vs_p_grid, _rhos_grid, vs_s_std_grid = section.to_grid(dz=DZ)

    if lateral_smoothing:
        vs_s_grid = smooth_laterally(vs_s_grid)
        vs_s_std_grid = smooth_laterally(vs_s_std_grid)

    return VelocitySection(
        positions=xs,
        elevations=zs,
        vs_grid=vs_s_grid,
        vs_std_grid=vs_s_std_grid,
    )


def save_velocity_section_plot(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> Path:
    """Save the Vs(x,z) + std(x,z) section plot in the profile's output folder."""
    section = _model_section(folder, model)
    fig = plot_velocity_and_std_section(section, dz=DZ, lateral_smoothing=lateral_smoothing)
    suffix = _section_suffix(model, lateral_smoothing)
    path = OUTPUT_DIR / folder / f"VelocitySection_0000{suffix}.png"
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path


def save_velocity_xzv(folder: str) -> Path:
    """Save Vs(x, z) section grids for every model variant (best, smooth_best,
    median, smooth_median, ensemble) into one HDF5 file in the profile's
    output folder, one group per variant.

    Best-effort per variant: a variant with too few inverted positions is
    skipped (logged) rather than blocking the others.
    """
    sections: dict[str, VelocityModelsSection] = {}
    for model in get_args(ModelName):
        try:
            sections[model] = _model_section(folder, model)
        except ValueError:
            logger.warning(
                "Could not build a '%s' velocity section for folder=%s;"
                " skipping it in the XZV export.",
                model,
                folder,
            )
    if not sections:
        raise ValueError(f"No model variant has at least two inverted positions in folder={folder}")
    path = OUTPUT_DIR / folder / "VelocitySection_0000.hdf5"
    save_velocity_models_sections(sections, path, dz=DZ)
    return path


def _observed_predicted_sections(
    folder: str, label: str, model: ModelName
) -> tuple[DispersionCurvesSection, DispersionCurvesSection]:
    """Observed picks and their model-forward-modeled prediction, one curve
    per position, for every position with both a pick and an inversion
    result for `label`."""
    xmids = get_xmid_folders(folder)
    mode = label_to_mode(label)

    observed_curves: list[DispersionCurve] = []
    predicted_curves: list[DispersionCurve] = []
    for xmid in xmids:
        try:
            observed = next(iter(curves_for_labels(folder, xmid, [label])))
        except ValueError:
            continue

        models = load_inversion_result(folder, xmid)
        if models is None:
            continue

        predicted_model = getattr(models, model)
        try:
            predicted = fwd_rayleigh_phase(
                thickness_per_layer=list(predicted_model.thicknesses),
                Vs_per_layer=list(predicted_model.vs_s),
                mode=mode.number,
                fs=observed.fs,
                Vp_Vs_ratio=VP_VS_RATIO,
            )
        except DispersionError:
            logger.warning(
                "Could not forward-model %s for folder=%s, xmid=%.2f, label=%s;"
                " omitting it from the pseudo-section comparison.",
                model,
                folder,
                xmid,
                label,
            )
            continue
        # fwd_rayleigh_phase doesn't know the real position; carry over the
        # observed curve's acquisition so the predicted curve sorts/groups by
        # the same xmid in the section below.
        predicted = DispersionCurve(
            fs=predicted.fs,
            vs=predicted.vs,
            mode=predicted.mode,
            acquisition=observed.acquisition,
            type=predicted.type,
        )
        observed_curves.append(observed)
        predicted_curves.append(predicted)

    if len(observed_curves) < 2:
        raise ValueError(
            f"At least two positions with both a pick and an inversion result are required "
            f"to build a pseudo-section comparison for label={label} in folder={folder}"
        )

    return (
        DispersionCurvesSection(dispersion_curves=tuple(observed_curves)),
        DispersionCurvesSection(dispersion_curves=tuple(predicted_curves)),
    )


def save_pseudo_section_comparison_plot(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> Path:
    """Save the observed-vs-predicted pseudo-section comparison for one label
    in the profile's output folder."""
    observed, predicted = _observed_predicted_sections(folder, label, model)
    fig = plot_pseudo_section_comparison(observed, predicted)
    suffix = _section_suffix(model, lateral_smoothing=False)
    path = OUTPUT_DIR / folder / f"PseudoSectionComparison_0000{suffix}_{label}.png"
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path


@dataclass(slots=True, frozen=True)
class PseudoSectionComparison:
    positions: np.ndarray
    fs: np.ndarray
    observed_grid: np.ndarray
    predicted_grid: np.ndarray
    residual_grid: np.ndarray


def get_pseudo_section_comparison(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> PseudoSectionComparison:
    """Observed-vs-predicted pseudo-section comparison grids for one label,
    for live display (the data behind `save_pseudo_section_comparison_plot`)."""
    observed, predicted = _observed_predicted_sections(folder, label, model)
    positions, fs, obs_grid, pred_grid, residual = pseudo_section_comparison_grids(
        observed, predicted
    )
    return PseudoSectionComparison(
        positions=positions,
        fs=fs,
        observed_grid=obs_grid,
        predicted_grid=pred_grid,
        residual_grid=residual,
    )


@dataclass(slots=True, frozen=True)
class PositionCurves:
    xmid: float
    observed_fs: np.ndarray | None
    observed_vs: np.ndarray | None
    observed_vs_err: np.ndarray | None
    predicted_fs: np.ndarray | None
    predicted_vs: np.ndarray | None
    velocity_type: str


def get_curves_by_position(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> list[PositionCurves]:
    xmids = get_xmid_folders(folder)
    if not xmids:
        raise ValueError(f"No xmid positions found in folder={folder}")

    mode = label_to_mode(label)
    result: list[PositionCurves] = []
    for xmid in xmids:
        try:
            observed = next(iter(curves_for_labels(folder, xmid, [label])))
        except ValueError:
            observed = None

        models = load_inversion_result(folder, xmid)
        predicted = None
        if models is not None and observed is not None:
            predicted_model = getattr(models, model)
            try:
                predicted = fwd_rayleigh_phase(
                    thickness_per_layer=list(predicted_model.thicknesses),
                    Vs_per_layer=list(predicted_model.vs_s),
                    mode=mode.number,
                    fs=observed.fs,
                    Vp_Vs_ratio=VP_VS_RATIO,
                )
            except DispersionError:
                logger.warning(
                    "Could not forward-model %s for folder=%s, xmid=%.2f, label=%s.",
                    model,
                    folder,
                    xmid,
                    label,
                )

        result.append(
            PositionCurves(
                xmid=xmid,
                observed_fs=observed.fs if observed is not None else None,
                observed_vs=observed.vs if observed is not None else None,
                observed_vs_err=observed.vs_err if observed is not None else None,
                predicted_fs=predicted.fs if predicted is not None else None,
                predicted_vs=predicted.vs if predicted is not None else None,
                velocity_type=observed.type.value if observed is not None else "",
            )
        )

    if not any(p.observed_fs is not None or p.predicted_fs is not None for p in result):
        raise ValueError(f"No curve for label '{label}' found in folder={folder}")

    return result
