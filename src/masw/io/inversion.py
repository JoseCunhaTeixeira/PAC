"""The inversions of an output folder (a run), as PAC's pages read them: sigpipe's inversion of a
window (sigpipe.masw.inversion), and the line's views (sigpipe.masw.inversion.section) over the
folder's windows."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from masw.io.dispersion_images import xmid_folder
from masw.io.folders import get_xmid_folders
from masw.io.paths import output_folder
from masw.io.quality.files import preset_stage, read_manifest
from masw.io.quality.inversion import (
    MEASURES_FILE,
    informed_to,
    thresholds_of,
    window_measures,
)
from sigpipe.base.dispersion_curve import Mode
from sigpipe.base.inversion import InversionResult
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.masw.inversion import InversionParameters, invert_window
from sigpipe.masw.inversion.measuring import measure_inversion
from sigpipe.masw.inversion.section import (
    ComparisonGrids,
    ModelName,
    VelocityGrid,
    comparison_grids,
    informed_levels,
    is_inverted,
    picked_curve,
    predicted_curve,
    save_comparison,
    save_section,
    save_sections_file,
    velocity_grid,
    window_model,
)

logger = logging.getLogger(__name__)


def _units(folder: str) -> list[str]:
    return [xmid_folder(folder, xmid).name for xmid in get_xmid_folders(folder)]


def invert_position(
    folder: str,
    xmid: float,
    labels: Sequence[str],
    parameters: InversionParameters,
    output_folder: Path | None = None,
    chain_jobs: int = 1,
) -> InversionResult:
    """Invert the window's curves of `labels`, its chains in `chain_jobs` processes, and write
    PAC's files beside them, or in `output_folder` (a staging folder: see
    sigpipe.masw.runs.stopping)."""
    modes = {Mode.from_label(label) for label in labels}
    return invert_window(
        xmid_folder(folder, xmid),
        parameters,
        modes,
        chain_jobs=chain_jobs,
        output_folder=output_folder,
    )


def list_inversion_status(folder: str) -> list[tuple[float, bool]]:
    return [(xmid, is_inverted(xmid_folder(folder, xmid))) for xmid in get_xmid_folders(folder)]


@dataclass(slots=True, frozen=True)
class SectionWindow:
    """A window's column in the sections: its middle, its ground's elevation, how deep its model
    reaches and how deep its data inform it (m; None when its measures do not say)."""

    x: float
    top: float
    depth: float
    informed: float | None


@dataclass(slots=True, frozen=True)
class VelocitySection:
    grid: VelocityGrid
    windows: list[SectionWindow]  # by position
    # Per column of the grid, the elevation down to which the data inform it (NaN: not known),
    # smoothed across positions as the grid is.
    levels: np.ndarray


def get_velocity_section(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> VelocitySection:
    """The section of the windows' model `model` on a grid, and each window's column: its model
    and the measures its inversion saved, each read once; nothing measured."""
    run_folder = output_folder(folder)
    found = {
        unit: one
        for unit in _units(folder)
        if (one := window_model(run_folder / unit, model)) is not None
    }
    if len(found) < 2:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    ordered = sorted(found.items(), key=lambda item: item[1].position.x)
    section = VelocityModelsSection(velocity_models=tuple(one for _, one in ordered))
    # Smoothed along the line over a window's length: what each window's model describes.
    window_m = window_length(run_folder)
    grid = velocity_grid(section, lateral_smoothing, window_m=window_m)
    windows = [_section_window(run_folder / unit, one) for unit, one in ordered]
    informed = [
        (one.x, one.top, None if one.informed is None else min(one.informed, one.depth))
        for one in windows
    ]
    levels = informed_levels(grid, informed, lateral_smoothing, window_m)
    return VelocitySection(grid=grid, windows=windows, levels=levels)


def window_length(run_folder: Path) -> float | None:
    """A window's length along the line (m), from its first receiver to its last; None for a
    folder without its run's settings."""
    manifest = read_manifest(run_folder)
    masw = preset_stage(manifest, "masw")
    if manifest is None or "length" not in masw:
        return None
    return (int(masw["length"]) - 1) * manifest.profile.receiver_spacing_m


def _section_window(window: Path, model: VelocityModel) -> SectionWindow:
    depth = round(float(np.sum(model.thicknesses)), 2)
    measured = window_measures(window)
    measures = measured[1] if measured is not None else None
    informed = None
    if measures is not None and (known := informed_to(measures)) is not None:
        # All of it: the model's own depth, the bottom the section draws.
        informed = depth if measures.useful_depth_m is None else known
    return SectionWindow(
        x=float(model.position.x), top=float(model.position.z), depth=depth, informed=informed
    )


def measure_position(
    folder: str, xmid: float, parameters: InversionParameters, output_folder: Path | None = None
) -> None:
    """Save the measures of the window's inversion (its fit, its chains, how deep its data
    inform it) beside its files, or in `output_folder` (a staging folder), where the assistant
    saves its own: Visualization reads them and never measures."""
    window = xmid_folder(folder, xmid)
    thresholds = thresholds_of(window.parent)
    measures = measure_inversion(
        window,
        parameters,
        n_bands=thresholds.n_bands,
        bound_edge=thresholds.bound_edge,
        std_ratio=thresholds.useful_std_ratio,
        output_folder=output_folder,
    )
    ((output_folder or window) / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))


def save_velocity_section_plot(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> Path:
    """Save the Vs(x,z) + std(x,z) section plot in the output folder."""
    path = save_section(output_folder(folder), _units(folder), model, lateral_smoothing)
    if path is None:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return path


def save_velocity_xzv(folder: str) -> Path:
    """Save Vs(x, z) section grids for every model variant into one HDF5 file in the output
    folder, one group per variant; a variant with fewer than two positions is left out."""
    path = save_sections_file(output_folder(folder), _units(folder))
    if path is None:
        raise ValueError(f"No model variant has at least two inverted positions in folder={folder}")
    return path


def save_pseudo_section_comparison_plot(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> Path:
    """Save the observed-vs-predicted pseudo-section comparison for one label in the output
    folder."""
    path = save_comparison(output_folder(folder), _units(folder), Mode.from_label(label), model)
    if path is None:
        raise ValueError(_too_few_comparisons(folder, label))
    return path


def get_pseudo_section_comparison(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> ComparisonGrids:
    grids = comparison_grids(output_folder(folder), _units(folder), Mode.from_label(label), model)
    if grids is None:
        raise ValueError(_too_few_comparisons(folder, label))
    return grids


def _too_few_comparisons(folder: str, label: str) -> str:
    return (
        "At least two positions with both a pick and an inversion result are required to build "
        f"a pseudo-section comparison for label={label} in folder={folder}"
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

    mode = Mode.from_label(label)
    result: list[PositionCurves] = []
    for xmid in xmids:
        observed = picked_curve(xmid_folder(folder, xmid), mode)
        predicted = (
            None
            if observed is None
            else predicted_curve(xmid_folder(folder, xmid), observed, model)
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
