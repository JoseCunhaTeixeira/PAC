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
from sigpipe.base.dispersion_curve import Mode
from sigpipe.base.inversion import InversionResult
from sigpipe.masw.inversion import InversionParameters, invert_window
from sigpipe.masw.inversion.section import (
    ComparisonGrids,
    ModelName,
    VelocityGrid,
    comparison_grids,
    is_inverted,
    models_section,
    picked_curve,
    predicted_curve,
    save_comparison,
    save_section,
    save_sections_file,
    velocity_grid,
)

logger = logging.getLogger(__name__)


def _units(folder: str) -> list[str]:
    return [xmid_folder(folder, xmid).name for xmid in get_xmid_folders(folder)]


def invert_position(
    folder: str,
    xmid: float,
    labels: Sequence[str],
    parameters: InversionParameters,
) -> InversionResult:
    """Invert the window's curves of `labels`, and write PAC's files beside them."""
    modes = {Mode.from_label(label) for label in labels}
    return invert_window(xmid_folder(folder, xmid), parameters, modes)


def list_inversion_status(folder: str) -> list[tuple[float, bool]]:
    return [(xmid, is_inverted(xmid_folder(folder, xmid))) for xmid in get_xmid_folders(folder)]


def get_velocity_section(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> VelocityGrid:
    section = models_section(output_folder(folder), _units(folder), model)
    if section is None:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return velocity_grid(section, lateral_smoothing)


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
