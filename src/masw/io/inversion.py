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
from masw.io.quality.inversion import thresholds_of
from sigpipe.base.dispersion_curve import Mode
from sigpipe.base.inversion import InversionResult
from sigpipe.masw.inversion import InversionParameters, invert_window
from sigpipe.masw.inversion.measuring import MEASURES_FILE, measure_inversion
from sigpipe.masw.inversion.section import (
    DEFAULT_MODEL,
    ComparisonGrids,
    ModelName,
    VelocitySection,
    comparison_grids,
    is_inverted,
    line_section,
    picked_curve,
    predicted_curve,
    save_comparison,
    save_section,
    save_sections_file,
)
from sigpipe.masw.inversion.summary import save_line_summary
from sigpipe.masw.runs import window_length

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


def get_velocity_section(
    folder: str, model: ModelName = DEFAULT_MODEL, lateral_smoothing: bool = False
) -> VelocitySection:
    """The section of the windows' model `model` on a grid, and each window's column: its model
    and the measures its inversion saved, each read once; nothing measured (sigpipe's
    line_section, the figures' own), smoothed over the run's window length."""
    run_folder = output_folder(folder)
    section = line_section(
        run_folder, _units(folder), model, lateral_smoothing, window_length(run_folder)
    )
    if section is None:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return section


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
        max_uncertainty=thresholds.useful_uncertainty,
        output_folder=output_folder,
    )
    ((output_folder or window) / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))


def save_velocity_section_plot(folder: str, model: ModelName = DEFAULT_MODEL) -> Path:
    """Save the section's figure in the output folder as Visualization shows it (Vs, its
    uncertainty, the interfaces; the depth informed), as the windows' columns and smoothed along
    the line (sigpipe's save_section): the first's path."""
    run_folder = output_folder(folder)
    path = save_section(run_folder, _units(folder), model, window_length(run_folder))
    if path is None:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return path


def save_line_summary_plot(folder: str) -> Path | None:
    """Save the figure of the line's inversions at a glance in the output folder (sigpipe's
    save_line_summary: per window, the depth informed, the misfit, R-hat and the layers), with
    the run's limits."""
    run_folder = output_folder(folder)
    thresholds = thresholds_of(run_folder)
    return save_line_summary(run_folder, _units(folder), thresholds.max_misfit, thresholds.max_rhat)


def save_velocity_xzv(folder: str) -> Path:
    """Save Vs(x, z) section grids for every model variant into one HDF5 file in the output
    folder, one group per variant; a variant with fewer than two positions is left out."""
    path = save_sections_file(output_folder(folder), _units(folder))
    if path is None:
        raise ValueError(f"No model variant has at least two inverted positions in folder={folder}")
    return path


def save_pseudo_section_comparison_plot(
    folder: str, label: str, model: ModelName = DEFAULT_MODEL
) -> Path:
    """Save the picked-vs-modelled pseudo-section comparison of one label in the output folder,
    by frequency and by wavelength (sigpipe's save_comparison): the first's path."""
    path = save_comparison(output_folder(folder), _units(folder), Mode.from_label(label), model)
    if path is None:
        raise ValueError(_too_few_comparisons(folder, label))
    return path


def get_pseudo_section_comparison(
    folder: str, label: str, model: ModelName = DEFAULT_MODEL
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
    folder: str, label: str, model: ModelName = DEFAULT_MODEL
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
