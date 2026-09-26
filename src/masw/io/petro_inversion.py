"""The petrophysical inversions of an output folder (a run), as PAC's pages read them: sigpipe's
inversion of a window (sigpipe.masw.petro), and the line's views (sigpipe.masw.petro.section)
over the folder's windows."""

from dataclasses import dataclass
from pathlib import Path

from masw.io.dispersion_images import xmid_folder
from masw.io.folders import get_xmid_folders
from masw.io.paths import output_folder
from sigpipe.algorithms.inversion.rayleigh.petro.silex_catalog import list_bundled_silex_models
from sigpipe.base.petro_model import PetroModel
from sigpipe.masw.inversion.section import ComparisonGrids
from sigpipe.masw.petro import (
    Quantity,
    fundamental_curve,
    invert_window_petro,
    load_modeled_curve,
    load_petro_model,
)
from sigpipe.masw.petro.section import (
    PetroGrid,
    RockPhysicsGrid,
    comparison_grids,
    petro_grid,
    petro_models,
    rock_physics_grid,
    save_petro_section,
    save_petro_sections_file,
    save_rock_physics_file,
    save_rock_physics_section,
)


def _units(folder: str) -> list[str]:
    return [xmid_folder(folder, xmid).name for xmid in get_xmid_folders(folder)]


def _too_few(folder: str) -> ValueError:
    return ValueError(
        f"At least two inverted positions are required to build a section in folder={folder}"
    )


def list_silex_models() -> list[str]:
    return list_bundled_silex_models()


def invert_position(folder: str, xmid: float, model_name: str) -> PetroModel:
    """Invert the window's fundamental mode, and write PAC's files beside it."""
    return invert_window_petro(xmid_folder(folder, xmid), model_name)


def list_petro_inversion_status(folder: str) -> list[tuple[float, bool]]:
    return [
        (xmid, load_petro_model(xmid_folder(folder, xmid)) is not None)
        for xmid in get_xmid_folders(folder)
    ]


@dataclass(slots=True, frozen=True)
class PositionCurves:
    xmid: float
    observed_fs: list[float] | None
    observed_vs: list[float] | None
    observed_vs_err: list[float] | None
    predicted_fs: list[float] | None
    predicted_vs: list[float] | None
    velocity_type: str


def get_curves_by_position(folder: str) -> list[PositionCurves]:
    xmids = get_xmid_folders(folder)
    if not xmids:
        raise ValueError(f"No xmid positions found in folder={folder}")

    result: list[PositionCurves] = []
    for xmid in xmids:
        window = xmid_folder(folder, xmid)
        try:
            observed = fundamental_curve(window)
        except ValueError:
            observed = None
        predicted = load_modeled_curve(window) if observed is not None else None

        result.append(
            PositionCurves(
                xmid=xmid,
                observed_fs=observed.fs.tolist() if observed is not None else None,
                observed_vs=observed.vs.tolist() if observed is not None else None,
                observed_vs_err=observed.vs_err.tolist()
                if observed is not None and observed.vs_err is not None
                else None,
                predicted_fs=predicted.fs.tolist() if predicted is not None else None,
                predicted_vs=predicted.vs.tolist() if predicted is not None else None,
                velocity_type=observed.type.value if observed is not None else "",
            )
        )

    if not any(p.observed_fs is not None or p.predicted_fs is not None for p in result):
        raise ValueError(f"No fundamental-mode curve found in folder={folder}")

    return result


def get_petro_section(folder: str) -> PetroGrid:
    section = petro_models(output_folder(folder), _units(folder))
    if section is None:
        raise _too_few(folder)
    return petro_grid(section)


def save_petro_section_plot(folder: str) -> Path:
    """Save the soil-type + N-value section plot in the output folder."""
    path = save_petro_section(output_folder(folder), _units(folder))
    if path is None:
        raise _too_few(folder)
    return path


def save_petro_section_hdf5(folder: str) -> Path:
    """Save the soil-type + N-value section grids into one HDF5 file in the output folder."""
    path = save_petro_sections_file(output_folder(folder), _units(folder))
    if path is None:
        raise _too_few(folder)
    return path


def _rock_physics(folder: str, quantity: Quantity) -> RockPhysicsGrid:
    grid = rock_physics_grid(output_folder(folder), _units(folder), quantity)
    if grid is None:
        raise _too_few(folder)
    return grid


def get_shear_modulus_section(folder: str) -> RockPhysicsGrid:
    return _rock_physics(folder, "shear_modulus")  # GPa


def get_vs_section(folder: str) -> RockPhysicsGrid:
    return _rock_physics(folder, "vs")


def _save_rock_physics(folder: str, quantity: Quantity, as_file: bool) -> Path:
    save = save_rock_physics_file if as_file else save_rock_physics_section
    path = save(output_folder(folder), _units(folder), quantity)
    if path is None:
        raise _too_few(folder)
    return path


def save_shear_modulus_section_plot(folder: str) -> Path:
    return _save_rock_physics(folder, "shear_modulus", as_file=False)


def save_shear_modulus_section_hdf5(folder: str) -> Path:
    return _save_rock_physics(folder, "shear_modulus", as_file=True)


def save_vs_section_plot(folder: str) -> Path:
    return _save_rock_physics(folder, "vs", as_file=False)


def save_vs_section_hdf5(folder: str) -> Path:
    return _save_rock_physics(folder, "vs", as_file=True)


def get_pseudo_section_comparison(folder: str) -> ComparisonGrids:
    grids = comparison_grids(output_folder(folder), _units(folder))
    if grids is None:
        raise ValueError(
            "At least two positions with both a pick and a petro inversion result are required "
            f"to build a pseudo-section comparison in folder={folder}"
        )
    return grids
