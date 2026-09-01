import json
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from santiludo import DEFAULT_FLUID_PROPERTIES, DEFAULT_GRAIN_PROPERTIES, Layer, RockPhysicsResult
from santiludo import compute_rock_physics as _compute_rock_physics_profile

from masw.adapters.petro_inversion import build_petro_inversion_pipeline
from masw.io.dispersion_images import load_dispersion_image, xmid_folder
from masw.io.folders import get_xmid_folders
from masw.io.paths import OUTPUT_DIR
from sigpipe.algorithms.inversion.rayleigh.petro.forward import fwd_petro_phase, parse_under_layers
from sigpipe.algorithms.inversion.rayleigh.petro.silex import (
    bundled_silex_model_dir,
    list_bundled_silex_models,
    silex_under_layers,
)
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import (
    DispersionCurve,
    DispersionCurves,
    DispersionCurvesSection,
    Mode,
)
from sigpipe.base.petro_model import PetroModel, PetroModelsSection, SoilType
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.dispersion.section import pseudo_section_comparison_grids
from sigpipe.dataio.petro_model.loading import load_petro_models
from sigpipe.dataio.petro_model.section import plot_petro_models_section
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, HEIGHT_CM
from sigpipe.transformers import Plot

logger = logging.getLogger(__name__)

# Silex requires the curve it's fed to carry exactly this Mode (sigpipe's
# inversion_silex matches on `dc.mode == Mode("R", 0)`). Picking labels in
# this app aren't tied to a wave-type letter -- users may label their
# fundamental-mode pick "M0", "L0", etc. -- so any picked curve whose mode
# *number* is 0 is treated as the fundamental mode and relabeled to this
# Mode before being handed to the Silex pipeline.
_FUNDAMENTAL_MODE = Mode("R", 0)


def _fundamental_curve(folder: str, xmid: float) -> DispersionCurve:
    image = load_dispersion_image(folder, xmid)
    if not image.dispersion_curves:
        raise ValueError(f"No picked curves for folder={folder}, xmid={xmid}")
    curve = next((c for c in image.dispersion_curves if c.mode.number == 0), None)
    if curve is None:
        raise ValueError(
            f"No fundamental-mode (mode number 0) pick for folder={folder}, xmid={xmid}"
        )
    if curve.mode != _FUNDAMENTAL_MODE:
        curve = replace(curve, mode=_FUNDAMENTAL_MODE)
    return curve


def list_silex_models() -> list[str]:
    return list_bundled_silex_models()


def _petro_model_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "PetroInversion_Model_0000.csv"


def _modeled_curve_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "PetroInversion_DispersionCurves_0000.csv"


def invert_position(folder: str, xmid: float, model_name: str) -> PetroModel:
    observed = _fundamental_curve(folder, xmid)
    model_dir = bundled_silex_model_dir(model_name)
    # sigpipe's own SilexModel._preprocess raises ValueError if `observed`
    # doesn't cover this checkpoint's trained frequency/velocity range --
    # no separate check needed here.
    curves = DispersionCurves(dispersion_curves=(observed,))
    output_folder = xmid_folder(folder, xmid)

    pipeline = build_petro_inversion_pipeline(model_dir, output_folder=output_folder)
    result: PetroModel = pipeline.run(data=[curves], show_log=False)[0]

    under_layers = parse_under_layers(silex_under_layers(model_dir))
    modeled_curve = fwd_petro_phase(
        result,
        mode=0,
        fs=observed.fs,
        under_layers=under_layers,
    )
    save_dispersion_curves(
        DispersionCurves(dispersion_curves=(modeled_curve,)),
        path=_modeled_curve_path(folder, xmid),
    )

    # Run santiludo's rock-physics chain once, here, and persist its
    # shear-modulus/Vs depth profiles -- so viewing the shear-modulus/Vs
    # sections later is just reading these files back, never re-running the
    # forward model on every view.
    rp = _compute_rock_physics(result)
    _save_rock_physics(folder, xmid, result.position, rp)

    return result


@lru_cache(maxsize=4096)
def load_petro_result(folder: str, xmid: float) -> PetroModel | None:
    path = _petro_model_path(folder, xmid)
    if not path.exists():
        return None
    return load_petro_models([path])[0][0]


def clear_petro_inversion_cache() -> None:
    load_petro_result.cache_clear()
    _load_modeled_curve.cache_clear()
    _load_shear_modulus.cache_clear()
    _load_vs.cache_clear()


def list_petro_inversion_status(folder: str) -> list[tuple[float, bool]]:
    xmids = get_xmid_folders(folder)
    return [(xmid, load_petro_result(folder, xmid) is not None) for xmid in xmids]


@lru_cache(maxsize=4096)
def _load_modeled_curve(folder: str, xmid: float) -> DispersionCurve | None:
    path = _modeled_curve_path(folder, xmid)
    if not path.exists():
        return None
    (curves,) = load_dispersion_curves([path])
    return curves[0]


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
        try:
            observed = _fundamental_curve(folder, xmid)
        except ValueError:
            observed = None

        predicted = _load_modeled_curve(folder, xmid) if observed is not None else None

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


# z cell budget for the live petro section grid, sized for the canvas that
# actually renders it -- mirrors io/inversion.py's _VIZ_GRID_NZ (same
# rationale: thousands of z-samples from a fine dz make no visible
# difference on screen).
_VIZ_GRID_NZ = 200


@dataclass(slots=True, frozen=True)
class PetroSection:
    positions: np.ndarray
    elevations: np.ndarray
    # sample_soil's no-data fill value round-trips through numpy's object
    # array as a plain "" str rather than SoilType.NONE -- cells are one or
    # the other depending on whether that position's own profile reaches
    # that elevation.
    soil_grid: list[list[SoilType | str]]
    n_grid: np.ndarray
    water_table_elevations: np.ndarray


def _petro_models(folder: str) -> list[PetroModel]:
    xmids = get_xmid_folders(folder)
    models = [model for xmid in xmids if (model := load_petro_result(folder, xmid)) is not None]
    if len(models) < 2:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    return sorted(models, key=lambda m: m.position.x)


def get_petro_section(folder: str) -> PetroSection:
    """One column per actually-inverted xmid (not an interpolated x-grid),
    same reasoning as io/inversion.py's get_velocity_section: soil/N are
    categorical, so there's nothing to interpolate between positions anyway.
    """
    models = _petro_models(folder)

    tops = [m.position.z for m in models]
    bottoms = [m.position.z - sum(m.thicknesses) for m in models]
    dz = min(min(m.thicknesses) for m in models) / 10
    nz = int(np.floor((max(tops) - min(bottoms)) / dz)) + 1
    zs_fine = (max(tops) - np.arange(nz, dtype=np.float32) * dz).astype(np.float32)

    xs = np.array([m.position.x for m in models], dtype=np.float32)
    soil_grid_fine = [list(m.sample_soil(zs_fine)) for m in models]
    n_grid_fine = np.array([m.sample_N(zs_fine) for m in models], dtype=np.float32)
    water_table_elevations = np.array(
        [m.position.z - m.water_table_depth for m in models], dtype=np.float32
    )

    z_stride = max(len(zs_fine) // _VIZ_GRID_NZ, 1)
    zs = zs_fine[::z_stride]
    soil_grid = [row[::z_stride] for row in soil_grid_fine]
    n_grid = n_grid_fine[:, ::z_stride]

    return PetroSection(
        positions=xs,
        elevations=zs,
        soil_grid=soil_grid,
        n_grid=n_grid,
        water_table_elevations=water_table_elevations,
    )


def save_petro_section_plot(folder: str) -> Path:
    """Save the soil-type + N-value section plot in the profile's output
    folder, mirroring io/inversion.py's save_velocity_section_plot."""
    section = PetroModelsSection(petro_models=tuple(_petro_models(folder)))
    fig = plot_petro_models_section(section)
    path = OUTPUT_DIR / folder / "PetroInversion_Section_0000.png"
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path


def save_petro_section_hdf5(folder: str) -> Path:
    """Save the soil-type + N-value section grids into one HDF5 file in the
    profile's output folder, mirroring io/inversion.py's save_velocity_xzv
    (sigpipe.dataio.velocity_model.section.save_velocity_models_sections) --
    sigpipe has no petro-section HDF5 saver of its own yet, so this builds the
    grid the same way plot_petro_models_section does (PetroModelsSection.to_grid,
    default dz/dx) and writes it out with the same x/z dataset naming
    convention as the velocity-section file, plus soil (string) and N."""
    section = PetroModelsSection(petro_models=tuple(_petro_models(folder)))
    xs, zs, soil_grid, n_grid, water_table_elevations = section.to_grid(dz=0.01, dx=None)
    # sample_soil's no-data fill value round-trips through numpy's object
    # array as a plain "" str rather than SoilType.NONE, so str(soil) (not
    # soil.value) has to handle both a SoilType member and a bare str.
    soil_str_grid = np.array(  # pyright: ignore[reportUnknownVariableType]
        [[str(soil) for soil in row] for row in soil_grid],
        dtype=h5py.string_dtype(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    path = OUTPUT_DIR / folder / "PetroInversion_Section_0000.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=xs)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("z", data=zs)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("soil", data=soil_str_grid)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("N", data=n_grid)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            "water_table_elevation", data=water_table_elevations
        )
    return path


# Matches fwd_petro_phase's own defaults (sigpipe's
# algorithms/inversion/rayleigh/petro/forward.py), so the persisted
# shear-modulus/Vs profile below represents exactly the same intermediate
# rock-physics stage used to build the forward-modeled comparison curve, not
# a separately parameterized recomputation that could quietly diverge from it.
_ROCK_PHYSICS_DZ = 0.01
_ROCK_PHYSICS_KK = 3
_ROCK_PHYSICS_FRAC = 0.3


def _compute_rock_physics(petro_model: PetroModel) -> RockPhysicsResult:
    layers = [
        Layer(soiltype=str(soil), thickness=thickness, N=float(n), frac=_ROCK_PHYSICS_FRAC)
        for soil, thickness, n in zip(
            petro_model.soils, petro_model.thicknesses, petro_model.Ns, strict=True
        )
    ]
    return _compute_rock_physics_profile(
        layers,
        WT=petro_model.water_table_depth,
        dz=_ROCK_PHYSICS_DZ,
        kk=_ROCK_PHYSICS_KK,
        grain_properties=DEFAULT_GRAIN_PROPERTIES,
        fluid_properties=DEFAULT_FLUID_PROPERTIES,
    )


def _shear_modulus_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "PetroInversion_ShearModulus_0000.csv"


def _vs_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "PetroInversion_Vs_0000.csv"


def _save_rock_physics_profile(
    path: Path, position: Coordinate, dz: float, column: str, values: np.ndarray
) -> None:
    """Same shape as sigpipe's own VelocityModel CSVs (a `position:` header,
    then a `thickness_m,<column>` table) -- here every row's "thickness" is
    just the fixed rock-physics sample spacing `dz`, since mu_HM/Vs vary
    continuously with depth rather than in discrete layers."""
    with path.open("w", encoding="utf-8") as file:
        file.write(f"position: {json.dumps(position.to_tuple())}\n")
        file.write(f"thickness_m,{column}\n")
        for v in values:
            file.write(f"{float(dz):.6f},{float(v):.6f}\n")


def _save_rock_physics(
    folder: str, xmid: float, position: Coordinate, rp: RockPhysicsResult
) -> None:
    _save_rock_physics_profile(
        _shear_modulus_path(folder, xmid), position, rp.dz, "mu_hm_pa", rp.muHMs
    )
    _save_rock_physics_profile(_vs_path(folder, xmid), position, rp.dz, "vs_m_s", rp.VSs)


def _load_rock_physics_profile(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(elevations, values), descending-elevation order (deepest last) --
    elevations are reconstructed from the position header + cumulative
    per-row thickness, the same convention VelocityModel CSVs use."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        position_line = file.readline()
    position_z = json.loads(position_line.removeprefix("position:").strip())[2]
    data = np.loadtxt(path, delimiter=",", skiprows=2, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    thicknesses, values = data[:, 0], data[:, 1]
    elevations = position_z - np.cumsum(thicknesses)
    return elevations, values


@lru_cache(maxsize=4096)
def _load_shear_modulus(folder: str, xmid: float) -> tuple[np.ndarray, np.ndarray] | None:
    return _load_rock_physics_profile(_shear_modulus_path(folder, xmid))


@lru_cache(maxsize=4096)
def _load_vs(folder: str, xmid: float) -> tuple[np.ndarray, np.ndarray] | None:
    return _load_rock_physics_profile(_vs_path(folder, xmid))


@dataclass(slots=True, frozen=True)
class ContinuousSection:
    positions: np.ndarray
    elevations: np.ndarray
    values: np.ndarray


def _rock_physics_section(
    folder: str, field: Literal["mu", "vs"], scale: float = 1.0
) -> ContinuousSection:
    """One column per actually-inverted xmid, each interpolated from that
    position's own (continuous, not piecewise-constant) saved rock-physics
    depth profile onto a shared elevation grid -- unlike soil/N, mu_HM and Vs
    vary continuously with depth even within one soil layer (saturation and
    effective pressure both vary with depth), so this interpolates within
    each position's own profile rather than doing a nearest-depth-bin lookup.
    Purely reads what invert_position already computed and saved; no
    rock-physics computation happens here.

    `scale` converts santiludo's own SI-unit output (`mu` in Pa) to whatever
    display unit the caller wants (e.g. 1e-9 for GPa); left at 1.0 for fields
    already in their display unit (`vs` is already m/s).
    """
    load_profile = _load_shear_modulus if field == "mu" else _load_vs

    xmids = get_xmid_folders(folder)
    entries: list[tuple[PetroModel, np.ndarray, np.ndarray]] = []
    for xmid in xmids:
        model = load_petro_result(folder, xmid)
        if model is None:
            continue
        loaded = load_profile(folder, xmid)
        if loaded is None:
            continue
        elevations, values = loaded
        entries.append((model, elevations, values))
    if len(entries) < 2:
        raise ValueError(
            f"At least two inverted positions are required to build a section in folder={folder}"
        )
    entries.sort(key=lambda entry: entry[0].position.x)

    tops = [m.position.z for m, _, _ in entries]
    bottoms = [elevations[-1] for _, elevations, _ in entries]
    nz_fine = int(np.floor((max(tops) - min(bottoms)) / _ROCK_PHYSICS_DZ)) + 1
    zs_fine = (max(tops) - np.arange(nz_fine, dtype=np.float32) * _ROCK_PHYSICS_DZ).astype(
        np.float32
    )

    xs = np.array([m.position.x for m, _, _ in entries], dtype=np.float32)
    values_fine = np.full((len(entries), nz_fine), np.nan, dtype=np.float32)
    for i, (_m, elevations_desc, values_desc) in enumerate(entries):
        # Saved in descending-elevation order (deepest last) -- reversed to
        # ascending for np.interp, which requires its xp argument sorted
        # ascending.
        elevations_asc = elevations_desc[::-1]
        values_asc = values_desc[::-1] * scale
        valid = (zs_fine >= elevations_asc[0]) & (zs_fine <= elevations_asc[-1])
        values_fine[i, valid] = np.interp(zs_fine[valid], elevations_asc, values_asc)

    z_stride = max(nz_fine // _VIZ_GRID_NZ, 1)
    return ContinuousSection(
        positions=xs,
        elevations=zs_fine[::z_stride],
        values=values_fine[:, ::z_stride],
    )


def get_shear_modulus_section(folder: str) -> ContinuousSection:
    return _rock_physics_section(folder, "mu", scale=1e-9)  # Pa -> GPa


def get_vs_section(folder: str) -> ContinuousSection:
    return _rock_physics_section(folder, "vs")


def _plot_continuous_section(
    section: ContinuousSection, *, colorbar_label: str, cmap: str
) -> Figure:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_CM * CM, HEIGHT_CM * CM), dpi=DISP_DPI)
    pcm = ax.pcolormesh(  # pyright: ignore[reportUnknownMemberType]
        section.positions, section.elevations, section.values.T, shading="nearest", cmap=cmap
    )
    ax.set_xlim(section.positions[0], section.positions[-1])
    fig.colorbar(pcm, ax=ax, label=colorbar_label)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Position [m]")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Elevation [m]")  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    return fig


def _save_continuous_section_hdf5(section: ContinuousSection, path: Path, value_key: str) -> None:
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=section.positions)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("z", data=section.elevations)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset(value_key, data=section.values)  # pyright: ignore[reportUnknownMemberType]


def save_shear_modulus_section_plot(folder: str) -> Path:
    fig = _plot_continuous_section(
        get_shear_modulus_section(folder),
        colorbar_label="Hertz-Mindlin shear modulus $\\mu_{HM}$ [GPa]",
        cmap="viridis",
    )
    path = OUTPUT_DIR / folder / "PetroInversion_ShearModulusSection_0000.png"
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path


def save_shear_modulus_section_hdf5(folder: str) -> Path:
    path = OUTPUT_DIR / folder / "PetroInversion_ShearModulusSection_0000.hdf5"
    _save_continuous_section_hdf5(get_shear_modulus_section(folder), path, "mu")
    return path


def save_vs_section_plot(folder: str) -> Path:
    fig = _plot_continuous_section(
        get_vs_section(folder), colorbar_label="$V_s$ [m/s]", cmap="terrain"
    )
    path = OUTPUT_DIR / folder / "PetroInversion_VsSection_0000.png"
    Plot.savefig(path=path, figure=fig)
    plt.close(fig)
    return path


def save_vs_section_hdf5(folder: str) -> Path:
    path = OUTPUT_DIR / folder / "PetroInversion_VsSection_0000.hdf5"
    _save_continuous_section_hdf5(get_vs_section(folder), path, "vs")
    return path


def _observed_predicted_sections(
    folder: str,
) -> tuple[DispersionCurvesSection, DispersionCurvesSection]:
    xmids = get_xmid_folders(folder)

    observed_curves: list[DispersionCurve] = []
    predicted_curves: list[DispersionCurve] = []
    for xmid in xmids:
        try:
            observed = _fundamental_curve(folder, xmid)
        except ValueError:
            continue

        predicted = _load_modeled_curve(folder, xmid)
        if predicted is None:
            continue

        observed_curves.append(observed)
        # fwd_petro_phase doesn't know the real position; carry over the
        # observed curve's acquisition so the predicted curve sorts/groups
        # by the same xmid downstream (mirrors io/inversion.py's
        # _predicted_curve).
        predicted_curves.append(replace(predicted, acquisition=observed.acquisition))

    if len(observed_curves) < 2:
        raise ValueError(
            f"At least two positions with both a pick and a petro inversion result are "
            f"required to build a pseudo-section comparison in folder={folder}"
        )

    return (
        DispersionCurvesSection(dispersion_curves=tuple(observed_curves)),
        DispersionCurvesSection(dispersion_curves=tuple(predicted_curves)),
    )


@dataclass(slots=True, frozen=True)
class PseudoSectionComparison:
    positions: np.ndarray
    fs: np.ndarray
    observed_grid: np.ndarray
    predicted_grid: np.ndarray
    residual_grid: np.ndarray


def get_pseudo_section_comparison(folder: str) -> PseudoSectionComparison:
    observed, predicted = _observed_predicted_sections(folder)
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
