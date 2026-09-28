import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from masw.api.jobs import Job, job_manager
from masw.api.routers.dispersion_images import nan_to_none, rounded
from masw.io import inversion as io
from masw.io.inversion import ModelName
from masw.models.inversion import InversionRunConfig
from sigpipe.masw.inversion import InversionParameters, ThicknessLayer, VsLayer
from sigpipe.masw.inversion.priors import PriorRules

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inversion"])


class InversionDefaultsOut(BaseModel):
    parameters: InversionParameters
    vs_layer: VsLayer  # a layer the form adds
    thickness_layer: ThicknessLayer
    half_space_layer: VsLayer  # the half-space's: it may be faster than the layers above


# The half-space as PACo's priors start it: up to 2,000 m/s, where the layers stop at 1,000.
HALF_SPACE = VsLayer(vs_max=PriorRules().half_space_vs_max)


class PositionStatusOut(BaseModel):
    xmid: float
    has_result: bool


class SectionWindowOut(BaseModel):
    """A window's column: its middle, its ground's elevation, how deep its model reaches and
    how deep its data inform it (m; None: not measured)."""

    x: float
    top: float
    depth: float
    informed: float | None


class VelocitySectionOut(BaseModel):
    positions: list[float]
    elevations: list[float]
    vs_grid: list[list[float | None]]
    vs_std_grid: list[list[float | None]]
    windows: list[SectionWindowOut]
    # Per column: the elevation down to which the data inform it (None: not known).
    informed_levels: list[float | None]


class PositionCurvesOut(BaseModel):
    xmid: float
    observed_fs: list[float] | None
    observed_vs: list[float] | None
    observed_vs_err: list[float] | None
    predicted_fs: list[float] | None
    predicted_vs: list[float] | None
    velocity_type: str


class PseudoSectionComparisonOut(BaseModel):
    """Picked, modelled and their residual by position and frequency, and by position and
    wavelength."""

    positions: list[float]
    fs: list[float]
    observed_grid: list[list[float | None]]
    predicted_grid: list[list[float | None]]
    residual_grid: list[list[float | None]]
    lambdas: list[float]
    observed_by_wavelength_grid: list[list[float | None]]
    predicted_by_wavelength_grid: list[list[float | None]]
    residual_by_wavelength_grid: list[list[float | None]]


@router.get("/inversion/defaults")
def get_inversion_defaults() -> InversionDefaultsOut:
    """The inversion form's starting values: sigpipe's, as PACo's agent gets them, the
    half-space up to 2,000 m/s as PACo's priors start it."""
    parameters = InversionParameters()
    return InversionDefaultsOut(
        parameters=parameters.model_copy(
            update={"vs_layers": (*parameters.vs_layers[:-1], HALF_SPACE)}
        ),
        vs_layer=VsLayer(),
        thickness_layer=ThicknessLayer(),
        half_space_layer=HALF_SPACE,
    )


@router.post("/inversion/run", status_code=202)
def start_inversion(config: InversionRunConfig) -> Job:
    return job_manager.submit_inversion(config)


@router.get("/inversion/status/{folder:path}")
def get_inversion_status(folder: str) -> list[PositionStatusOut]:
    try:
        status = io.list_inversion_status(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [PositionStatusOut(xmid=xmid, has_result=has_result) for xmid, has_result in status]


@router.get("/inversion/velocity_section/{folder:path}")
def get_velocity_section(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> VelocitySectionOut:
    try:
        section = io.get_velocity_section(folder, model, lateral_smoothing)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    grid = section.grid
    return VelocitySectionOut(
        positions=rounded(grid.positions, 3),
        elevations=rounded(grid.elevations, 3),
        vs_grid=nan_to_none(grid.vs, 1),
        vs_std_grid=nan_to_none(grid.vs_std, 1),
        windows=[
            SectionWindowOut(x=one.x, top=one.top, depth=one.depth, informed=one.informed)
            for one in section.windows
        ],
        informed_levels=nan_to_none(section.levels[None, :], 3)[0],
    )


@router.get("/inversion/curves/{folder:path}/{label}")
def get_curves_by_position(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> list[PositionCurvesOut]:
    try:
        curves = io.get_curves_by_position(folder, label, model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        PositionCurvesOut(
            xmid=c.xmid,
            observed_fs=c.observed_fs.tolist() if c.observed_fs is not None else None,
            observed_vs=c.observed_vs.tolist() if c.observed_vs is not None else None,
            observed_vs_err=c.observed_vs_err.tolist() if c.observed_vs_err is not None else None,
            predicted_fs=c.predicted_fs.tolist() if c.predicted_fs is not None else None,
            predicted_vs=c.predicted_vs.tolist() if c.predicted_vs is not None else None,
            velocity_type=c.velocity_type,
        )
        for c in curves
    ]


@router.get("/inversion/pseudo_section_comparison/{folder:path}/{label}")
def get_pseudo_section_comparison(
    folder: str, label: str, model: ModelName = "smooth_median"
) -> PseudoSectionComparisonOut:
    try:
        comparison = io.get_pseudo_section_comparison(folder, label, model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PseudoSectionComparisonOut(
        positions=comparison.positions.tolist(),
        fs=comparison.fs.tolist(),
        observed_grid=nan_to_none(comparison.observed, 1),
        predicted_grid=nan_to_none(comparison.predicted, 1),
        residual_grid=nan_to_none(comparison.residual, 2),  # %
        lambdas=comparison.lambdas.tolist(),
        observed_by_wavelength_grid=nan_to_none(comparison.observed_by_wavelength, 1),
        predicted_by_wavelength_grid=nan_to_none(comparison.predicted_by_wavelength, 1),
        residual_by_wavelength_grid=nan_to_none(comparison.residual_by_wavelength, 2),
    )
