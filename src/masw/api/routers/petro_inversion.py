import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from masw.api.jobs import Job, job_manager
from masw.api.routers.dispersion_images import nan_to_none
from masw.io import petro_inversion as io
from masw.models.petro_inversion import PetroInversionRunConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["petro_inversion"])


class PositionStatusOut(BaseModel):
    xmid: float
    has_result: bool


class PositionCurvesOut(BaseModel):
    xmid: float
    observed_fs: list[float] | None
    observed_vs: list[float] | None
    observed_vs_err: list[float] | None
    predicted_fs: list[float] | None
    predicted_vs: list[float] | None
    velocity_type: str


class PetroSectionOut(BaseModel):
    positions: list[float]
    elevations: list[float]
    soil_grid: list[list[str | None]]
    n_grid: list[list[float | None]]
    water_table_elevations: list[float]


class PseudoSectionComparisonOut(BaseModel):
    positions: list[float]
    fs: list[float]
    observed_grid: list[list[float | None]]
    predicted_grid: list[list[float | None]]
    residual_grid: list[list[float | None]]


class ContinuousSectionOut(BaseModel):
    positions: list[float]
    elevations: list[float]
    values: list[list[float | None]]


@router.get("/petro_inversion/models")
def get_silex_models() -> list[str]:
    return io.list_silex_models()


@router.post("/petro_inversion/run", status_code=202)
def start_petro_inversion(config: PetroInversionRunConfig) -> Job:
    return job_manager.submit_petro_inversion(config)


@router.get("/petro_inversion/status/{folder}")
def get_petro_inversion_status(folder: str) -> list[PositionStatusOut]:
    try:
        status = io.list_petro_inversion_status(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [PositionStatusOut(xmid=xmid, has_result=has_result) for xmid, has_result in status]


@router.get("/petro_inversion/curves/{folder}")
def get_curves_by_position(folder: str) -> list[PositionCurvesOut]:
    try:
        curves = io.get_curves_by_position(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        PositionCurvesOut(
            xmid=c.xmid,
            observed_fs=c.observed_fs,
            observed_vs=c.observed_vs,
            observed_vs_err=c.observed_vs_err,
            predicted_fs=c.predicted_fs,
            predicted_vs=c.predicted_vs,
            velocity_type=c.velocity_type,
        )
        for c in curves
    ]


@router.get("/petro_inversion/section/{folder}")
def get_petro_section(folder: str) -> PetroSectionOut:
    try:
        section = io.get_petro_section(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PetroSectionOut(
        positions=section.positions.tolist(),
        elevations=section.elevations.tolist(),
        # sample_soil's no-data fill value round-trips through numpy's object
        # array as a plain "" str rather than SoilType.NONE, so str(s) (not
        # s.value) has to handle both a SoilType member and a bare str.
        soil_grid=[[None if str(s) == "" else str(s) for s in row] for row in section.soil_grid],
        n_grid=nan_to_none(section.n_grid),
        water_table_elevations=section.water_table_elevations.tolist(),
    )


@router.get("/petro_inversion/shear_modulus_section/{folder}")
def get_shear_modulus_section(folder: str) -> ContinuousSectionOut:
    try:
        section = io.get_shear_modulus_section(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ContinuousSectionOut(
        positions=section.positions.tolist(),
        elevations=section.elevations.tolist(),
        values=nan_to_none(section.values),
    )


@router.get("/petro_inversion/vs_section/{folder}")
def get_vs_section(folder: str) -> ContinuousSectionOut:
    try:
        section = io.get_vs_section(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ContinuousSectionOut(
        positions=section.positions.tolist(),
        elevations=section.elevations.tolist(),
        values=nan_to_none(section.values),
    )


@router.get("/petro_inversion/pseudo_section_comparison/{folder}")
def get_pseudo_section_comparison(folder: str) -> PseudoSectionComparisonOut:
    try:
        comparison = io.get_pseudo_section_comparison(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PseudoSectionComparisonOut(
        positions=comparison.positions.tolist(),
        fs=comparison.fs.tolist(),
        observed_grid=nan_to_none(comparison.observed_grid),
        predicted_grid=nan_to_none(comparison.predicted_grid),
        residual_grid=nan_to_none(comparison.residual_grid),
    )
