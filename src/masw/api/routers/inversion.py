import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from masw.api.jobs import Job, job_manager
from masw.api.routers.dispersion_images import nan_to_none
from masw.io import inversion as io
from masw.io.inversion import ModelName
from masw.models.inversion import InversionRunConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inversion"])


class PositionStatusOut(BaseModel):
    xmid: float
    has_result: bool


class VelocitySectionOut(BaseModel):
    positions: list[float]
    elevations: list[float]
    vs_grid: list[list[float | None]]
    vs_std_grid: list[list[float | None]]


class PositionCurvesOut(BaseModel):
    xmid: float
    observed_fs: list[float] | None
    observed_vs: list[float] | None
    observed_vs_err: list[float] | None
    predicted_fs: list[float] | None
    predicted_vs: list[float] | None
    velocity_type: str


class SaveImagesIn(BaseModel):
    labels: list[str]
    model: ModelName = "smooth_median"
    lateral_smoothing: bool = False


class SaveImagesOut(BaseModel):
    saved_paths: list[str]
    errors: list[str]


class PseudoSectionComparisonOut(BaseModel):
    positions: list[float]
    fs: list[float]
    observed_grid: list[list[float | None]]
    predicted_grid: list[list[float | None]]
    residual_grid: list[list[float | None]]


@router.post("/inversion/run", status_code=202)
def start_inversion(config: InversionRunConfig) -> Job:
    return job_manager.submit_inversion(config)


@router.get("/inversion/status/{folder}")
def get_inversion_status(folder: str) -> list[PositionStatusOut]:
    try:
        status = io.list_inversion_status(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [PositionStatusOut(xmid=xmid, has_result=has_result) for xmid, has_result in status]


@router.get("/inversion/velocity_section/{folder}")
def get_velocity_section(
    folder: str, model: ModelName = "smooth_median", lateral_smoothing: bool = False
) -> VelocitySectionOut:
    try:
        section = io.get_velocity_section(folder, model, lateral_smoothing)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return VelocitySectionOut(
        positions=section.positions.tolist(),
        elevations=section.elevations.tolist(),
        vs_grid=nan_to_none(section.vs_grid),
        vs_std_grid=nan_to_none(section.vs_std_grid),
    )


@router.get("/inversion/curves/{folder}/{label}")
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


@router.get("/inversion/pseudo_section_comparison/{folder}/{label}")
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
        observed_grid=nan_to_none(comparison.observed_grid),
        predicted_grid=nan_to_none(comparison.predicted_grid),
        residual_grid=nan_to_none(comparison.residual_grid),
    )


@router.post("/inversion/save_images/{folder}")
def save_images(folder: str, body: SaveImagesIn) -> SaveImagesOut:
    """Save the Vs/std section and per-label pseudo-section comparison plots
    for the given model/smoothing choice into the profile's output folder.

    Best-effort per artifact (matching the runner's end-of-run auto-save):
    one label without enough data doesn't block the others.
    """
    saved_paths: list[str] = []
    errors: list[str] = []

    try:
        section_path = io.save_velocity_section_plot(folder, body.model, body.lateral_smoothing)
        saved_paths.append(str(section_path))
    except ValueError as exc:
        errors.append(str(exc))

    for label in body.labels:
        try:
            pseudo_path = io.save_pseudo_section_comparison_plot(folder, label, body.model)
            saved_paths.append(str(pseudo_path))
        except ValueError as exc:
            errors.append(str(exc))

    return SaveImagesOut(saved_paths=saved_paths, errors=errors)
