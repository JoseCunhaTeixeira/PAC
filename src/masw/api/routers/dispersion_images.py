import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from masw.io import dispersion_images as io
from masw.io.folders import get_output_folders

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispersion_images"])

# Up to 3 capital letters followed by a number, e.g. "M0", "AB12".
LABEL_PATTERN = r"^[A-Z]{1,3}[0-9]+$"


class DispersionCurveOut(BaseModel):
    label: str
    fs: list[float]
    vs: list[float]


class DispersionImageOut(BaseModel):
    fv_map: list[list[float]]
    fs: list[float]
    vs: list[float]
    type: str
    curves: list[DispersionCurveOut]


class LassoPickRequest(BaseModel):
    polygon: list[tuple[float, float]]
    label: str = Field(pattern=LABEL_PATTERN)


class BoxPickRequest(BaseModel):
    fmin: float | None = None
    fmax: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    lbdmin: float | None = None
    lbdmax: float | None = None
    label: str = Field(pattern=LABEL_PATTERN)


class PositionPicksOut(BaseModel):
    xmid: float
    labels: list[str]


class PseudoSectionOut(BaseModel):
    positions: list[float]
    fs_grid: list[float]
    velocities_by_frequency: list[list[float | None]]
    lambdas_grid: list[float]
    velocities_by_wavelength: list[list[float | None]]


def _nan_to_none(rows: np.ndarray) -> list[list[float | None]]:
    return [[None if math.isnan(v) else float(v) for v in row] for row in rows]


def _to_image_out(image) -> DispersionImageOut:  # noqa: ANN001
    curves = (
        [
            DispersionCurveOut(label=c.label, fs=c.fs.tolist(), vs=c.vs.tolist())
            for c in image.dispersion_curves
        ]
        if image.dispersion_curves
        else []
    )
    return DispersionImageOut(
        fv_map=image.fv_map.tolist(),
        fs=image.fs.tolist(),
        vs=image.vs.tolist(),
        type=image.type,
        curves=curves,
    )


@router.get("/output_folders")
def list_output_folders() -> list[str]:
    return get_output_folders()


@router.get("/xmids/{folder}")
def get_xmids(folder: str) -> list[float]:
    try:
        return io.get_xmid_folders(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_images/{folder}/{xmid}")
def get_dispersion_image(folder: str, xmid: float) -> DispersionImageOut:
    try:
        return _to_image_out(io.load_dispersion_image(folder, xmid))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/dispersion_images/{folder}/{xmid}/pick/lasso")
def pick_lasso(folder: str, xmid: float, request: LassoPickRequest) -> DispersionImageOut:
    try:
        return _to_image_out(io.pick_lasso(folder, xmid, request.polygon, request.label))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/dispersion_images/{folder}/{xmid}/pick/box")
def pick_box(folder: str, xmid: float, request: BoxPickRequest) -> DispersionImageOut:
    try:
        return _to_image_out(
            io.pick_box(
                folder,
                xmid,
                fmin=request.fmin,
                fmax=request.fmax,
                vmin=request.vmin,
                vmax=request.vmax,
                lbdmin=request.lbdmin,
                lbdmax=request.lbdmax,
                label=request.label,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/dispersion_images/{folder}/{xmid}/pick/{label}")
def delete_pick(folder: str, xmid: float, label: str) -> DispersionImageOut:
    try:
        return _to_image_out(io.delete_curve(folder, xmid, label))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_image_labels/{folder}")
def get_dispersion_image_labels(folder: str) -> dict[str, int]:
    try:
        return io.list_labels(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_picks_by_position/{folder}")
def get_dispersion_picks_by_position(folder: str) -> list[PositionPicksOut]:
    try:
        picks = io.list_labels_by_position(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [PositionPicksOut(xmid=xmid, labels=labels) for xmid, labels in picks]


@router.get("/dispersion_pseudo_section/{folder}/{label}")
def get_pseudo_section(folder: str, label: str) -> PseudoSectionOut:
    try:
        section = io.get_pseudo_section(folder, label)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PseudoSectionOut(
        positions=section.positions.tolist(),
        fs_grid=section.fs_grid.tolist(),
        velocities_by_frequency=_nan_to_none(section.velocities_by_frequency),
        lambdas_grid=section.lambdas_grid.tolist(),
        velocities_by_wavelength=_nan_to_none(section.velocities_by_wavelength),
    )
