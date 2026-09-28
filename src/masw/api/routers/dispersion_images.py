import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from masw.io import dispersion_images as io
from masw.io.folders import get_output_folders, get_xmid_folders
from masw.io.pick_origin import Origin
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_image import DispersionImage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispersion_images"])

# Up to 3 capital letters followed by a number, e.g. "M0", "AB12".
LABEL_PATTERN = r"^[A-Z]{1,3}[0-9]+$"


class DispersionCurveOut(BaseModel):
    label: str
    fs: list[float]
    vs: list[float]
    vs_std: list[float] | None = None


class DispersionImageOut(BaseModel):
    fv_map: list[list[float]]
    fs: list[float]
    vs: list[float]
    type: str
    curves: list[DispersionCurveOut]
    lambda_min: float | None
    lambda_max: float | None


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
    picked_by: Origin | None = None  # PACo's picker ("auto") or a person in PAC ("hand")


class PseudoSectionOut(BaseModel):
    positions: list[float]
    fs_grid: list[float]
    velocities_by_frequency: list[list[float | None]]
    lambdas_grid: list[float]
    velocities_by_wavelength: list[list[float | None]]


def rounded(values: np.ndarray, decimals: int) -> list[float]:
    """`values` to `decimals` decimals, as JSON writes them short: float32 values made float64
    first, or 0.1234 would come out 0.12340000271797180."""
    result: list[float] = np.round(np.asarray(values, dtype=np.float64), decimals).tolist()
    return result


def nan_to_none(rows: np.ndarray, decimals: int | None = None) -> list[list[float | None]]:
    """`rows` with None for NaN, to `decimals` decimals when given: what a plot can show, and a
    response a few times shorter than full precision."""
    # A per-cell Python loop (math.isnan + float() on every element) is fine
    # for a small dispersion-image map but multi-second for a fine-grained
    # velocity section (hundreds of thousands to millions of cells) --
    # astype(object) + boolean-mask assignment does the same NaN->None swap
    # in vectorized C instead.
    arr = np.asarray(rows, dtype=np.float64)
    if decimals is not None:
        arr = np.round(arr, decimals)
    out = arr.astype(object)
    out[np.isnan(arr)] = None
    result: list[list[float | None]] = out.tolist()
    return result


def _to_image_out(image: DispersionImage) -> DispersionImageOut:
    curves = (
        [
            DispersionCurveOut(
                label=c.mode.label,
                fs=c.fs.tolist(),
                vs=c.vs.tolist(),
                vs_std=c.vs_err.tolist() if c.vs_err is not None else None,
            )
            for c in sorted(image.dispersion_curves, key=lambda c: c.mode)
        ]
        if image.dispersion_curves
        else []
    )

    # Where the checks' flags start (the user, 2026-09-28): under lambda_min (twice the smallest
    # spacing) the aliasing zone, over lambda_max (three window lengths) beyond the window's
    # reach. Both along the ground, and undefined for an unknown geometry.
    # The map normalized (0 to 1): four decimals finer than any colour step.
    return DispersionImageOut(
        fv_map=[rounded(row, 4) for row in image.fv_map],
        fs=rounded(image.fs, 3),
        vs=rounded(image.vs, 3),
        type=image.type,
        curves=curves,
        lambda_min=min_resolvable_wavelength(image.acquisition),
        lambda_max=longest_reached_wavelength(image.acquisition),
    )


@router.get("/output_folders")
def list_output_folders() -> list[str]:
    return get_output_folders()


@router.get("/xmids/{folder:path}")
def get_xmids(folder: str) -> list[float]:
    try:
        return get_xmid_folders(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_images/{folder:path}/{xmid}")
def get_dispersion_image(folder: str, xmid: float) -> DispersionImageOut:
    try:
        return _to_image_out(io.load_dispersion_image(folder, xmid))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/dispersion_images/{folder:path}/{xmid}/pick/lasso")
def pick_lasso(folder: str, xmid: float, request: LassoPickRequest) -> DispersionImageOut:
    try:
        return _to_image_out(io.pick_lasso(folder, xmid, request.polygon, request.label))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/dispersion_images/{folder:path}/{xmid}/pick/box")
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


@router.post("/dispersion_images/{folder:path}/{xmid}/pick/auto")
def pick_auto(folder: str, xmid: float) -> DispersionImageOut:
    """The window's M0 picked automatically, replacing its M0."""
    try:
        return _to_image_out(io.auto_pick_m0(folder, xmid))
    except io.NoCurveFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/dispersion_images/{folder:path}/{xmid}/pick/{label}")
def delete_pick(folder: str, xmid: float, label: str) -> DispersionImageOut:
    try:
        return _to_image_out(io.delete_curve(folder, xmid, label))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_image_labels/{folder:path}")
def get_dispersion_image_labels(folder: str) -> dict[str, int]:
    try:
        return io.list_labels(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/dispersion_picks_by_position/{folder:path}")
def get_dispersion_picks_by_position(folder: str) -> list[PositionPicksOut]:
    try:
        picks = io.list_labels_by_position(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        PositionPicksOut(xmid=xmid, labels=labels, picked_by=origin)
        for xmid, labels, origin in picks
    ]


@router.get("/dispersion_pseudo_section/{folder:path}/{label}")
def get_pseudo_section(folder: str, label: str) -> PseudoSectionOut:
    try:
        section = io.get_pseudo_section(folder, label)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PseudoSectionOut(
        positions=section.positions.tolist(),
        fs_grid=section.fs_grid.tolist(),
        velocities_by_frequency=nan_to_none(section.velocities_by_frequency, 1),
        lambdas_grid=section.lambdas_grid.tolist(),
        velocities_by_wavelength=nan_to_none(section.velocities_by_wavelength, 1),
    )
