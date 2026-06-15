import logging

from fastapi import APIRouter, HTTPException

from masw.io.acquisition import load_acquisition
from masw.io.folders import get_input_folders
from masw.models.acquisition import AcquisitionParameters

logger = logging.getLogger(__name__)

router = APIRouter(tags=["acquisitions"])


@router.get("/folders")
def list_folders() -> list[str]:
    return get_input_folders()


@router.get("/acquisitions/{folder}")
def get_acquisition(folder: str) -> AcquisitionParameters:
    try:
        return load_acquisition(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
