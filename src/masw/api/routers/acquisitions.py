import logging

from fastapi import APIRouter, HTTPException

from masw.io.folders import get_input_folders
from masw.io.profiles import Acquisition, load_acquisition

logger = logging.getLogger(__name__)

router = APIRouter(tags=["acquisitions"])


@router.get("/input_folders")
def list_input_folders() -> list[str]:
    return get_input_folders()


@router.get("/acquisitions/{folder}")
def get_acquisition(folder: str) -> Acquisition:
    try:
        return load_acquisition(folder)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
