import logging

from fastapi import APIRouter
from pydantic import BaseModel

from masw.adapters.windows import build_windows
from masw.models.acquisition import AcquisitionParameters
from masw.models.masw import MASWParameters

logger = logging.getLogger(__name__)

router = APIRouter(tags=["windows"])


class WindowRequest(BaseModel):
    acquisition_params: AcquisitionParameters
    masw_params: MASWParameters


class WindowSummary(BaseModel):
    xmid: float
    start_index: int
    end_index: int
    n_shots: int


@router.post("/windows")
def preview_windows(request: WindowRequest) -> list[WindowSummary]:
    windows = build_windows(request.acquisition_params, request.masw_params)
    return [
        WindowSummary(
            xmid=w.xmid,
            start_index=w.receiver_indices[0],
            end_index=w.receiver_indices[-1],
            n_shots=len(w.selected_files),
        )
        for w in windows
    ]
