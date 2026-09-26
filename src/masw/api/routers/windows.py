import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from masw.io.paths import workspace
from sigpipe.masw.profiles import ProfileError, load_profile
from sigpipe.masw.windows import MASWParameters, build_windows

logger = logging.getLogger(__name__)

router = APIRouter(tags=["windows"])


class WindowRequest(BaseModel):
    profile: str
    masw: MASWParameters


class WindowSummary(BaseModel):
    xmid: float
    start_index: int
    end_index: int
    n_shots: int


@router.post("/windows")
def preview_windows(request: WindowRequest) -> list[WindowSummary]:
    try:
        windows = build_windows(load_profile(request.profile, workspace()), request.masw)
    except (ProfileError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [
        WindowSummary(
            xmid=w.xmid,
            start_index=w.receiver_indices[0],
            end_index=w.receiver_indices[-1],
            n_shots=len(w.selected_files),
        )
        for w in windows
    ]
