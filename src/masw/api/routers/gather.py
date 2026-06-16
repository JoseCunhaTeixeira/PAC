import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from obspy import read
from pydantic import BaseModel

from masw.io.paths import INPUT_DIR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gather"])


class GatherResponse(BaseModel):
    dt: float
    n_samples: int
    traces: list[list[float]]


@router.get("/gather/{folder}/{file}")
def get_gather(folder: str, file: str) -> GatherResponse:
    path = INPUT_DIR / folder / file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")

    stream = read(str(path))
    dt = float(stream[0].stats.delta)

    data = np.array([tr.data for tr in stream], dtype=float)

    # Downsample in time to keep the payload small.
    step = max(1, data.shape[1] // 800)
    data = data[:, ::step]
    dt *= step

    # Normalise each trace to [-1, 1] for display.
    peak = np.max(np.abs(data), axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    norm = data / peak

    return GatherResponse(
        dt=dt, n_samples=norm.shape[1], traces=np.round(norm, 4).tolist()
    )
