import logging
from typing import Literal

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
def get_gather(
    folder: str, file: str, norm: Literal["trace", "global"] = "trace"
) -> GatherResponse:
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

    # Normalise to [-1, 1] for display, either trace-by-trace or globally.
    if norm == "trace":
        peak = np.max(np.abs(data), axis=1, keepdims=True)
    else:
        peak = np.full((data.shape[0], 1), np.max(np.abs(data)))
    peak[peak == 0] = 1.0
    normalized = data / peak

    return GatherResponse(
        dt=dt, n_samples=normalized.shape[1], traces=np.round(normalized, 4).tolist()
    )
