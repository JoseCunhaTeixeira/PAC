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


class SpectrumResponse(BaseModel):
    freqs: list[float]  # Hz
    power_db: list[float]  # the traces' mean power, in dB of its largest


# The most frequencies a spectrum's preview gets: a plot shows no more.
SPECTRUM_POINTS = 1000


@router.get("/spectrum/{folder}/{file}")
def get_spectrum(folder: str, file: str) -> SpectrumResponse:
    """The record's mean power spectrum over its traces, each trace's mean removed, in dB of its
    largest: where its energy lies, for a filter's band to be chosen by it (a computing page's
    preview). The largest of each group of frequencies kept, at most SPECTRUM_POINTS."""
    path = INPUT_DIR / folder / file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")
    stream = read(str(path))
    data = np.array([tr.data for tr in stream], dtype=float)
    data -= data.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(data, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(stream[0].stats.delta))
    group = max(1, -(-freqs.size // SPECTRUM_POINTS))
    n = freqs.size // group * group
    peaks = np.concatenate((power[:n].reshape(-1, group).max(axis=1), power[n:][:1]))
    kept = np.concatenate((freqs[:n:group], freqs[n:][:1]))
    db = 10 * np.log10(peaks / (peaks.max() or 1.0) + 1e-30)
    return SpectrumResponse(
        freqs=np.round(kept, 3).tolist(), power_db=np.round(np.maximum(db, -120.0), 2).tolist()
    )
