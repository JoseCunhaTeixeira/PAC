import logging
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from obspy import read
from pydantic import BaseModel

from masw.io.paths import INPUT_DIR
from sigpipe.dataio.signal_plotting import trace_spectra

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
    freqs: list[float]  # Hz, 0 to Nyquist
    # Each trace's amplitude spectrum, in the file's order (traces x freqs), 0 to 1 of its largest.
    amplitude: list[list[float]]


@router.get("/spectrum/{folder}/{file}")
def get_spectrum(folder: str, file: str) -> SpectrumResponse:
    """Each trace's amplitude spectrum as the record's saved figure has it (sigpipe's
    trace_spectra: its mean removed, the whole of it, 0 to Nyquist, each scaled to its own
    largest): where its energy lies, for a filter's band to be chosen by it (a computing page's
    preview)."""
    path = INPUT_DIR / folder / file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")
    stream = read(str(path))
    data = np.array([tr.data for tr in stream], dtype=float)
    freqs, amplitude = trace_spectra(data, 1.0 / float(stream[0].stats.delta))
    return SpectrumResponse(
        freqs=np.round(freqs, 3).tolist(), amplitude=np.round(amplitude, 3).tolist()
    )
