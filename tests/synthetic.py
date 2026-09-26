"""Synthetic profiles for the MASW layer, written as PAC keeps them: records (MiniSEED here), a
receiver_positions.yaml, and for shots a source_positions.yaml."""

from pathlib import Path

import numpy as np
import yaml
from obspy import Stream as ObspyStream
from obspy import Trace, UTCDateTime

SAMPLING = 1000.0  # Hz
N_RECEIVERS = 12
SPACING = 1.0  # m
VELOCITY = 200.0  # m/s, the surface wave of the shots
FREQUENCY = 20.0  # Hz, the wavelet's centre
SOURCES = {"1.mseed": -2.0, "2.mseed": 13.0}  # m, before the line and past it


def _write(path: Path, xt: np.ndarray) -> None:
    traces = [
        Trace(
            data=row.astype(np.float32),
            header={"sampling_rate": SAMPLING, "starttime": UTCDateTime(0), "station": f"R{i:02d}"},
        )
        for i, row in enumerate(xt)
    ]
    ObspyStream(traces).write(str(path), format="MSEED")


def _shot(source_x: float, rng: np.random.Generator) -> np.ndarray:
    """A causal wavelet moving out at VELOCITY from `source_x`, over weak noise, 1 s long."""
    ts = np.arange(int(SAMPLING)) / SAMPLING
    offsets = np.abs(np.arange(N_RECEIVERS) * SPACING - source_x)
    xt = rng.standard_normal((N_RECEIVERS, ts.size)) * 0.01
    for i, offset in enumerate(offsets):
        tau = ts - offset / VELOCITY
        pulse = np.where(
            tau >= 0, tau * np.exp(-2 * FREQUENCY * tau) * np.cos(2 * np.pi * FREQUENCY * tau), 0.0
        )
        xt[i] += pulse / (np.exp(-1) / (2 * FREQUENCY)) / np.sqrt(offset)
    return xt


def write_profiles(root: Path) -> None:
    """Two profiles in `root`: `shots` (active: two shots, one at each end) and `noise` (passive:
    4 s of ambient noise)."""
    rng = np.random.default_rng(0)
    receivers = [{"x": i * SPACING, "z": 0.0} for i in range(N_RECEIVERS)]

    shots = root / "shots"
    shots.mkdir()
    for name, x in SOURCES.items():
        _write(shots / name, _shot(x, rng))
    (shots / "receiver_positions.yaml").write_text(yaml.safe_dump(receivers))
    (shots / "source_positions.yaml").write_text(
        yaml.safe_dump({name: {"x": x, "z": 0.0} for name, x in SOURCES.items()})
    )

    noise = root / "noise"
    noise.mkdir()
    _write(noise / "1.mseed", rng.standard_normal((N_RECEIVERS, int(4 * SAMPLING))))
    (noise / "receiver_positions.yaml").write_text(yaml.safe_dump(receivers))
