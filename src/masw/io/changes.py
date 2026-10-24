"""What PAC's pages change in a run's windows, recorded in each window's own history (sigpipe's
runs.history `Change`): a curve picked with the lasso, with a box or automatically, or deleted; a
seismic inversion; a soil column. Each with what it took, a summary of what it made, where what it
replaced was set aside, and the later stages set aside with it. The history PAC's pages show
reads them (masw.io.quality.history)."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from masw.io.quality.petro import MEASURES_FILE as PETRO_MEASURES
from sigpipe.base.dispersion_curve import DispersionCurve, Mode
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import informed_depth, saved_measures
from sigpipe.masw.petro.measuring import PetroMeasures
from sigpipe.masw.runs.history import Change, Stage, record_change


def curve_result(curve: DispersionCurve | None) -> dict[str, Any]:
    """What a pick made, as its history says it: its points, band and wavelengths; empty for
    none (a curve deleted)."""
    if curve is None or len(curve.fs) == 0:
        return {}
    fs = np.asarray(curve.fs, dtype=float)
    lengths = np.asarray(curve.vs, dtype=float) / fs
    return {
        "n_points": int(fs.size),
        "band_hz": [round(float(fs.min()), 3), round(float(fs.max()), 3)],
        "wavelength_m": [round(float(lengths.min()), 2), round(float(lengths.max()), 2)],
    }


def record_pick(
    window: Path,
    how: str,
    mode: Mode,
    inputs: Mapping[str, Any],
    curve: DispersionCurve | None,
    aside: str | None,
    later: Sequence[Stage],
) -> None:
    """A curve of `mode` picked in PAC (`how`: "lasso", "box", "auto") or deleted ("delete"),
    with `inputs`, giving `curve` (None: deleted), recorded in `window`'s history."""
    record_change(
        window,
        Change(
            at=datetime.now(UTC),
            stage="picking",
            how=how,
            mode=mode.label,
            inputs=dict(inputs),
            result=curve_result(curve),
            aside=aside,
            later=tuple(later),
        ),
    )


def record_inversion(
    window: Path, labels: Sequence[str], parameters: InversionParameters, aside: str | None
) -> None:
    """A seismic inversion of `window`'s curves of `labels` with `parameters`, made in PAC's
    page, recorded in its history with its model's summary."""
    record_change(
        window,
        Change(
            at=datetime.now(UTC),
            stage="inversion",
            how="inversion",
            inputs={"modes": list(labels), "parameters": parameters.model_dump(mode="json")},
            result=model_result(window),
            aside=aside,
        ),
    )


def model_result(window: Path) -> dict[str, Any]:
    """A model's summary, as its history says it: its layers, the depth it was modelled to and
    informed to, its worst misfit; empty without its measures."""
    measures = saved_measures(window)
    if measures is None:
        return {}
    informed = informed_depth(measures)
    misfits = [band.misfit for band in measures.fits[0].bands if band.misfit is not None]
    return {
        "layers": len(measures.vs_layers),
        "depth_m": round(measures.depth_max_m, 2),
        "informed_m": round(informed, 2) if informed is not None else None,
        "misfit": round(max(misfits), 3) if misfits else None,
    }


def record_soils(window: Path, model_name: str, aside: str | None) -> None:
    """A soil column of `window` made with Silex model `model_name` in PAC's page, recorded in
    its history with its summary."""
    record_change(
        window,
        Change(
            at=datetime.now(UTC),
            stage="petro_inversion",
            how="petro",
            inputs={"model": model_name},
            result=soils_result(window),
            aside=aside,
        ),
    )


def soils_result(window: Path) -> dict[str, Any]:
    """A soil column's summary: its soils, top down, and the water table's depth; empty
    without its measures."""
    path = window / PETRO_MEASURES
    if not path.exists():
        return {}
    measures = PetroMeasures.model_validate_json(path.read_text())
    return {"soils": list(measures.soils), "water_table_m": round(measures.water_table_m, 2)}
