"""A run's files as the Quality views read them: its folder and manifest (run.json), the values
its preset gave each stage, each window's picked fundamental mode and window.json (the records it
stacks), and the line's geometry: where every receiver and every shot of the profile lies."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from masw.io.paths import INPUT_DIR, output_folder
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.masw.picks import load_curves
from sigpipe.masw.pipelines.common import PREPROCESSED
from sigpipe.masw.profiles.loading import RECEIVER_POSITIONS_FILE, SOURCE_POSITIONS_FILE
from sigpipe.masw.runs import RunManifest
from sigpipe.masw.windows import MASWWindow

MANIFEST_FILE = "run.json"
WINDOW_FILE = "window.json"


def folder_path(folder: str) -> Path:
    """Output folder `folder`; raises ValueError when there is none."""
    path = output_folder(folder)
    if not path.is_dir():
        raise ValueError(f"Output folder not found: {folder}")
    return path


def read_manifest(folder: Path) -> RunManifest | None:
    """The run's manifest; None for a folder of the older layout, which has none."""
    path = folder / MANIFEST_FILE
    return RunManifest.model_validate_json(path.read_text()) if path.exists() else None


def preset_stage(manifest: RunManifest | None, stage: str) -> dict[str, Any]:
    """The values the run's pipelines received for one stage of its preset (masw, dispersion...)."""
    if manifest is None:
        return {}
    values: object = manifest.preset.model_dump(mode="json").get(stage)
    return cast(dict[str, Any], values) if isinstance(values, dict) else {}


def merged(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`base` with `overrides` on it, nested dictionaries merged key by key."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], cast(dict[str, Any], value))
        else:
            result[key] = value
    return result


def fundamental(window: Path) -> DispersionCurve | None:
    """The window's picked fundamental mode (M0, or another wave's mode 0); None when none is
    picked."""
    curves = load_curves(window)
    if curves is None:
        return None
    return next((curve for curve in curves.dispersion_curves if curve.mode.number == 0), None)


def wavelengths(curve: DispersionCurve) -> np.ndarray:
    return np.asarray(curve.vs, dtype=float) / np.asarray(curve.fs, dtype=float)


def read_window(window: Path) -> MASWWindow | None:
    """The records window folder `window` stacks, as the run built it; None without its
    window.json."""
    path = window / WINDOW_FILE
    return MASWWindow.model_validate_json(path.read_text()) if path.exists() else None


@dataclass(frozen=True)
class Line:
    """The profile's geometry along the line."""

    receivers: tuple[float, ...]  # every receiver's x, in order
    sources: dict[str, float]  # each record's source x, by file name; none for a passive line


def line_geometry(run_folder: Path, manifest: RunManifest) -> Line:
    """The line of the run's profile: from its input folder, or when that is gone, from the
    run's preprocessed records (each holds its source and every receiver)."""
    stamp = (run_folder / MANIFEST_FILE).stat().st_mtime_ns
    return _line(str(run_folder), stamp, manifest.model_dump_json())


@lru_cache(maxsize=16)
def _line(run_folder: str, stamp: int, manifest_json: str) -> Line:
    """The line, kept while run.json stays the same (`stamp`, its modification time)."""
    del stamp
    manifest = RunManifest.model_validate_json(manifest_json)
    active = manifest.profile.kind == "active"
    profile = INPUT_DIR / manifest.profile.name
    receivers = _receivers(profile / RECEIVER_POSITIONS_FILE)
    sources = _sources(profile / SOURCE_POSITIONS_FILE)
    if receivers and (sources or not active):
        return Line(receivers, sources)
    streams = [
        (record.name, Path(run_folder) / record.folder / PREPROCESSED)
        for record in manifest.records
        if record.status == "succeeded"
    ]
    loaded = {name: load_stream([path])[0] for name, path in streams if path.exists()}
    if not receivers and loaded:
        first = next(iter(loaded.values()))
        receivers = tuple(float(receiver.x) for receiver in first.acquisition.receivers)
    if active:
        sources = {name: float(stream.acquisition.source.x) for name, stream in loaded.items()}
    return Line(receivers, sources)


def _receivers(path: Path) -> tuple[float, ...]:
    raw: object = yaml.safe_load(path.read_text()) if path.exists() else None
    if not isinstance(raw, list):
        return ()
    found: list[float] = []
    for entry in cast(list[object], raw):
        x = cast(dict[str, object], entry).get("x") if isinstance(entry, dict) else None
        if isinstance(x, int | float):
            found.append(float(x))
    return tuple(found)


def _sources(path: Path) -> dict[str, float]:
    raw: object = yaml.safe_load(path.read_text()) if path.exists() else None
    if not isinstance(raw, dict):
        return {}
    found: dict[str, float] = {}
    for name, position in cast(dict[object, object], raw).items():
        x = cast(dict[str, object], position).get("x") if isinstance(position, dict) else None
        if isinstance(x, int | float):
            found[str(name)] = float(x)
    return found
