"""The shots a window stacks, and every other shot of the line with why the window leaves it out:
rejected by the assistant's signal check (G1), failed to preprocess, inside the window, too near
or too far from its middle, or left with too few receivers once its bad traces were left out.
From the window's window.json (the records it stacks, as the run built it), the run's manifest
(its exclusions and distances) and, for a run the assistant made, its log (the reasons)."""

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import (
    WINDOW_FILE,
    Line,
    folder_path,
    line_geometry,
    preset_stage,
    read_manifest,
    read_window,
    shot_distances,
)
from masw.io.quality.log import LINE, QCLog, line_receivers, read_log
from masw.io.quality.view import Sentence, flag_text, number, plural, span
from sigpipe.masw.runs import RunManifest, window_folders
from sigpipe.masw.windows import MASWWindow

# How a window uses a shot: stacks it (with all its receivers, or with some left out), or not,
# and why.
type Use = Literal["used", "part", "excluded", "failed", "inside", "near", "far", "traces"]


class Shot(BaseModel):
    """A shot of the line, as the selected window uses it."""

    model_config = ConfigDict(frozen=True)

    name: str  # its record's file name
    x: float
    use: Use
    why: str
    receivers: int | None = None  # of the window's receivers, those it gives: when stacked


class WindowSources(BaseModel):
    """What the window's image stacks."""

    model_config = ConfigDict(frozen=True)

    key: str
    xmid: float
    first: float  # m, its first and last receivers
    last: float
    receivers: int
    passive: bool  # every record serves every window: no shot to choose
    stacked: int  # the records it stacks
    shots: tuple[Shot, ...]  # every shot of the line, by position
    sentences: tuple[Sentence, ...]


def window_sources(folder: str, xmid: float) -> WindowSources:
    run_folder = folder_path(folder)
    unit = f"xmid_{xmid:.2f}"
    window = read_window(run_folder / unit)
    manifest = read_manifest(run_folder)
    if window is None or manifest is None:
        raise ValueError(f"No window.json or run.json for folder={folder}, xmid={xmid}")
    line = line_geometry(run_folder, manifest)
    receivers = [
        line.receivers[index] for index in window.receiver_indices if index < len(line.receivers)
    ]
    first, last = (min(receivers), max(receivers)) if receivers else (xmid, xmid)
    passive = manifest.profile.kind == "passive"
    log = read_log(run_folder)
    shots = () if passive else _shots(manifest, window, line, log, first, last)
    return WindowSources(
        key=unit,
        xmid=xmid,
        first=first,
        last=last,
        receivers=len(window.receiver_indices),
        passive=passive,
        stacked=len(window.selected_files),
        shots=shots,
        sentences=_sentences(manifest, window, shots, line, log),
    )


def _shots(
    manifest: RunManifest,
    window: MASWWindow,
    line: Line,
    log: QCLog | None,
    first: float,
    last: float,
) -> tuple[Shot, ...]:
    masw = preset_stage(manifest, "masw")
    near, farthest = shot_distances(masw)
    far = math.inf if farthest is None else farthest
    stacked = {path.name: index for index, path in enumerate(window.selected_files)}
    width = len(window.receiver_indices)
    # The receivers left out of every window are the line's, not a shot's.
    line_out, _ = line_receivers(log)
    field = near_field(log)
    shots: list[Shot] = []
    for record in manifest.records:
        x = line.sources.get(record.name)
        if x is None:
            continue
        distance = abs(x - window.xmid)
        away = f"{number(distance, 4)} m from the middle"
        edge = max(first - x, x - last)  # from the nearest receiver
        in_field = field is not None and 0 < edge < field
        if record.name in stacked:
            own = window.record_receivers[stacked[record.name]] if window.record_receivers else None
            given = len(own) if own is not None else width
            dropped = sorted(set(window.receiver_indices) - set(own or ()) - line_out)
            if own is not None and dropped:
                why = f"stacked with {given} of the {width} receivers, {away}: " + _traces_text(
                    dropped, line, log, record.name
                )
                shots.append(Shot(name=record.name, x=x, use="part", why=why, receivers=given))
            else:
                why = f"stacked, {away}" + (
                    ": in its near field, the window has no farther shot" if in_field else ""
                )
                shots.append(Shot(name=record.name, x=x, use="used", why=why, receivers=given))
        elif record.name in manifest.exclusions.records:
            why = "left out of every window by the signal check (G1)"
            reasons = _record_reasons(log, record.name)
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="excluded",
                    why=why + (f": {reasons}" if reasons else ""),
                )
            )
        elif record.status == "failed":
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="failed",
                    why=f"failed to preprocess ({record.error or 'no error saved'})",
                )
            )
        elif first < x < last:
            shots.append(Shot(name=record.name, x=x, use="inside", why="inside the window"))
        elif in_field and field is not None:
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="near",
                    why=f"{number(edge, 4)} m from the window's nearest receiver: in its near "
                    f"field (nearer than {number(field, 4)} m, half the longest wavelength)",
                )
            )
        elif distance <= near:
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="near",
                    why=f"{away}: within the {number(near, 4)} m kept clear",
                )
            )
        elif distance >= far:
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="far",
                    why=f"{away}: beyond the {number(far, 4)} m the windows stack",
                )
            )
        else:
            shots.append(
                Shot(
                    name=record.name,
                    x=x,
                    use="traces",
                    why=f"{away}, but its traces left out leave the window too few receivers",
                )
            )
    return tuple(sorted(shots, key=lambda shot: shot.x))


def _sentences(
    manifest: RunManifest,
    window: MASWWindow,
    shots: tuple[Shot, ...],
    line: Line,
    log: QCLog | None,
) -> tuple[Sentence, ...]:
    if manifest.profile.kind == "passive":
        return (
            Sentence(
                mark="info",
                text=f"Its image stacks the cross-correlations of "
                f"{plural(len(window.selected_files), 'record')}: every record of a passive line "
                "serves every window.",
            ),
        )
    used = [shot for shot in shots if shot.use in ("used", "part")]
    what = "correlation gathers" if manifest.preset.mode == "passive-active" else "images"
    said: list[Sentence] = []
    if used:
        distances = [abs(shot.x - window.xmid) for shot in used]
        left = sum(shot.x < window.xmid for shot in used)
        sides = ", ".join(
            part
            for part in (
                f"{left} on the left" if left else "",
                f"{len(used) - left} on the right" if len(used) > left else "",
            )
            if part
        )
        said.append(
            Sentence(
                mark="info",
                text=f"Its dispersion image stacks the {what} of {plural(len(used), 'shot')}, "
                f"{span(min(distances), max(distances), 'm', digits=4)} from its middle: {sides}.",
            )
        )
    line_out, _ = line_receivers(log)
    out = [index for index in window.receiver_indices if index in line_out]
    if out:
        where = ", ".join(
            f"{number(line.receivers[index], 4)} m" for index in out if index < len(line.receivers)
        )
        its = "Its receiver" if len(out) == 1 else "Its receivers"
        said.append(
            Sentence(
                mark="info",
                text=f"{its} at {where} {'is' if len(out) == 1 else 'are'} left out, as of every "
                "window: in most of the line's records, too weak or too strong for "
                f"{'its' if len(out) == 1 else 'their'} distance from the shot (a bad geophone).",
            )
        )
    part = [shot for shot in used if shot.use == "part"]
    if part:
        gives = "gives" if len(part) == 1 else "give"
        said.append(
            Sentence(
                mark="info",
                text=f"{plural(len(part), 'shot')} {gives} only some of its "
                f"{len(window.receiver_indices)} receivers: bad traces left out.",
            )
        )
    near, far = shot_distances(preset_stage(manifest, "masw"))
    reasons = {
        "excluded": "rejected by the signal check",
        "failed": "failed to preprocess",
        "far": f"beyond the {number(far, 4)} m reach" if far is not None else "beyond the reach",
        "near": (
            f"in the near field (nearer than {number(field, 4)} m to the window)"
            if (field := near_field(log)) is not None
            else f"within {number(near, 4)} m"
        ),
        "inside": "inside the window",
        "traces": "left with too few receivers",
    }
    # In short: how many for each reason; the reasons in full, and the shots named, on hover.
    short = {
        "excluded": "rejected (G1)",
        "failed": "failed",
        "far": "too far",
        "near": "too near",
        "inside": "inside",
        "traces": "too few receivers",
    }
    left_out: list[str] = []
    detail: list[str] = []
    for use, reason in reasons.items():
        names = [shot.name for shot in shots if shot.use == use]
        if names:
            listed = f" ({', '.join(names)})" if use in ("excluded", "failed") else ""
            left_out.append(f"{len(names)} {short[use]}")
            detail.append(f"{len(names)} {reason}{listed}")
    if left_out:
        excluded = any(shot.use in ("excluded", "failed") for shot in shots)
        said.append(
            Sentence(
                mark="warn" if excluded else "info",
                text=f"Shots left out: {', '.join(left_out)}.",
                detail=f"Shots left out: {', '.join(detail)}.",
            )
        )
    elif not used:
        said.append(Sentence(mark="fail", text="No shot stacked."))
    return tuple(said)


def near_field(log: QCLog | None) -> float | None:
    """How near a window's receivers the assistant let no shot stand, where the window has a
    farther one: half the longest wavelength of the line's trial curves; None when it set no
    such distance."""
    line = log.latest(LINE, "phase_shift") if log is not None else None
    field = line.parameters.get("near_field") if line is not None else None
    distance = cast(dict[str, object], field).get("distance_m") if isinstance(field, dict) else None
    return float(distance) if isinstance(distance, int | float) else None


def _record_reasons(log: QCLog | None, name: str) -> str:
    """The flags G1 left record `name` out for, but the spent budget."""
    result = log.result(name, "preprocessing", "G1") if log is not None else None
    if result is None:
        return ""
    names = dict.fromkeys(flag.name for flag in result.flags if flag.name != "budget_spent")
    return ", ".join(flag_text(name) for name in names)


def _traces_text(dropped: list[int], line: Line, log: QCLog | None, record: str) -> str:
    """The receivers a record's traces were left out at, with the flag that left each out."""
    why: dict[int, str] = {}
    for flag in log.raised(record, "preprocessing") if log is not None else ():
        if flag.action.kind == "exclude_traces":
            for trace in flag.action.traces or ():
                why.setdefault(trace, flag_text(flag.name))
    said = [
        f"{number(line.receivers[index], 4)} m" if index < len(line.receivers) else f"#{index}"
        for index in dropped
    ]
    reasons = dict.fromkeys(why[index] for index in dropped if index in why)
    return f"{'its trace' if len(said) == 1 else 'its traces'} at {', '.join(said)} left out" + (
        f" ({', '.join(reasons)})" if reasons else ""
    )


def stacking_windows(run_folder: Path) -> dict[str, tuple[str, ...]]:
    """The windows that stack each record, by its file name, from their window.json."""
    units = window_folders(run_folder)
    files = [run_folder / unit / WINDOW_FILE for unit in units]
    stamp = max((path.stat().st_mtime_ns for path in files if path.exists()), default=0)
    return _stacking(str(run_folder), stamp)


@lru_cache(maxsize=8)
def _stacking(run_folder: str, stamp: int) -> dict[str, tuple[str, ...]]:
    """The windows of each record, kept while no window.json changes (`stamp`, the latest)."""
    del stamp
    found: dict[str, list[str]] = {}
    for unit in window_folders(Path(run_folder)):
        window = read_window(Path(run_folder) / unit)
        for path in window.selected_files if window is not None else ():
            found.setdefault(path.name, []).append(unit)
    return {name: tuple(units) for name, units in found.items()}
