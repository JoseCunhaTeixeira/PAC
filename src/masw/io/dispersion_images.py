import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from masw.io.changes import record_pick
from masw.io.folders import get_xmid_folders
from masw.io.history import curve_changed, writing
from masw.io.paths import output_folder
from masw.io.pick_origin import pick_origin
from masw.io.quality.history import Maker, picked
from masw.io.quality.log import read_log
from masw.io.quality.view import How
from sigpipe.algorithms.picking.dispersion.curve import pick_curves
from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso as lasso
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurves, Mode
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import (
    PseudoSection,
    load_curves,
    pseudo_section,
    remove_pick,
    save_curves,
    save_pick,
)
from sigpipe.masw.runs import load_image
from sigpipe.masw.runs.finding import IMAGE_FILE
from sigpipe.masw.runs.history import set_aside
from sigpipe.masw.runs.origin import Origin, mark_auto, mark_edited


def xmid_folder(folder: str, xmid: float) -> Path:
    return output_folder(folder) / f"xmid_{xmid:.2f}"


def load_dispersion_image(folder: str, xmid: float) -> DispersionImage:
    """The window's dispersion image, with its picked curves."""
    path = xmid_folder(folder, xmid)
    if not (path / IMAGE_FILE).exists():
        raise ValueError(f"No dispersion image for folder={folder}, xmid={xmid}")
    return replace(load_image(path), dispersion_curves=load_curves(path))


def pick_lasso(
    folder: str,
    xmid: float,
    polygon: Sequence[tuple[float, float]],
    label: str,
) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    mode = Mode.from_label(label)
    updated = lasso(image, polygon, mode)
    window = xmid_folder(folder, xmid)
    with writing(output_folder(folder)):
        aside = set_aside(window, "picking", copy=True)
        save_curves(window, image, updated.dispersion_curves)
        inputs = {"polygon": [[float(f), float(v)] for f, v in polygon]}
        _edited(window, mode, "lasso", inputs, updated.dispersion_curves, aside)
    return updated


def pick_box(
    folder: str,
    xmid: float,
    fmin: float | None,
    fmax: float | None,
    vmin: float | None,
    vmax: float | None,
    lbdmin: float | None,
    lbdmax: float | None,
    label: str,
) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    mode = Mode.from_label(label)
    # A mode picked again replaces its curve.
    updated = pick_curves(
        image,
        fmins=[fmin],
        fmaxs=[fmax],
        vmins=[vmin],
        vmaxs=[vmax],
        lbdmins=[lbdmin],
        lbdmaxs=[lbdmax],
        labels=[mode.wave],
        modes=[mode.number],
        resample_over_wavelength=True,
    )
    window = xmid_folder(folder, xmid)
    box = {
        "fmin": fmin,
        "fmax": fmax,
        "vmin": vmin,
        "vmax": vmax,
        "lbdmin": lbdmin,
        "lbdmax": lbdmax,
    }
    with writing(output_folder(folder)):
        aside = set_aside(window, "picking", copy=True)
        save_curves(window, image, updated.dispersion_curves)
        _edited(window, mode, "box", box, updated.dispersion_curves, aside)
    return updated


def delete_curve(folder: str, xmid: float, label: str) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    mode = Mode.from_label(label)
    window = xmid_folder(folder, xmid)
    with writing(output_folder(folder)):
        aside = set_aside(window, "picking", copy=True)
        try:
            remaining = remove_pick(window, image, mode)
        except ValueError:
            if aside is not None:  # no such curve: nothing replaced, nothing to keep
                shutil.rmtree(window / aside)
            raise
        _edited(window, mode, "delete", {}, remaining, aside)
    return replace(image, dispersion_curves=remaining)


def _edited(
    window: Path,
    mode: Mode,
    how: str,
    inputs: Mapping[str, Any],
    curves: DispersionCurves | None,
    aside: str | None,
) -> None:
    """`window`'s curve of `mode` changed by hand (`how`, with `inputs`; `curves`: the window's
    now, `aside`: where its curves before were set aside): said so, what was made of the older
    curve forgotten (see masw.io.history), and the change recorded in its history."""
    mark_edited(window, mode)
    later = curve_changed(window, mode)
    record_pick(window, how, mode, inputs, _curve_of(curves, mode), aside, later)


def _curve_of(curves: DispersionCurves | None, mode: Mode) -> DispersionCurve | None:
    return next((curve for curve in curves or () if curve.mode == mode), None)


def picked_by(folder: str, xmid: float) -> dict[str, tuple[Maker, How | None]]:
    """Who picked each of the window's curves and how, by its mode's label (quality.history)."""
    run_folder = output_folder(folder)
    return picked(run_folder, xmid_folder(folder, xmid).name, read_log(run_folder))


def list_labels(folder: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for xmid in get_xmid_folders(folder):
        for curve in load_curves(xmid_folder(folder, xmid)) or ():
            counts[curve.mode.label] = counts.get(curve.mode.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: Mode.from_label(item[0])))


def list_labels_by_position(folder: str) -> list[tuple[float, list[str], Origin | None]]:
    """Each window's picked labels, and who picked them: PACo automatically, or by hand."""
    log = read_log(output_folder(folder))
    found: list[tuple[float, list[str], Origin | None]] = []
    for xmid in get_xmid_folders(folder):
        window = xmid_folder(folder, xmid)
        labels = sorted(
            (curve.mode.label for curve in load_curves(window) or ()), key=Mode.from_label
        )
        found.append(
            (xmid, labels, pick_origin(window, log, [Mode.from_label(one) for one in labels]))
        )
    return found


def get_pseudo_section(folder: str, label: str) -> PseudoSection:
    xmids = get_xmid_folders(folder)
    if not xmids:
        raise ValueError(f"No xmid positions found in folder={folder}")

    mode = Mode.from_label(label)
    curves: list[DispersionCurve | None] = [
        next(
            (one for one in load_curves(xmid_folder(folder, xmid)) or () if one.mode == mode), None
        )
        for xmid in xmids
    ]
    if not any(curve is not None for curve in curves):
        raise ValueError(f"No curve labelled '{label}' found in folder={folder}")
    return pseudo_section(xmids, curves)


class NoCurveFound(ValueError):
    """The automatic picking found no ridge to follow in a window."""


def auto_pick_m0(folder: str, xmid: float) -> DispersionImage:
    """The window's M0 picked automatically, as PACo picks it (sigpipe's tracking picker, its
    defaults), replacing any M0 it holds, its other modes kept; the window marked picked
    automatically, and what the old M0 led to forgotten (as a hand edit does). Raises
    NoCurveFound when the picker finds none: the window's curves stay as they are."""
    window = xmid_folder(folder, xmid)
    if not (window / IMAGE_FILE).exists():
        raise ValueError(f"No dispersion image for folder={folder}, xmid={xmid}")
    image = load_image(window)
    modes = pick_modes(image, PickingParameters())
    curve = modes[0].curve if modes else None
    if curve is None:
        raise NoCurveFound(f"The automatic picking found no fundamental mode at xmid {xmid:g} m.")
    with writing(output_folder(folder)):
        aside = set_aside(window, "picking", copy=True)
        save_pick(window, image, curve)
        later = curve_changed(window, curve.mode)
        mark_auto(window)
        picking = {"picking": PickingParameters().model_dump(mode="json")}
        record_pick(window, "auto", curve.mode, picking, curve, aside, later)
    return load_dispersion_image(folder, xmid)
