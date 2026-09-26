from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from masw.io.folders import get_xmid_folders
from masw.io.paths import output_folder
from sigpipe.algorithms.picking.dispersion.curve import pick_curves
from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso as lasso
from sigpipe.base.dispersion_curve import DispersionCurve, Mode
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import PseudoSection, load_curves, pseudo_section, remove_pick, save_curves
from sigpipe.masw.runs import load_image
from sigpipe.masw.runs.finding import IMAGE_FILE


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
    updated = lasso(image, polygon, Mode.from_label(label))
    save_curves(xmid_folder(folder, xmid), image, updated.dispersion_curves)
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
    save_curves(xmid_folder(folder, xmid), image, updated.dispersion_curves)
    return updated


def delete_curve(folder: str, xmid: float, label: str) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    remaining = remove_pick(xmid_folder(folder, xmid), image, Mode.from_label(label))
    return replace(image, dispersion_curves=remaining)


def list_labels(folder: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for xmid in get_xmid_folders(folder):
        for curve in load_curves(xmid_folder(folder, xmid)) or ():
            counts[curve.mode.label] = counts.get(curve.mode.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: Mode.from_label(item[0])))


def list_labels_by_position(folder: str) -> list[tuple[float, list[str]]]:
    return [
        (
            xmid,
            sorted(
                (curve.mode.label for curve in load_curves(xmid_folder(folder, xmid)) or ()),
                key=Mode.from_label,
            ),
        )
        for xmid in get_xmid_folders(folder)
    ]


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
