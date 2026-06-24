from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sigproc.algorithms.picking.dispersion.curve import min_resolvable_wavelength, pick_curves
from sigproc.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigproc.base.dispersion_image import DispersionImage
from sigproc.dataio.dispersion.loading import load_dispersion_curves
from sigproc.dataio.dispersion.loading import load_dispersion_image as _load_dispersion_image
from sigproc.dataio.dispersion.plotting import plot_dispersion_image
from sigproc.dataio.dispersion.saving import save_dispersion_curves
from sigproc.transformers import Plot

from masw.algorithms.dispersion_picking import label_to_mode, mode_to_label, pick_curve_lasso
from masw.io.folders import get_xmid_folders
from masw.io.paths import OUTPUT_DIR

PSEUDO_SECTION_POINTS = 200


def xmid_folder(folder: str, xmid: float) -> Path:
    return OUTPUT_DIR / folder / f"xmid_{xmid:.2f}"


def _image_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "DispersionImage_0000.hdf5"


def _curves_path(folder: str, xmid: float) -> Path:
    return xmid_folder(folder, xmid) / "DispersionCurves_0000.csv"


def _save_picked_plot(folder: str, xmid: float, image: DispersionImage) -> None:
    lbmin = min_resolvable_wavelength(image.acquisition)
    fig = plot_dispersion_image(image, lbmin=lbmin, normalize=True, show_errorbars=True)
    Plot.savefig(path=xmid_folder(folder, xmid) / "DispersionImage_0000.png", figure=fig)
    plt.close(fig)


def load_dispersion_image(folder: str, xmid: float) -> DispersionImage:
    image_path = _image_path(folder, xmid)
    if not image_path.exists():
        raise ValueError(f"No dispersion image for folder={folder}, xmid={xmid}")
    curves_path = _curves_path(folder, xmid)
    curves_paths = [curves_path] if curves_path.exists() else None
    return _load_dispersion_image([image_path], curves_paths=curves_paths)[0]


def pick_lasso(
    folder: str,
    xmid: float,
    polygon: Sequence[tuple[float, float]],
    label: str,
) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    updated = pick_curve_lasso(image, polygon, label=label)
    assert updated.dispersion_curves is not None
    save_dispersion_curves(updated.dispersion_curves, path=_curves_path(folder, xmid))
    _save_picked_plot(folder, xmid, updated)
    return updated


def _dedupe_curves_by_mode(curves: Iterable[DispersionCurve]) -> DispersionCurvesImage:
    # sigproc's pick_curves appends the new pick without checking for an
    # existing curve with the same mode; keep only the latest curve per
    # mode so re-picking a label replaces it instead of duplicating it.
    by_mode: dict[Mode, DispersionCurve] = {}
    for curve in curves:
        by_mode[curve.mode] = curve
    return DispersionCurvesImage(dispersion_curves=tuple(by_mode.values()))


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
    mode = label_to_mode(label)
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
    assert updated.dispersion_curves is not None
    deduped_curves = _dedupe_curves_by_mode(updated.dispersion_curves)
    updated = DispersionImage(
        fv_map=updated.fv_map,
        fs=updated.fs,
        vs=updated.vs,
        type=updated.type,
        acquisition=updated.acquisition,
        dispersion_curves=deduped_curves,
    )
    save_dispersion_curves(deduped_curves, path=_curves_path(folder, xmid))
    _save_picked_plot(folder, xmid, updated)
    return updated


def delete_curve(folder: str, xmid: float, label: str) -> DispersionImage:
    image = load_dispersion_image(folder, xmid)
    if not image.dispersion_curves:
        raise ValueError(f"No picked curves for folder={folder}, xmid={xmid}")

    mode = label_to_mode(label)
    remaining = tuple(c for c in image.dispersion_curves if c.mode != mode)
    if len(remaining) == len(image.dispersion_curves):
        raise ValueError(f"No curve labelled '{label}' for folder={folder}, xmid={xmid}")

    if remaining:
        save_dispersion_curves(
            DispersionCurvesImage(dispersion_curves=remaining), path=_curves_path(folder, xmid)
        )
    else:
        _curves_path(folder, xmid).unlink(missing_ok=True)

    updated = DispersionImage(
        fv_map=image.fv_map,
        fs=image.fs,
        vs=image.vs,
        type=image.type,
        acquisition=image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=remaining) if remaining else None,
    )
    _save_picked_plot(folder, xmid, updated)
    return updated


def list_labels(folder: str) -> dict[str, int]:
    xmids = get_xmid_folders(folder)
    counts: dict[str, int] = {}
    for xmid in xmids:
        curves_path = _curves_path(folder, xmid)
        if not curves_path.exists():
            continue
        for curve in load_dispersion_curves([curves_path])[0]:
            label = mode_to_label(curve.mode)
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: label_to_mode(item[0])))


def list_labels_by_position(folder: str) -> list[tuple[float, list[str]]]:
    xmids = get_xmid_folders(folder)
    result: list[tuple[float, list[str]]] = []
    for xmid in xmids:
        curves_path = _curves_path(folder, xmid)
        labels = (
            sorted(
                (mode_to_label(curve.mode) for curve in load_dispersion_curves([curves_path])[0]),
                key=label_to_mode,
            )
            if curves_path.exists()
            else []
        )
        result.append((xmid, labels))
    return result


@dataclass(slots=True, frozen=True)
class PseudoSection:
    positions: np.ndarray
    fs_grid: np.ndarray
    velocities_by_frequency: np.ndarray
    lambdas_grid: np.ndarray
    velocities_by_wavelength: np.ndarray


def get_pseudo_section(folder: str, label: str) -> PseudoSection:
    xmids = get_xmid_folders(folder)
    if not xmids:
        raise ValueError(f"No xmid positions found in folder={folder}")

    mode = label_to_mode(label)
    curve_fs: list[np.ndarray | None] = []
    curve_vs: list[np.ndarray | None] = []
    for xmid in xmids:
        curves_path = _curves_path(folder, xmid)
        curve = None
        if curves_path.exists():
            curve = next(
                (c for c in load_dispersion_curves([curves_path])[0] if c.mode == mode),
                None,
            )
        curve_fs.append(curve.fs if curve is not None else None)
        curve_vs.append(curve.vs if curve is not None else None)

    if not any(fs is not None for fs in curve_fs):
        raise ValueError(f"No curve labelled '{label}' found in folder={folder}")

    return build_pseudo_section(xmids, curve_fs, curve_vs)


def build_pseudo_section(
    positions: Sequence[float],
    curve_fs: Sequence[np.ndarray | None],
    curve_vs: Sequence[np.ndarray | None],
) -> PseudoSection:
    fmin = min(float(fs.min()) for fs in curve_fs if fs is not None)
    fmax = max(float(fs.max()) for fs in curve_fs if fs is not None)
    fs_grid = np.linspace(fmin, fmax, PSEUDO_SECTION_POINTS, dtype=np.float32)
    velocities_by_frequency = np.full(
        (len(positions), PSEUDO_SECTION_POINTS), np.nan, dtype=np.float32
    )
    for i, (fs, vs) in enumerate(zip(curve_fs, curve_vs, strict=True)):
        if fs is None or vs is None:
            continue
        mask = (fs_grid >= fs.min()) & (fs_grid <= fs.max())
        velocities_by_frequency[i, mask] = np.interp(fs_grid[mask], fs, vs)

    lambdas = [
        vs / fs if fs is not None and vs is not None else None
        for fs, vs in zip(curve_fs, curve_vs, strict=True)
    ]
    lmin = min(float(lbd.min()) for lbd in lambdas if lbd is not None)
    lmax = max(float(lbd.max()) for lbd in lambdas if lbd is not None)
    lambdas_grid = np.linspace(lmin, lmax, PSEUDO_SECTION_POINTS, dtype=np.float32)
    velocities_by_wavelength = np.full(
        (len(positions), PSEUDO_SECTION_POINTS), np.nan, dtype=np.float32
    )
    for i, (lbd, vs) in enumerate(zip(lambdas, curve_vs, strict=True)):
        if lbd is None or vs is None:
            continue
        order = np.argsort(lbd)
        lbd_sorted = lbd[order]
        vs_sorted = vs[order]
        mask = (lambdas_grid >= lbd_sorted.min()) & (lambdas_grid <= lbd_sorted.max())
        velocities_by_wavelength[i, mask] = np.interp(lambdas_grid[mask], lbd_sorted, vs_sorted)

    return PseudoSection(
        positions=np.asarray(positions, dtype=np.float32),
        fs_grid=fs_grid,
        velocities_by_frequency=velocities_by_frequency,
        lambdas_grid=lambdas_grid,
        velocities_by_wavelength=velocities_by_wavelength,
    )
