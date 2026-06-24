from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path
from scipy.signal import medfilt, savgol_filter
from sigproc.algorithms.picking.dispersion.curve import lorentzian_uncertainty, resample_wavelength
from sigproc.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigproc.base.dispersion_image import DispersionImage


def label_to_mode(label: str) -> Mode:
    wave = label.rstrip("0123456789")
    return Mode(wave, int(label[len(wave) :]))


def mode_to_label(mode: Mode) -> str:
    return f"{mode.wave}{mode.number}"


def pick_curve_lasso(
    dispersion_image: DispersionImage,
    polygon: Sequence[tuple[float, float]],
    label: str = "unknown",
) -> DispersionImage:
    if len(polygon) < 3:
        raise ValueError("polygon must have at least 3 points")

    fs = dispersion_image.fs
    vs = dispersion_image.vs
    fv_map = dispersion_image.fv_map

    poly = np.asarray(polygon, dtype=np.float64).copy()
    # Force the closing edge vertical at the starting frequency, so the implicit
    # segment from the last lasso point back to the first doesn't bias the mask
    # at the low-frequency end.
    poly[-1, 0] = poly[0, 0]

    f_indices = np.clip(np.searchsorted(fs, poly[:, 0]), 0, len(fs) - 1)
    v_indices = np.clip(np.searchsorted(vs, poly[:, 1]), 0, len(vs) - 1)
    f_start_i, f_end_i = int(f_indices.min()), int(f_indices.max())
    v_start_i, v_end_i = int(v_indices.min()), int(v_indices.max())

    if f_end_i <= f_start_i or v_end_i <= v_start_i + 1:
        raise ValueError("lasso selection is too small to pick a curve")

    F, V = np.meshgrid(fs, vs, indexing="ij")
    coords = np.column_stack([F.ravel(), V.ravel()])
    mask = Path(poly).contains_points(coords).reshape(fv_map.shape)
    fv_masked = np.where(mask, fv_map, 0.0)

    f_picked: list[float] = []
    v_picked: list[float] = []
    for row_i in range(f_start_i, f_end_i):
        window = fv_masked[row_i, v_start_i + 1 : v_end_i]
        if not np.any(window):
            continue
        local_max_i = int(np.argmax(window))
        true_v_i = v_start_i + 1 + local_max_i
        # A pick sitting exactly on the window's upper edge is usually an
        # artifact of the mask boundary rather than a real spectral maximum.
        if true_v_i == v_end_i - 1 and v_picked:
            v_picked.append(v_picked[-1])
        else:
            v_picked.append(float(vs[true_v_i]))
        f_picked.append(float(fs[row_i]))

    # Drop the first row: it sits right where the lasso's vertical closing
    # edge was forced, so its pick is the least reliable.
    f_picked_arr = np.asarray(f_picked[1:], dtype=np.float32)
    v_picked_arr = np.asarray(v_picked[1:], dtype=np.float32)

    if len(f_picked_arr) < 2:
        raise ValueError("lasso selection did not cover enough frequency rows to pick a curve")

    if len(v_picked_arr) >= 5:
        median_vs = medfilt(v_picked_arr, kernel_size=5)
        residual = np.abs(v_picked_arr - median_vs)
        threshold = 2.5 * np.median(residual)
        outliers = residual > threshold
        if np.any(outliers):
            valid = ~outliers
            v_picked_arr[outliers] = np.interp(
                f_picked_arr[outliers], f_picked_arr[valid], v_picked_arr[valid]
            )

        wl = (
            len(v_picked_arr) // 2 + 1 if len(v_picked_arr) / 2 % 2 == 0 else len(v_picked_arr) // 2
        )
        v_picked_arr = np.asarray(
            savgol_filter(v_picked_arr, window_length=wl, polyorder=3),
            dtype=np.float32,
        )

    mode = label_to_mode(label)
    new_curve = resample_wavelength(
        DispersionCurve(
            fs=f_picked_arr,
            vs=v_picked_arr,
            mode=mode,
            type=dispersion_image.type,
            acquisition=dispersion_image.acquisition,
            vs_err=lorentzian_uncertainty(f_picked_arr, v_picked_arr, dispersion_image.acquisition),
        )
    )

    existing = (
        list(dispersion_image.dispersion_curves) if dispersion_image.dispersion_curves else []
    )
    existing = [c for c in existing if c.mode != mode]
    existing.append(new_curve)

    return DispersionImage(
        fv_map=dispersion_image.fv_map,
        fs=dispersion_image.fs,
        vs=dispersion_image.vs,
        type=dispersion_image.type,
        acquisition=dispersion_image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=tuple(existing)),
    )
