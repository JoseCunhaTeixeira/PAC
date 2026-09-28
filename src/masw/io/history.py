"""A window's stage done again by hand in PAC replaces all it had of it: the assistant's earlier
attempts (their archived results, the QC log's lines of them) are erased with sigpipe's `forget`,
and so is what was made of an old result the new one makes wrong: changing a curve erases the
seismic inversion that inverted it and, the fundamental mode's, the soil column."""

from pathlib import Path

from sigpipe.base.dispersion_curve import Mode
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs.history import Stage, forget


def redone(window: Path, stage: Stage) -> None:
    """`window`'s `stage` done again by hand, its new result in place: its earlier attempts at
    it forgotten, with their archived results and the checks the assistant made of them."""
    forget(window.parent, window.name, stage, later=False, results=False)


def curve_changed(window: Path, mode: Mode) -> None:
    """`window`'s curve of `mode` picked again or deleted by hand: its earlier picks forgotten,
    with the picking's other results (the assistant's check of its pick), and what was made of
    the old curve: the seismic inversion that inverted that mode, the soil column (made of the
    fundamental mode)."""
    run_folder, unit = window.parent, window.name
    forget(run_folder, unit, "picking", later=False, keep=(CURVES_FILE,))
    if _inverted(window, mode):
        forget(run_folder, unit, "inversion", later=False)
    if mode.number == 0 and any(window.glob("PetroInversion_*")):
        forget(run_folder, unit, "petro_inversion", later=False)


def _inverted(window: Path, mode: Mode) -> bool:
    """Whether `window`'s seismic inversion inverted `mode`'s curve: its models' curves hold that
    mode (the forward model labels them by number); an inversion whose curves cannot tell, as if
    it did."""
    if not any(window.glob("SeismicInversion_*")):
        return False
    for path in sorted(window.glob("SeismicInversion_DispersionCurves_0000_*.csv")):
        try:
            (curves,) = load_dispersion_curves([path])
        except OSError, ValueError:
            continue
        return any(curve.mode.number == mode.number for curve in curves)
    return True
