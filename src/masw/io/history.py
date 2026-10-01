"""A window's seismic or petrophysical inversion done again by hand in PAC replaces all it had
of it: the assistant's earlier attempts (their archived results, the QC log's lines of them) are
erased with sigpipe's `forget`. A curve changed in PAC keeps the assistant's picks and checks of the window, its
history (sigpipe's masw.runs.origin tells whose each curve is now), and erases what was made of
the older curve, which the new one makes wrong: the seismic inversion that inverted it and, the
fundamental mode's, the soil column."""

import logging
import threading
from pathlib import Path

from sigpipe.base.dispersion_curve import Mode
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.masw.picks import save_picks_figures
from sigpipe.masw.runs import window_folders
from sigpipe.masw.runs.history import Stage, forget

logger = logging.getLogger(__name__)


def redone(window: Path, stage: Stage) -> None:
    """`window`'s `stage` done again by hand, its new result in place: its earlier attempts at
    it forgotten, with their archived results and the checks the assistant made of them."""
    forget(window.parent, window.name, stage, later=False, results=False)


def curve_changed(window: Path, mode: Mode) -> None:
    """`window`'s curve of `mode` picked again or deleted in PAC (by hand, or by PAC's automatic
    picking): what was made of the older curve forgotten, the seismic inversion that inverted
    that mode and the soil column (made of the fundamental mode); the assistant's picks and
    checks of the window kept, its history."""
    run_folder, unit = window.parent, window.name
    if _inverted(window, mode):
        forget(run_folder, unit, "inversion", later=False)
    if mode.number == 0 and any(window.glob("PetroInversion_*")):
        forget(run_folder, unit, "petro_inversion", later=False)
    redraw_picks(run_folder, mode)


# One redraw of a run's picks figures at a time.
_PICKS_FIGURES = threading.Lock()


def redraw_picks(run_folder: Path, mode: Mode) -> threading.Thread:
    """The run's pseudo-sections of `mode`'s picked curves drawn again (sigpipe's
    save_picks_figures), in a thread: the pick's answer does not wait for them."""

    def draw() -> None:
        with _PICKS_FIGURES:
            try:
                save_picks_figures(run_folder, window_folders(run_folder), [mode])
            except Exception:
                logger.exception("Could not draw the picks' figures of %s", run_folder)

    thread = threading.Thread(target=draw, daemon=True)
    thread.start()
    return thread


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
