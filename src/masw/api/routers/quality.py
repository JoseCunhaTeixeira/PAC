"""What Visualization reads of a run: the runs, a run's card (its settings and why, its line),
each stage's units along the line, and the selected unit's card."""

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from masw.io.quality import dispersion, figures, inversion, petro, records, runs, sources
from masw.io.quality.dispersion import DispersionCard
from masw.io.quality.inversion import Chains, FigureName, InversionCard
from masw.io.quality.petro import PetroCard
from masw.io.quality.records import RecordCard, RecordGather, SavedSpectra
from masw.io.quality.runs import ProfileRuns, RunCard
from masw.io.quality.sources import WindowSources
from masw.io.quality.view import Overview
from sigpipe.dataio.selection_plotting import SelectionScores
from sigpipe.masw.inversion.section import DEFAULT_MODEL, ModelName

router = APIRouter(tags=["quality"])


def _found[T](read: Callable[[], T]) -> T:
    """`read()`, a ValueError (no such run, window or record) as a 404."""
    try:
        return read()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/quality/runs")
def get_runs() -> tuple[ProfileRuns, ...]:
    """Every profile, with its runs, newest first."""
    return runs.list_runs()


@router.get("/quality/run/{folder:path}")
def get_run(folder: str) -> RunCard:
    """Who made the run, what its windows and stages ran with and why, and its line."""
    return _found(lambda: runs.run_card(folder))


@router.get("/quality/sources/{folder:path}/{xmid}")
def get_sources(folder: str, xmid: float) -> WindowSources:
    """The shots the window stacks, and why it leaves out the others."""
    return _found(lambda: sources.window_sources(folder, xmid))


@router.get("/quality/records/overview/{folder:path}")
def get_records_overview(folder: str) -> Overview:
    return _found(lambda: records.records_overview(folder))


@router.get("/quality/records/card/{folder:path}/{name}")
def get_record_card(folder: str, name: str) -> RecordCard:
    return _found(lambda: records.record_card(folder, name))


@router.get("/quality/records/gather/{folder:path}/{name}")
def get_record_gather(
    folder: str, name: str, norm: Literal["trace", "global"] = "trace"
) -> RecordGather:
    """The record as the windows used it, each trace at its receiver along the line."""
    return _found(lambda: records.record_gather(folder, name, norm))


@router.get("/quality/records/spectrum/{folder:path}/{name}")
def get_record_spectra(folder: str, name: str) -> SavedSpectra:
    """The record's spectra, preprocessed, as its job saved them."""
    return _found(lambda: records.record_spectra(folder, name))


@router.get("/quality/dispersion/overview/{folder:path}")
def get_dispersion_overview(folder: str) -> Overview:
    return _found(lambda: dispersion.dispersion_overview(folder))


@router.get("/quality/dispersion/card/{folder:path}/{xmid}")
def get_dispersion_card(folder: str, xmid: float) -> DispersionCard:
    return _found(lambda: dispersion.dispersion_card(folder, xmid))


@router.get("/quality/inversion/overview/{folder:path}")
def get_inversion_overview(folder: str) -> Overview:
    return _found(lambda: inversion.inversion_overview(folder))


@router.get("/quality/inversion/card/{folder:path}/{xmid}")
def get_inversion_card(folder: str, xmid: float, model: ModelName = DEFAULT_MODEL) -> InversionCard:
    """The window's model (`model`, against depth), its fit, its chains' convergence, and what
    it ran with."""
    return _found(lambda: inversion.inversion_card(folder, xmid, model))


@router.get("/quality/inversion/chains/{folder:path}/{xmid}")
def get_inversion_chains(folder: str, xmid: float) -> Chains:
    """Each parameter's samples along each chain, and its marginal between the prior's bounds."""
    return _found(lambda: inversion.inversion_chains(folder, xmid))


@router.get("/quality/inversion/figure/{folder:path}/{xmid}/{name}")
def get_inversion_figure(folder: str, xmid: float, name: FigureName) -> FileResponse:
    """A figure the window's inversion saved: its posterior marginals, density curves, or the
    dispersion image with the model's modes."""
    path = _found(lambda: inversion.figure_path(folder, xmid, name))
    return FileResponse(path, media_type="image/png")


@router.get("/quality/run_figures/{folder:path}")
def get_run_figures(folder: str, xmid: float | None = None, record: str | None = None) -> list[str]:
    """The figures the run saved at its root (its sections, its pseudo-section comparisons), in
    its window at `xmid` or in its record `record`, by name."""
    return _found(lambda: figures.run_figures(folder, xmid, record))


@router.get("/quality/run_figure/{folder:path}/{name}")
def get_run_figure(
    folder: str, name: str, xmid: float | None = None, record: str | None = None
) -> FileResponse:
    """One of the figures the run saved (run_figures)."""
    path = _found(lambda: figures.run_figure_path(folder, name, xmid, record))
    return FileResponse(path, media_type="image/png")


@router.get("/quality/dispersion/gather/{folder:path}/{xmid}")
def get_window_gather(
    folder: str, xmid: float, norm: Literal["trace", "global"] = "trace"
) -> RecordGather:
    """The stacked correlations a passive or passive-active window's image was made of, each
    trace at its receiver along the line."""
    return _found(lambda: dispersion.window_gather(folder, xmid, norm))


@router.get("/quality/dispersion/selection/{folder:path}/{xmid}")
def get_window_selection(folder: str, xmid: float) -> SelectionScores:
    """A passive window's fk segment selection, as its job saved it: each segment's f-k ratio,
    kept or not, around the threshold."""
    return _found(lambda: dispersion.window_selection(folder, xmid))


@router.get("/quality/dispersion/spectrum/{folder:path}/{xmid}")
def get_window_spectra(folder: str, xmid: float) -> SavedSpectra:
    """The spectra of a passive or passive-active window's stacked correlations, as its job
    saved them."""
    return _found(lambda: dispersion.window_spectra(folder, xmid))


@router.get("/quality/petro/overview/{folder:path}")
def get_petro_overview(folder: str) -> Overview:
    return _found(lambda: petro.petro_overview(folder))


@router.get("/quality/petro/card/{folder:path}/{xmid}")
def get_petro_card(folder: str, xmid: float) -> PetroCard:
    return _found(lambda: petro.petro_card(folder, xmid))
