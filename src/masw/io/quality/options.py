"""An option of a run's comparisons, its dispersion images as the run's card shows them, small,
at the windows it was judged on: the autonomous mode's search's thumbnails of a finalist, or those
the study kept of a trial (PACo's tidy), or, for the option the run was made with when none were
kept, the run's own windows' there; a workflow trial's (a window length, a mute), its sample
windows' nearest the places along the line the search shows its own at. Each made as PACo makes
its sheets'."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import folder_path, read_manifest
from masw.io.quality.journal import (
    EXPERIMENT_IMAGES,
    EXPERIMENTS,
    SEARCH_IMAGES,
    ExperimentDone,
    study_of,
)
from masw.io.quality.log import LENGTH_TRIALS, MUTE_TRIALS
from masw.io.quality.tried import LENGTH_OPTION, MUTE_OPTION
from sigpipe.masw.runs import window_folders, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE


class OptionImage(BaseModel):
    """A window's dispersion image, small, as the agent compared it, no pick on it: where it is;
    its image, each frequency scaled from its least value to its largest, 0 to 255 (rows the
    velocities from the lowest, columns the frequencies); the frequencies and velocities it
    spans."""

    model_config = ConfigDict(frozen=True)

    xmid: float
    image: list[list[int]]
    extent: tuple[float, float, float, float]  # fmin, fmax, vmin, vmax


def option_images(folder: str, trial: str, line: bool) -> tuple[OptionImage, ...]:
    """The dispersion images of the option `trial` names in the study of run `folder` (a search's
    variant by its label, or a trial by its key) or among its workflow trials (a window length, a
    mute), in order along the line; `line`: the option the run was made with, its own windows'
    when the study kept none. None for a run of no study and of no such trial."""
    from paco.autonomous.tools.grid import read_thumbs, thumbnail  # PACo's sheets'

    run_folder = folder_path(folder)
    tried = _trial_folder(run_folder, trial)
    if tried is not None:
        return tuple(
            OptionImage(xmid=one.xmid, image=one.image.tolist(), extent=one.extent)
            for one in (
                thumbnail(tried / name, xmid_of(name)) for name in _along(run_folder, tried)
            )
        )
    study = study_of(run_folder.parent, run_folder.name)
    if study is None:
        return ()
    kept = study / EXPERIMENT_IMAGES / f"{trial}.npz"
    searched = study / SEARCH_IMAGES / f"{trial}.npz"
    if searched.exists():
        thumbs = read_thumbs(searched)
    elif kept.exists():
        thumbs = read_thumbs(kept)
    elif line:
        names = {round(xmid_of(name), 2): name for name in window_folders(run_folder)}
        found = [names.get(round(xmid, 2)) for xmid in _windows(study, trial)]
        thumbs = [
            thumbnail(run_folder / name, xmid_of(name))
            for name in found
            if name is not None and (run_folder / name / IMAGE_FILE).exists()
        ]
    else:
        thumbs = []
    return tuple(
        OptionImage(xmid=one.xmid, image=one.image.tolist(), extent=one.extent) for one in thumbs
    )


def _trial_folder(run_folder: Path, trial: str) -> Path | None:
    """The windows of the workflow trial `trial` names (length-7, mute-standard); None for no
    such trial."""
    if trial.startswith(LENGTH_OPTION) and trial.removeprefix(LENGTH_OPTION).isdigit():
        found = run_folder / LENGTH_TRIALS / trial.removeprefix(LENGTH_OPTION)
    elif trial.startswith(MUTE_OPTION) and trial.removeprefix(MUTE_OPTION).isalpha():
        found = run_folder / MUTE_TRIALS / trial.removeprefix(MUTE_OPTION) / "windows"
    else:
        return None
    return found if found.is_dir() else None


def _along(run_folder: Path, tried: Path) -> list[str]:
    """The trial's windows nearest the places along the line the search shows its own at, each
    once, in order along it, those with an image."""
    from paco.autonomous.tools.grid import PLACES  # PACo's sheets'

    manifest = read_manifest(run_folder)
    names = [name for name in window_folders(tried) if (tried / name / IMAGE_FILE).exists()]
    if manifest is None or not names:
        return []
    first, last = manifest.profile.receiver_x_range_m
    places = [first + (last - first) * (2 * i + 1) / (2 * PLACES) for i in range(PLACES)]
    nearest = {min(names, key=lambda name: abs(xmid_of(name) - place)) for place in places}
    return sorted(nearest, key=xmid_of)


def _windows(study: Path, trial: str) -> tuple[float, ...]:
    """The windows the study's experiments tried `trial` on; none when none did."""
    for path in sorted((study / EXPERIMENTS).glob("E*.json")):
        try:
            found = ExperimentDone.model_validate_json(path.read_text())
        except ValueError:
            continue
        for one in found.tried:
            if one.trial == trial:
                return one.scorecard.windows
    return ()
