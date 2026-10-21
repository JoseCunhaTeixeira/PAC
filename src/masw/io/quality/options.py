"""An option of the autonomous mode's comparisons, its dispersion images as the run's card shows
them, small, at the windows it was tried on: the thumbnails the study kept of its trial (PACo's
tidy), or, for the option the run was made with when none were kept, the run's own windows' there,
made as PACo makes its sheets'."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from masw.io.quality.files import folder_path
from masw.io.quality.journal import EXPERIMENT_IMAGES, EXPERIMENTS, ExperimentDone, study_of
from sigpipe.masw.runs import window_folders, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE


class OptionImage(BaseModel):
    """A window's dispersion image, small: where it is; its image, each frequency scaled from its
    least value to its largest, 0 to 255 (rows the velocities from the lowest, columns the
    frequencies); the frequencies and velocities it spans; its picked curve."""

    model_config = ConfigDict(frozen=True)

    xmid: float
    image: list[list[int]]
    extent: tuple[float, float, float, float]  # fmin, fmax, vmin, vmax
    pick_fs: list[float]
    pick_vs: list[float]


def option_images(folder: str, trial: str, line: bool) -> tuple[OptionImage, ...]:
    """The dispersion images of the option tried as `trial` in the study of run `folder`, in order
    along the line; `line`: the option the run was made with, its own windows' when the study kept
    none. None for a run of no study."""
    from paco.autonomous.tools.grid import read_thumbs, thumbnail  # PACo's sheets'

    run_folder = folder_path(folder)
    study = study_of(run_folder.parent, run_folder.name)
    if study is None:
        return ()
    kept = study / EXPERIMENT_IMAGES / f"{trial}.npz"
    if kept.exists():
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
        OptionImage(
            xmid=one.xmid,
            image=one.image.tolist(),
            extent=one.extent,
            pick_fs=one.pick[0].tolist(),
            pick_vs=one.pick[1].tolist(),
        )
        for one in thumbs
    )


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
