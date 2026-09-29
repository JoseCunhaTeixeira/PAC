"""The figures a run saves (PAC's jobs and the assistant draw them), as Visualization lists and
serves them: read, never drawn here. At the run's root, its sections and pseudo-section
comparisons; in a window's folder, its processing's (the stacked correlations, the segments
selected, the dispersion image: the inversion's are its card's); in a record's, its
preprocessed traces."""

from pathlib import Path

from masw.io.quality.files import folder_path, read_manifest

# The figures of a run's root Visualization shows: the picks', the seismic and petrophysical
# inversions'.
RUN_FIGURE_PREFIXES = ("DispersionPicking_", "SeismicInversion_", "PetroInversion_")
# A window's figures Visualization lists (the inversion's are its card's), in the order its steps
# made them: its processing's, then its petrophysical inversion's.
WINDOW_FIGURES = ("Selection_", "Stream_", "DispersionImage_", "PetroInversion_")


def unit_folder(folder: str, xmid: float | None = None, record: str | None = None) -> Path:
    """Run `folder`'s root, its window at `xmid` or its record `record` (by name, as its
    manifest has it); ValueError when it has none."""
    run_folder = folder_path(folder)
    if xmid is not None:
        path = run_folder / f"xmid_{xmid:.2f}"
    elif record is not None:
        manifest = read_manifest(run_folder)
        found = next(
            (one for one in (manifest.records if manifest else ()) if one.name == record), None
        )
        if found is None:
            raise ValueError(f"No record {record} in folder={folder}")
        path = run_folder / found.folder
    else:
        return run_folder
    if not path.is_dir():
        raise ValueError(f"No folder {path.name} in folder={folder}")
    return path


def run_figures(folder: str, xmid: float | None = None, record: str | None = None) -> list[str]:
    """The figures saved at run `folder`'s root (its sections first, then its pseudo-section
    comparisons; each view as the windows' columns or by frequency before its smoothed one or
    by wavelength), in its window at `xmid` (its processing's, in their steps' order) or in its
    record `record`, by name."""
    base = unit_folder(folder, xmid, record)
    names = [path.name for path in base.glob("*.png") if path.is_file()]
    if xmid is None and record is None:
        names = [name for name in names if name.startswith(RUN_FIGURE_PREFIXES)]
        return sorted(names, key=lambda name: ("PseudoSection" in name, name))
    if xmid is not None:
        names = [name for name in names if name.startswith(WINDOW_FIGURES)]
        return sorted(
            names,
            key=lambda name: (
                next(i for i, prefix in enumerate(WINDOW_FIGURES) if name.startswith(prefix)),
                name,
            ),
        )
    return sorted(names)


def run_figure_path(
    folder: str, name: str, xmid: float | None = None, record: str | None = None
) -> Path:
    """Where run `folder` keeps its figure `name` (run_figures'), nothing else."""
    if name not in run_figures(folder, xmid, record):
        raise ValueError(f"No figure {name} for folder={folder}")
    return unit_folder(folder, xmid, record) / name
