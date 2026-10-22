"""Who made each window's Vs profile and soil column in a run an agent made: the agent, as its QC
log records, or the user in PAC after it. PAC writes its configuration into the run when it
inverts; read from the files' contents first, their dates last: a copy of the run dates them
anew."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from masw.io.quality.files import pac_inverted
from masw.io.quality.log import QCLog
from sigpipe.masw.inversion.window import PARAMETERS_FILE, SAMPLES_FILE
from sigpipe.masw.petro.window import MODEL_FILE

PAC_INVERSION_CONFIG = "seismic_inversion_config.json"  # what PAC's page inverted with
PAC_PETRO_CONFIG = "petro_inversion_config.json"
# A file written by the agent's attempt at most this long after the log says it ended.
LATER = timedelta(minutes=1)


def inversion_by_agent(log: QCLog | None, unit: str, run_folder: Path) -> bool:
    """Whether the inversion in window `unit` is the agent's: it logged one, and PAC did not
    invert the window again after it. Without PAC's configuration, the agent's; with it, the
    window's saved parameters other than the agent's last are PAC's; the same, PAC's when its last
    job inverted the window after the agent."""
    latest = log.latest(unit, "inversion") if log is not None else None
    if latest is None:
        return False
    config = run_folder / PAC_INVERSION_CONFIG
    if not config.exists():
        return True
    attempts = log.of(unit, "inversion") if log is not None else ()
    done = next((one for one in reversed(attempts) if one.status == "succeeded"), None)
    saved = run_folder / unit / PARAMETERS_FILE
    if saved.exists():
        given = cast(dict[str, Any], json.loads(saved.read_text()).get("parameters") or {})
        # The agent's parameters as its log holds them, against the window's.
        if done is None or any(given.get(key) != value for key, value in done.parameters.items()):
            return False
    if not pac_inverted(config, unit):
        return True
    samples = run_folder / unit / SAMPLES_FILE
    ended = latest.finished_at or latest.started_at
    return not (samples.exists() and samples.stat().st_mtime > (ended + LATER).timestamp())


def soil_by_agent(log: QCLog | None, window: Path) -> bool:
    """Whether the soil column in window folder `window` is the agent's: it logged one, and PAC
    did not invert the window again after it. Without PAC's configuration, the agent's, however a
    copy of the run dated its files."""
    latest = log.latest(window.name, "petro_inversion") if log is not None else None
    if latest is None:
        return False
    if not (window.parent / PAC_PETRO_CONFIG).exists():
        return True
    model = window / MODEL_FILE
    ended = latest.finished_at or latest.started_at
    return not (model.exists() and model.stat().st_mtime > (ended + LATER).timestamp())
