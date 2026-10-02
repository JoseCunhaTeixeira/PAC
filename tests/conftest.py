"""PAC's tests read and write their own folders: two synthetic profiles in a temporary input
folder, runs in a temporary output folder. Set before any masw module reads them."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib
import pytest

_ROOT = Path(tempfile.mkdtemp(prefix="pac-tests-"))
# Removed when the session ends: every session writes about 120 MB there.
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
os.environ["MASW_INPUT_DIR"] = str(_ROOT / "input")
os.environ["MASW_OUTPUT_DIR"] = str(_ROOT / "output")
(_ROOT / "input").mkdir()
(_ROOT / "output").mkdir()

from .synthetic import write_profiles  # noqa: E402

write_profiles(_ROOT / "input")

# The API saves figures from worker threads: never a GUI backend.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The assistant's settings from the environment alone, never from a `.env` in the folder the
    tests run from: a developer's own (their model, its context) would make the tests pass where
    CI, which has none, fails."""
    try:
        from paco.agent.settings import AgentSettings
        from paco.settings import Settings
    except ImportError:  # PAC without its assistant
        return
    for settings in (AgentSettings, Settings):
        monkeypatch.setitem(settings.model_config, "env_file", None)
