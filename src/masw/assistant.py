"""The assistant's settings, as its settings popup sets them: the mode the chat runs (autonomous or
workflow), the model (on a provider's API: its name and the key; or on a server speaking
OpenAI's chat API, the user's or any other: its address, its name, its key, its context), and
whether figures may reach it. Kept in PAC's data folder (data/settings/assistant.env, its owner's
alone on every system: it holds the API key) and applied over the environment's (compose's .env),
where PACo reads them as PACO_*. Pydantic and sigpipe's text reading only: PACo may be missing."""

import logging
import os
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from masw.io.paths import PROJECT_ROOT
from masw.io.private import is_private, make_private, write_private
from sigpipe.text import read_hand_written

logger = logging.getLogger(__name__)

SETTINGS_FILE = (
    Path(os.environ.get("PAC_SETTINGS_DIR", PROJECT_ROOT / "data/settings")) / "assistant.env"
)
# Held while the settings are saved and applied, and while a conversation takes its copy of them
# (masw.agent): a conversation never runs with half of a save.
LOCK = threading.RLock()

type Mode = Literal["workflow", "autonomous"]
MODES: tuple[Mode, ...] = ("workflow", "autonomous")

# The popup's settings, by the names PACo reads them under; the key apart, never shown.
NAMES = {
    "mode": "PACO_AGENT_MODE",
    "provider": "PACO_LLM_PROVIDER",
    "model": "PACO_LLM_MODEL",
    "base_url": "PACO_LLM_BASE_URL",
    "context": "PACO_LLM_CONTEXT",
}
KEY = "PACO_LLM_API_KEY"
# The key of a server started without one, which takes any: PACo's "no key".
NO_KEY = "EMPTY"


class AssistantSettings(BaseModel):
    """The settings in force, as the popup shows them: those set (saved by the popup, in the
    environment or in .env), none made up; the key never, only whether one is set."""

    mode: Mode | None = None
    provider: str | None = None
    model: str = ""
    base_url: str = ""
    context: int | None = None
    key_set: bool = False


class AssistantSettingsIn(BaseModel):
    """The settings as the popup sends them: the mode may be missing from a connection test, never
    from a save; `api_key` None keeps the key set, empty removes it."""

    mode: Mode | None = None
    provider: str = Field(min_length=1)
    model: str = ""
    base_url: str = ""
    context: int | None = Field(default=None, ge=1)
    api_key: str | None = None


def load() -> None:
    """The kept settings applied over the environment's, and over a .env file's, which PACo reads
    after the environment: an empty one is one not set, as PACo reads it. When PAC starts, and
    after each save."""
    os.environ.update(kept())


def kept() -> dict[str, str]:
    """The settings the popup saved, by PACo's names; none before the first save. A file others
    may read (copied in, or kept by a PAC that left that to its folder) is made its owner's alone
    first."""
    if not SETTINGS_FILE.is_file():
        return {}
    try:
        if not is_private(SETTINGS_FILE):
            make_private(SETTINGS_FILE)
    except OSError as error:
        logger.warning("%s could not be made its owner's alone: %s", SETTINGS_FILE, error)
    found: dict[str, str] = {}
    for line in SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() in (*NAMES.values(), KEY):
            found[name.strip()] = value.strip()
    return found


def current() -> AssistantSettings:
    """The settings in force, as PACo reads them: the environment's (the kept ones applied over
    them), else the .env file's of the folder PAC runs from."""
    file = _dotenv()

    def setting(name: str) -> str:
        return os.environ[name] if name in os.environ else file.get(name, "")

    mode, context = (setting(NAMES[name]) for name in ("mode", "context"))
    return AssistantSettings(
        mode=mode if mode in MODES else None,
        provider=setting(NAMES["provider"]) or None,
        model=setting(NAMES["model"]),
        base_url=setting(NAMES["base_url"]),
        context=int(context) if context.isdigit() and int(context) > 0 else None,
        key_set=setting(KEY) not in ("", NO_KEY),
    )


def _dotenv() -> dict[str, str]:
    """The popup's settings in the .env file of the folder PAC runs from."""
    return dotenv((*NAMES.values(), KEY))


def dotenv(names: Collection[str]) -> dict[str, str]:
    """Settings `names` in the .env file of the folder PAC runs from, which PACo reads after the
    environment (compose's .env, when PAC runs outside its container)."""
    path = Path(".env")
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    for line in read_hand_written(path).splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() in names:
            found[name.strip()] = value.strip().strip("'\"")
    return found


def values_of(given: AssistantSettingsIn) -> dict[str, str]:
    """`given` by PACo's names, the key the one in force when the popup sent none."""
    values = {
        NAMES["mode"]: given.mode or "",
        NAMES["provider"]: given.provider.strip(),
        NAMES["model"]: given.model.strip(),
        NAMES["base_url"]: given.base_url.strip(),
        NAMES["context"]: str(given.context) if given.context is not None else "",
    }
    key = (
        os.environ.get(KEY, _dotenv().get(KEY, ""))
        if given.api_key is None
        else given.api_key.strip()
    )
    values[KEY] = "" if key == NO_KEY else key
    if any("\n" in value for value in values.values()):
        raise ValueError("A setting holds a line break.")
    return values


def save(given: AssistantSettingsIn) -> AssistantSettings:
    """`given` kept, in a file its owner alone may read or write (masw.io.private: private from
    its creation, before the key is written in it), and applied. The file is written whole
    beside, then put in place: never read half-written."""
    values = values_of(given)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        write_private(SETTINGS_FILE, "".join(f"{name}={value}\n" for name, value in values.items()))
        load()
        return current()
