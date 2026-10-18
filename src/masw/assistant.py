"""The assistant's settings, as its settings page sets them: which agent the chat runs (PACo's guided
workflow, or its autonomous geophysicist), the model both talk to (any provider: its name, its key
or the machine's cloud credentials, its address, a cloud region and project, its context), and
whether figures may reach it. Kept in PAC's data folder (data/settings/assistant.env, readable by
its owner alone: it holds the API key) and applied over the environment's (compose's .env), where
PACo reads them as PACO_*. Pydantic only: PACo may be missing."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from masw.io.paths import PROJECT_ROOT

SETTINGS_FILE = (
    Path(os.environ.get("PAC_SETTINGS_DIR", PROJECT_ROOT / "data/settings")) / "assistant.env"
)

type Mode = Literal["guided", "autonomous"]

# The page's settings, by the names PACo reads them under; the key apart, never shown.
NAMES = {
    "mode": "PACO_AGENT_MODE",
    "provider": "PACO_LLM_PROVIDER",
    "model": "PACO_LLM_MODEL",
    "base_url": "PACO_LLM_BASE_URL",
    "region": "PACO_LLM_REGION",
    "project": "PACO_LLM_PROJECT",
    "context": "PACO_LLM_CONTEXT",
    "vision": "PACO_LLM_VISION",
}
KEY = "PACO_LLM_API_KEY"
# The key of a server started without one, which takes any: PACo's "no key".
NO_KEY = "EMPTY"
# PACo's defaults: a vLLM server of the user's, the guided workflow.
DEFAULT_PROVIDER = "vllm"
TRUE = ("1", "true", "yes", "on")


class AssistantSettings(BaseModel):
    """The settings in force, as the page shows them: the key never, only whether one is set."""

    mode: Mode = "guided"
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    base_url: str = ""
    region: str = ""
    project: str = ""
    context: int | None = None
    vision: bool = False
    key_set: bool = False


class AssistantSettingsIn(BaseModel):
    """The settings as the page sends them; `api_key` None keeps the key set, empty removes it."""

    mode: Mode = "guided"
    provider: str = Field(default=DEFAULT_PROVIDER, min_length=1)
    model: str = ""
    base_url: str = ""
    region: str = ""
    project: str = ""
    context: int | None = Field(default=None, ge=1)
    vision: bool = False
    api_key: str | None = None


def load() -> None:
    """The kept settings applied over the environment's, and over a .env file's, which PACo reads
    after the environment: an empty one is one not set, as PACo reads it. When PAC starts, and
    after each save."""
    os.environ.update(kept())


def kept() -> dict[str, str]:
    """The settings the page saved, by PACo's names; none before the first save."""
    if not SETTINGS_FILE.is_file():
        return {}
    found: dict[str, str] = {}
    for line in SETTINGS_FILE.read_text().splitlines():
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

    context = setting(NAMES["context"])
    return AssistantSettings(
        mode="autonomous" if setting(NAMES["mode"]) == "autonomous" else "guided",
        provider=setting(NAMES["provider"]) or DEFAULT_PROVIDER,
        model=setting(NAMES["model"]),
        base_url=setting(NAMES["base_url"]),
        region=setting(NAMES["region"]),
        project=setting(NAMES["project"]),
        context=int(context) if context.isdigit() and int(context) > 0 else None,
        vision=setting(NAMES["vision"]).lower() in TRUE,
        key_set=setting(KEY) not in ("", NO_KEY),
    )


def _dotenv() -> dict[str, str]:
    """The page's settings in the .env file of the folder PAC runs from, which PACo reads after
    the environment (compose's .env, when PAC runs outside its container)."""
    path = Path(".env")
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() in (*NAMES.values(), KEY):
            found[name.strip()] = value.strip().strip("'\"")
    return found


def values_of(given: AssistantSettingsIn) -> dict[str, str]:
    """`given` by PACo's names, the key the one in force when the page sent none."""
    values = {
        NAMES["mode"]: given.mode,
        NAMES["provider"]: given.provider.strip(),
        NAMES["model"]: given.model.strip(),
        NAMES["base_url"]: given.base_url.strip(),
        NAMES["region"]: given.region.strip(),
        NAMES["project"]: given.project.strip(),
        NAMES["context"]: str(given.context) if given.context is not None else "",
        NAMES["vision"]: "true" if given.vision else "false",
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
    """`given` kept, in a file readable by its owner alone, and applied."""
    values = values_of(given)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Created readable by its owner alone before the key is written in it.
    descriptor = os.open(SETTINGS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write("".join(f"{name}={value}\n" for name, value in values.items()))
    SETTINGS_FILE.chmod(0o600)
    load()
    return current()
