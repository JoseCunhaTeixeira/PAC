"""PACo's agent in PAC: the chat page's conversations, each an agent in this process whose tools
are PACo's, called in-process on PAC's own folders, so that its runs are PAC's runs. The model is
served elsewhere, behind an OpenAI-compatible API (vLLM serving Qwen3-8B): PACo's settings
PACO_LLM_BASE_URL and PACO_LLM_MODEL.

PACo is optional (PAC's `agent` extra): without it, without a model, or with a model server that
does not answer, `status` says what is missing and the rest of PAC is unaffected. The only
module of PAC that imports PACo, and only once a conversation starts."""

from __future__ import annotations

import importlib.util
import logging
import os
import queue
import threading
import uuid
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.to_thread
import httpx
from pydantic import BaseModel, ValidationError

from masw.io.paths import INPUT_DIR, OUTPUT_DIR

if TYPE_CHECKING:
    from paco.agent import AgentSettings, ChatModel

logger = logging.getLogger(__name__)

# Where the conversations are saved, one JSON file each, when they end.
LOG_DIR = OUTPUT_DIR / "agent_logs"
# Conversations kept at once: a new one past this closes the oldest idle one.
MAX_SESSIONS = 8

type EventKind = Literal["user", "step", "answer", "error"]


class Event(BaseModel):
    index: int
    kind: EventKind  # the user's message, a tool call (or its failure), the answer, an error
    text: str


class AgentStatus(BaseModel):
    available: bool
    reason: str | None = None  # what is missing, and how to fix it
    model: str | None = None


class AgentUnavailable(RuntimeError):
    """The agent cannot run: the message says why and what to do."""


class SessionBusy(RuntimeError):
    """A conversation is still answering the last message."""


def installed() -> bool:
    """Whether PAC was installed with its assistant (the agent extra)."""
    return importlib.util.find_spec("paco") is not None


def status() -> AgentStatus:
    """Whether the chat can run: PACo installed, its model set, the model server answering."""
    if not installed():
        return AgentStatus(
            available=False,
            reason="PACo, PAC's agent, is not installed: install PAC with its agent extra "
            "(uv sync --extra agent; with Docker, build with PAC_EXTRAS=agent).",
        )
    try:
        settings = _settings()
    except AgentUnavailable as error:
        return AgentStatus(available=False, reason=str(error))
    url = settings.llm_base_url.rstrip("/")
    try:
        response = httpx.get(
            f"{url}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
            timeout=3.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        return AgentStatus(
            available=False,
            reason=f"The model server at {url} does not answer ({type(error).__name__}): start "
            "it (docker compose --profile agent up), or set PACO_LLM_BASE_URL to a server that "
            "runs, on a machine whose GPU can serve the model.",
            model=settings.llm_model,
        )
    return AgentStatus(available=True, model=settings.llm_model)


def _settings() -> AgentSettings:
    from paco.agent import AgentSettings

    try:
        return AgentSettings()  # pyright: ignore[reportCallIssue]  # from the environment
    except ValidationError as error:
        missing = ", ".join(f"PACO_{problem['loc'][0]}".upper() for problem in error.errors())
        raise AgentUnavailable(
            f"The agent's model is not set: set {missing} (the model server's address, e.g. "
            "http://127.0.0.1:8001/v1, and the name it serves the model under)."
        ) from error


def _use_pacs_folders() -> None:
    """PACo's tools read PAC's profiles and write PAC's runs; its worker processes, half the
    machine's cores unless PACO_WORKERS says otherwise."""
    from paco.settings import get_settings

    os.environ["PACO_INPUT_DIR"] = str(INPUT_DIR)
    os.environ["PACO_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ.setdefault("PACO_WORKERS", str(max(1, (os.cpu_count() or 2) // 2)))
    get_settings.cache_clear()


class Session:
    """One conversation: its agent answers in a thread of its own, one message at a time; the
    page polls the events."""

    def __init__(self, model: ChatModel | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.events: list[Event] = []
        self.busy = False
        self.progress: str | None = None  # the running tool's latest progress, while busy
        self.closed = False
        self._model = model
        self._questions: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"agent-{self.id[:8]}", daemon=True)
        self._thread.start()

    def ask(self, text: str) -> None:
        with self._lock:
            if self.closed:
                raise AgentUnavailable("This conversation has ended: start a new one.")
            if self.busy:
                raise SessionBusy("The agent is still answering the last message.")
            self.busy = True
            self._add("user", text)
        self._questions.put(text)

    def events_after(self, after: int) -> list[Event]:
        with self._lock:
            return self.events[after:]

    def close(self) -> None:
        self._questions.put(None)

    def _add(self, kind: EventKind, text: str) -> None:
        self.events.append(Event(index=len(self.events), kind=kind, text=text))

    def _on_event(self, line: str) -> None:
        """PACo's loop reports tool calls and failures (steps), and a running tool's progress
        (indented lines): the latter is shown live, not kept."""
        text = line.strip()
        with self._lock:
            if line.startswith("   ") and not text.startswith(("failed:", "(")):
                self.progress = text
            else:
                self._add("step", text)

    def _finish(self, kind: EventKind, text: str) -> None:
        with self._lock:
            self._add(kind, text)
            self.busy = False
            self.progress = None

    def _run(self) -> None:
        anyio.run(self._converse)

    async def _converse(self) -> None:
        from mcp import Client
        from openai import AsyncOpenAI
        from paco import server
        from paco.agent import Agent, OpenAIChat, save_transcript

        agent: Agent | None = None
        name = "scripted"
        try:
            settings = _settings() if self._model is None else None
            if settings is not None:
                name = settings.llm_model
                client = AsyncOpenAI(
                    base_url=settings.llm_base_url, api_key=settings.llm_api_key.get_secret_value()
                )
                model: ChatModel = OpenAIChat(client, settings.llm_model)
            else:
                assert self._model is not None
                model = self._model
            _use_pacs_folders()
            async with Client(server.server) as tools:
                agent = await Agent.start(tools, model, on_event=self._on_event)
                while (question := await anyio.to_thread.run_sync(self._questions.get)) is not None:
                    try:
                        self._finish("answer", await agent.answer(question))
                    except Exception as error:
                        logger.exception("The agent failed to answer in session %s", self.id)
                        self._finish("error", f"{type(error).__name__}: {error}")
        except Exception as error:
            logger.exception("The agent's session %s stopped", self.id)
            self._finish("error", f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self.closed = True
            if agent is not None and agent.steps:
                try:
                    save_transcript(agent.transcript(name), LOG_DIR)
                except Exception:
                    logger.exception("Could not save the conversation of session %s", self.id)


class Sessions:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, model: ChatModel | None = None) -> Session:
        """A new conversation; raises AgentUnavailable, with the reason, when the chat cannot
        run (a scripted `model` needs no model server)."""
        if model is None and not (found := status()).available:
            raise AgentUnavailable(found.reason or "The agent is not available.")
        session = Session(model)
        with self._lock:
            idle = [one for one in self._sessions.values() if not one.busy]
            while len(self._sessions) >= MAX_SESSIONS and idle:
                oldest = idle.pop(0)
                oldest.close()
                del self._sessions[oldest.id]
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True


sessions = Sessions()
