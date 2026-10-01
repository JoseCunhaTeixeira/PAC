"""PACo's agent in PAC: the chat page's conversations, each an agent in this process whose tools
are PACo's, called in-process on PAC's own folders, so that its runs are PAC's runs. The model is
served elsewhere, behind an OpenAI-compatible API (vLLM serving Qwen3-8B): PACo's settings
PACO_LLM_BASE_URL and PACO_LLM_MODEL.

PACo is optional (PAC's `agent` extra): without it, without a model, or with a model server that
does not answer, `status` says what is missing and the rest of PAC is unaffected. The only
module of PAC that imports PACo, and only once a conversation starts."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

import anyio
import anyio.from_thread
import anyio.lowlevel
import anyio.to_thread
import httpx
from pydantic import BaseModel, ValidationError

from masw.io.paths import INPUT_DIR, OUTPUT_DIR

if TYPE_CHECKING:
    from paco.agent import AgentSettings, ChatModel
    from paco.stopping import Signal

logger = logging.getLogger(__name__)

# Where the conversations are saved, one JSON file each, when they end.
LOG_DIR = OUTPUT_DIR / "agent_logs"
# Conversations kept at once: a new one past this closes the oldest idle one.
MAX_SESSIONS = 8

type EventKind = Literal["user", "step", "answer", "error", "stopped"]
# What an answer did on a run: PAC's pages show each.
type RunStage = Literal["processing", "picking", "inversion", "petro"]

# What a stopped answer says, in the conversation.
STOPPED = "Stopped. What had finished is kept; the rest is as it was."
# The tools that work on a run, and what each does there (redo: by the stage it goes back to).
RUN_STAGES: dict[str, RunStage] = {
    "run_processing": "processing",
    "pick": "picking",
    "judge": "picking",
    "invert": "inversion",
    "job_status": "inversion",
    "invert_petro": "petro",
}


class RunResult(BaseModel):
    """A run an answer worked on, and what it did there, in order."""

    run_id: str
    folder: str  # <profile>/<run_id>, as PAC's pages name a run
    stages: tuple[RunStage, ...]


class Event(BaseModel):
    index: int
    # The user's message, a tool call (or its failure), the answer, an error, a stop.
    kind: EventKind
    text: str
    # An answer's: the runs it worked on, and how long it took.
    results: tuple[RunResult, ...] = ()
    seconds: float | None = None


class SessionInfo(BaseModel):
    """A conversation as the chat page lists them."""

    id: str
    title: str  # its first question; empty before any
    busy: bool
    closed: bool
    started_at: datetime
    updated_at: datetime
    # The answers ended so far (failed or stopped ones too), for the menu to show a new one,
    # and whether the last asks the user to choose among the options a tool offered.
    answers: int = 0
    asks: bool = False


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
    page polls the events. An answer can be stopped: at once, and everything it started (see
    paco.stopping), what had finished kept, nothing half-written."""

    def __init__(self, model: ChatModel | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.events: list[Event] = []
        self.busy = False
        self.stopping = False  # a stop was asked, and the answer has not ended yet
        self.progress: str | None = None  # the running tool's latest progress, while busy
        self.closed = False
        self.answers = 0  # the answers ended so far
        self.asks = False  # the last one asks the user to choose
        self.started_at = self.updated_at = datetime.now(UTC)
        self._model = model
        self._questions: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        # Once the conversation runs: PACo's stop, the running answer's scope, and its loop.
        self._signal: Signal | None = None
        self._scope: anyio.CancelScope | None = None
        self._loop: anyio.lowlevel.EventLoopToken | None = None
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

    def info(self) -> SessionInfo:
        with self._lock:
            title = next((event.text for event in self.events if event.kind == "user"), "")
            return SessionInfo(
                id=self.id,
                title=title,
                busy=self.busy,
                closed=self.closed,
                started_at=self.started_at,
                updated_at=self.updated_at,
                answers=self.answers,
                asks=self.asks,
            )

    def stop(self) -> bool:
        """Stop the answer running, and everything it started: at once, what had finished kept,
        nothing half-written. False when no answer runs."""
        with self._lock:
            if not self.busy:
                return False
            self.stopping = True
            signal, scope, loop = self._signal, self._scope, self._loop
        # The answer first (this waits until its loop has it), then its tools: a tool that ended
        # first would have its answer go on without it.
        if scope is not None and loop is not None:
            with contextlib.suppress(RuntimeError):  # the conversation's loop has just ended
                anyio.from_thread.run_sync(scope.cancel, token=loop)
        if signal is not None:
            signal.stop()
        return True

    def close(self) -> None:
        self._questions.put(None)

    def _add(
        self,
        kind: EventKind,
        text: str,
        results: tuple[RunResult, ...] = (),
        seconds: float | None = None,
    ) -> None:
        self.events.append(
            Event(index=len(self.events), kind=kind, text=text, results=results, seconds=seconds)
        )
        self.updated_at = datetime.now(UTC)

    def _on_event(self, line: str) -> None:
        """PACo's loop reports tool calls and failures (steps), and a running tool's progress
        (indented lines): the latter is shown live, not kept."""
        text = line.strip()
        with self._lock:
            if line.startswith("   ") and not text.startswith(("failed:", "(")):
                self.progress = text
            else:
                self._add("step", text)

    def _finish(
        self,
        kind: EventKind,
        text: str,
        results: tuple[RunResult, ...] = (),
        seconds: float | None = None,
        asks: bool = False,
    ) -> None:
        """The answer ended (`kind`: answered, failed or stopped); `asks` when it asks the user
        to choose among the options a tool offered."""
        with self._lock:
            self._add(kind, text, results, seconds)
            self.answers += 1
            self.asks = asks
            self.busy = False
            self.stopping = False
            self.progress = None

    def _run(self) -> None:
        anyio.run(self._converse)

    async def _converse(self) -> None:
        from mcp import Client
        from openai import AsyncOpenAI
        from paco import server, stopping
        from paco.agent import Agent, Limits, OpenAIChat, save_transcript
        from paco.agent.record import ToolStep

        agent: Agent | None = None
        name = "scripted"
        try:
            settings = _settings() if self._model is None else None
            if settings is not None:
                name = settings.llm_model
                client = AsyncOpenAI(
                    base_url=settings.llm_base_url, api_key=settings.llm_api_key.get_secret_value()
                )
                model: ChatModel = OpenAIChat(
                    client, settings.llm_model, settings.llm_temperature, settings.llm_seed
                )
            else:
                assert self._model is not None
                model = self._model
            _use_pacs_folders()
            # PACo's stop for this conversation, set before its tools' server starts: their
            # calls, run by the server's tasks, inherit it.
            signal = stopping.Signal()
            stopping.SIGNAL.set(signal)
            with self._lock:
                self._signal = signal
                self._loop = anyio.lowlevel.current_token()
            async with Client(server.server) as tools:
                limits = Limits.of(settings) if settings is not None else None
                agent = await Agent.start(tools, model, on_event=self._on_event, limits=limits)
                while (question := await anyio.to_thread.run_sync(self._questions.get)) is not None:
                    signal.renew()
                    answer: str | None = None
                    failed: Exception | None = None
                    first, started = len(agent.steps), time.monotonic()
                    with anyio.CancelScope() as scope:
                        with self._lock:
                            self._scope = scope
                            if self.stopping:  # stopped before the answer began
                                signal.stop()
                                scope.cancel()
                        try:
                            answer = await agent.answer(question)
                        except Exception as error:
                            logger.exception("The agent failed to answer in session %s", self.id)
                            failed = error
                    with self._lock:
                        self._scope = None
                    if failed is not None:
                        self._finish("error", f"{type(failed).__name__}: {failed}")
                    elif answer is None:
                        self._finish("stopped", STOPPED)
                    else:
                        done = [
                            (step.name, step.arguments, step.result)
                            for step in agent.steps[first:]
                            if isinstance(step, ToolStep) and step.called and not step.is_error
                        ]
                        self._finish(
                            "answer",
                            answer,
                            run_results(done),
                            time.monotonic() - started,
                            asks=bool(agent.offers),
                        )
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

    def listed(self) -> list[SessionInfo]:
        """The conversations kept, the latest first."""
        with self._lock:
            found = list(self._sessions.values())
        return sorted((one.info() for one in found), key=lambda info: info.updated_at, reverse=True)

    def close(self, session_id: str) -> bool:
        """Delete conversation `session_id`: its answer stopped first if one runs (its work kept
        or undone as a stop does); its transcript is saved as it ends."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop()
        session.close()
        return True


sessions = Sessions()


def run_results(calls: Sequence[tuple[str, str, str]]) -> tuple[RunResult, ...]:
    """The runs the tool calls of an answer worked on, each call its name, arguments and
    result: those of a tool that works on a run, its result naming it; with what each did."""
    done: dict[str, list[RunStage]] = {}
    for name, arguments, result in calls:
        stage = RUN_STAGES.get(name)
        if name == "redo":
            asked = _json(arguments)
            stage = "inversion" if asked.get("stage") == "inversion" else "picking"
        run_id = _json(result).get("run_id")
        if stage is None or not isinstance(run_id, str):
            continue
        stages = done.setdefault(run_id, [])
        if stage not in stages:
            stages.append(stage)
    results: list[RunResult] = []
    for run_id, stages in done.items():
        found = sorted(OUTPUT_DIR.glob(f"*/{run_id}/run.json"))
        if found:
            folder = f"{found[0].parent.parent.name}/{run_id}"
            results.append(RunResult(run_id=run_id, folder=folder, stages=tuple(stages)))
    return tuple(results)


def _json(text: str) -> dict[str, object]:
    """`text` as a JSON object; empty when it is none."""
    try:
        value: object = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, object], value) if isinstance(value, dict) else {}
