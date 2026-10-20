"""The chat page: conversations with PACo's agent (masw.agent), polled like PAC's jobs; and the
assistant's settings popup: the mode, the model and its test (masw.assistant)."""

import logging
import time
from typing import TYPE_CHECKING

import anyio
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError

from masw import assistant
from masw.agent import (
    AgentStatus,
    AgentUnavailable,
    Event,
    SessionBusy,
    SessionInfo,
    save_settings,
    sessions,
    status,
)

if TYPE_CHECKING:
    from paco.llm.settings import ModelSettings

# How long a connection test waits for the model, in seconds.
TEST_S = 60.0

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


class SessionOut(BaseModel):
    id: str


class MessageIn(BaseModel):
    text: str = Field(min_length=1)


class EventsOut(BaseModel):
    events: list[Event]  # after the index asked for
    busy: bool  # the agent is answering
    stopping: bool = False  # a stop was asked, and the answer has not ended yet
    progress: str | None  # the running tool's latest progress
    closed: bool  # the conversation ended (an error stopped it): start a new one


@router.get("/agent/status")
def get_status() -> AgentStatus:
    return status()


@router.get("/agent/sessions")
def list_sessions() -> list[SessionInfo]:
    """The conversations this server keeps, the latest first: the page finds the one it left."""
    return sessions.listed()


@router.post("/agent/sessions", status_code=201)
def create_session() -> SessionOut:
    try:
        return SessionOut(id=sessions.create().id)
    except AgentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/agent/sessions/{session_id}/messages", status_code=202)
def send_message(session_id: str, message: MessageIn) -> Response:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {session_id}")
    try:
        session.ask(message.text)
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AgentUnavailable as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    return Response(status_code=202)


@router.get("/agent/sessions/{session_id}/events")
def get_events(session_id: str, after: int = 0) -> EventsOut:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {session_id}")
    return EventsOut(
        events=session.events_after(after),
        busy=session.busy,
        stopping=session.stopping,
        progress=session.progress,
        closed=session.closed,
    )


@router.post("/agent/sessions/{session_id}/stop", status_code=204)
def stop_session(session_id: str) -> Response:
    """Stop the answer running and everything it started: at once, what had finished kept,
    nothing half-written. Nothing to stop is no error."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {session_id}")
    session.stop()
    return Response(status_code=204)


@router.delete("/agent/sessions/{session_id}", status_code=204)
def close_session(session_id: str) -> Response:
    if not sessions.close(session_id):
        raise HTTPException(status_code=404, detail=f"Unknown conversation: {session_id}")
    return Response(status_code=204)


class TestOut(BaseModel):
    """A connection test: whether the model answered, how long it took, whether it takes images
    (the autonomous mode needs them), or the error."""

    ok: bool
    seconds: float | None = None
    images: bool | None = None
    error: str | None = None


@router.get("/agent/settings")
def get_settings() -> assistant.AssistantSettings:
    """The assistant's settings in force, the API key left out (whether one is set)."""
    return assistant.current()


@router.put("/agent/settings")
def put_settings(given: assistant.AssistantSettingsIn) -> assistant.AssistantSettings:
    """The assistant's settings saved and applied, once PACo takes them: refused, saying what is
    missing or wrong, otherwise; refused while an answer runs (409). Changed, they clear the
    conversations (masw.agent.save_settings); the next ones take them."""
    if given.mode is None:
        raise HTTPException(status_code=422, detail="Choose a mode: autonomous or workflow.")
    _model_settings(given)
    try:
        return save_settings(given)
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent/settings/test")
async def test_settings(given: assistant.AssistantSettingsIn) -> TestOut:
    """One short call to the model the popup's settings name (saved or not), and whether it takes
    images: what the popup says of the connection. A few tokens, never a figure."""
    from paco.llm.chat import chat_model, tried

    settings = _model_settings(given)
    started = time.monotonic()
    try:
        # Asked once, within a minute: the popup counts it down.
        with anyio.fail_after(TEST_S):
            chat = chat_model(settings, reconnect_s=0)
            await tried(chat)
            images = await chat.takes_images()
    except TimeoutError:
        return TestOut(ok=False, error=f"No answer within {TEST_S:g} s.")
    except Exception as exc:  # a key refused, a model not there, a server down: said as such
        return TestOut(ok=False, error=f"{type(exc).__name__}: {exc}")
    return TestOut(ok=True, seconds=round(time.monotonic() - started, 2), images=images)


def _model_settings(given: assistant.AssistantSettingsIn) -> ModelSettings:
    """PACo's settings of the model `given` names (the key set when the popup sent none); 422,
    saying what is missing or wrong, when PACo refuses them."""
    from paco.llm.settings import ModelSettings

    values = assistant.values_of(given)
    try:
        return ModelSettings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            llm_provider=values[assistant.NAMES["provider"]],  # pyright: ignore[reportArgumentType]
            llm_model=values[assistant.NAMES["model"]],
            llm_base_url=values[assistant.NAMES["base_url"]] or None,
            llm_api_key=values[assistant.KEY] or assistant.NO_KEY,  # pyright: ignore[reportArgumentType]
            llm_context=given.context,
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"PACO_{problem['loc'][0]}".upper() + f" ({problem['msg']})" for problem in exc.errors()
        )
        raise HTTPException(status_code=422, detail=f"Set {problems}.") from exc
