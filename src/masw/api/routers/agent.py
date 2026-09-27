"""The chat page: conversations with PACo's agent (masw.agent), polled like PAC's jobs."""

import logging

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from masw.agent import (
    AgentStatus,
    AgentUnavailable,
    Event,
    SessionBusy,
    SessionInfo,
    installed,
    sessions,
    status,
)

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


@router.get("/agent/installed")
def get_installed() -> dict[str, bool]:
    """Whether the menu shows the assistant: PAC installed with it."""
    return {"installed": installed()}


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
