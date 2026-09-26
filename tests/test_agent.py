"""The chat page's backend with a scripted model standing in for Qwen: PACo's tools run in this
process, on PAC's own folders, and the page reads the conversation as events."""

import json
import time
from typing import Any

import pytest

# The assistant is PAC's optional agent extra: without PACo, nothing to test here.
pytest.importorskip("paco")

from fastapi.testclient import TestClient
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from paco.agent import Reply, ToolCall

from masw import agent
from masw.api.main import app

client = TestClient(app)


class ScriptedModel:
    """Asks for list_profiles, then answers with what the tool returned."""

    def __init__(self) -> None:
        self.seen: list[list[ChatCompletionMessageParam]] = []
        self.tools: list[ChatCompletionFunctionToolParam] = []

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        self.seen.append(list(messages))
        self.tools = tools
        last: Any = messages[-1]
        if last["role"] == "user":
            return Reply(content="", tool_calls=(ToolCall("call_0", "list_profiles", "{}"),))
        profiles = json.loads(last["content"])["result"]  # a list comes wrapped
        return Reply(content=f"Your profiles: {', '.join(profiles)}.", tool_calls=())


def _events(session: str, after: int = 0) -> dict[str, Any]:
    for _ in range(200):
        body = client.get(f"/agent/sessions/{session}/events", params={"after": after}).json()
        if not body["busy"]:
            return body
        time.sleep(0.1)
    raise AssertionError("the agent did not answer")


def test_the_menu_shows_the_assistant_where_it_is_installed() -> None:
    assert client.get("/agent/installed").json() == {"installed": True}


def test_without_a_model_the_page_says_what_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PACO_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PACO_LLM_MODEL", raising=False)
    monkeypatch.chdir("/")  # no .env to read the settings from

    found = client.get("/agent/status").json()

    assert not found["available"]
    assert "PACO_LLM_BASE_URL" in found["reason"] and "PACO_LLM_MODEL" in found["reason"]
    refused = client.post("/agent/sessions")
    assert refused.status_code == 503 and "PACO_LLM_BASE_URL" in refused.json()["detail"]


def test_a_model_server_that_does_not_answer_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens there
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-8B")

    found = client.get("/agent/status").json()

    assert not found["available"] and found["model"] == "Qwen/Qwen3-8B"
    assert "http://127.0.0.1:9/v1 does not answer" in found["reason"]


def test_a_conversation_runs_pacos_tools_on_pacs_folders() -> None:
    model = ScriptedModel()
    session = agent.sessions.create(model).id

    sent = client.post(f"/agent/sessions/{session}/messages", json={"text": "What can I process?"})
    assert sent.status_code == 202
    body = _events(session)

    kinds = [event["kind"] for event in body["events"]]
    assert kinds == ["user", "step", "answer"]
    assert body["events"][1]["text"] == "-> list_profiles({})"
    # PACo's tool listed PAC's own profiles (the tests' synthetic ones).
    assert body["events"][2]["text"] == "Your profiles: noise, shots."
    assert not body["closed"] and body["progress"] is None
    # The model was offered PACo's tools.
    names = {tool["function"]["name"] for tool in model.tools}
    assert {"list_profiles", "run_processing", "pick", "invert"} <= names
    # The page asks for what it has not seen yet.
    later = client.get(f"/agent/sessions/{session}/events", params={"after": 2}).json()
    assert [event["kind"] for event in later["events"]] == ["answer"]

    assert client.delete(f"/agent/sessions/{session}").status_code == 204
    assert client.get(f"/agent/sessions/{session}/events").status_code == 404


def test_one_message_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    session = agent.sessions.create(ScriptedModel())
    monkeypatch.setattr(session, "busy", True)

    busy = client.post(f"/agent/sessions/{session.id}/messages", json={"text": "and now?"})

    assert busy.status_code == 409
    assert client.post("/agent/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    agent.sessions.close(session.id)
