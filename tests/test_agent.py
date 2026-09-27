"""The chat page's backend with a scripted model standing in for Qwen: PACo's tools run in this
process, on PAC's own folders, and the page reads the conversation as events."""

import json
import threading
import time
from pathlib import Path
from typing import Any

import anyio
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
from masw.io.paths import OUTPUT_DIR

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


def _shots_runs() -> set[Path]:
    """The runs of the shots profile, now."""
    runs = OUTPUT_DIR / "shots"
    return set(runs.iterdir()) if runs.exists() else set()


class Stopped:
    """Asks to process the shots finely, long enough to be stopped, then answers plainly."""

    def __init__(self) -> None:
        self.seen: list[list[ChatCompletionMessageParam]] = []

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        del tools
        # A model's call waits on the network: where a stop reaches the answer, once the tool
        # it stopped has returned (the tool call itself cannot be cut).
        await anyio.sleep(0)
        self.seen.append(list(messages))
        if len(self.seen) == 1:
            finely = {"profile": "shots", "overrides": {"masw": {"length": 6, "step": 1}}}
            return Reply(
                content="",
                tool_calls=(ToolCall("call_0", "run_processing", json.dumps(finely)),),
            )
        return Reply(content="Still here.", tool_calls=())


def test_an_answer_stopped_undoes_its_work_and_the_conversation_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paco.agent.loop import STOPPED as NOTED
    from paco.qc import line

    # The records' preprocessing made slow enough to be stopped midway, whatever the machine.
    started = threading.Event()
    real = line.preprocess_records

    def slow(*args: object, **kwargs: object) -> object:
        started.set()
        time.sleep(2.0)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(line, "preprocess_records", slow)
    before = _shots_runs()
    model = Stopped()
    session = agent.sessions.create(model).id
    client.post(f"/agent/sessions/{session}/messages", json={"text": "Process the shots finely"})
    assert started.wait(20)  # PACo's processing runs

    assert client.post(f"/agent/sessions/{session}/stop").status_code == 204
    body = _events(session)

    # The tool's call ends first, said stopped (a call cannot be cut), then the answer.
    assert [event["kind"] for event in body["events"]] == ["user", "step", "step", "stopped"]
    assert body["events"][2]["text"].startswith("failed: Stopped on the user's request")
    assert body["events"][-1]["text"] == agent.STOPPED
    for _ in range(100):  # the tool, in its own thread, done undoing its run
        if _shots_runs() == before:
            break
        time.sleep(0.1)
    assert _shots_runs() == before  # the run removed whole
    # The conversation goes on, from a history the model can read.
    client.post(f"/agent/sessions/{session}/messages", json={"text": "Are you there?"})
    body = _events(session, 4)
    assert [event["text"] for event in body["events"]] == ["Are you there?", "Still here."]
    assert model.seen[1][-2:] == [
        {"role": "assistant", "content": NOTED},
        {"role": "user", "content": "Are you there?"},
    ]
    assert client.post("/agent/sessions/nope/stop").status_code == 404
    agent.sessions.close(session)


def test_a_conversation_deleted_while_answering_is_stopped_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paco.qc import line

    started = threading.Event()
    real = line.preprocess_records

    def slow(*args: object, **kwargs: object) -> object:
        started.set()
        time.sleep(2.0)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(line, "preprocess_records", slow)
    before = _shots_runs()
    session = agent.sessions.create(Stopped())
    client.post(f"/agent/sessions/{session.id}/messages", json={"text": "Process the shots"})
    assert started.wait(20)

    assert client.delete(f"/agent/sessions/{session.id}").status_code == 204

    assert session.id not in {one["id"] for one in client.get("/agent/sessions").json()}
    for _ in range(100):  # its answer stopped, its run removed whole
        if not session.busy and _shots_runs() == before:
            break
        time.sleep(0.1)
    assert not session.busy and session.events[-1].kind == "stopped"
    assert _shots_runs() == before


def test_the_conversations_are_listed_the_latest_first() -> None:
    first = agent.sessions.create(ScriptedModel()).id
    second = agent.sessions.create(ScriptedModel()).id
    client.post(f"/agent/sessions/{first}/messages", json={"text": "What can I process?"})
    _events(first)

    listed = client.get("/agent/sessions").json()

    mine = [one for one in listed if one["id"] in (first, second)]
    assert [(one["id"], one["title"], one["busy"]) for one in mine] == [
        (first, "What can I process?", False),
        (second, "", False),
    ]
    agent.sessions.close(first)
    agent.sessions.close(second)
