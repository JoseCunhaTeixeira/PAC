"""The chat page's backend with a scripted model standing in for Qwen: PACo's tools run in this
process, on PAC's own folders, and the page reads the conversation as events."""

import json
import shutil
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
from paco.agent import Filled, Reply, ToolCall
from paco.agent.answer import SCHEMA as ANSWER_SCHEMA
from paco.agent.loop import from_data

from masw import agent
from masw.api.main import app
from masw.io.paths import OUTPUT_DIR

client = TestClient(app)
# A message's scope asking every stage: PACo's tools behave as without a scope.
EVERYTHING = {
    "process": True,
    "pick": True,
    "invert": True,
    "soils": True,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "length_receivers": None,
    "length_m": None,
    "step_receivers": None,
    "step_m": None,
    "compare_lengths_receivers": [],
    "compare_lengths_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
}


# How an answer to such a message starts.
SCOPE_LINE = "Scope: process, pick, invert, soils."


class FillsEverything:
    """A stand-in's forms: every message asks every stage; every answer is its draft."""

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        if schema == ANSWER_SCHEMA:
            draft = str(messages[-1].get("content")).split("\nAnswer: ", 1)[1]
            return Filled(content=json.dumps({"said": draft, "question": None}))
        return Filled(content=json.dumps(EVERYTHING))


class ScriptedModel(FillsEverything):
    """Asks for the profiles (inspect), then answers with what the tool returned."""

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
            asked = ToolCall("call_0", "inspect", '{"what": "profiles"}')
            return Reply(content="", tool_calls=(asked,))
        # A line per profile: "noise: a profile to process, no run yet", in PACo's data block.
        profiles = [line.split(":")[0] for line in from_data(last["content"]).splitlines()]
        return Reply(content=f"Your profiles: {', '.join(profiles)}.", tool_calls=())


def _events(session: str, after: int = 0) -> dict[str, Any]:
    # A minute at most: processing runs its trials (the window lengths, the mutes) first.
    for _ in range(600):
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


def test_an_empty_model_is_one_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # As compose passes it when .env names none: the model is the user's choice, no default.
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://model:8000/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "")

    found = client.get("/agent/status").json()

    assert not found["available"] and "set PACO_LLM_MODEL" in found["reason"]


def test_a_model_server_that_does_not_answer_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens there
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-14B-FP8")

    found = client.get("/agent/status").json()

    assert not found["available"] and found["model"] == "Qwen/Qwen3-14B-FP8"
    assert "http://127.0.0.1:9/v1 does not answer" in found["reason"]


def test_a_conversation_runs_pacos_tools_on_pacs_folders() -> None:
    model = ScriptedModel()
    session = agent.sessions.create(model).id

    sent = client.post(f"/agent/sessions/{session}/messages", json={"text": "What can I process?"})
    assert sent.status_code == 202
    body = _events(session)

    kinds = [event["kind"] for event in body["events"]]
    assert kinds == ["user", "step", "answer"]
    assert body["events"][1]["text"] == '-> inspect({"what": "profiles"})'
    # PACo's tool listed PAC's own profiles (the tests' synthetic ones).
    # The answer, after the scope PACo read from the message.
    assert body["events"][2]["text"] == f"{SCOPE_LINE}\n\nYour profiles: noise, shots."
    assert not body["closed"] and body["progress"] is None
    # The menu learns of the answer: one ended, asking nothing.
    listed = {one["id"]: one for one in client.get("/agent/sessions").json()}
    assert (listed[session]["answers"], listed[session]["asks"]) == (1, False)
    # The model was offered PACo's tools.
    names = {tool["function"]["name"] for tool in model.tools}
    assert {"inspect", "run_processing", "pick", "judge", "invert"} <= names
    # Nothing done on a run: nothing to show of one.
    assert body["events"][2]["results"] == []
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


class Stopped(FillsEverything):
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
    assert [event["text"] for event in body["events"]] == [
        "Are you there?",
        f"{SCOPE_LINE}\n\nStill here.",
    ]
    from paco.agent.scope import Scope

    # The new message, with the scope PACo read from it.
    read = Scope.model_validate(EVERYTHING).for_model(None)
    assert model.seen[1][-2:] == [
        {"role": "assistant", "content": NOTED},
        {"role": "user", "content": f"Are you there?\n\n{read}"},
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


def test_an_answer_says_the_runs_it_worked_on_and_what_it_did() -> None:
    run = OUTPUT_DIR / "shots" / "20260930-120000-abcd"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text("{}")
    calls = [
        ("inspect", '{"what": "runs"}', "Run 20260930-120000-abcd: shots"),
        ("run_processing", '{"profile": "shots"}', '{"run_id": "20260930-120000-abcd"}'),
        ("pick", '{"run_id": "20260930-120000-abcd"}', '{"run_id": "20260930-120000-abcd"}'),
        ("redo", '{"stage": "inversion"}', '{"run_id": "20260930-120000-abcd"}'),
        ("job_status", '{"job_id": "j"}', '{"run_id": "20260930-120000-abcd"}'),
        ("pick", '{"run_id": "gone"}', '{"run_id": "gone"}'),  # no such run: nothing to show
    ]

    try:
        (result,) = agent.run_results(calls)
    finally:
        shutil.rmtree(run)  # a run without a manifest, which the other tests would meet

    assert result.folder == "shots/20260930-120000-abcd"
    assert result.stages == ("processing", "picking", "inversion")


class ProcessesShots(FillsEverything):
    """Asks to process the shots, then answers plainly."""

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        if messages[-1]["role"] == "user":
            asked = ToolCall("call_0", "run_processing", '{"profile": "shots"}')
            return Reply(content="", tool_calls=(asked,))
        return Reply(content="Done.", tool_calls=())


def test_an_answer_that_asks_to_choose_is_said_to_the_menu() -> None:
    # A second conversation asked to process the shots meets the first one's run: the tool
    # offers the options (process again, or go on), and the answer asks the user to choose.
    said: list[str] = []
    for _ in range(2):
        session = agent.sessions.create(ProcessesShots()).id
        client.post(f"/agent/sessions/{session}/messages", json={"text": "Process the shots."})
        said += [event["text"][:300] for event in _events(session)["events"]]

    listed = {one["id"]: one for one in client.get("/agent/sessions").json()}
    assert (listed[session]["answers"], listed[session]["asks"]) == (1, True), said
