"""The chat page's backend with a scripted model standing in for Qwen: PACo's tools run in this
process, on PAC's own folders, and the page reads the conversation as events."""

import http.server
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
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
# The scope of a request to process: that stage asked, nothing named.
PROCESS = {
    "process": True,
    "pick": False,
    "invert": False,
    "soils": False,
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
    "workers": None,
    "mode": None,
}
# A message asking no stage (what exists, a greeting): a look.
LOOK = {**PROCESS, "process": False}


# How an answer to a look starts.
LOOK_LINE = "Scope: look only."


class FillsForms:
    """A stand-in's forms: each message's scope from `scopes`, in order (the last one again
    once they run out); every answer is its draft."""

    def __init__(self, *scopes: dict[str, Any]) -> None:
        self._scopes = list(scopes)

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        if schema == ANSWER_SCHEMA:
            draft = str(messages[-1].get("content")).split("\nAnswer: ", 1)[1]
            return Filled(content=json.dumps({"said": draft, "question": None}))
        scope = self._scopes.pop(0) if len(self._scopes) > 1 else self._scopes[0]
        return Filled(content=json.dumps(scope))


class ScriptedModel(FillsForms):
    """Asks for the profiles (inspect), then answers with what the tool returned."""

    def __init__(self) -> None:
        super().__init__(LOOK)  # its messages ask what there is
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

    assert not found["available"] and found["reason"] == (
        "The agent has no model yet: choose one in its settings (PACO_LLM_MODEL)."
    )


def test_a_model_server_that_does_not_answer_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens there
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-14B-FP8")

    found = client.get("/agent/status").json()

    assert not found["available"] and found["model"] == "Qwen/Qwen3-14B-FP8"
    assert "http://127.0.0.1:9/v1 does not answer" in found["reason"]


def test_a_gateway_serving_no_model_list_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    # A company's gateway may serve the chat and no /models (404), or guard it (401): it
    # answers, all the status asks.
    class NoList(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(404)
            self.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", 0), NoList)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PACO_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "company-model")
    try:
        found = client.get("/agent/status").json()
    finally:
        server.shutdown()
        server.server_close()

    assert found["available"], found


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
    assert body["events"][2]["text"] == f"{LOOK_LINE}\n\nYour profiles: noise, shots."
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


def test_a_closed_conversation_empties_pacos_images_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def closed(answering: bool) -> Path:
        """A conversation's answer, an image kept in PACo's cache, then the conversation deleted
        while another conversation answers or not: the image."""
        session = agent.sessions.create(ScriptedModel())
        client.post(f"/agent/sessions/{session.id}/messages", json={"text": "What can I process?"})
        _events(session.id)
        kept = OUTPUT_DIR / ".cache" / "ab" / "ab12"
        kept.mkdir(parents=True, exist_ok=True)
        (kept / "image.h5").write_bytes(b"x" * 100)
        monkeypatch.setattr(agent.sessions, "answering", lambda: answering)
        assert client.delete(f"/agent/sessions/{session.id}").status_code == 204
        session._thread.join(20)  # pyright: ignore[reportPrivateUsage]  # its end, tidied
        return kept

    # Another conversation answering: its images may be in use, the cache stays.
    assert closed(answering=True).exists()
    # None: only the results stay after a session.
    assert not closed(answering=False).exists()


def test_pac_stopping_waits_for_each_conversations_end() -> None:
    session = agent.sessions.create(ScriptedModel())
    client.post(f"/agent/sessions/{session.id}/messages", json={"text": "What can I process?"})
    _events(session.id)
    kept = OUTPUT_DIR / ".cache" / "cd" / "cd34"
    kept.mkdir(parents=True, exist_ok=True)
    (kept / "image.h5").write_bytes(b"x" * 100)

    left = agent.sessions.shut(20)

    # Every conversation closed, and its end done before PAC goes on stopping: its transcript
    # saved and PACo's images' cache emptied, only the results staying.
    assert left == [] and session.closed
    assert agent.sessions.get(session.id) is None
    assert not kept.exists()


def test_pac_starting_removes_what_no_conversation_goes_on_with() -> None:
    from paco.autonomous.tools.trials import TRIALS

    profile = OUTPUT_DIR / "swept"
    study = profile / "studies" / "20261021-150000-abcd"
    (study / TRIALS / "t1").mkdir(parents=True)
    (study / TRIALS / "t1" / "window.npz").write_bytes(b"x" * 100)
    (study / "journal.jsonl").write_text("{}\n")
    cached = OUTPUT_DIR / ".cache" / "ef" / "ef56"
    cached.mkdir(parents=True, exist_ok=True)
    (cached / "image.h5").write_bytes(b"x" * 100)
    try:
        freed = agent.sweep()

        # A conversation is its study, and none outlives PAC: the studies' working files and
        # PACo's images' cache removed, their results kept.
        assert freed >= 200
        assert not (study / TRIALS).exists() and (study / "journal.jsonl").exists()
        assert not cached.exists()
    finally:
        shutil.rmtree(profile)


def test_one_message_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    session = agent.sessions.create(ScriptedModel())
    monkeypatch.setattr(session, "busy", True)

    busy = client.post(f"/agent/sessions/{session.id}/messages", json={"text": "and now?"})

    assert busy.status_code == 409
    assert client.post("/agent/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    agent.sessions.close(session.id)


class CountsTokens(FillsForms):
    """Answers at once, each call's tokens told, as a provider's API tells them."""

    def __init__(self) -> None:
        super().__init__(LOOK)
        self.calls = 0

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        self.calls += 1
        return Reply(
            content="Nothing to process.",
            tool_calls=(),
            prompt_tokens=1200,
            completion_tokens=80,
            cached_tokens=1000,
        )


def test_each_answer_and_its_conversation_say_the_tokens() -> None:
    model = CountsTokens()
    session = agent.sessions.create(model).id
    for text in ("What can I process?", "And now?"):
        client.post(f"/agent/sessions/{session}/messages", json={"text": text})
        _events(session)

    answers = [one["tokens"] for one in _events(session)["events"] if one["kind"] == "answer"]
    listed = next(one for one in client.get("/agent/sessions").json() if one["id"] == session)
    agent.sessions.close(session)

    # Each answer its calls' tokens; the conversation's, all of them.
    assert len(answers) == 2 and all(one["prompt"] >= 1200 for one in answers)
    total = {
        "prompt": 1200 * model.calls,
        "cached": 1000 * model.calls,
        "completion": 80 * model.calls,
    }
    assert listed["tokens"] == total
    assert {key: sum(one[key] for one in answers) for key in total} == total


def _shots_runs() -> set[Path]:
    """The runs of the shots profile, now."""
    runs = OUTPUT_DIR / "shots"
    return set(runs.iterdir()) if runs.exists() else set()


class Stopped(FillsForms):
    """Asks to process the shots finely, long enough to be stopped, then answers plainly."""

    def __init__(self) -> None:
        super().__init__(PROCESS, LOOK)  # the request, then messages asking nothing
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
        f"{LOOK_LINE}\n\nStill here.",
    ]
    from paco.agent.scope import Scope

    # The new message, with the scope PACo read from it.
    read = Scope.model_validate(LOOK).for_model(None)
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


class ProcessesShots(FillsForms):
    """Asks to process the shots, then answers plainly."""

    def __init__(self) -> None:
        super().__init__(PROCESS)

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


def test_the_autonomous_answer_says_the_runs_it_worked_on() -> None:
    run = OUTPUT_DIR / "shots" / "20261018-120000-abcd"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text("{}")
    calls = [
        ("survey", "{}", "active line, 25 shots"),
        ("commit", "{}", "L1 made run 20261018-120000-abcd. 7 windows picked."),
        ("decide", "{}", "Accepted 18.75."),  # the line's run, as the study keeps it
        ("invert", "{}", "Job inv-20261018-130000-ef01 started on 7 windows."),  # a job, no run
        ("invert_petro", "{}", '{"run_id": "20261018-120000-abcd", "summary": "7 soils."}'),
    ]

    try:
        (result,) = agent.autonomous_run_results(calls, "20261018-120000-abcd")
    finally:
        shutil.rmtree(run)

    assert result.folder == "shots/20261018-120000-abcd"
    assert result.stages == ("processing", "picking", "inversion", "petro")


def test_each_mode_has_its_tools() -> None:
    workflow = anyio.run(agent._tools_server("workflow").list_tools)  # pyright: ignore[reportPrivateUsage]
    autonomous = anyio.run(agent._tools_server("autonomous").list_tools)  # pyright: ignore[reportPrivateUsage]

    assert {"run_processing", "judge"} <= {tool.name for tool in workflow}
    assert {"try_settings", "commit", "freeze"} <= {tool.name for tool in autonomous}


def test_the_autonomous_mode_needs_a_long_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_AGENT_MODE", "autonomous")
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-14B-FP8")
    monkeypatch.delenv("PACO_LLM_CONTEXT", raising=False)

    found = client.get("/agent/status").json()

    # A server of the user's: a small model's context unless set, too short for it.
    assert (found["available"], found["mode"]) == (False, "autonomous")
    assert found["reason"].startswith("The autonomous mode needs a model with a long")


def test_the_autonomous_mode_refuses_a_model_that_takes_no_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paco.llm import chat

    async def blind(_: object) -> str:
        return "claude-opus-5-5 takes no images: the autonomous mode judges every image by eye."

    monkeypatch.setenv("PACO_AGENT_MODE", "autonomous")
    monkeypatch.setenv("PACO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PACO_LLM_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("PACO_LLM_API_KEY", "sk-ant-test")
    monkeypatch.setattr(chat, "blind", blind)

    session = agent.sessions.create().id
    client.post(f"/agent/sessions/{session}/messages", json={"text": "Give me the Vs."})
    said = _events(session)["events"][-1]

    # Its reason alone, before any call to the model.
    assert (said["kind"], said["text"]) == (
        "error",
        "claude-opus-5-5 takes no images: the autonomous mode judges every image by eye.",
    )


# --------------------------------------------------------------- the assistant's settings popup


@pytest.fixture
def fresh_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No settings saved, none in the environment or in a .env: as a new install."""
    from masw import assistant

    monkeypatch.chdir(tmp_path)  # away from a developer's own .env
    assistant.SETTINGS_FILE.unlink(missing_ok=True)
    for name in (*assistant.NAMES.values(), assistant.KEY):
        monkeypatch.delenv(name, raising=False)
    return assistant.SETTINGS_FILE


def test_the_settings_are_saved_applied_and_the_key_never_shown(
    fresh_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing set: nothing shown, no mode or provider made up.
    assert client.get("/agent/settings").json() == {
        "mode": None,
        "provider": None,
        "model": "",
        "base_url": "",
        "context": None,
        "key_set": False,
    }

    sent = {
        "mode": "autonomous",
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "api_key": "sk-ant-secret",
    }
    saved = client.put("/agent/settings", json=sent)

    assert saved.status_code == 200
    body = saved.json()
    assert (body["mode"], body["provider"], body["model"], body["key_set"]) == (
        "autonomous",
        "anthropic",
        "claude-opus-5-5",
        True,
    )
    assert "sk-ant-secret" not in saved.text
    assert "vision" not in body
    # Kept for the next start, readable by its owner alone; in force now.
    assert fresh_settings.stat().st_mode & 0o777 == 0o600
    assert "PACO_LLM_API_KEY=sk-ant-secret" in fresh_settings.read_text()
    import os

    assert (os.environ["PACO_AGENT_MODE"], os.environ["PACO_LLM_PROVIDER"]) == (
        "autonomous",
        "anthropic",
    )
    # Saved again without a key: the key kept.
    again = client.put("/agent/settings", json={**sent, "api_key": None, "mode": "workflow"})
    assert again.json()["key_set"] is True and os.environ["PACO_LLM_API_KEY"] == "sk-ant-secret"
    for name in (
        "PACO_AGENT_MODE",
        "PACO_LLM_PROVIDER",
        "PACO_LLM_MODEL",
        "PACO_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_paco_refuses_are_not_saved(fresh_settings: Path) -> None:
    # A vLLM server without its address.
    refused = client.put(
        "/agent/settings",
        json={"mode": "workflow", "provider": "server", "model": "Qwen/Qwen3-14B-FP8"},
    )
    # No mode chosen.
    no_mode = client.put(
        "/agent/settings",
        json={
            "provider": "server",
            "model": "Qwen/Qwen3-14B-FP8",
            "base_url": "http://gpu:8001/v1",
        },
    )

    assert refused.status_code == 422
    assert refused.json()["detail"].startswith("Set PACO_LLM_BASE_URL (")
    assert (no_mode.status_code, no_mode.json()["detail"]) == (
        422,
        "Choose a mode: autonomous or workflow.",
    )
    assert not fresh_settings.exists()


@pytest.mark.usefixtures("fresh_settings")
def test_a_connection_test_says_why_it_failed() -> None:
    tried = client.post(
        "/agent/settings/test",
        json={
            "provider": "server",
            "model": "Qwen/Qwen3-14B-FP8",
            "base_url": "http://127.0.0.1:9/v1",
        },
    )

    assert tried.status_code == 200
    body = tried.json()
    assert body["ok"] is False and body["error"].startswith("APIConnectionError")


# The settings a save applies, then others.
SERVED = {
    "mode": "workflow",
    "provider": "server",
    "model": "Qwen/Qwen3-14B-FP8",
    "base_url": "http://gpu.test:8001/v1",
}
OTHER = {**SERVED, "mode": "autonomous"}


class Thinking(FillsForms):
    """Thinks until told to answer: an answer running for as long as a test needs."""

    def __init__(self) -> None:
        super().__init__(LOOK)
        self.thinking = threading.Event()
        self.done = threading.Event()

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        del messages, tools
        self.thinking.set()
        while not self.done.is_set():
            await anyio.sleep(0.05)
        return Reply(content="Done.", tool_calls=())


@pytest.mark.usefixtures("fresh_settings")
def test_new_settings_clear_the_conversations_unchanged_ones_keep_them() -> None:
    assert client.put("/agent/settings", json=SERVED).status_code == 200
    session = agent.sessions.create(ScriptedModel()).id
    client.post(f"/agent/sessions/{session}/messages", json={"text": "What is there?"})
    _events(session)

    # Saved unchanged: the conversation goes on.
    assert client.put("/agent/settings", json=SERVED).status_code == 200
    assert session in {one["id"] for one in client.get("/agent/sessions").json()}
    # Changed: the conversations are cleared.
    assert client.put("/agent/settings", json=OTHER).status_code == 200

    assert session not in {one["id"] for one in client.get("/agent/sessions").json()}
    assert client.get(f"/agent/sessions/{session}/events").status_code == 404


def test_the_settings_cannot_change_while_an_answer_runs() -> None:
    model = Thinking()
    session = agent.sessions.create(model).id
    client.post(f"/agent/sessions/{session}/messages", json={"text": "Think it over."})
    assert model.thinking.wait(20)

    refused = client.put("/agent/settings", json=OTHER)

    assert (refused.status_code, refused.json()["detail"]) == (409, agent.ANSWERING)
    # Nothing saved, the answer goes on.
    model.done.set()
    assert _events(session)["events"][-1]["kind"] == "answer"
    agent.sessions.close(session)


def test_a_conversation_made_while_the_settings_change_is_made_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[agent.Session] = []

    class SavedMeanwhile(agent.Session):
        """A conversation the settings change under, while it is made."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(*args, **kwargs)
            made.append(self)
            changes = agent._changes  # pyright: ignore[reportPrivateUsage]
            monkeypatch.setattr(agent, "_changes", changes + 1)

    monkeypatch.setattr(agent, "Session", SavedMeanwhile)

    session = agent.sessions.create(ScriptedModel())

    # The first, made with the settings before, deleted; the second kept.
    listed = {one["id"] for one in client.get("/agent/sessions").json()}
    assert [one.id for one in made] == [made[0].id, session.id]
    assert made[0].id not in listed and session.id in listed
    agent.sessions.close(session.id)


@pytest.mark.usefixtures("fresh_settings")
def test_the_popups_settings_win_over_a_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PAC run from a folder whose .env sets a small model's context.
    (tmp_path / ".env").write_text("PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8\nPACO_LLM_CONTEXT=16384\n")
    monkeypatch.chdir(tmp_path)
    from paco.agent.settings import AgentSettings

    monkeypatch.setitem(AgentSettings.model_config, "env_file", ".env")
    assert client.get("/agent/settings").json()["model"] == "Qwen/Qwen3-14B-FP8"

    sent = {
        "mode": "autonomous",
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "context": None,
    }
    assert client.put("/agent/settings", json=sent).status_code == 200

    # The popup left the context empty: the provider's own, not .env's.
    assert AgentSettings().context() == 131_072  # pyright: ignore[reportCallIssue]
    for name in (
        "PACO_AGENT_MODE",
        "PACO_LLM_PROVIDER",
        "PACO_LLM_MODEL",
        "PACO_LLM_BASE_URL",
        "PACO_LLM_CONTEXT",
        "PACO_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_the_workers_set_in_dotenv_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.delenv("PACO_WORKERS", raising=False)
    monkeypatch.chdir(tmp_path)
    # No .env: half the machine's cores.
    agent._use_pacs_folders()  # pyright: ignore[reportPrivateUsage]
    assert os.environ["PACO_WORKERS"] == str(max(1, (os.cpu_count() or 2) // 2))
    # Set in .env: left to PACo, which reads it there.
    monkeypatch.delenv("PACO_WORKERS")
    (tmp_path / ".env").write_text("PACO_WORKERS=3\n")
    agent._use_pacs_folders()  # pyright: ignore[reportPrivateUsage]
    assert "PACO_WORKERS" not in os.environ
