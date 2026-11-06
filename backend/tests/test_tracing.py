"""Turn tracing, hop propagation, context capture and offline replay (#62).

One turn is driven through the real loop, the real llama.cpp provider (over
``httpx.MockTransport``) and the real delegate client to a fake app. Its trace
shows turn → model calls → hops → the app's request ids, its model calls are
captured byte for byte, and the pair replays offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog

from app.ai import trace as run_trace
from app.ai.context_capture import JsonlContextCapture, read_captures
from app.ai.loop import AgentLoop, AgentRunFailed
from app.ai.provider import ProviderRequestError
from app.ai.providers.llamacpp import LlamaCppProvider
from app.ai.replay import ReplayError, replay
from app.fleet import context as fleet_context
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import DelegateClient
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.tools import build_delegate_tools
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn

_LLM = "http://llm.test/v1"


def _completion(
    *, content: str | None = None, call: tuple[str, dict[str, Any]] | None = None
) -> str:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish = "stop"
    if call is not None:
        name, arguments = call
        message["tool_calls"] = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
        finish = "tool_calls"
    return json.dumps(
        {
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
        }
    )


def _model(turns: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    """A fake llama-swap: chat completions in order, plus apply-template."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/apply-template"):
            return httpx.Response(200, json={"prompt": "<rendered prompt>"})
        return httpx.Response(200, text=turns.pop(0))

    return handle


class _TasksApp:
    """A fake PCC delegate API that checks the trace headers and answers with
    its own request id."""

    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(
            {
                "turn": request.headers.get("X-Agent-Turn", ""),
                "hop": request.headers.get("X-Agent-Hop", ""),
            }
        )
        headers = {"X-Request-ID": f"pcc-req-{len(self.seen)}"}
        if request.url.path.endswith("/conversations"):
            return httpx.Response(201, json={"id": 5}, headers=headers)
        turn = {"conversation_id": 5, "created_at": "2026-11-05T00:00:00Z"}
        body = {
            "user_message": {**turn, "id": 1, "role": "user", "content": "add pay rent"},
            "assistant_message": {
                **turn,
                "id": 2,
                "role": "assistant",
                "content": "Added task #41 pay rent.",
                "tool_calls": [{"tool": "create_task", "arguments": {}, "result": "41"}],
                "stop_reason": "completed",
            },
        }
        return httpx.Response(200, json=body, headers=headers)


def _tasks_fleet(app: _TasksApp) -> None:
    fleet = Fleet(
        apps=(
            FleetApp(
                name="tasks",
                title="Tasks",
                upstream="127.0.0.1:8100",
                agent=AgentSpec(description="Tasks.", api="/api/agent", examples=("add a task",)),
            ),
        )
    )
    build_delegate_tools(
        fleet,
        lambda _app: DelegateClient(
            "http://tasks.test/api/agent",
            client=httpx.Client(transport=httpx.MockTransport(app.handle)),
        ),
    )


@pytest.fixture
def delegation() -> Any:
    context = DelegationContext(
        master_conversation_id="7", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield context


def _traced_turn(tmp_path: Path) -> tuple[Path, Path, str, _TasksApp]:
    """One traced and captured turn: ask_tasks, then a text reply."""
    app = _TasksApp()
    _tasks_fleet(app)
    trace_path, context_path = tmp_path / "trace.jsonl", tmp_path / "context.jsonl"
    turns = [
        _completion(call=("ask_tasks", {"message": "add a task to pay rent"})),
        _completion(content="Done: pay rent is on your list."),
    ]
    provider = LlamaCppProvider(
        _LLM,
        "gemma-4-12b",
        client=httpx.Client(transport=httpx.MockTransport(_model(turns))),
        capture=JsonlContextCapture(context_path),
    )
    loop = AgentLoop(provider, tracer=run_trace.JsonlTracer(trace_path))
    result = loop.run("add a task to pay rent", conversation_id="7")
    return trace_path, context_path, result.run_id, app


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- one trace: turn → hops → app request ids --------------------------------------


def test_one_trace_shows_turn_hops_and_app_request_ids(
    tmp_path: Path, delegation: DelegationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A throwaway logger proxy, so this test's delegate_call never pins the
    # module's cached one (see test_conversations_api).
    monkeypatch.setattr(fleet_context, "logger", structlog.get_logger(fleet_context.__name__))
    with structlog.testing.capture_logs() as logs:
        trace_path, _, run_id, app = _traced_turn(tmp_path)

    [record] = _records(trace_path)
    assert record["schema"] == run_trace.TRACE_SCHEMA
    assert record["run_id"] == run_id
    assert record["conversation_id"] == "7"
    assert record["stop_reason"] == "completed"
    assert [call["status"] for call in record["calls"]] == ["ok", "ok"]
    assert record["prompt_tokens"] == 1800
    assert record["serving"]["model"] == "gemma-4-12b"
    assert record["serving"]["tools_offered"] == 2  # ask_tasks + list_agents

    [hop] = record["hops"]
    assert hop["app"] == "tasks"
    assert hop["run_id"] == run_id
    assert hop["thread_id"] == 5
    assert hop["app_request_id"] == "pcc-req-2"  # the send_message response's
    assert hop["app_stop_reason"] == "completed"
    assert hop["app_tools"] == ["create_task"]
    # Every request of the hop carried the turn and the hop.
    assert app.seen == [{"turn": run_id, "hop": hop["hop_id"]}] * 2

    [event] = [log for log in logs if log["event"] == "delegate_call"]
    assert event["hop_id"] == hop["hop_id"]
    assert event["app_request_id"] == "pcc-req-2"


def test_a_failed_turn_is_traced_with_its_error(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    loop = AgentLoop(
        ScriptedProvider([ProviderRequestError("llama-server request failed: timed out")]),
        tracer=run_trace.JsonlTracer(path),
    )
    with pytest.raises(AgentRunFailed) as caught:
        loop.run("hi")

    [record] = _records(path)
    assert record["run_id"] == caught.value.result.run_id
    assert record["stop_reason"] == "provider_error"
    assert "timed out" in record["error"]
    assert [call["status"] for call in record["calls"]] == ["failed"]


def test_a_trace_write_that_fails_never_costs_the_turn() -> None:
    class Broken:
        def write(self, record: dict[str, Any]) -> None:
            raise OSError("disk full")

    result = AgentLoop(ScriptedProvider([text_turn("hi")]), tracer=Broken()).run("hello")
    assert result.stop_reason == "completed"


def test_tracing_off_writes_nothing_and_still_names_the_turn(tmp_path: Path) -> None:
    result = AgentLoop(ScriptedProvider([text_turn("hi")])).run("hello")
    assert len(result.run_id) == 32
    assert list(tmp_path.iterdir()) == []


def test_a_bound_run_id_is_reused() -> None:
    with structlog.contextvars.bound_contextvars(run_id="abc123"):
        result = AgentLoop(ScriptedProvider([text_turn("hi")])).run("hello")
    assert result.run_id == "abc123"


# --- context capture ------------------------------------------------------------------


def test_every_model_call_is_captured_byte_for_byte(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    trace_path, context_path, run_id, _ = _traced_turn(tmp_path)

    captures = read_captures(context_path, run_id)
    assert [c["seq"] for c in captures] == [0, 1]
    first = captures[0]
    assert first["status_code"] == 200
    assert first["error"] == ""
    assert first["template"] == {"prompt": "<rendered prompt>"}
    request = json.loads(first["request"])
    assert request["max_tokens"] == 2048
    assert request["messages"][-1] == {"role": "user", "content": "add a task to pay rent"}
    assert json.loads(first["response"])["choices"][0]["message"]["tool_calls"]
    # The capture joins the trace by run id and seq.
    [record] = _records(trace_path)
    assert len(record["calls"]) == len(captures)


def test_a_capture_that_fails_never_costs_the_call() -> None:
    class Broken:
        def begin(self) -> Any:
            raise OSError("disk full")

        def record(self, stamp: Any, call: dict[str, Any]) -> None:
            raise OSError("disk full")

    provider = LlamaCppProvider(
        _LLM,
        "gemma-4-12b",
        client=httpx.Client(transport=httpx.MockTransport(_model([_completion(content="hi")]))),
        capture=Broken(),
    )
    assert provider.chat([{"role": "user", "content": "hello"}]).content == "hi"


# --- offline replay -------------------------------------------------------------------


def test_a_captured_turn_replays_offline(tmp_path: Path, delegation: DelegationContext) -> None:
    trace_path, context_path, run_id, app = _traced_turn(tmp_path)
    calls_to_app = len(app.seen)

    report = replay(trace_path, context_path, run_id)

    assert report.faithful, report
    assert report.calls_replayed == 2
    assert len(app.seen) == calls_to_app  # no app was called


def test_replay_names_the_first_request_that_drifted(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    trace_path, context_path, run_id, _ = _traced_turn(tmp_path)
    # Today's code sends a different cap than the turn did.
    record = _records(trace_path)[0]
    record["limits"]["max_tokens"] = 1024
    trace_path.write_text(json.dumps(record) + "\n")

    report = replay(trace_path, context_path, run_id)

    assert not report.faithful
    assert report.first_request_divergence == 0
    assert "max_tokens" in report.divergence_detail


def test_replay_of_an_unknown_turn_says_so(tmp_path: Path, delegation: DelegationContext) -> None:
    trace_path, context_path, _, _ = _traced_turn(tmp_path)
    with pytest.raises(ReplayError):
        replay(trace_path, context_path, "nope")


def test_hop_headers_go_out_with_tracing_off(delegation: DelegationContext) -> None:
    app = _TasksApp()
    _tasks_fleet(app)
    loop = AgentLoop(
        ScriptedProvider(
            [tool_calls_turn(("ask_tasks", {"message": "add pay rent"})), text_turn("ok")]
        )
    )
    loop.run("add pay rent")
    assert all(seen["hop"] and seen["turn"] for seen in app.seen)
