"""Turn tracing, hop propagation, context capture and offline replay (#62).

One turn is driven through the real interpreter, the real llama.cpp provider
(over ``httpx.MockTransport``) and the real delegate client to a fake app. Its trace
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

from app.ai import compiler
from app.ai import trace as run_trace
from app.ai.context_capture import JsonlContextCapture, read_captures
from app.ai.interpreter import Interpreter
from app.ai.provider import ProviderRequestError
from app.ai.providers.llamacpp import LlamaCppProvider
from app.ai.replay import ReplayError, replay
from app.ai.runtime import AgentRunFailed
from app.fleet import context as fleet_context
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import DelegateClient
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.tools import build_delegate_tools
from tests.scripted_provider import ScriptedProvider, text_turn

_LLM = "http://llm.test/v1"


_PLAN = {
    "steps": [
        {
            "id": "s1",
            "do": "ask",
            "app": "tasks",
            "request": "add a task to pay rent",
            "destructive": False,
        }
    ]
}


def _completion(content: str) -> str:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    return json.dumps(
        {
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
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


def _fleet(description: str = "Tasks.") -> Fleet:
    return Fleet(
        apps=(
            FleetApp(
                name="tasks",
                title="Tasks",
                upstream="127.0.0.1:8100",
                agent=AgentSpec(
                    description=description, api="/api/agent", examples=("add a task",)
                ),
            ),
        )
    )


def _tasks_fleet(app: _TasksApp) -> Fleet:
    fleet = _fleet()
    build_delegate_tools(
        fleet,
        lambda _app: DelegateClient(
            "http://tasks.test/api/agent",
            client=httpx.Client(transport=httpx.MockTransport(app.handle)),
        ),
    )
    return fleet


@pytest.fixture
def delegation() -> Any:
    context = DelegationContext(
        master_conversation_id="7", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield context


def _traced_turn(tmp_path: Path, *, narrate: bool = False) -> tuple[Path, Path, str, _TasksApp]:
    """One traced and captured turn: a plan that asks tasks (and, with
    ``narrate``, the narrator's reply; without, code's record)."""
    app = _TasksApp()
    fleet = _tasks_fleet(app)
    trace_path, context_path = tmp_path / "trace.jsonl", tmp_path / "context.jsonl"
    turns = [_completion(json.dumps(_PLAN))]
    if narrate:
        turns.append(_completion("Pay rent is on your list."))
    provider = LlamaCppProvider(
        _LLM,
        "gemma-4-12b",
        client=httpx.Client(transport=httpx.MockTransport(_model(turns))),
        capture=JsonlContextCapture(context_path),
    )
    conductor = Interpreter(
        provider, fleet, tracer=run_trace.JsonlTracer(trace_path), narrate=narrate
    )
    result = conductor.run("add a task to pay rent", conversation_id="7", surface="voice")
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
    assert [(call["role"], call["status"]) for call in record["calls"]] == [("compiler", "ok")]
    assert record["prompt_tokens"] == 900
    assert record["serving"]["model"] == "gemma-4-12b"
    planner = compiler.planner_prompt(_fleet())
    assert record["serving"]["system_prompt_sha256"] == run_trace.sha256(planner)
    assert record["plan"]["body"] == _PLAN and record["plan"]["status"] == "ran"
    # No loop since #75: the record keeps none of its fields.
    assert not {"history", "iterations", "corrections", "limits"} & set(record)

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


def test_a_failed_turn_is_traced_with_its_error(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    path = tmp_path / "trace.jsonl"
    conductor = Interpreter(
        ScriptedProvider([ProviderRequestError("llama-server request failed: timed out")]),
        _fleet(),
        tracer=run_trace.JsonlTracer(path),
    )
    with pytest.raises(AgentRunFailed) as caught:
        conductor.run("hi")

    [record] = _records(path)
    assert record["run_id"] == caught.value.result.run_id
    assert record["stop_reason"] == "provider_error"
    assert "timed out" in record["error"]
    assert [call["status"] for call in record["calls"]] == ["failed"]


# A direct decline is code's line: one model call, the compile.
_DECLINE = '{"answer": "decline"}'


def test_a_trace_write_that_fails_never_costs_the_turn(delegation: DelegationContext) -> None:
    class Broken:
        def write(self, record: dict[str, Any]) -> None:
            raise OSError("disk full")

    conductor = Interpreter(ScriptedProvider([text_turn(_DECLINE)]), _fleet(), tracer=Broken())
    result = conductor.run("hello")
    assert result.stop_reason == "completed"


def test_tracing_off_writes_nothing_and_still_names_the_turn(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    result = Interpreter(ScriptedProvider([text_turn(_DECLINE)]), _fleet()).run("hello")
    assert len(result.run_id) == 32
    assert list(tmp_path.iterdir()) == []


def test_a_bound_run_id_is_reused(delegation: DelegationContext) -> None:
    with structlog.contextvars.bound_contextvars(run_id="abc123"):
        result = Interpreter(ScriptedProvider([text_turn(_DECLINE)]), _fleet()).run("hello")
    assert result.run_id == "abc123"


# --- context capture ------------------------------------------------------------------


def test_every_model_call_is_captured_byte_for_byte(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    trace_path, context_path, run_id, _ = _traced_turn(tmp_path)

    captures = read_captures(context_path, run_id)
    assert [c["seq"] for c in captures] == [0]
    first = captures[0]
    assert first["status_code"] == 200
    assert first["error"] == ""
    assert first["template"] == {"prompt": "<rendered prompt>"}
    request = json.loads(first["request"])
    assert request["max_tokens"] == compiler.MAX_TOKENS
    assert request["response_format"]["json_schema"]["name"] == "plan"
    assert "tools" not in request  # no loop: the planner is offered no tools
    assert request["messages"][-1] == {
        "role": "user",
        "content": "The person said: add a task to pay rent",
    }
    answer = json.loads(first["response"])["choices"][0]["message"]["content"]
    assert json.loads(answer) == _PLAN
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

    report = replay(trace_path, context_path, run_id, _fleet())

    assert report.faithful, report
    assert report.calls_replayed == 1
    assert len(app.seen) == calls_to_app  # no app was called


def test_a_narrated_turn_replays_its_narrator_too(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    trace_path, context_path, run_id, _ = _traced_turn(tmp_path, narrate=True)
    [record] = _records(trace_path)
    assert record["narration"]["path"] == "narrated" and record["surface"] == "voice"

    report = replay(trace_path, context_path, run_id, _fleet())

    assert report.faithful, report
    assert report.calls_replayed == 2


def test_replay_names_the_first_request_that_drifted(
    tmp_path: Path, delegation: DelegationContext
) -> None:
    trace_path, context_path, run_id, _ = _traced_turn(tmp_path)

    # Today's manifests describe tasks differently: the planner prompt drifted.
    report = replay(trace_path, context_path, run_id, _fleet("Manages tasks now."))

    assert not report.faithful
    assert report.first_request_divergence == 0
    assert "messages" in report.divergence_detail


def test_replay_of_an_unknown_turn_says_so(tmp_path: Path, delegation: DelegationContext) -> None:
    trace_path, context_path, _, _ = _traced_turn(tmp_path)
    with pytest.raises(ReplayError):
        replay(trace_path, context_path, "nope", _fleet())


def test_hop_headers_go_out_with_tracing_off(delegation: DelegationContext) -> None:
    app = _TasksApp()
    fleet = _tasks_fleet(app)
    Interpreter(ScriptedProvider([text_turn(json.dumps(_PLAN))]), fleet).run("add pay rent")
    assert app.seen and all(seen["hop"] and seen["turn"] for seen in app.seen)
