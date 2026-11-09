"""The turn runtime (#75): STANDARD.md §3's safety nets, kept from the retired loop.

``max_tokens`` on every model call, one step's dispatch (a refusal is the
record's error, a bad call is "not run"), the acting identity threaded to the
tools, and failure stops that carry the partial turn. The loop's iteration,
correction, stall and per-run call caps went with it; a plan's own ceiling
(12 steps) is in ``test_compiled_plans.py``.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.ai import compiler, narrator, run_record
from app.ai.interpreter import Interpreter
from app.ai.provider import ChatResult, ProviderRequestError, ToolCall, ToolSpec
from app.ai.runtime import ACTOR, FAILURE_STATUS, AgentRunFailed, Runtime
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.tools import runtime as tool_runtime
from app.tools.registry import ToolError, tool
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


@pytest.fixture(autouse=True)
def delegation() -> Generator[None, None, None]:
    context = DelegationContext(
        master_conversation_id="1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield


def _plan(*steps: dict[str, Any]) -> ChatResult:
    body = {"steps": [{"id": f"s{n}", **step} for n, step in enumerate(steps, start=1)]}
    return text_turn(json.dumps(body))


def _ask(app_name: str, request: str) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": False}


class _Capped:
    """Records the ``max_tokens`` of every call, then answers with ``turns``."""

    def __init__(self, turns: Sequence[ChatResult | Exception]) -> None:
        self._turns = list(turns)
        self.max_tokens: list[int | None] = []

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        assert not tools, "conductor offers a model no tools (#75)"
        self.max_tokens.append(max_tokens)
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def test_every_model_call_of_a_turn_carries_max_tokens(fleet: Fleet, world: World) -> None:
    said = text_turn("Added “water the plants”; say undo to move it to the trash.")
    provider = _Capped([_plan(_ask("tasks", "add a task to water the plants")), said])
    Interpreter(provider, fleet, narrate=True).run("add a task to water the plants")
    assert provider.max_tokens == [compiler.MAX_TOKENS, narrator.MAX_TOKENS]


def test_the_runtime_cap_fills_any_call_that_names_none() -> None:
    provider = _Capped([text_turn("hi")])
    Runtime(provider, max_tokens=321).chat([], _trace(), role="speaker")
    assert provider.max_tokens == [321]
    with pytest.raises(ValueError):
        Runtime(provider, max_tokens=0)


def _trace() -> Any:
    from app.ai import trace as run_trace  # noqa: PLC0415

    return run_trace.RunTrace(run_id="r", actor=ACTOR, utterance="hi")


@pytest.fixture
def scratch() -> Generator[list[str], None, None]:
    seen: list[str] = []

    @tool
    def note(text: str) -> str:
        """Append a note."""
        seen.append(tool_runtime.current_tool_actor.get())
        return f"noted: {text}"

    @tool
    def refuse(reason: str) -> str:
        """Always refuse."""
        raise ToolError(f"cannot: {reason}")

    yield seen


def test_a_refusal_is_the_records_error_and_a_bad_call_never_ran(scratch: list[str]) -> None:
    runtime = Runtime(ScriptedProvider([]))
    refused = runtime.dispatch(ToolCall(id="1", name="refuse", arguments={"reason": "no"}), ACTOR)
    assert (refused.result, refused.error) == (None, "cannot: no")
    assert run_record.classify(refused) == "failed"
    ghost = runtime.dispatch(ToolCall(id="2", name="ghost", arguments={}), ACTOR)
    bad = runtime.dispatch(ToolCall(id="3", name="note", arguments={"text": 7}), ACTOR)
    assert [run_record.classify(r) for r in (ghost, bad)] == ["not_run", "not_run"]
    assert (bad.error or "").startswith(run_record.INVALID_ARGUMENTS_PREFIX)
    assert scratch == []


def test_the_acting_identity_is_threaded_to_the_tool(scratch: list[str]) -> None:
    runtime = Runtime(ScriptedProvider([]))
    runtime.dispatch(ToolCall(id="1", name="note", arguments={"text": "a"}), ACTOR)
    runtime.dispatch(ToolCall(id="2", name="note", arguments={"text": "b"}), "agent:conductor")
    assert scratch == [ACTOR, "agent:conductor"] and ACTOR == "agent:loop"


def test_a_model_outage_while_compiling_fails_the_turn_with_no_reply(fleet: Fleet) -> None:
    provider = ScriptedProvider([ProviderRequestError("llama-server unreachable")])
    with pytest.raises(AgentRunFailed) as failed:
        Interpreter(provider, fleet).run("hi")
    assert failed.value.http_status == 502
    result = failed.value.result
    assert (result.stop_reason, result.reply, result.tool_calls) == ("provider_error", None, [])


def test_a_step_that_crashes_fails_the_turn_with_what_ran_first(
    fleet: Fleet, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ScriptedProvider(
        [_plan(_ask("tasks", "add a task to water the plants"), _ask("music", "grab Swim"))]
    )
    conductor = Interpreter(provider, fleet)
    real = conductor._call

    def crash_on_music(call: ToolCall, actor: str) -> Any:
        if call.name == "ask_music":
            raise RuntimeError("kaboom")
        return real(call, actor)

    monkeypatch.setattr(conductor, "_call", crash_on_music)
    with pytest.raises(AgentRunFailed) as failed:
        conductor.run("add the plants task and grab Swim")
    result = failed.value.result
    assert failed.value.http_status == 500 and result.stop_reason == "internal_error"
    assert [r.tool for r in result.tool_calls] == ["ask_tasks", "ask_music"]
    assert run_record.classify(result.tool_calls[0]) == "done"
    # The reply is the record, naming the write that happened: it never vanishes.
    assert result.reply is not None and "ask_tasks" in result.reply
    assert result.plan is not None and result.plan.status == "ran"
    assert [w.kind for w in world.writes()] == ["create"]


def test_failure_statuses_cover_exactly_the_failure_stops() -> None:
    assert FAILURE_STATUS == {"provider_error": 502, "timed_out": 504, "internal_error": 500}
