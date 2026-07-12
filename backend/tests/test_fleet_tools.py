"""Delegate tools, list_agents, the prompt fleet layer, and the loop path.

All fakes — no live delegate calls. The ask_<app> tools are exercised against a
scriptable fake client so the contract behaviors (thread reuse, 404
recreate-once, typed faults → ToolError, the per-turn budget, and the audit
event) can be asserted deterministically.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from datetime import date, datetime

import pytest
import structlog

from app.ai.loop import AgentLoop, build_system_prompt
from app.fleet.context import (
    DelegationContext,
    InMemoryThreadStore,
    use_delegation_context,
)
from app.fleet.delegate import (
    DelegateRateLimited,
    DelegateThreadGone,
    DelegateUnavailable,
    MessageExchange,
    MessageRead,
    ToolCallRead,
)
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.tools import (
    build_delegate_tools,
    render_fleet_section,
)
from app.tools import registry
from app.tools.registry import ToolError
from tests.scripted_provider import ScriptedProvider
from tests.scripted_provider import text_turn as _text
from tests.scripted_provider import tool_calls_turn as _calls


def _fleet() -> Fleet:
    return Fleet(
        apps=(
            FleetApp(
                name="chess",
                title="Chess",
                upstream="127.0.0.1:8000",
                agent=AgentSpec(
                    description="Plays chess.",
                    api="/api/agent",
                    examples=("move my knight to f3", "castle kingside"),
                ),
            ),
            FleetApp(
                name="tasks",
                title="Project Command Center",
                upstream="127.0.0.1:8100",
                agent=AgentSpec(
                    description="Manages projects and tasks.",
                    api="/api/agent",
                    examples=("what's due today",),
                ),
            ),
            FleetApp(name="odysseus", title="Odysseus", upstream="127.0.0.1:7000", agent=None),
        )
    )


def _exchange(
    content: str | None = "played e4",
    *,
    tool_calls: list[ToolCallRead] | None = None,
    stop_reason: str | None = "completed",
) -> MessageExchange:
    when = datetime(2026, 7, 11, 0, 0, 0)
    return MessageExchange(
        user_message=MessageRead(
            id=1, conversation_id=1, role="user", content="msg", created_at=when
        ),
        assistant_message=MessageRead(
            id=2,
            conversation_id=1,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            created_at=when,
        ),
    )


class FakeClient:
    """A scriptable stand-in for DelegateClient, shared across ask_ calls.

    ``send_effects`` is a queue of what each ``send_message`` yields (a
    MessageExchange to return, or an exception to raise); it defaults to a
    plain successful exchange. Every create/send/delete is recorded in ``log``.
    """

    def __init__(
        self, log: list[tuple[object, ...]], send_effects: Sequence[object] | None = None
    ) -> None:
        self._log = log
        self._next_id = 1
        self._send_effects = list(send_effects or [])

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        cid = self._next_id
        self._next_id += 1
        self._log.append(("create", cid))
        return cid

    def send_message(self, conversation_id: int, message: str) -> MessageExchange:
        self._log.append(("send", conversation_id, message))
        effect: object = self._send_effects.pop(0) if self._send_effects else _exchange()
        if isinstance(effect, Exception):
            raise effect
        assert isinstance(effect, MessageExchange)
        return effect


def _factory(client: FakeClient) -> Callable[[FleetApp], FakeClient]:
    """A client_factory that always hands back the same shared fake client."""

    def factory(app: FleetApp) -> FakeClient:
        return client

    return factory


@pytest.fixture
def bound_context() -> Generator[DelegationContext, None, None]:
    context = DelegationContext(
        master_conversation_id="m1",
        thread_store=InMemoryThreadStore(),
        calls_per_turn_per_app=3,
    )
    with use_delegation_context(context):
        yield context


def _ask(name: str, message: str) -> str:
    result = registry.get_tool(name).fn(message=message)
    assert isinstance(result, str)
    return result


# --- registry integration -----------------------------------------------------


def test_tools_are_registered_with_manifest_descriptions_and_message_schema() -> None:
    names = build_delegate_tools(_fleet(), _factory(FakeClient([])))
    assert names == ["ask_chess", "ask_tasks", "list_agents"]

    specs = {spec.name: spec for spec in registry.tool_specs()}
    assert set(specs) == {"ask_chess", "ask_tasks", "list_agents"}
    assert specs["ask_chess"].description == "Plays chess."
    assert specs["ask_tasks"].description == "Manages projects and tasks."
    # Each ask_ tool takes a single required string `message`.
    for name in ("ask_chess", "ask_tasks"):
        schema = specs[name].parameters
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["required"] == ["message"]


# --- thread creation + reuse ---------------------------------------------------


def test_first_call_creates_thread_then_reuses_it(bound_context: DelegationContext) -> None:
    log: list[tuple[object, ...]] = []
    build_delegate_tools(_fleet(), _factory(FakeClient(log)))

    _ask("ask_chess", "play e4")
    _ask("ask_chess", "now develop")

    # One create (first call), then the second call reuses the same thread id.
    assert log == [("create", 1), ("send", 1, "play e4"), ("send", 1, "now develop")]
    assert bound_context.thread_for("chess") == 1


def test_each_app_gets_its_own_thread(bound_context: DelegationContext) -> None:
    log: list[tuple[object, ...]] = []
    build_delegate_tools(_fleet(), _factory(FakeClient(log)))

    _ask("ask_chess", "play e4")
    _ask("ask_tasks", "what's due")

    assert bound_context.thread_for("chess") == 1
    assert bound_context.thread_for("tasks") == 2


# --- reply + activity note formatting -----------------------------------------


def test_reply_relays_text_and_a_compact_activity_note(bound_context: DelegationContext) -> None:
    exchange = _exchange(
        content="two tasks due: rent, dentist",
        tool_calls=[
            ToolCallRead(tool="list_tasks", arguments={}, result="[...]"),
            ToolCallRead(tool="get_focus_plan", arguments={}, error="blocked"),
        ],
    )
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=[exchange])))

    reply = _ask("ask_tasks", "what's due today")
    assert reply.startswith("two tasks due: rent, dentist")
    # Compact note names the tools; the failed one is marked, no raw transcript.
    assert "[tasks did: list_tasks, get_focus_plan(failed)]" in reply
    assert "[...]" not in reply  # the raw tool result never leaks into history


def test_empty_reply_reports_stop_reason(bound_context: DelegationContext) -> None:
    exchange = _exchange(content=None, stop_reason="max_iterations")
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=[exchange])))

    reply = _ask("ask_chess", "do a thing")
    assert "without a text reply" in reply
    assert "max_iterations" in reply


# --- 404 → recreate once -------------------------------------------------------


def test_404_recreates_thread_and_retries_once(bound_context: DelegationContext) -> None:
    log: list[tuple[object, ...]] = []
    effects = [DelegateThreadGone("gone"), _exchange(content="ok now")]
    build_delegate_tools(_fleet(), _factory(FakeClient(log, send_effects=effects)))

    reply = _ask("ask_chess", "play e4")

    assert reply.startswith("ok now")
    # create(1) → send(1) 404 → recreate(2) → send(2) ok.
    assert log == [
        ("create", 1),
        ("send", 1, "play e4"),
        ("create", 2),
        ("send", 2, "play e4"),
    ]
    assert bound_context.thread_for("chess") == 2  # mapping updated to the fresh thread


def test_double_404_becomes_a_tool_error(bound_context: DelegationContext) -> None:
    effects = [DelegateThreadGone("gone"), DelegateThreadGone("gone again")]
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=effects)))

    with pytest.raises(ToolError, match="chess"):
        _ask("ask_chess", "play e4")


# --- typed faults → informative ToolError -------------------------------------


def test_rate_limited_becomes_tool_error_with_retry_hint(
    bound_context: DelegationContext,
) -> None:
    effects = [DelegateRateLimited("429", retry_after=9)]
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=effects)))

    with pytest.raises(ToolError) as excinfo:
        _ask("ask_tasks", "what's due")
    message = str(excinfo.value)
    assert "rate-limiting" in message
    assert "9s" in message
    assert "Do not retry" in message


def test_unavailable_becomes_tool_error(bound_context: DelegationContext) -> None:
    effects = [DelegateUnavailable("down")]
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=effects)))

    with pytest.raises(ToolError, match="unavailable"):
        _ask("ask_chess", "play e4")


# --- per-turn budget -----------------------------------------------------------


def test_per_turn_budget_exhaustion_raises_tool_error(
    bound_context: DelegationContext,
) -> None:
    # calls_per_turn_per_app=3 (fixture): the 4th call to one app is blocked.
    build_delegate_tools(_fleet(), _factory(FakeClient([])))

    for _ in range(3):
        _ask("ask_chess", "again")
    with pytest.raises(ToolError, match="per-turn limit"):
        _ask("ask_chess", "again")
    # A different app is unaffected by chess's budget.
    _ask("ask_tasks", "what's due")


def test_budget_zero_disables_the_limit() -> None:
    context = DelegationContext(
        master_conversation_id="mcp",
        thread_store=InMemoryThreadStore(),
        calls_per_turn_per_app=0,
    )
    build_delegate_tools(_fleet(), _factory(FakeClient([])))
    with use_delegation_context(context):
        for _ in range(10):
            _ask("ask_chess", "again")  # never blocked


# --- audit ---------------------------------------------------------------------


def test_every_delegate_call_emits_one_audit_event(bound_context: DelegationContext) -> None:
    build_delegate_tools(_fleet(), _factory(FakeClient([])))

    with structlog.testing.capture_logs() as logs:
        _ask("ask_chess", "play e4")

    events = [log for log in logs if log["event"] == "delegate_call"]
    assert len(events) == 1
    event = events[0]
    assert event["app"] == "chess"
    assert event["subagent_conversation_id"] == 1
    assert event["stop_reason"] == "completed"
    assert event["error"] is None
    assert isinstance(event["latency_ms"], int)


def test_failed_delegate_call_audits_the_error_class(bound_context: DelegationContext) -> None:
    effects = [DelegateUnavailable("down")]
    build_delegate_tools(_fleet(), _factory(FakeClient([], send_effects=effects)))

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(ToolError):
            _ask("ask_chess", "play e4")

    events = [log for log in logs if log["event"] == "delegate_call"]
    assert len(events) == 1
    assert events[0]["error"] == "DelegateUnavailable"
    assert events[0]["stop_reason"] is None


# --- list_agents ---------------------------------------------------------------


def test_list_agents_reports_fleet_from_manifests() -> None:
    build_delegate_tools(_fleet(), _factory(FakeClient([])))

    listing = registry.get_tool("list_agents").fn()
    apps = {entry["app"]: entry for entry in listing["agents"]}
    assert set(apps) == {"chess", "tasks"}
    assert apps["chess"]["tool"] == "ask_chess"
    assert apps["chess"]["description"] == "Plays chess."
    assert apps["tasks"]["examples"] == ["what's due today"]
    assert listing["non_agent_apps"] == ["odysseus"]


# --- prompt fleet layer --------------------------------------------------------


def test_render_fleet_section_lists_tools_and_examples() -> None:
    section = render_fleet_section(_fleet())
    assert "ask_chess" in section
    assert "ask_tasks" in section
    assert "Plays chess." in section
    # Examples are present as routing hints.
    assert "move my knight to f3" in section
    # Non-agent members are named as un-delegable.
    assert "odysseus" in section


def test_render_fleet_section_empty_without_agents() -> None:
    fleet = Fleet(apps=(FleetApp("odysseus", "Odysseus", "127.0.0.1:7000", agent=None),))
    assert render_fleet_section(fleet) == ""


def test_system_prompt_composes_base_glitch_fleet_then_date() -> None:
    section = render_fleet_section(_fleet())
    prompt = build_system_prompt(date(2026, 7, 11), fleet_section=section)

    base_at = prompt.index("Act only through your tools")
    glitch_at = prompt.index("you are Glitch")
    fleet_at = prompt.index("Your fleet")
    date_at = prompt.index("Today's date is 2026-07-11.")
    assert base_at < glitch_at < fleet_at < date_at
    # Routing examples ride in the prompt, not in tool docstrings.
    assert "move my knight to f3" in prompt


def test_system_prompt_without_fleet_section_is_unchanged() -> None:
    prompt = build_system_prompt(date(2026, 7, 11))
    assert "Your fleet" not in prompt
    assert prompt.rstrip().endswith("Today's date is 2026-07-11.")


# --- loop-level path -----------------------------------------------------------


def test_loop_delegates_via_context_then_answers() -> None:
    """Scripted provider calls ask_chess, then answers — the full path through
    the DelegationContext with a fake client."""
    log: list[tuple[object, ...]] = []
    exchange = _exchange(
        content="e4 played",
        tool_calls=[ToolCallRead(tool="make_move", arguments={"uci": "e2e4"}, result="{}")],
    )
    fleet = _fleet()
    build_delegate_tools(fleet, _factory(FakeClient(log, send_effects=[exchange])))
    section = render_fleet_section(fleet)

    provider = ScriptedProvider(
        [
            _calls(("ask_chess", {"message": "play e4"})),
            _text("done — e4 is on the board."),
        ]
    )
    loop = AgentLoop(provider, fleet_section=section)

    context = DelegationContext(
        master_conversation_id="master-1",
        thread_store=InMemoryThreadStore(),
        calls_per_turn_per_app=3,
    )
    with use_delegation_context(context):
        result = loop.run("play e4 in chess")

    assert result.stop_reason == "completed"
    assert result.reply == "done — e4 is on the board."
    assert [record.tool for record in result.tool_calls] == ["ask_chess"]
    assert result.tool_calls[0].error is None
    # The delegated reply + activity note was fed back to the model as a tool result.
    tool_feedback = provider.requests[1]["messages"][-1]
    assert tool_feedback["role"] == "tool"
    assert "e4 played" in tool_feedback["content"]
    assert "[chess did: make_move]" in tool_feedback["content"]
    # The system prompt carried the fleet layer.
    system_prompt = provider.requests[0]["messages"][0]["content"]
    assert "Your fleet" in system_prompt
    assert context.thread_for("chess") == 1
