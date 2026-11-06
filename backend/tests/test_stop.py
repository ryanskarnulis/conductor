"""Stop (#66): conductor's cancel, passed on to the app's cancel.

The loop stops at its next boundary and replies from the record; the route
reaches the in-flight turn; the ask tool hands the Stop to the app; the
client speaks the app's ``…/cancel``. The whole path, mid-delegate, is the
trajectory walk ``test_stop_mid_delegate`` and session eval S10.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai.loop import AgentLoop
from app.ai.stop import STOPPED_ERROR, Stop, use_stop
from app.api import routes_agent, turn_activity
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import (
    DelegateClient,
    DelegateThreadGone,
    EntityRef,
    MessageExchange,
    MessageRead,
    RecordEntry,
)
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.tools import registry
from app.tools.registry import ToolError
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn

# --- the loop ---------------------------------------------------------------------


@pytest.fixture
def stop() -> Stop:
    return Stop()


@pytest.fixture
def notes(stop: Stop) -> Generator[list[str], None, None]:
    store: list[str] = []

    @registry.tool
    def add_note(text: str) -> str:
        """Append a note."""
        store.append(text)
        return f"added: {text}"

    @registry.tool
    def add_then_stop(text: str) -> str:
        """Append a note; the person presses Stop while it runs."""
        store.append(text)
        stop.request()
        return f"added: {text}"

    yield store


def test_a_stop_before_the_run_calls_nothing(stop: Stop) -> None:
    provider = ScriptedProvider([text_turn("never")])
    stop.request()
    run = AgentLoop(provider).run("hi", stop=stop)
    assert run.stop_reason == "cancelled"
    assert run.reply is None  # nothing ran: the client shows its own stop line
    assert provider._turns, "a model call started after Stop"


def test_a_stop_mid_batch_skips_the_rest_and_replies_from_the_record(
    notes: list[str], stop: Stop
) -> None:
    provider = ScriptedProvider(
        [
            tool_calls_turn(("add_then_stop", {"text": "milk"}), ("add_note", {"text": "eggs"})),
            text_turn("never"),
        ]
    )
    run = AgentLoop(provider).run("add milk and eggs", stop=stop)
    assert run.stop_reason == "cancelled"
    assert notes == ["milk"]
    assert [(r.tool, r.error) for r in run.tool_calls] == [
        ("add_then_stop", None),
        ("add_note", STOPPED_ERROR),
    ]
    assert run.reply is not None
    assert run.reply.startswith("I stopped before finishing: you stopped me.")
    assert "- #1 add_then_stop" in run.reply.split("Not run:")[0]
    assert "#2 add_note" in run.reply.split("Not run:")[1]
    # Every call has its one answer in the transcript.
    assert sum(1 for m in run.messages if m["role"] == "tool") == 2
    assert provider._turns, "a model call started after Stop"


def test_a_stopped_run_is_never_narrated(notes: list[str], stop: Stop) -> None:
    class _StopsOnTheText(ScriptedProvider):
        def chat(self, *args: Any, **kwargs: Any) -> Any:
            result = super().chat(*args, **kwargs)
            if not result.tool_calls:
                stop.request()  # Stop lands while the planner writes its reply
            return result

    provider = _StopsOnTheText(
        [
            tool_calls_turn(("add_note", {"text": "milk"})),
            text_turn("Added milk."),
            text_turn("the narrator's words"),
        ]
    )
    run = AgentLoop(provider, narrate=True).run("add milk", stop=stop)
    assert run.stop_reason == "cancelled"
    assert run.reply is not None and "the narrator's words" not in run.reply
    assert "- #1 add_note" in run.reply
    assert len(provider._turns) == 1, "the narrator was called after Stop"


# --- the registry and the route ---------------------------------------------------------


def test_a_turn_has_a_fresh_stop_and_a_late_one_does_nothing() -> None:
    first = turn_activity.begin(1)
    assert turn_activity.request_stop(1) is first and first.is_set()
    turn_activity.end(1, first)
    assert turn_activity.request_stop(1) is None  # nothing left to trip the next turn
    second = turn_activity.begin(1)
    assert not second.is_set()
    turn_activity.end(1, first)  # a stale end never clears the live turn
    assert turn_activity.get(1) is not None
    turn_activity.end(1, second)


def _conversation(client: TestClient) -> int:
    return int(client.post("/api/agent/conversations", json={}).json()["id"])


def test_cancel_with_nothing_running_says_so(client: TestClient) -> None:
    conversation = _conversation(client)
    response = client.post(f"/api/agent/conversations/{conversation}/cancel")
    assert response.status_code == 200
    assert response.json() == {"cancel_requested": False}
    assert client.post("/api/agent/conversations/999/cancel").status_code == 404


def test_cancel_stops_the_turn_in_flight(client: TestClient) -> None:
    conversation = _conversation(client)
    answers: list[dict[str, Any]] = []

    @registry.tool
    def slow_write(text: str) -> str:
        """A write the person stops while it runs."""
        answers.append(client.post(f"/api/agent/conversations/{conversation}/cancel").json())
        return f"wrote {text}"

    provider = ScriptedProvider(
        [tool_calls_turn(("slow_write", {"text": "a"})), text_turn("never")]
    )
    app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(provider)
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "write a"}
    )
    assert response.status_code == 200
    assert answers == [{"cancel_requested": True}]
    assistant = response.json()["assistant_message"]
    assert assistant["stop_reason"] == "cancelled"
    assert "- #1 slow_write" in assistant["content"]
    assert turn_activity.get(conversation) is None


# --- the ask tool passes it on ----------------------------------------------------------


class _CancellableClient:
    """A delegate client whose app the person stops mid-message."""

    def __init__(self, stop: Stop) -> None:
        self.stop = stop
        self.cancelled: list[int] = []
        self.sent = 0

    def __enter__(self) -> _CancellableClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        return 7

    def send_message(
        self, conversation_id: int, message: str, *, confirm_token: str | None = None
    ) -> MessageExchange:
        self.sent += 1
        self.stop.request()
        self.stop.pass_on()  # what the route's background task does
        when = datetime(2026, 11, 6)
        done = RecordEntry(
            tool="create_task",
            outcome="done",
            touched=[EntityRef(app="tasks", kind="task", id="52", title="Book dentist")],
        )
        return MessageExchange(
            user_message=MessageRead(
                id=1, conversation_id=7, role="user", content=message, created_at=when
            ),
            assistant_message=MessageRead(
                id=2,
                conversation_id=7,
                role="assistant",
                content=None,  # a cancelled app run has no text (delegate-api)
                stop_reason="cancelled",
                created_at=when,
                record=[done],
            ),
        )

    def cancel(self, conversation_id: int) -> bool:
        self.cancelled.append(conversation_id)
        return True


def _tasks_fleet() -> Fleet:
    return Fleet(
        apps=(
            FleetApp(
                name="tasks",
                title="Tasks",
                upstream="127.0.0.1:8100",
                agent=AgentSpec(description="Manages tasks.", api="/api/agent", examples=()),
            ),
        )
    )


@pytest.fixture
def delegation() -> Generator[DelegationContext, None, None]:
    context = DelegationContext(
        master_conversation_id="m1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield context


def test_the_ask_tool_passes_the_stop_on_and_says_what_the_app_did(
    delegation: DelegationContext,
) -> None:
    stop = Stop()
    fake = _CancellableClient(stop)
    build_delegate_tools(_tasks_fleet(), lambda _: fake)
    with use_stop(stop):
        result = registry.get_tool("ask_tasks").fn(message="book the dentist")
    assert fake.cancelled == [7]
    assert isinstance(result, str)
    assert "stop reason: cancelled" in result
    assert "Done: create_task (Book dentist)." in result


def test_a_stopped_run_never_starts_a_delegate_call(delegation: DelegationContext) -> None:
    stop = Stop()
    fake = _CancellableClient(stop)
    build_delegate_tools(_tasks_fleet(), lambda _: fake)
    stop.request()
    with use_stop(stop), pytest.raises(ToolError, match=STOPPED_ERROR):
        registry.get_tool("ask_tasks").fn(message="book the dentist")
    assert fake.sent == 0
    assert delegation._calls_this_turn == {}  # never charged the budget


# --- the client ---------------------------------------------------------------------------


def _client(handler: Any) -> DelegateClient:
    transport = httpx.MockTransport(handler)
    return DelegateClient("http://app.test/api/agent", client=httpx.Client(transport=transport))


def test_the_client_posts_the_apps_cancel() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"cancel_requested": True})

    assert _client(handler).cancel(12) is True
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/agent/conversations/12/cancel"
    assert seen[0].headers["X-Agent-Actor"] == "agent:conductor"
    assert seen[0].extensions["timeout"]["read"] == 5.0
    assert json.loads(seen[0].content or b"null") is None


def test_an_app_without_a_cancel_route_is_a_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(DelegateThreadGone):
        _client(handler).cancel(12)


def test_passing_on_never_raises() -> None:
    stop = Stop()

    def broken() -> None:
        raise DelegateThreadGone("no cancel route")

    with stop.delegating(broken) as live:
        assert live
        stop.request()
        stop.pass_on()  # logged, not raised
