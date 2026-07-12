"""The conversations API: CRUD, the message → loop round trip, and progress.

The loop dependency is overridden with a real ``AgentLoop`` over a
``ScriptedProvider`` (fake model, real machinery) — PCC's pattern.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.ai.loop import AgentLoop
from app.ai.provider import ProviderRequestError
from app.api import routes_agent, turn_activity
from app.config import get_settings
from app.fleet.context import current_delegation_context
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.tools import registry
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn


def _use_loop(provider: ScriptedProvider) -> None:
    """Route the loop dependency at a scripted provider (client fixture clears it)."""
    app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(provider)


def _create_conversation(client: TestClient, title: str | None = None) -> int:
    body: dict[str, Any] = {} if title is None else {"title": title}
    response = client.post("/api/agent/conversations", json=body)
    assert response.status_code == 201
    return int(response.json()["id"])


def test_conversation_crud_and_recency_order(client: TestClient) -> None:
    first = _create_conversation(client, title="errands")
    second = _create_conversation(client)

    listed = client.get("/api/agent/conversations").json()
    assert [item["id"] for item in listed] == [second, first]
    assert listed[1]["title"] == "errands"
    assert listed[0]["title"] is None

    detail = client.get(f"/api/agent/conversations/{first}").json()
    assert detail["id"] == first
    assert detail["messages"] == []

    assert client.delete(f"/api/agent/conversations/{first}").status_code == 204
    assert client.get(f"/api/agent/conversations/{first}").status_code == 404
    assert [item["id"] for item in client.get("/api/agent/conversations").json()] == [second]


def test_missing_conversation_is_404_everywhere(client: TestClient) -> None:
    assert client.get("/api/agent/conversations/999").status_code == 404
    assert client.delete("/api/agent/conversations/999").status_code == 404
    assert client.get("/api/agent/conversations/999/activity").status_code == 404
    _use_loop(ScriptedProvider([text_turn("unused")]))
    response = client.post("/api/agent/conversations/999/messages", json={"content": "hi"})
    assert response.status_code == 404


def test_post_message_runs_loop_and_persists_exchange(client: TestClient) -> None:
    @registry.tool
    def scratch_lookup(topic: str) -> str:
        """Scratch lookup tool for the test loop."""
        return f"facts about {topic}"

    provider = ScriptedProvider(
        [tool_calls_turn(("scratch_lookup", {"topic": "chess"})), text_turn("All done.")]
    )
    _use_loop(provider)
    conversation_id = _create_conversation(client)

    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": "look up chess"},
    )
    assert response.status_code == 200
    exchange = response.json()
    assert exchange["user_message"]["role"] == "user"
    assert exchange["user_message"]["content"] == "look up chess"
    assistant = exchange["assistant_message"]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "All done."
    assert assistant["stop_reason"] == "completed"
    assert assistant["tool_calls"] == [
        {
            "tool": "scratch_lookup",
            "arguments": {"topic": "chess"},
            "result": "facts about chess",
            "error": None,
        }
    ]

    # The exchange is persisted, and the untitled conversation took its title
    # from the first user message.
    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert detail["title"] == "look up chess"
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]


def test_history_replays_text_turns_only(client: TestClient) -> None:
    @registry.tool
    def scratch_noop() -> str:
        """Scratch no-op tool."""
        return "ok"

    _use_loop(ScriptedProvider([tool_calls_turn(("scratch_noop", {})), text_turn("First reply")]))
    conversation_id = _create_conversation(client)
    client.post(f"/api/agent/conversations/{conversation_id}/messages", json={"content": "one"})

    follow_up = ScriptedProvider([text_turn("Second reply")])
    _use_loop(follow_up)
    client.post(f"/api/agent/conversations/{conversation_id}/messages", json={"content": "two"})

    # The second run's context: system + prior text turns + new user turn —
    # no role:tool messages, no replayed tool trajectory.
    replayed = follow_up.requests[0]["messages"]
    assert [message["role"] for message in replayed] == ["system", "user", "assistant", "user"]
    assert replayed[1]["content"] == "one"
    assert replayed[2]["content"] == "First reply"
    assert replayed[3]["content"] == "two"


def test_provider_failure_is_502_and_keeps_user_message(client: TestClient) -> None:
    _use_loop(ScriptedProvider([ProviderRequestError("llama-server unreachable")]))
    conversation_id = _create_conversation(client)

    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "hello?"}
    )
    assert response.status_code == 502

    messages = client.get(f"/api/agent/conversations/{conversation_id}").json()["messages"]
    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"] == "hello?"
    # The failed turn left no dangling in-flight activity.
    activity = client.get(f"/api/agent/conversations/{conversation_id}/activity").json()
    assert activity == {
        "active": False,
        "kind": None,
        "tool": None,
        "iteration": None,
        "elapsed_seconds": None,
    }


def test_post_message_is_rate_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "agent_messages_per_min", 1)
    _use_loop(ScriptedProvider([text_turn("ok")]))
    conversation_id = _create_conversation(client)

    first = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "one"}
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "two"}
    )
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_message_length_cap(client: TestClient) -> None:
    conversation_id = _create_conversation(client)
    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "x" * 8001}
    )
    assert response.status_code == 422


def test_run_binds_delegation_context_over_db_thread_store(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A tool running inside the HTTP loop sees the request's delegation
    context, and threads it remembers land in the delegate_threads table."""
    seen: dict[str, Any] = {}

    @registry.tool
    def scratch_delegate() -> str:
        """Scratch tool standing in for an ask_<app> body."""
        context = current_delegation_context()
        seen["master_conversation_id"] = context.master_conversation_id
        seen["budget"] = context.calls_per_turn_per_app
        context.charge_call("chess")
        context.remember_thread("chess", 42)
        return "delegated"

    _use_loop(ScriptedProvider([tool_calls_turn(("scratch_delegate", {})), text_turn("done")]))
    conversation_id = _create_conversation(client)
    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "go"}
    )
    assert response.status_code == 200

    assert seen["master_conversation_id"] == str(conversation_id)
    assert seen["budget"] == get_settings().conductor_delegate_calls_per_turn_per_app
    # The remembered thread went through the DB-backed store, so a fresh
    # store instance (fresh session) reads it back.
    assert DbThreadStore(session_factory).get(str(conversation_id), "chess") == 42


def test_activity_is_reported_while_the_run_is_in_flight(client: TestClient) -> None:
    """While POST …/messages blocks, the activity registry carries the loop's
    current beat — observed from inside a tool body, exactly when a poll of
    GET …/activity would see it."""
    observed: list[turn_activity.TurnActivity | None] = []
    conversation_id_holder: list[int] = []

    @registry.tool
    def scratch_probe() -> str:
        """Scratch tool that records the in-flight activity beat."""
        observed.append(turn_activity.get(conversation_id_holder[0]))
        return "probed"

    _use_loop(ScriptedProvider([tool_calls_turn(("scratch_probe", {})), text_turn("done")]))
    conversation_id = _create_conversation(client)
    conversation_id_holder.append(conversation_id)

    idle = client.get(f"/api/agent/conversations/{conversation_id}/activity").json()
    assert idle["active"] is False

    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages", json={"content": "probe"}
    )
    assert response.status_code == 200

    assert len(observed) == 1
    beat = observed[0]
    assert beat is not None
    assert beat.kind == "tool"
    assert beat.tool == "scratch_probe"
    assert beat.iteration == 1

    # The turn is over: the poll target reports idle again.
    after = client.get(f"/api/agent/conversations/{conversation_id}/activity").json()
    assert after["active"] is False
