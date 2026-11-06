"""The conversations API: CRUD, the message → loop round trip, and progress.

The loop dependency is overridden with a real ``AgentLoop`` over a
``ScriptedProvider`` (fake model, real machinery) — PCC's pattern.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import structlog
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.ai.loop import AgentLoop
from app.ai.provider import ProviderRequestError
from app.api import routes_agent, turn_activity
from app.config import get_settings
from app.fleet import context as fleet_context
from app.fleet.context import current_delegation_context
from app.fleet.delegate import DelegateClient
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.thread_store import DbThreadStore
from app.fleet.tools import build_delegate_tools
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
            # A local tool, not a delegate call — nothing else ran.
            "app_tools": None,
            "app_record": None,
        }
    ]

    # The exchange is persisted, and the untitled conversation took its title
    # from the first user message.
    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert detail["title"] == "look up chess"
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]


def test_no_chat_turn_is_replayed_the_turn_opens_with_state_and_recall(
    client: TestClient,
) -> None:
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

    # The second run's context (#64): the system prompt, then one user message
    # holding the state block, recall of the first exchange as data (the reply
    # labelled as words, not a record), and the new words last. No replayed
    # assistant turn, no role:tool messages.
    replayed = follow_up.requests[0]["messages"]
    assert [message["role"] for message in replayed] == ["system", "user"]
    opening = replayed[1]["content"]
    assert opening.startswith("[State at the start of this turn")
    assert 'The person said: "one"' in opening
    assert 'What you said then (your words, not a record): "First reply"' in opening
    assert opening.endswith("\n\ntwo")


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


# --- notes: what the person did without saying anything ------------------------


def test_a_note_is_stored_as_the_persons_turn_with_no_model_run(client: TestClient) -> None:
    """The sort panel files a song through the app directly; the thread still
    has to show that it was answered."""
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/agent/conversations/{conversation}/notes",
        json={"content": "Zeds Dead → Dubstep (5 songs)"},
    )

    assert response.status_code == 201
    note = response.json()
    assert note["role"] == "user", "the person acted; nothing is put in conductor's mouth"
    assert note["content"] == "Zeds Dead → Dubstep (5 songs)"
    assert note["tool_calls"] is None and note["stop_reason"] is None

    detail = client.get(f"/api/agent/conversations/{conversation}").json()
    assert [message["content"] for message in detail["messages"]] == [
        "Zeds Dead → Dubstep (5 songs)"
    ]


def test_a_note_needs_no_provider_at_all(client: TestClient) -> None:
    """No loop dependency is overridden here on purpose: a note that could 502
    because a model is cold is not a record, it is a second thing to go wrong."""
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/agent/conversations/{conversation}/notes", json={"content": "filed 5"}
    )

    assert response.status_code == 201


def test_a_note_on_a_missing_conversation_is_404(client: TestClient) -> None:
    assert (
        client.post("/api/agent/conversations/999/notes", json={"content": "x"}).status_code == 404
    )


def test_a_note_is_carried_into_the_next_turns_context(client: TestClient) -> None:
    """Otherwise conductor's next reply contradicts what is on screen."""
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]
    client.post(
        f"/api/agent/conversations/{conversation}/notes",
        json={"content": "Zeds Dead → Dubstep (5 songs)"},
    )
    provider = ScriptedProvider([text_turn("nice one")])
    _use_loop(provider)

    client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "what's left?"}
    )

    sent = provider.requests[0]["messages"]
    assert any("Zeds Dead → Dubstep" in str(message.get("content")) for message in sent)


def test_a_provider_death_after_a_delegate_write_keeps_the_write_in_the_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#52 / B2: conductor's model dies after ``ask_tasks`` wrote. The turn is
    stored naming the write, the route still answers 502, and the next turn's
    context carries it."""
    # The client's lifespan turns on structlog's cache-on-first-use, so the
    # first delegate_call logged here would pin the module's logger and blind
    # later ``capture_logs`` audits. Log through a throwaway proxy instead.
    monkeypatch.setattr(fleet_context, "logger", structlog.get_logger(fleet_context.__name__))
    writes: list[str] = []

    def tasks_app(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/conversations"):
            return httpx.Response(201, json={"id": 5})
        writes.append(request.read().decode())
        turn = {"conversation_id": 5, "created_at": "2026-11-05T00:00:00Z"}
        return httpx.Response(
            200,
            json={
                "user_message": {**turn, "id": 1, "role": "user", "content": "add pay rent"},
                "assistant_message": {
                    **turn,
                    "id": 2,
                    "role": "assistant",
                    "content": "Added task #41 pay rent.",
                    "tool_calls": [
                        {"tool": "create_task", "arguments": {"title": "pay rent"}, "result": "41"}
                    ],
                    "stop_reason": "completed",
                },
            },
        )

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
        lambda app: DelegateClient(
            "http://tasks.test/api/agent",
            client=httpx.Client(transport=httpx.MockTransport(tasks_app)),
        ),
    )
    dying = ScriptedProvider(
        [
            tool_calls_turn(("ask_tasks", {"message": "add a task to pay rent"})),
            ProviderRequestError("llama-server request failed: timed out"),
        ]
    )
    _use_loop(dying)
    conversation_id = _create_conversation(client)

    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": "add a task to pay rent"},
    )
    # Still a provider death to the client (and to the eval harness's infra check).
    assert response.status_code == 502
    assert response.json()["detail"] == "agent run failed: llama-server request failed: timed out"
    assert len(writes) == 1

    messages = client.get(f"/api/agent/conversations/{conversation_id}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    stored = messages[1]
    assert stored["stop_reason"] == "provider_error"
    assert "ask_tasks" in stored["content"] and "Added task #41 pay rent." in stored["content"]
    assert stored["tool_calls"][0]["tool"] == "ask_tasks"

    follow_up = ScriptedProvider([text_turn("It's on your list.")])
    _use_loop(follow_up)
    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": "did that go through?"},
    )
    assert response.status_code == 200
    # No chat turns replayed (#64): the write reaches the next turn as recall,
    # the code-kept record of what the app did.
    replayed = follow_up.requests[0]["messages"]
    assert [m["role"] for m in replayed] == ["system", "user"]
    opening = replayed[1]["content"]
    assert "The record of earlier turns" in opening
    assert "Added task #41 pay rent." in opening
    assert opening.endswith("did that go through?")
