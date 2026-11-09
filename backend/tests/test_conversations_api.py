"""The conversations API: CRUD, the message → turn round trip, and progress.

The turn dependency is overridden with a real ``Interpreter`` over a
``ScriptedProvider`` (fake model, real machinery) — PCC's pattern. The planner
is scripted with plan JSON; ``ask_tasks`` is a scratch tool or a fake app.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import structlog
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.ai.interpreter import Interpreter
from app.ai.provider import ChatResult, ProviderRequestError
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
from tests.scripted_provider import ScriptedProvider, text_turn

_FLEET = Fleet(
    apps=(
        FleetApp(
            name="tasks",
            title="Tasks",
            upstream="127.0.0.1:8100",
            agent=AgentSpec(description="Tasks.", api="/api/agent", examples=("add a task",)),
        ),
    )
)
# A direct decline: code's line, one model call (the compile).
_DECLINE = text_turn('{"answer": "decline"}')


def _plan(*requests: str) -> ChatResult:
    steps = [
        {"id": f"s{n}", "do": "ask", "app": "tasks", "request": request, "destructive": False}
        for n, request in enumerate(requests, start=1)
    ]
    return text_turn(json.dumps({"steps": steps}))


def _use(provider: ScriptedProvider) -> None:
    """Route the turn dependency at a scripted provider (client fixture clears it)."""
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        provider, _FLEET, narrate=True
    )


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
    _use(ScriptedProvider([_DECLINE]))
    response = client.post("/api/agent/conversations/999/messages", json={"content": "hi"})
    assert response.status_code == 404


def test_post_message_runs_the_turn_and_persists_exchange(client: TestClient) -> None:
    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch ask for the test turn."""
        return f"facts about {message}"

    provider = ScriptedProvider([_plan("chess"), text_turn("All done.")])
    _use(provider)
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
            "tool": "ask_tasks",
            "arguments": {"message": "chess"},
            "result": "facts about chess",
            "error": None,
            # A scratch body, not a real delegate call — nothing else ran.
            "app_tools": None,
            "app_record": None,
            "app_confirm": None,
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
    def ask_tasks(message: str) -> str:
        """Scratch ask."""
        return "ok"

    _use(ScriptedProvider([_plan("one"), text_turn("First reply")]))
    conversation_id = _create_conversation(client)
    client.post(f"/api/agent/conversations/{conversation_id}/messages", json={"content": "one"})

    follow_up = ScriptedProvider([_DECLINE])
    _use(follow_up)
    client.post(f"/api/agent/conversations/{conversation_id}/messages", json={"content": "two"})

    # The second turn's planner call (#64): the planner prompt, then one user
    # message holding the state block, recall of the first exchange as data
    # (the reply labelled as words, not a record), and the new words last. No
    # replayed assistant turn, no role:tool messages, no tools offered.
    replayed = follow_up.requests[0]["messages"]
    assert [message["role"] for message in replayed] == ["system", "user"]
    opening = replayed[1]["content"]
    assert opening.startswith("[State at the start of this turn")
    assert 'The person said: "one"' in opening
    assert 'What you said then (your words, not a record): "First reply"' in opening
    assert opening.endswith("\n\nThe person said: two")
    assert follow_up.requests[0]["tools"] == []


def test_provider_failure_is_502_and_keeps_user_message(client: TestClient) -> None:
    _use(ScriptedProvider([ProviderRequestError("llama-server unreachable")]))
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
        "ack": None,
    }


def test_post_message_is_rate_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "agent_messages_per_min", 1)
    _use(ScriptedProvider([_DECLINE]))
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
    """A step running inside the HTTP turn sees the request's delegation
    context, and threads it remembers land in the delegate_threads table."""
    seen: dict[str, Any] = {}

    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch tool standing in for an ask_<app> body."""
        context = current_delegation_context()
        seen["master_conversation_id"] = context.master_conversation_id
        seen["budget"] = context.calls_per_turn_per_app
        context.charge_call("chess")
        context.remember_thread("chess", 42)
        return "delegated"

    _use(ScriptedProvider([_plan("go"), text_turn("done")]))
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
    """While POST …/messages blocks, the activity registry carries the turn's
    current beat — observed from inside a step's body, exactly when a poll of
    GET …/activity would see it."""
    observed: list[turn_activity.TurnActivity | None] = []
    conversation_id_holder: list[int] = []

    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch ask that records the in-flight activity beat."""
        observed.append(turn_activity.get(conversation_id_holder[0]))
        return "probed"

    _use(ScriptedProvider([_plan("probe"), text_turn("done")]))
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
    assert beat.tool == "ask_tasks"
    assert beat.iteration == 1
    assert beat.ack == "On it, asking tasks."  # the honest ack stays for the turn

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
    """No turn dependency is overridden here on purpose: a note that could 502
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
    provider = ScriptedProvider([_DECLINE])
    _use(provider)

    client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "what's left?"}
    )

    sent = provider.requests[0]["messages"]
    assert any("Zeds Dead → Dubstep" in str(message.get("content")) for message in sent)


class _TasksApp:
    """A fake PCC delegate API: the first message writes, the second hits a
    bug in conductor's client path (a tool that raises)."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/conversations"):
            return httpx.Response(201, json={"id": 5})
        if self.writes:
            raise RuntimeError("a bug, not a DelegateError")
        self.writes.append(request.read().decode())
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


def _real_tasks(monkeypatch: pytest.MonkeyPatch) -> _TasksApp:
    # The client's lifespan turns on structlog's cache-on-first-use, so the
    # first delegate_call logged here would pin the module's logger and blind
    # later ``capture_logs`` audits. Log through a throwaway proxy instead.
    monkeypatch.setattr(fleet_context, "logger", structlog.get_logger(fleet_context.__name__))
    tasks = _TasksApp()
    build_delegate_tools(
        _FLEET,
        lambda app: DelegateClient(
            "http://tasks.test/api/agent",
            client=httpx.Client(transport=httpx.MockTransport(tasks.handle)),
        ),
    )
    return tasks


def test_a_crash_after_a_delegate_write_keeps_the_write_in_the_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#52 / B2 on the plan path: ``ask_tasks`` wrote, then the next step
    crashed inside conductor. The turn is stored naming the write before the
    route answers 500, and the next turn's context carries it."""
    tasks = _real_tasks(monkeypatch)
    _use(ScriptedProvider([_plan("add a task to pay rent", "add a task to pay the gas bill")]))
    conversation_id = _create_conversation(client)

    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": "add tasks to pay rent and the gas bill"},
    )
    assert response.status_code == 500
    assert response.json()["detail"].startswith("agent run crashed in ask_tasks")
    assert len(tasks.writes) == 1

    messages = client.get(f"/api/agent/conversations/{conversation_id}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    stored = messages[1]
    assert stored["stop_reason"] == "internal_error"
    assert "ask_tasks" in stored["content"] and "Added task #41 pay rent." in stored["content"]
    assert [call["tool"] for call in stored["tool_calls"]] == ["ask_tasks", "ask_tasks"]

    follow_up = ScriptedProvider([_DECLINE])
    _use(follow_up)
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


def test_a_narrator_death_after_a_delegate_write_still_names_the_write(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conductor's model dies after the app wrote, while the narrator speaks:
    the turn completes on code's record reply, never a 502 that hides it."""
    tasks = _real_tasks(monkeypatch)
    _use(
        ScriptedProvider(
            [
                _plan("add a task to pay rent"),
                ProviderRequestError("llama-server request failed: timed out"),
            ]
        )
    )
    conversation_id = _create_conversation(client)
    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": "add a task to pay rent"},
    )
    assert response.status_code == 200
    assert len(tasks.writes) == 1
    assistant = response.json()["assistant_message"]
    assert assistant["stop_reason"] == "completed"
    assert "Added task #41 pay rent." in (assistant["content"] or "")
