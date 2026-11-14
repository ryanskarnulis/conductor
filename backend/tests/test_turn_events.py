"""The live push channel (#38): the registry's stream, the step-event seam,
and ``GET …/events``.

The stream is followed for real (``turn_activity.follow`` on its own event
loop, or over HTTP); the turns are scripted (``ScriptedProvider``) over the
session evals' fake fleet, through the real route, interpreter and ledger.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import turn_events
from app.ai.interpreter import Interpreter
from app.api import routes_agent, turn_activity
from app.api.turn_activity import Heartbeat, TurnEvent
from app.db.models import LedgerStep, LedgerTurn
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import AgentSpec, Fleet, FleetApp, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.tools import registry
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn

Item = TurnEvent | Heartbeat


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _rest(stream: Any) -> list[Item]:
    return [item async for item in stream]


def _kinds(items: list[Item]) -> list[str]:
    return [item.kind for item in items if isinstance(item, TurnEvent)]


def _publish(cid: int, stop: Any, kind: str, **data: Any) -> None:
    turn_activity.publish(cid, stop, kind, data)


# --- the registry's stream ----------------------------------------------------------------


def test_a_stream_opens_on_a_snapshot_and_ends_when_no_turn_comes() -> None:
    async def go() -> list[Item]:
        return await _rest(turn_activity.follow(1, heartbeat=0.01, idle=0.03))

    items = _run(go())
    first = items[0]
    assert isinstance(first, TurnEvent)
    assert (first.kind, first.data) == ("snapshot", {"active": False})
    assert all(isinstance(item, Heartbeat) for item in items[1:])
    assert turn_activity.subscribers(1) == 0
    assert 1 not in turn_activity._CHANNELS


def test_events_arrive_in_order_and_the_stream_ends_with_the_turn() -> None:
    async def go() -> list[Item]:
        stream = turn_activity.follow(1, heartbeat=5)
        items: list[Item] = [await anext(stream)]
        stop = turn_activity.begin(1)
        turn_activity.update(1, kind="model", tool=None, iteration=1, ack="On it, asking tasks.")
        _publish(1, stop, "plan", steps=[{"id": "s1", "do": "ask", "app": "tasks"}], start=0)
        _publish(1, stop, "step_started", seq=1, tool="ask_tasks", arguments={"message": "x"})
        turn_activity.update(1, kind="tool", tool="ask_tasks", iteration=1)
        _publish(1, stop, "receipt", seq=1, tool="ask_tasks", arguments={}, outcome="done")
        _publish(1, stop, "paused", reason="app_confirm", question="Trash it?")
        turn_activity.end(1, stop, stop_reason="completed", message_id=7)
        items += await _rest(stream)
        return items

    items = _run(go())
    assert _kinds(items) == [
        "snapshot",
        "turn_started",
        "activity",
        "plan",
        "step_started",
        "activity",
        "receipt",
        "paused",
        "turn_ended",
    ]
    ids = [item.id for item in items if isinstance(item, TurnEvent)]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    last = items[-1]
    assert isinstance(last, TurnEvent)
    assert last.data == {"stop_reason": "completed", "message_id": 7}
    assert turn_activity.get(1) is None
    assert turn_activity.subscribers(1) == 0
    assert 1 not in turn_activity._CHANNELS


def test_a_quiet_turn_gets_heartbeats_and_never_idles_out() -> None:
    async def go() -> list[Item]:
        stop = turn_activity.begin(1)
        stream = turn_activity.follow(1, heartbeat=0.01, idle=0.01)
        items: list[Item] = [await anext(stream)]
        await asyncio.sleep(0.06)
        items += [await anext(stream), await anext(stream)]
        turn_activity.end(1, stop, stop_reason="completed")
        items += await _rest(stream)
        return items

    items = _run(go())
    snapshot = items[0]
    assert isinstance(snapshot, TurnEvent) and snapshot.data["active"] is True
    assert isinstance(items[1], Heartbeat) and isinstance(items[2], Heartbeat)
    assert _kinds(items)[-1] == "turn_ended"


def test_a_closed_stream_unsubscribes_at_once() -> None:
    async def go() -> None:
        stream = turn_activity.follow(1)
        await anext(stream)
        assert turn_activity.subscribers(1) == 1
        await stream.aclose()
        assert turn_activity.subscribers(1) == 0
        assert 1 not in turn_activity._CHANNELS

        # A client gone mid-wait: the stream's task is cancelled.
        async def consume() -> None:
            async for _ in turn_activity.follow(2):
                pass

        task = asyncio.create_task(consume())
        while turn_activity.subscribers(2) == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert turn_activity.subscribers(2) == 0

    _run(go())


def test_a_reconnect_starts_from_a_snapshot_of_the_turn_so_far() -> None:
    stop = turn_activity.begin(1)
    turn_activity.update(1, kind="model", tool=None, iteration=1, ack="On it, asking tasks.")
    plan = {"steps": [{"id": "s1"}, {"id": "s2"}], "start": 0}
    turn_activity.publish(1, stop, "plan", plan)
    _publish(1, stop, "step_started", seq=1, tool="ask_tasks", arguments={"message": "a"})
    _publish(
        1, stop, "receipt", seq=1, tool="ask_tasks", arguments={"message": "a"}, outcome="done"
    )
    _publish(1, stop, "step_started", seq=2, tool="ask_music", arguments={"message": "b"})
    turn_activity.update(1, kind="tool", tool="ask_music", iteration=1)

    async def go() -> TurnEvent:
        stream = turn_activity.follow(1)
        first = await anext(stream)
        await stream.aclose()
        assert isinstance(first, TurnEvent)
        return first

    snapshot = _run(go())
    assert snapshot.kind == "snapshot"
    data = snapshot.data
    assert data["active"] is True
    assert data["plan"] == plan
    assert data["activity"]["tool"] == "ask_music"
    assert data["activity"]["ack"] == "On it, asking tasks."
    first, second = data["steps"]
    assert (first["seq"], first["receipt"]["outcome"]) == (1, "done")
    # Started, no receipt: the snapshot never says more than the record.
    assert (second["seq"], second["tool"], second["receipt"]) == (2, "ask_music", None)
    assert data["paused"] is None


def test_a_resumed_turn_says_so_and_a_reconnect_still_knows() -> None:
    stop = turn_activity.begin(1)
    resumed = {"reason": "ask_user", "step": "s1", "answer": "“Book dentist”"}
    _publish(1, stop, "resumed", **resumed)
    _publish(1, stop, "plan", steps=[{"id": "s1"}, {"id": "s2"}], start=1)

    async def go() -> TurnEvent:
        stream = turn_activity.follow(1)
        first = await anext(stream)
        await stream.aclose()
        assert isinstance(first, TurnEvent)
        return first

    data = _run(go()).data
    assert data["resumed"] == resumed
    # A turn that resumed nothing says so too, and a fresh turn forgets it.
    turn_activity.begin(1)
    assert _run(go()).data["resumed"] is None


def test_a_follower_that_falls_behind_is_resynced_never_grown() -> None:
    async def go() -> list[Item]:
        stream = turn_activity.follow(1, heartbeat=5)
        items: list[Item] = [await anext(stream)]
        stop = turn_activity.begin(1)
        for iteration in range(1, 3 * turn_activity.QUEUE_LIMIT + 1):
            turn_activity.update(1, kind="model", tool=None, iteration=iteration)
            await asyncio.sleep(0)
        (follower,) = turn_activity._CHANNELS[1].followers
        assert follower.queue.qsize() <= turn_activity.QUEUE_LIMIT
        items.append(await anext(stream))
        turn_activity.end(1, stop, stop_reason="completed")
        items += await _rest(stream)
        return items

    items = _run(go())
    resync = items[1]
    assert isinstance(resync, TurnEvent) and resync.kind == "snapshot"
    assert resync.data["activity"]["iteration"] == 3 * turn_activity.QUEUE_LIMIT
    # Nothing older than the snapshot follows it.
    assert _kinds(items[2:]) == ["turn_ended"]


def test_a_turns_state_and_followers_are_bounded() -> None:
    stop = turn_activity.begin(1)
    for seq in range(1, turn_activity.MAX_STEPS + 20):
        _publish(1, stop, "step_started", seq=seq, tool="ask_tasks", arguments={})

    async def go() -> None:
        streams = [turn_activity.follow(1) for _ in range(turn_activity.MAX_SUBSCRIBERS + 1)]
        firsts = [await anext(stream) for stream in streams]
        assert isinstance(firsts[0], TurnEvent)
        assert len(firsts[0].data["steps"]) == turn_activity.MAX_STEPS
        # One too many: the oldest stream is closed.
        assert await _rest(streams[0]) == []
        assert turn_activity.subscribers(1) == turn_activity.MAX_SUBSCRIBERS
        for stream in streams[1:]:
            await stream.aclose()
        assert turn_activity.subscribers(1) == 0

    _run(go())


def test_a_stale_turns_events_are_dropped_and_the_first_plan_stands() -> None:
    old = turn_activity.begin(1)
    new = turn_activity.begin(1)
    _publish(1, old, "step_started", seq=1, tool="ask_tasks", arguments={})
    turn_activity.publish(1, new, "plan", {"steps": [{"id": "s1"}], "start": 0})
    turn_activity.publish(1, new, "plan", {"steps": [{"id": "s9"}], "start": 3})
    turn_activity.end(1, old, stop_reason="completed")
    assert turn_activity.get(1) is not None  # the old turn's end leaves the new one alone

    async def go() -> TurnEvent:
        stream = turn_activity.follow(1)
        first = await anext(stream)
        await stream.aclose()
        assert isinstance(first, TurnEvent)
        return first

    data = _run(go()).data
    assert data["steps"] == []
    assert data["plan"] == {"steps": [{"id": "s1"}], "start": 0}


# --- the seam: no step lands ahead of its record ----------------------------------------


def _tasks_only() -> Fleet:
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
        master_conversation_id="1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield context


def test_a_step_lands_only_once_its_record_exists(delegation: DelegationContext) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    seen_while_running: list[list[str]] = []

    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch ask that looks at the stream while it runs."""
        seen_while_running.append([kind for kind, _ in events])
        return "ok"

    provider = ScriptedProvider([_plan_turn(_ask("tasks", "go"))])
    with turn_events.publishing(lambda kind, data: events.append((kind, data))):
        run = Interpreter(provider, _tasks_only()).run("go")

    assert run.stop_reason == "completed"
    # Running: the plan and the step's start are out, its receipt isn't.
    assert seen_while_running == [["plan", "step_started"]]
    (_, plan), (_, started), (_, landed) = events
    assert [step["id"] for step in plan["steps"]] == ["s1"]
    assert (started["seq"], started["tool"]) == (1, "ask_tasks")
    assert (landed["seq"], landed["kind"], landed["outcome"]) == (1, "ask", "done")
    # An answer with no record: conductor's unknown receipt, as the ledger keeps it.
    assert [(r["source"], r["outcome"]) for r in landed["receipts"]] == [("conductor", "unknown")]


def test_no_sink_is_a_no_op_and_a_broken_sink_never_costs_the_turn(
    delegation: DelegationContext,
) -> None:
    turn_events.receipt(1, _record())  # nothing bound: nothing happens

    def explode(_kind: str, _data: dict[str, Any]) -> None:
        raise RuntimeError("sink bug")

    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch ask."""
        return "ok"

    provider = ScriptedProvider([_plan_turn(_ask("tasks", "go"))])
    with turn_events.publishing(explode):
        run = Interpreter(provider, _tasks_only()).run("go")
    assert run.stop_reason == "completed"
    assert [record.tool for record in run.tool_calls] == ["ask_tasks"]


def _record(**fields: Any) -> Any:
    from app.ai.runtime import ToolCallRecord  # noqa: PLC0415

    return ToolCallRecord(**{"tool": "ask_tasks", "arguments": {"message": "x"}, **fields})


def test_a_live_step_carries_the_ledgers_receipts_and_never_a_token() -> None:
    from app.services import ledger  # noqa: PLC0415

    entry = {"tool": "create_task", "outcome": "done", "touched": [{"id": "1", "title": "milk"}]}
    step = ledger.live_step(1, _record(result="ok", app_record=[entry]))
    assert (step["kind"], step["app"], step["outcome"]) == ("ask", "tasks", "done")
    assert step["receipts"] == [{**entry, "source": "app"}]
    # A v1 answer with no record: conductor's unknown receipt, never "nothing changed".
    (unknown,) = ledger.live_step(2, _record(result="ok"))["receipts"]
    assert (unknown["source"], unknown["outcome"]) == ("conductor", "unknown")
    asked = {"app": "tasks", "thread_id": 4, "token": "secret", "tool": "trash", "question": "Q?"}
    waiting = ledger.live_step(3, _record(result="ok", app_confirm=asked))
    assert waiting["confirm"] == {"app": "tasks", "tool": "trash", "question": "Q?"}
    assert "secret" not in json.dumps(waiting)


# --- through the route ---------------------------------------------------------------------


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


class _Follower:
    """Follows one conversation's stream on its own event loop, in a thread."""

    def __init__(self, conversation_id: int) -> None:
        self.items: list[Item] = []
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._collect(conversation_id)), daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 5
        while turn_activity.subscribers(conversation_id) == 0:
            assert time.monotonic() < deadline, "the stream never subscribed"
            time.sleep(0.001)

    async def _collect(self, conversation_id: int) -> None:
        async for item in turn_activity.follow(conversation_id, heartbeat=0.05, idle=5):
            self.items.append(item)

    def events(self) -> list[TurnEvent]:
        self._thread.join(timeout=5)
        assert not self._thread.is_alive(), "the stream never ended"
        return [item for item in self.items if isinstance(item, TurnEvent)]


@pytest.fixture
def turn(
    client: TestClient, fleet: Fleet, world: World
) -> Generator[Callable[[ScriptedProvider, str], tuple[dict[str, Any], list[TurnEvent]]]]:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])

    def run(provider: ScriptedProvider, words: str) -> tuple[dict[str, Any], list[TurnEvent]]:
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider, fleet, narrate=True
        )
        follower = _Follower(conversation)
        response = client.post(
            f"/api/agent/conversations/{conversation}/messages", json={"content": words}
        )
        assert response.status_code == 200, response.text
        return dict(response.json()["assistant_message"]), follower.events()

    yield run


def _plan_turn(*steps: dict[str, Any]) -> Any:
    body = {"steps": [{"id": f"s{n}", **step} for n, step in enumerate(steps, start=1)]}
    return text_turn(json.dumps(body))


def _plan_turn_resume(answer: str) -> Any:
    return text_turn(json.dumps({"resume": answer}))


def _ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


def test_a_compiled_turn_streams_plan_steps_and_receipts_as_the_ledger_keeps_them(
    turn: Any, db_session: Session
) -> None:
    provider = ScriptedProvider(
        [_plan_turn(_ask("tasks", "add a task to water the plants")), text_turn("Added it.")]
    )
    reply, events = turn(provider, "add a task to water the plants")
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds == ["snapshot", "turn_started", "plan", "step_started", "receipt", "turn_ended"]
    by_kind = {event.kind: event.data for event in events}
    assert by_kind["plan"]["start"] == 0
    assert [step["id"] for step in by_kind["plan"]["steps"]] == ["s1"]
    assert any(e.kind == "activity" and e.data["ack"] for e in events)  # the honest ack
    receipt = by_kind["receipt"]
    assert by_kind["turn_ended"] == {"stop_reason": "completed", "message_id": reply["id"]}

    # The receipt on the stream is the ledger's step, field for field.
    db_session.expire_all()
    (step,) = db_session.execute(select(LedgerStep)).scalars()
    assert (receipt["seq"], receipt["kind"], receipt["app"], receipt["outcome"]) == (
        step.seq,
        step.kind,
        step.app,
        step.outcome,
    )
    assert [r["outcome"] for r in receipt["receipts"]] == [r.outcome for r in step.receipts]
    assert [r["tool"] for r in receipt["receipts"]] == [r.tool for r in step.receipts]


def test_a_gated_plan_streams_its_pause_and_a_yes_resumes_from_the_gated_step(
    turn: Any,
) -> None:
    gated = _plan_turn(_ask("music", "delete the album download", destructive=True))
    _, events = turn(ScriptedProvider([gated]), "delete the album download")
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds == ["snapshot", "turn_started", "plan", "paused", "turn_ended"]
    paused = next(event.data for event in events if event.kind == "paused")
    assert (paused["reason"], paused["app"], paused["step"]) == ("gate", "music", "s1")
    assert paused["question"].startswith("Just to check")

    yes = ScriptedProvider(
        [_plan_turn({"do": "confirm", "answer": "yes"}), text_turn("Already gone.")]
    )
    _, events = turn(yes, "yes")
    plan = next(event.data for event in events if event.kind == "plan")
    resumed = [event for event in events if event.kind == "resumed"]
    assert [event.data for event in resumed] == [
        {"reason": "gate", "step": "s1", "answer": None, "untrusted": False, "entity": None}
    ]
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds.index("resumed") < kinds.index("plan")  # said before the plan runs again
    # The paused plan, from the gated step: what this turn runs.
    assert plan["start"] == 0 and plan["steps"][0]["app"] == "music"
    started = next(event.data for event in events if event.kind == "step_started")
    assert (started["seq"], started["tool"]) == (1, "ask_music")


def test_an_answer_to_ask_user_resumes_the_same_plan_and_says_what_s1_becomes(
    turn: Any,
) -> None:
    question = {"do": "ask_user", "question": "Which dentist thing?", "options": ["a", "b"]}
    first = _plan_turn(question, _ask("tasks", "push {s1} to friday"))
    _, events = turn(ScriptedProvider([first]), "push the dentist thing to friday")
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds == ["snapshot", "turn_started", "plan", "paused", "turn_ended"]
    assert not any(event.kind == "resumed" for event in events)

    answer = ScriptedProvider([_plan_turn_resume("the {appointment} one"), text_turn("Moved.")])
    _, events = turn(answer, "the appointment one")
    resumed = next(event.data for event in events if event.kind == "resumed")
    # The step it waited on, and the answer as `{s1}` will read (no braces).
    assert resumed == {
        "reason": "ask_user",
        "step": "s1",
        "answer": "the (appointment) one",
        "untrusted": False,
        "entity": None,
    }
    plan = next(event.data for event in events if event.kind == "plan")
    assert plan["start"] == 1  # the question was asked last turn


def test_the_pick_fills_s1_as_the_entity_state_named_and_resumed_carries_it(
    turn: Any, world: World
) -> None:
    # Two dentist tasks on record in this conversation (#120).
    added = _plan_turn(_ask("tasks", "add a task to call the dentist about the bill"))
    turn(ScriptedProvider([added, text_turn("Added.")]), "add a task to call the dentist")
    read = _plan_turn(_ask("tasks", "what dentist tasks do I have?"))
    turn(ScriptedProvider([read, text_turn("Two.")]), "what dentist tasks do I have?")
    created = max(world.live_tasks(), key=lambda task: task.id)
    assert "dentist" in created.title

    question = {
        "do": "ask_user",
        "question": "Which dentist task?",
        "options": ["dentist appointment", created.title],
    }
    first = _plan_turn(question, _ask("tasks", "push {s1} to friday"))
    turn(ScriptedProvider([first]), "push the dentist thing to friday")

    answer = ScriptedProvider([_plan_turn_resume("the first one"), text_turn("Moved.")])
    reply, events = turn(answer, "the first one")
    resumed = next(event.data for event in events if event.kind == "resumed")
    assert resumed == {
        "reason": "ask_user",
        "step": "s1",
        "answer": "“dentist appointment”",
        "untrusted": False,
        "entity": {
            "app": "tasks",
            "kind": "task",
            "ref": "3",
            "title": "dentist appointment",
            "untrusted": False,
        },
    }
    (call,) = reply["tool_calls"]
    assert call["arguments"]["message"] == "push “dentist appointment” to friday"
    # tasks takes v1.2 (#202): the picked task also goes as a ref, with its id.
    assert [ref["id"] for ref in call["arguments"]["refs"]][:1] == ["3"]
    moved = [w for w in world.writes() if w.kind == "update"]
    assert [w.ref for w in moved] == ["task:3"]


def test_the_yes_to_an_apps_own_question_resumes_too(turn: Any) -> None:
    trash = _plan_turn(_ask("tasks", "delete the groceries task", destructive=True))
    _, events = turn(ScriptedProvider([trash]), "delete the groceries task")
    assert [e.data["reason"] for e in events if e.kind == "paused"] == ["app_confirm"]
    yes = ScriptedProvider([_plan_turn({"do": "confirm", "answer": "yes"}), text_turn("Done.")])
    _, events = turn(yes, "yes")
    resumed = [e.data for e in events if e.kind == "resumed"]
    assert resumed == [
        {"reason": "app_confirm", "step": None, "answer": None, "untrusted": False, "entity": None}
    ]


def test_the_turn_ends_on_the_stream_only_after_the_ledger_committed_it(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_at_end: list[str] = []
    real_end = turn_activity.end

    def spy(conversation_id: int, stop: Any = None, **ended: Any) -> None:
        db_session.expire_all()
        last = db_session.execute(select(LedgerTurn).order_by(LedgerTurn.id.desc())).scalar()
        ledger_at_end.append(last.status if last else "none")
        real_end(conversation_id, stop, **ended)

    monkeypatch.setattr(turn_activity, "end", spy)
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        ScriptedProvider([text_turn('{"answer": "decline"}')]), Fleet(apps=())
    )
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    follower = _Follower(conversation)
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "hi"}
    )
    assert response.status_code == 200
    assert ledger_at_end == ["ended"]
    assert follower.events()[-1].data["stop_reason"] == "completed"


def test_a_failed_turn_ends_the_stream_with_its_stop_reason(client: TestClient) -> None:
    from app.ai.provider import ProviderRequestError  # noqa: PLC0415

    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        ScriptedProvider([ProviderRequestError("model down")]), Fleet(apps=())
    )
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    follower = _Follower(conversation)
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "hi"}
    )
    assert response.status_code == 502
    ended = follower.events()[-1]
    assert (ended.kind, ended.data) == (
        "turn_ended",
        {"stop_reason": "provider_error", "message_id": None},
    )
    assert turn_activity.get(conversation) is None


# --- over HTTP -------------------------------------------------------------------------------


def test_the_events_route_404s_for_an_unknown_conversation(client: TestClient) -> None:
    assert client.get("/api/agent/conversations/999/events").status_code == 404


def test_the_events_route_speaks_server_sent_events(client: TestClient) -> None:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    stop = turn_activity.begin(conversation)

    def the_turn() -> None:
        deadline = time.monotonic() + 5
        while turn_activity.subscribers(conversation) == 0:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        _publish(conversation, stop, "step_started", seq=1, tool="ask_tasks", arguments={})
        turn_activity.end(conversation, stop, stop_reason="completed", message_id=3)

    thread = threading.Thread(target=the_turn, daemon=True)
    thread.start()
    response = client.get(f"/api/agent/conversations/{conversation}/events")
    thread.join(timeout=5)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    parsed = [
        dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        for block in blocks
    ]
    assert [block["event"] for block in parsed] == ["snapshot", "step_started", "turn_ended"]
    assert parsed[0]["retry"] == str(turn_activity.RETRY_MS)
    assert json.loads(parsed[0]["data"])["active"] is True
    assert json.loads(parsed[2]["data"]) == {"stop_reason": "completed", "message_id": 3}
    ids = [int(block["id"]) for block in parsed]
    assert ids == sorted(ids)
    assert turn_activity.subscribers(conversation) == 0


# --- control steps (#74) --------------------------------------------------------------------


def test_an_undo_streams_its_steps_and_receipts_and_waits_on_the_apps_question(
    turn: Any,
) -> None:
    s3 = _plan_turn(
        _ask("music", "download the new Caribou album"),
        _ask("tasks", "add a task to sort {s1} this weekend"),
    )
    turn(ScriptedProvider([s3, text_turn("Queued “Honey”.")]), "download and add a task")
    _, events = turn(ScriptedProvider([_plan_turn({"do": "undo", "target": "last"})]), "undo")
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds == [
        "snapshot",
        "turn_started",
        "plan",
        "receipt",  # the download can't be undone: a step that won't run
        "step_started",
        "receipt",
        "paused",
        "turn_ended",
    ]
    plan = next(event.data for event in events if event.kind == "plan")
    assert [(step["app"], step["destructive"]) for step in plan["steps"]] == [
        ("music", False),
        ("tasks", True),
    ]
    note, asked = [event.data for event in events if event.kind == "receipt"]
    assert (note["seq"], note["app"], note["outcome"]) == (1, "music", "not_run")
    assert (asked["seq"], asked["app"], asked["outcome"]) == (2, "tasks", "done")
    assert asked["confirm"]["question"].startswith("Trash ")
    paused = next(event.data for event in events if event.kind == "paused")
    assert (paused["reason"], paused["app"]) == ("app_confirm", "tasks")

    yes = ScriptedProvider([_plan_turn({"do": "confirm", "answer": "yes"}), text_turn("Done.")])
    _, events = turn(yes, "yes")
    landed = [event.data for event in events if event.kind == "receipt"]
    assert [(r["tool"], r["outcome"]) for r in landed] == [
        ("confirm", "done"),
        ("ask_music", "not_run"),
    ]
    assert "undone_by_step_id" not in json.dumps(landed)  # ledger internals stay home
    assert events[-1].kind == "turn_ended"


def test_a_cancelled_pause_ends_the_turn_with_nothing_left_waiting(turn: Any) -> None:
    gated = _plan_turn(_ask("music", "delete the album download", destructive=True))
    turn(ScriptedProvider([gated]), "delete the album download")
    reply, events = turn(
        ScriptedProvider([_plan_turn({"do": "cancel", "target": "pending"})]), "cancel"
    )
    kinds = [event.kind for event in events if event.kind != "activity"]
    assert kinds == ["snapshot", "turn_started", "turn_ended"]
    assert events[-1].data == {"stop_reason": "completed", "message_id": reply["id"]}
    assert reply["content"].startswith("Okay, cancelled: I won't ask music")
