"""Jobs in the UI (#90): the live channel finds the jobs a page missed, and a
card's Cancel is the direct route (no model), recorded as the person's note.

Built on ``test_jobs``'s fixtures: the session evals' fake fleet and world,
the real routes, interpreter, ledger and watcher (driven with ``sweep()``).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api import routes_agent, turn_activity
from app.api.turn_activity import Heartbeat, TurnEvent
from app.db.models import LedgerStep, LedgerTurn
from app.fleet.jobs import JobError, JobGone, JobRead
from app.fleet.manifests import Fleet
from app.main import app
from tests.fake_world import World
from tests import test_jobs as base
from tests.test_jobs import (
    Clock,
    FakeCancel,
    WorldJobs,
    _job,
    _journal_job,
    _notes,
    _queue_playlist,
    _watcher,
)

# The jobs tests' fixtures, shared by assignment.
events = base.events
fleet = base.fleet
jobs = base.jobs
run_with = base.run_with
world = base.world

# --- the live channel finds the jobs a page missed -------------------------------------------


def _seed_of(*jobs: dict[str, Any]) -> Any:
    return lambda: list(jobs)


def _running(ref: str, done: int, **more: Any) -> dict[str, Any]:
    return {"app": "music", "ref": ref, "status": "working", "done": done, "ended": False} | more


def _parse(text: str) -> list[dict[str, str]]:
    blocks = [block for block in text.split("\n\n") if block.strip()]
    return [
        dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        for block in blocks
    ]


def test_a_seeded_snapshot_lists_the_ledgers_jobs_and_the_stream_follows_them_to_their_end() -> (
    None
):
    async def go() -> list[TurnEvent | Heartbeat]:
        stream = turn_activity.follow(21, heartbeat=5, seed=_seed_of(_running("1", 5)))
        items: list[TurnEvent | Heartbeat] = [await anext(stream)]
        turn_activity.publish_job(21, _running("1", 20))
        turn_activity.publish_job(21, _running("1", 40, ended=True, note_message_id=9))
        async for item in stream:
            items.append(item)
        return items

    items = asyncio.run(go())
    snapshot = items[0]
    assert isinstance(snapshot, TurnEvent)
    assert snapshot.data["jobs"] == [_running("1", 5)]
    kinds = [item.kind for item in items if isinstance(item, TurnEvent)]
    assert kinds == ["snapshot", "job", "job"]
    assert 21 not in turn_activity._CHANNELS


def test_a_seed_never_overrules_the_channels_newer_report_and_a_failed_seed_adds_nothing() -> None:
    turn_activity.publish_job(22, _running("1", 30))

    async def snapshot(seed: Any) -> TurnEvent:
        stream = turn_activity.follow(22, heartbeat=5, seed=seed)
        try:
            first = await anext(stream)
        finally:
            await stream.aclose()
        assert isinstance(first, TurnEvent)
        return first

    newer = asyncio.run(snapshot(_seed_of(_running("1", 5), _running("2", 1))))
    assert [(job["ref"], job["done"]) for job in newer.data["jobs"]] == [("1", 30), ("2", 1)]

    def broken() -> list[dict[str, Any]]:
        raise RuntimeError("no database")

    failed = asyncio.run(snapshot(broken))
    assert [(job["ref"], job["done"]) for job in failed.data["jobs"]] == [("1", 30)]


def test_the_events_route_finds_a_running_job_after_its_turn_ended(
    run_with: Any, client: TestClient, world: World
) -> None:
    _queue_playlist(run_with)
    conversation = run_with.conversation
    assert turn_activity.get(conversation) is None

    def the_end() -> None:
        deadline = time.monotonic() + 5
        while turn_activity.subscribers(conversation) == 0:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        time.sleep(0.2)  # past the snapshot and its seed
        turn_activity.publish_job(
            conversation, _running("1", 38, status="completed", ended=True, note_message_id=5)
        )

    thread = threading.Thread(target=the_end, daemon=True)
    thread.start()
    response = client.get(f"/api/agent/conversations/{conversation}/events")
    thread.join(timeout=5)

    parsed = _parse(response.text)
    assert [block["event"] for block in parsed] == ["snapshot", "job"]
    seeded = json.loads(parsed[0]["data"])
    assert seeded["active"] is False
    assert [(j["app"], j["ref"], j["status"], j["done"], j["total"]) for j in seeded["jobs"]] == [
        ("music", "1", "submitted", 0, 40)
    ]
    assert (seeded["jobs"][0]["title"], seeded["jobs"][0]["untrusted"]) == ("your playlist", True)


# --- cancel from the card ----------------------------------------------------------------------


@pytest.fixture
def card(
    client: TestClient,
    fleet: Fleet,
    world: World,
    jobs: WorldJobs,
) -> Generator[Any, None, None]:
    """The card's cancel over the real route. No model is reachable: a model
    call fails the test."""

    def no_model() -> Any:
        raise AssertionError("a card's cancel never calls the model")

    app.dependency_overrides[routes_agent.get_conductor] = no_model
    app.dependency_overrides[routes_agent.get_job_fleet] = lambda: fleet
    state: dict[str, Any] = {"canceller": jobs.cancel}
    app.dependency_overrides[routes_agent.get_job_canceller] = lambda: state["canceller"]

    def cancel_card(conversation: int, ref: str, app_name: str = "music") -> Any:
        return client.post(
            f"/api/agent/conversations/{conversation}/jobs/cancel",
            json={"app": app_name, "ref": ref},
        )

    cancel_card.use = lambda canceller: state.update(canceller=canceller)  # type: ignore[attr-defined]
    yield cancel_card


def _thread(client: TestClient, conversation: int) -> list[dict[str, Any]]:
    return list(client.get(f"/api/agent/conversations/{conversation}").json()["messages"])


def test_a_card_cancel_stops_the_job_with_no_model_and_records_the_persons_note(
    card: Any,
    client: TestClient,
    run_with: Any,
    db_session: Session,
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 1480)
    card.use(
        FakeCancel(JobRead(ref="31", status="working", done=164, total=1480, cancel_requested=True))
    )
    response = card(run_with.conversation, "31")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["status"], body["done"], body["total"], body["cancel_requested"]) == (
        "working",
        164,
        1480,
        True,
    )
    # The person's own note, with the done count at the cancel: no assistant reply invented.
    note = body["message"]
    assert note["role"] == "user"
    assert note["content"] == (
        "Asked music to stop “Library sort” from its card: 164 of 1,480 done when I asked."
    )
    assert _thread(client, run_with.conversation)[-1]["id"] == note["id"]
    # On the ledger: a note turn whose one step is the cancel, with its receipt.
    turn = db_session.execute(
        select(LedgerTurn).where(LedgerTurn.user_message_id == note["id"])
    ).scalar_one()
    assert (turn.kind, turn.status) == ("note", "ended")
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).scalar_one()
    assert (step.app, step.tool, step.outcome) == ("music", "cancel_job_music", "done")
    assert [(r.tool, r.outcome) for r in step.receipts] == [("cancel_job", "done")]
    job = _job(db_session)
    assert (job.status, job.done, job.total, job.noted_at) == ("working", 164, 1480, None)
    # The card hears it as asked to stop, not as stopped.
    conversation, payload = events[-1]
    assert conversation == run_with.conversation
    assert payload["cancel_requested"] is True and payload["ended"] is False
    assert (payload["status"], payload["done"]) == ("working", 164)


def test_a_card_cancel_over_the_world_stops_it_and_the_watcher_notes_it_once(
    card: Any,
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    response = card(run_with.conversation, "1")
    assert response.status_code == 200, response.text
    assert response.json()["message"]["content"] == (
        "Stopped “your playlist” from its card: 0 of 40 done when I asked."
    )
    assert jobs.cancels == ["1"]
    assert [r.kind for r in world.receipts] == ["queue", "cancel"]
    watcher, _ = _watcher(
        session_factory, fleet, jobs, Clock(), "“your playlist” was cancelled: 0 of 40."
    )
    assert watcher.sweep() == 1
    assert len(_notes(run_with)) == 1
    assert watcher.sweep(force=True) == 0


def test_a_card_cancel_on_a_job_that_already_ended_says_so_and_reopens_nothing(
    card: Any,
    run_with: Any,
    db_session: Session,
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    card.use(
        FakeCancel(
            JobRead(ref="31", status="completed", done=400, total=400, cancel_requested=False)
        )
    )
    response = card(run_with.conversation, "31")
    assert response.status_code == 200, response.text
    assert response.json()["cancel_requested"] is False
    assert response.json()["message"]["content"] == (
        "Tried to stop “Library sort” from its card, but it had already finished: 400 of 400 done."
    )
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).scalar_one()
    assert [r.outcome for r in step.receipts] == ["looked_up"]
    assert events == []  # the watcher's note is what ends a card, never this tap


def test_a_card_cancel_for_an_unknown_job_or_another_conversations_job_is_404(
    card: Any,
    client: TestClient,
    run_with: Any,
    db_session: Session,
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    other = int(client.post("/api/agent/conversations", json={}).json()["id"])
    before = len(_thread(client, run_with.conversation))
    for conversation, ref in ((run_with.conversation, "999"), (other, "31")):
        response = card(conversation, ref)
        assert response.status_code == 404, response.text
    assert card(run_with.conversation, "31", app_name="chess").status_code == 404
    assert card(999, "31").status_code == 404
    assert len(_thread(client, run_with.conversation)) == before
    assert _thread(client, other) == []


def test_a_card_cancel_that_cant_reach_the_app_records_nothing(
    card: Any,
    client: TestClient,
    run_with: Any,
    db_session: Session,
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    before = len(_thread(client, run_with.conversation))
    card.use(FakeCancel(JobError("music couldn't be reached")))
    response = card(run_with.conversation, "31")
    assert response.status_code == 502
    assert response.json()["detail"] == "music couldn't be reached"
    card.use(FakeCancel(JobGone("music doesn't know that job")))
    assert card(run_with.conversation, "31").status_code == 410
    assert len(_thread(client, run_with.conversation)) == before
    assert db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).first() is None
    job = _job(db_session)
    assert (job.status, job.done) == ("working", 112)  # as last reported: nothing happened
