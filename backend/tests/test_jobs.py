"""Jobs (#76): long steps detach, the watcher follows them, and one note posts
when each ends; ``cancel``/``amend`` on a running job go through the jobs
runtime.

The model is scripted (``ScriptedProvider``); the apps are the session evals'
fake fleet (``fake_world.py``), reached through the real route, interpreter,
delegate tools and ledger. The watcher is driven with ``sweep()`` on a fake
clock over the test database; its job reader is the world's jobs endpoint.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.ai import confirmation, interpreter, narrator, turn_events
from app.ai import plan as plans
from app.ai.interpreter import Interpreter
from app.ai.runtime import AgentRunResult, ToolCallRecord
from app.ai.narrator import JobEnd
from app.ai.provider import ChatResult, ProviderRequestError
from app.api import routes_agent, turn_activity
from app.api.turn_activity import Heartbeat, TurnEvent
from app.config import get_settings
from app.db.models import (
    Conversation,
    ConversationMessage,
    ConversationRole,
    LedgerJob,
    LedgerReceipt,
    LedgerStep,
    LedgerTurn,
)
from app.fleet import jobs as fleet_jobs
from app.fleet.jobs import JobError, JobGone, JobRead
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.services import conversations as conversations_service
from app.services import job_watcher, ledger
from app.services.job_watcher import JobWatcher
from tests.fake_world import PLAYLIST_MISSES, World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn

BACKEND = Path(__file__).resolve().parents[1]
PLAYLIST = "download this playlist https://example.com/playlist?list=PL40"


def plan_turn(body: dict[str, Any]) -> ChatResult:
    return text_turn(json.dumps(body))


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]}


def ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


def cancel(target: str = "pending") -> dict[str, Any]:
    return steps({"do": "cancel", "target": target})


def amend(change: str, target: str = "pending") -> dict[str, Any]:
    return steps({"do": "amend", "target": target, "change": change})


class Clock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class WorldJobs:
    """The world's jobs endpoint as a ``JobReader``/``JobCanceller``, counting calls."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.reads: list[str] = []
        self.cancels: list[str] = []

    def read(self, fleet_app: Any, ref: str) -> JobRead:
        self.reads.append(ref)
        body = self.world.job_read(ref)
        if body is None:
            raise JobGone("music doesn't know that job")
        return JobRead.model_validate(body)

    def cancel(self, fleet_app: Any, ref: str) -> JobRead:
        self.cancels.append(ref)
        body = self.world.job_cancel(ref)
        if body is None:
            raise JobGone("music doesn't know that job")
        return JobRead.model_validate(body)


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


@pytest.fixture
def jobs(world: World, monkeypatch: pytest.MonkeyPatch) -> WorldJobs:
    endpoint = WorldJobs(world)
    monkeypatch.setattr(interpreter, "cancel_job", endpoint.cancel)
    return endpoint


@pytest.fixture
def events() -> Generator[list[tuple[int, dict[str, Any]]], None, None]:
    seen: list[tuple[int, dict[str, Any]]] = []
    turn_events.set_job_sink(lambda cid, payload: seen.append((cid, payload)))
    try:
        yield seen
    finally:
        turn_events.set_job_sink(turn_activity.publish_job)


@pytest.fixture
def run_with(client: TestClient, fleet: Fleet, world: World) -> Generator[Any, None, None]:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])

    def run(provider: ScriptedProvider, words: str) -> dict[str, Any]:
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider, fleet, narrate=True
        )
        response = client.post(
            f"/api/agent/conversations/{conversation}/messages", json={"content": words}
        )
        assert response.status_code == 200, response.text
        assert provider._turns == [], "the script wasn't fully consumed"
        return dict(response.json()["assistant_message"])

    def messages() -> list[dict[str, Any]]:
        detail = client.get(f"/api/agent/conversations/{conversation}")
        return list(detail.json()["messages"])

    run.conversation = conversation  # type: ignore[attr-defined]
    run.messages = messages  # type: ignore[attr-defined]
    yield run


def _watcher(
    session_factory: sessionmaker[Session],
    fleet: Fleet,
    jobs: WorldJobs,
    clock: Clock,
    *notes: str | Exception,
) -> tuple[JobWatcher, ScriptedProvider]:
    provider = ScriptedProvider([n if isinstance(n, Exception) else text_turn(n) for n in notes])
    watcher = JobWatcher(
        session_factory, fleet, reader=jobs.read, provider=lambda: provider, clock=clock
    )
    return watcher, provider


def _queue_playlist(run_with: Any) -> dict[str, Any]:
    reply: dict[str, Any] = run_with(
        ScriptedProvider(
            [
                plan_turn(steps(ask("music", PLAYLIST))),
                text_turn("Queued your playlist: 40 tracks."),
            ]
        ),
        "grab this whole playlist",
    )
    return reply


def _notes(run_with: Any) -> list[dict[str, Any]]:
    return [m for m in run_with.messages() if m["stop_reason"] == narrator.JOB_NOTE]


def _job(db: Session) -> LedgerJob:
    db.expire_all()
    return db.execute(select(LedgerJob)).scalar_one()


GOOD_NOTE = (
    "“Your playlist” is in: 38 of 40 downloaded; “Lost Signal” and “Paper Moons” didn't match."
)


# --- detaching -----------------------------------------------------------------------------


def test_a_step_with_a_job_ref_detaches_and_the_ledger_tracks_the_job(
    run_with: Any, world: World, db_session: Session
) -> None:
    reply = _queue_playlist(run_with)
    assert reply["stop_reason"] == "completed"
    assert reply["content"] == "Queued “your playlist”: 40 tracks."
    job = _job(db_session)
    assert (job.app, job.ref, job.status, job.done, job.total) == ("music", "1", "submitted", 0, 40)
    assert (job.title, job.untrusted, job.noted_at) == ("your playlist", True, None)
    assert job.started_by_step_id is not None
    assert world.jobs[1].state == "queued"
    # The state block reads the job from the ledger, its title marked as outside text.
    lines = ledger.running_job_lines(db_session, run_with.conversation)
    assert lines == ["music job 1, “your playlist” (outside text): submitted, 0 of 40 done"]


# --- the watcher ---------------------------------------------------------------------------


def test_the_poll_interval_backs_off_after_two_minutes() -> None:
    assert job_watcher.poll_interval(timedelta(seconds=10)) == timedelta(seconds=5)
    assert job_watcher.poll_interval(timedelta(minutes=2)) == timedelta(seconds=30)


def test_the_watcher_polls_every_5s_then_every_30s(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _queue_playlist(run_with)
    clock = Clock()
    watcher, _ = _watcher(session_factory, fleet, jobs, clock)
    start = clock.now
    seen = []
    for offset in (0, 2, 5, 9, 10, 130, 140, 159, 161):
        clock.now = start + timedelta(seconds=offset)
        watcher.sweep()
        seen.append(len(jobs.reads))
    # Polled at once, then 5 s apart while young, then 30 s apart once the
    # job is past 2 minutes old.
    assert seen == [1, 1, 2, 2, 3, 4, 4, 4, 5]
    assert _notes(run_with) == []
    # Polls that moved nothing push nothing: the job sat at 0 of 40.
    assert events == [] or all(not payload["ended"] for _, payload in events)


def test_progress_is_pushed_as_it_changes_and_the_job_ends_with_one_note(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _queue_playlist(run_with)
    clock = Clock()
    watcher, provider = _watcher(session_factory, fleet, jobs, clock, GOOD_NOTE)
    world.jobs[1].done = 7
    assert watcher.sweep() == 0
    progress = [payload for _, payload in events]
    assert progress[-1] | {"title": None} == {
        "app": "music",
        "ref": "1",
        "status": "working",
        "done": 7,
        "total": 40,
        "title": None,
        "untrusted": True,
        "ended": False,
    }
    world.advance()
    clock.advance(seconds=5)
    assert watcher.sweep() == 1
    notes = _notes(run_with)
    assert [note["content"] for note in notes] == [GOOD_NOTE]
    ended = events[-1]
    assert ended[0] == run_with.conversation
    assert ended[1]["ended"] is True and ended[1]["status"] == "completed"
    assert (ended[1]["done"], ended[1]["total"]) == (38, 40)
    assert ended[1]["note_message_id"] == notes[0]["id"]
    job = _job(db_session)
    assert (job.status, job.done, job.note_message_id) == ("completed", 38, notes[0]["id"])
    assert job.noted_at is not None
    # The narrator read the app's final report, misses and all, as data.
    opening = provider.requests[0]["messages"][1]["content"]
    assert '"failed_items"' in opening and "Lost Signal" in opening
    # Ended: never polled or noted again.
    clock.advance(minutes=5)
    assert watcher.sweep(force=True) == 0
    assert len(_notes(run_with)) == 1


def test_a_restarted_watcher_never_posts_the_note_twice(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    world.advance()
    clock = Clock()
    first, _ = _watcher(session_factory, fleet, jobs, clock, GOOD_NOTE)
    assert first.sweep() == 1
    # A new process: a fresh watcher over the same ledger.
    second, provider = _watcher(session_factory, fleet, jobs, clock)
    assert second.sweep(force=True) == 0
    assert provider.requests == []
    assert len(_notes(run_with)) == 1


def test_two_watchers_racing_post_one_note(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    world.advance()
    clock = Clock()
    other, _ = _watcher(session_factory, fleet, jobs, clock, "The other one's note: 38 of 40.")

    class Racing(ScriptedProvider):
        def chat(self, messages: Any, **kwargs: Any) -> ChatResult:
            # While this watcher narrates, the other one finishes first.
            assert other.sweep(force=True) == 1
            return super().chat(messages, **kwargs)

    racing = Racing([text_turn(GOOD_NOTE)])
    watcher = JobWatcher(
        session_factory, fleet, reader=jobs.read, provider=lambda: racing, clock=clock
    )
    assert watcher.sweep() == 0
    notes = _notes(run_with)
    assert len(notes) == 1
    assert notes[0]["content"].startswith("The other one's note")


def test_the_note_names_what_the_narrator_dropped(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    world.advance()
    watcher, _ = _watcher(session_factory, fleet, jobs, Clock(), "Your playlist finished.")
    watcher.sweep()
    note = _notes(run_with)[0]["content"]
    # The outside title is quoted, and what the narrator left out is appended.
    assert note.startswith("“Your playlist” finished. For the record:")
    assert "38 of 40 done" in note
    for title in PLAYLIST_MISSES:
        assert f"“{title}”" in note


def test_a_narrator_failure_posts_codes_note(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    world.advance()
    error = ProviderRequestError("down")
    watcher, _ = _watcher(session_factory, fleet, jobs, Clock(), error)
    assert watcher.sweep() == 1
    assert _notes(run_with)[0]["content"] == (
        "“your playlist” finished: 38 of 40 done. “Lost Signal” and “Paper Moons” didn't make "
        "it. music said: “2 tracks couldn't be downloaded”"
    )


def test_after_six_hours_the_watcher_gives_up_honestly(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _queue_playlist(run_with)
    clock = Clock()
    watcher, provider = _watcher(session_factory, fleet, jobs, clock)
    world.jobs[1].done = 12
    watcher.sweep()
    clock.advance(hours=5, minutes=59)
    assert watcher.sweep() == 0
    clock.advance(minutes=2)
    assert watcher.sweep() == 1
    note = _notes(run_with)[0]["content"]
    assert note.startswith("I've lost track of “your playlist”: music hasn't said it finished")
    assert "don't know how it ended" in note and "12 of 40 done" in note
    assert "fail" not in note
    assert provider.requests == []  # code's words, never narrated
    reads = len(jobs.reads)
    clock.advance(hours=1)
    assert watcher.sweep(force=True) == 0
    assert len(jobs.reads) == reads  # it stopped polling
    assert events[-1][1]["ended"] is True and events[-1][1]["how"] == "lost"
    # What the job did is unknown: conductor's receipt says so.
    turn = db_session.execute(select(LedgerTurn).where(LedgerTurn.kind == "job")).scalar_one()
    receipts = turn.plans[0].steps[0].receipts
    assert [(r.source, r.outcome) for r in receipts] == [("conductor", "unknown")]
    assert ledger.running_jobs(db_session, run_with.conversation) == []


def test_a_job_its_app_forgot_is_lost_not_failed(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    del world.jobs[1]
    watcher, _ = _watcher(session_factory, fleet, jobs, Clock())
    assert watcher.sweep() == 1
    note = _notes(run_with)[0]["content"]
    assert note.startswith("music no longer knows about “your playlist”")


def test_an_unreachable_app_is_polled_again_later(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _queue_playlist(run_with)
    calls: list[str] = []

    def down(fleet_app: Any, ref: str) -> JobRead:
        calls.append(ref)
        raise JobError("music couldn't be reached")

    clock = Clock()
    watcher = JobWatcher(session_factory, fleet, reader=down, clock=clock)
    assert watcher.sweep() == 0
    assert watcher.sweep() == 0  # not due again yet
    clock.advance(seconds=5)
    assert watcher.sweep() == 0
    assert calls == ["1", "1"]
    assert _job(db_session).status == "submitted"


# --- the ledger invariant --------------------------------------------------------------------


def test_every_app_write_of_a_job_has_a_receipt_row(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _queue_playlist(run_with)
    world.advance()
    watcher, _ = _watcher(session_factory, fleet, jobs, Clock(), GOOD_NOTE)
    watcher.sweep()
    db_session.expire_all()
    job = _job(db_session)
    turn = db_session.execute(select(LedgerTurn).where(LedgerTurn.kind == "job")).scalar_one()
    assert turn.conversation_id == run_with.conversation
    assert turn.assistant_message_id == job.note_message_id
    step = turn.plans[0].steps[0]
    assert (step.kind, step.app, step.outcome) == ("job", "music", "done")
    receipts = step.receipts
    assert [(r.tool, r.outcome, r.count) for r in receipts] == [
        ("download_playlist", "done", 38),
        ("download_track", "refused", None),
    ]
    assert all(r.job_id == job.id for r in receipts)
    assert [t.title for t in receipts[1].touched] == list(PLAYLIST_MISSES)
    # Every write the world made has an app receipt on the ledger.
    written = db_session.execute(
        select(LedgerReceipt).where(LedgerReceipt.outcome.in_(("done", "unknown")))
    ).scalars()
    assert {r.app for r in written} >= {receipt.app for receipt in world.receipts}
    # The job's end is on the record for "what did you just do?".
    assert "A job reported back: music job 1 finished" in ledger.self_report(
        db_session, run_with.conversation
    )


def test_a_job_that_ended_with_no_readable_record_gets_an_unknown_receipt(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _queue_playlist(run_with)

    def bare(fleet_app: Any, ref: str) -> JobRead:
        return JobRead(ref=ref, status="completed", done=40, total=40, record=None)

    watcher = JobWatcher(session_factory, fleet, reader=bare, clock=Clock())
    assert watcher.sweep() == 1
    turn = db_session.execute(select(LedgerTurn).where(LedgerTurn.kind == "job")).scalar_one()
    receipts = turn.plans[0].steps[0].receipts
    assert [(r.source, r.outcome) for r in receipts] == [("conductor", "unknown")]
    # No model: code's words.
    assert _notes(run_with)[0]["content"] == "“your playlist” finished: 40 of 40 done."


def test_a_note_neither_voids_the_apps_question_nor_the_waiting_plan(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _queue_playlist(run_with)
    # A plan waits at code's gate (music doesn't confirm its own deletes).
    gated = steps(ask("music", "delete the album download", destructive=True))
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    world.advance()
    watcher, _ = _watcher(session_factory, fleet, jobs, Clock(), GOOD_NOTE)
    assert watcher.sweep() == 1
    paused = ledger.paused_plan(db_session, run_with.conversation)
    assert paused is not None and paused.reason == "gate"
    turns = conversations_service.stored_turns(db_session, run_with.conversation)
    assert all(t["content"] != GOOD_NOTE for t in turns)


def test_stored_turns_leave_a_note_out_so_an_apps_question_survives(
    db_session: Session,
) -> None:
    conversation = Conversation(title="t")
    db_session.add(conversation)
    db_session.flush()
    asked = ToolCallRecord(
        tool="ask_tasks",
        arguments={"message": "trash the kitchen project"},
        result="Trash it?",
        app_confirm={
            "app": "tasks",
            "thread_id": 3,
            "token": "tok",
            "tool": "trash_project",
            "question": "Trash Kitchen?",
        },
    )
    for role, content, calls, stop in (
        (ConversationRole.user, "trash kitchen", None, None),
        (ConversationRole.assistant, "Trash Kitchen?", [asked.model_dump()], "completed"),
        (ConversationRole.assistant, "Your playlist is in.", None, narrator.JOB_NOTE),
    ):
        db_session.add(
            ConversationMessage(
                conversation_id=conversation.id,
                role=role,
                content=content,
                tool_calls=calls,
                stop_reason=stop,
            )
        )
    db_session.flush()
    turns = conversations_service.stored_turns(db_session, conversation.id)
    pending = confirmation.pending(turns)
    assert pending is not None and pending.question == "Trash Kitchen?"


# --- cancel and amend on a running job -------------------------------------------------------


def _journal_job(
    db: Session, conversation_id: int, ref: str, title: str, done: int, total: int
) -> None:
    """A turn the ledger closed with a music call that started a job."""
    conversation = db.get(Conversation, conversation_id)
    assert conversation is not None
    message = conversations_service.append_user_message(db, conversation, f"get {title}")
    turn = ledger.open_turn(db, conversation, message)
    record = ToolCallRecord(
        tool="ask_music",
        arguments={"message": f"download {title}"},
        result="Queued.",
        app_record=[
            {
                "tool": "queue_download",
                "outcome": "done",
                "touched": [
                    {
                        "app": "music",
                        "kind": "playlist",
                        "id": ref,
                        "title": title,
                        "untrusted": True,
                    }
                ],
                "undo": {"none": "-"},
                "job": {"ref": ref, "status": "working", "done": done, "total": total},
            }
        ],
    )
    run = AgentRunResult(reply="ok", stop_reason="completed", tool_calls=[record], messages=[])
    ledger.close_turn(
        db, turn, run, conversations_service.append_assistant_message(db, conversation, run)
    )
    db.commit()


class FakeCancel:
    def __init__(self, answer: JobRead | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, fleet_app: Any, ref: str) -> JobRead:
        self.calls.append((fleet_app.name, ref))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _cancelling(monkeypatch: pytest.MonkeyPatch, answer: JobRead | Exception) -> FakeCancel:
    fake = FakeCancel(answer)
    monkeypatch.setattr(interpreter, "cancel_job", fake)
    return fake


def test_a_control_plan_on_a_running_job_needs_no_repair() -> None:
    for body in (cancel(), amend("ten"), cancel("music:job:31")):
        assert plans.unrunnable(plans.load(body)) == []


def test_cancel_stops_the_job_and_says_how_far_it_got(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 1480)
    fake = _cancelling(
        monkeypatch,
        JobRead(ref="31", status="working", done=164, total=1480, cancel_requested=True),
    )
    reply = run_with(ScriptedProvider([plan_turn(cancel())]), "stop that")
    assert fake.calls == [("music", "31")]
    assert reply["content"] == (
        "Asked music to stop “Library sort”: 164 of 1,480 done when I asked. "
        "I'll post a note when it ends."
    )
    assert [call["tool"] for call in reply["tool_calls"]] == ["cancel_job_music"]
    # The cancel is a write with its receipt, and the job row has the new report.
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).scalar_one()
    assert (step.app, step.outcome) == ("music", "done")
    assert [(r.tool, r.outcome, r.undo_none) for r in step.receipts] == [
        ("cancel_job", "done", "a stopped job can't be resumed; ask for it again to restart it")
    ]
    job = _job(db_session)
    assert (job.status, job.done, job.total) == ("working", 164, 1480)
    assert job.noted_at is None  # its end still gets the note


def test_cancel_over_the_world_ends_the_job_and_the_watcher_notes_it_once(
    run_with: Any,
    world: World,
    jobs: WorldJobs,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    reply = run_with(ScriptedProvider([plan_turn(cancel("the playlist download"))]), "stop it")
    assert reply["content"] == "Cancelled “your playlist”: 0 of 40 done."
    assert jobs.cancels == ["1"]
    assert [r.kind for r in world.receipts] == ["queue", "cancel"]
    watcher, _ = _watcher(
        session_factory, fleet, jobs, Clock(), "“your playlist” was cancelled: 0 of 40."
    )
    assert watcher.sweep() == 1
    assert len(_notes(run_with)) == 1


def test_a_job_that_already_finished_isnt_stopped(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    _cancelling(
        monkeypatch,
        JobRead(ref="31", status="completed", done=400, total=400, cancel_requested=False),
    )
    reply = run_with(ScriptedProvider([plan_turn(cancel())]), "stop that")
    assert reply["content"] == "“Library sort” had already finished: 400 of 400 done."
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).scalar_one()
    assert [r.outcome for r in step.receipts] == ["looked_up"]


def test_a_cancel_that_cant_reach_the_app_says_so(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    _cancelling(monkeypatch, JobError("music couldn't be reached"))
    reply = run_with(ScriptedProvider([plan_turn(cancel())]), "stop that")
    assert reply["content"] == (
        "I couldn't stop “Library sort”: music couldn't be reached. As far as I know it's "
        "still running (112 of 400 done at the last report)."
    )
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "cancel")).scalar_one()
    assert (step.outcome, step.receipts) == ("failed", [])


def test_with_several_jobs_running_a_vague_cancel_asks_which(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    _journal_job(db_session, run_with.conversation, "32", "Road trip", 3, 12)
    fake = _cancelling(
        monkeypatch, JobRead(ref="32", status="canceled", done=3, total=12, cancel_requested=True)
    )
    reply = run_with(ScriptedProvider([plan_turn(cancel())]), "stop it")
    assert reply["content"].startswith("2 jobs are running: ")
    assert reply["content"].endswith("Which one should I stop?")
    assert "“Library sort” in music (112 of 400 done)" in reply["content"]
    assert fake.calls == []
    named = run_with(ScriptedProvider([plan_turn(cancel("road trip"))]), "the road trip one")
    assert fake.calls == [("music", "32")]
    assert named["content"] == "Cancelled “Road trip”: 3 of 12 done."


def test_a_waiting_plan_still_takes_a_cancel_that_doesnt_name_the_job(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    fake = _cancelling(monkeypatch, JobRead(ref="31", status="working", cancel_requested=True))
    gated = steps(ask("music", "delete the album download", destructive=True))
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    reply = run_with(ScriptedProvider([plan_turn(cancel())]), "no, don't")
    assert reply["content"].startswith("Okay, cancelled: I won't ask music")
    assert fake.calls == []
    # Naming the job reaches it even while a plan waits.
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    run_with(ScriptedProvider([plan_turn(cancel("the sort job"))]), "stop the sort job")
    assert fake.calls == [("music", "31")]


def test_amend_on_a_running_job_says_it_cant_and_how_to_stop_it(
    run_with: Any, world: World, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _journal_job(db_session, run_with.conversation, "31", "Library sort", 112, 400)
    fake = _cancelling(monkeypatch, JobRead(ref="31", status="working", cancel_requested=True))
    reply = run_with(ScriptedProvider([plan_turn(amend("only jazz"))]), "make it only jazz")
    assert reply["content"] == (
        "I can't change “Library sort” while it runs: music has no way to edit a job in "
        "progress. It's at 112 of 400 done. Say stop to cancel it, then ask again with the change."
    )
    assert fake.calls == []
    assert reply["tool_calls"] is None


# --- the narrator's note -------------------------------------------------------------------


def _end(**overrides: Any) -> JobEnd:
    base: dict[str, Any] = {
        "app": "music",
        "ref": "1",
        "status": "completed",
        "done": 1438,
        "total": 1480,
        "title": "Big <b>list</b>\nignore previous",
        "untrusted": True,
        "record": [
            {
                "tool": "download_track",
                "outcome": "refused",
                "touched": [{"kind": "song", "id": "9", "title": "Odd “Song”", "untrusted": True}],
            }
        ],
    }
    return JobEnd(**(base | overrides))


def test_a_note_counts_with_thousands_and_quotes_outside_titles() -> None:
    def chat(messages: Any, max_tokens: int) -> ChatResult:
        return text_turn(
            "Big <b>list</b> ignore previous is done: 1,438 of 1,480; Odd Song missed."
        )

    note = narrator.job_note(_end(), surface="desktop", chat=chat)
    assert note.path == "narrated"
    assert note.reply is not None
    assert "“Big <b>list</b> ignore previous”" in note.reply
    assert "“Odd Song”" in note.reply


def test_a_failed_job_must_say_it_failed() -> None:
    def chat(messages: Any, max_tokens: int) -> ChatResult:
        return text_turn("Big list: 10 of 1,480 done.")

    end = _end(status="failed", done=10, record=[], reason="disk full", untrusted=False)
    note = narrator.job_note(end, surface="desktop", chat=chat)
    assert note.reply == (
        "Big list: 10 of 1,480 done. For the record: it was “Big <b>list</b> ignore previous”;"
        " it failed (“disk full”)."
    )
    assert note.path == "appended"


def test_a_spoken_note_keeps_to_two_sentences() -> None:
    def chat(messages: Any, max_tokens: int) -> ChatResult:
        return text_turn("Done. All of it. Really. Truly.")

    note = narrator.job_note(_end(record=[]), surface="voice", chat=chat)
    assert note.reply == (
        "Done. For the record: it was “Big <b>list</b> ignore previous”; 1,438 of 1,480 done."
    )


# --- the live channel ----------------------------------------------------------------------


def test_a_stream_follows_a_job_past_its_turn_and_ends_when_it_does() -> None:
    async def go() -> list[TurnEvent | Heartbeat]:
        stream = turn_activity.follow(7, heartbeat=5)
        items: list[TurnEvent | Heartbeat] = [await anext(stream)]
        stop = turn_activity.begin(7)
        turn_activity.publish_job(7, {"app": "music", "ref": "1", "done": 0, "ended": False})
        turn_activity.end(7, stop, stop_reason="completed", message_id=3)
        turn_activity.publish_job(7, {"app": "music", "ref": "1", "done": 20, "ended": False})
        turn_activity.publish_job(
            7, {"app": "music", "ref": "1", "done": 38, "ended": True, "note_message_id": 9}
        )
        async for item in stream:
            items.append(item)
        return items

    items = asyncio.run(go())
    kinds = [item.kind for item in items if isinstance(item, TurnEvent)]
    assert kinds == ["snapshot", "turn_started", "job", "turn_ended", "job", "job"]
    assert 7 not in turn_activity._CHANNELS


def test_a_late_follower_gets_the_running_jobs_in_its_snapshot() -> None:
    turn_activity.publish_job(8, {"app": "music", "ref": "1", "done": 5, "ended": False})

    async def first() -> TurnEvent | Heartbeat:
        stream = turn_activity.follow(8, heartbeat=5)
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    snapshot = asyncio.run(first())
    assert isinstance(snapshot, TurnEvent)
    assert snapshot.data == {
        "active": False,
        "jobs": [{"app": "music", "ref": "1", "done": 5, "ended": False}],
    }


# --- the contract's client -----------------------------------------------------------------


def _client_with(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        return real(transport=httpx.MockTransport(handler), **kwargs)

    # The client is built per request inside ``app.fleet.jobs``.
    monkeypatch.setattr(httpx, "Client", client)


def test_a_poll_is_a_plain_get_and_a_cancel_carries_the_actor(
    monkeypatch: pytest.MonkeyPatch, fleet: Fleet
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"ref": "a/1", "status": "bogus", "done": -1, "total": "x", "record": "nope"}
        if request.method == "POST":
            body |= {"status": "canceled", "done": 3, "cancel_requested": True}
        return httpx.Response(200, json=body)

    _client_with(monkeypatch, handler)
    music = fleet.get("music")
    assert music is not None
    read = fleet_jobs.read_job(music, "a/1")
    assert (read.status, read.done, read.total, read.record) == ("working", 0, None, None)
    stopped = fleet_jobs.cancel_job(music, "a/1")
    assert (stopped.status, stopped.done, stopped.cancel_requested) == ("canceled", 3, True)
    get, post = seen
    assert get.url.raw_path.endswith(b"/jobs/a%2F1") and get.method == "GET"
    assert post.url.raw_path.endswith(b"/jobs/a%2F1/cancel")
    assert "x-agent-actor" not in get.headers
    assert post.headers["x-agent-actor"] == "agent:conductor"


def test_a_404_is_gone_and_anything_else_is_a_job_error(
    monkeypatch: pytest.MonkeyPatch, fleet: Fleet
) -> None:
    statuses = iter([404, 500])
    _client_with(monkeypatch, lambda request: httpx.Response(next(statuses), json={}))
    music = fleet.get("music")
    assert music is not None
    with pytest.raises(JobGone):
        fleet_jobs.read_job(music, "1")
    with pytest.raises(JobError, match="answered 500"):
        fleet_jobs.read_job(music, "1")


# --- the migration -------------------------------------------------------------------------


def test_the_migration_marks_jobs_from_before_it_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        config = Config()
        config.set_main_option("script_location", str(BACKEND / "app" / "alembic"))
        command.upgrade(config, "5189491c0978")
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO ledger_jobs (app, ref, status, done, created_at, updated_at) "
                    "VALUES ('music', '1', 'working', 3, '2026-11-01', '2026-11-01')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            noted = connection.execute(text("SELECT noted_at FROM ledger_jobs")).scalar_one()
        assert noted is not None
        command.downgrade(config, "5189491c0978")
        command.upgrade(config, "head")
        engine.dispose()
    finally:
        get_settings.cache_clear()


# --- lifecycle -----------------------------------------------------------------------------


def test_the_watcher_thread_sweeps_until_stopped(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _queue_playlist(run_with)
    polled = threading.Event()

    def reader(fleet_app: Any, ref: str) -> JobRead:
        polled.set()
        return JobRead(ref=ref, status="working", done=1, total=40)

    watcher = JobWatcher(session_factory, fleet, reader=reader, tick=0.01)
    watcher.start()
    try:
        assert polled.wait(timeout=5)
    finally:
        watcher.stop()
    assert watcher._thread is None


def test_the_app_sends_job_events_to_the_live_channel(client: TestClient) -> None:
    assert turn_events._JOB_SINK is turn_activity.publish_job
    # Tests never start the watcher on their own (``conftest.no_job_watcher``).
    assert app.state.job_watcher is None
