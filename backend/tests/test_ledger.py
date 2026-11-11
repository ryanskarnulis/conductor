"""The ledger (#69): turns, plans, steps, receipts, jobs and entities."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app.ai import state_block, turn_context
from app.ai.interpreter import Interpreter
from app.ai.plan import TurnPlan
from app.ai.provider import ProviderRequestError
from app.ai.runtime import AgentRunResult, ToolCallRecord
from app.ai.state_block import Entity, Now
from app.api import routes_agent
from app.config import get_settings
from app.db.models import (
    Base,
    Conversation,
    ConversationMessage,
    ConversationRole,
    LedgerEntity,
    LedgerJob,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTouch,
    LedgerTurn,
)
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.tools import OUTCOME_UNKNOWN
from app.main import app
from app.services import ledger
from app.tools import registry
from tests.scripted_provider import ScriptedProvider, text_turn

BACKEND = Path(__file__).resolve().parents[1]


def _conversation(db: Session) -> Conversation:
    conversation = Conversation(title="t")
    db.add(conversation)
    db.flush()
    return conversation


def _user(db: Session, conversation: Conversation, words: str) -> ConversationMessage:
    message = ConversationMessage(
        conversation_id=conversation.id, role=ConversationRole.user, content=words
    )
    db.add(message)
    db.flush()
    return message


def _run(
    *records: ToolCallRecord, stop: str = "completed", plan: TurnPlan | None = None
) -> AgentRunResult:
    return AgentRunResult(
        reply="ok",
        stop_reason=stop,  # type: ignore[arg-type]
        tool_calls=list(records),
        messages=[],
        run_id="run-1",
        plan=plan or TurnPlan(status="ran"),
    )


def _turn(db: Session, conversation: Conversation, *records: ToolCallRecord) -> LedgerTurn:
    turn = ledger.open_turn(db, conversation, _user(db, conversation, "words"))
    assert turn is not None and turn.status == "running"
    ledger.close_turn(db, turn, _run(*records))
    db.commit()
    return turn


def _ref(kind: str, ident: str, title: str | None, **extra: Any) -> dict[str, Any]:
    return {"app": "tasks", "kind": kind, "id": ident, "title": title, **extra}


def _ask(app_record: list[dict[str, Any]] | None, **fields: Any) -> ToolCallRecord:
    fields.setdefault("result", "done")
    return ToolCallRecord(
        tool="ask_tasks", arguments={"message": "do it"}, app_record=app_record, **fields
    )


def test_a_turn_journals_its_steps_and_the_apps_receipts(db_session: Session) -> None:
    conversation = _conversation(db_session)
    record: list[dict[str, Any]] = [
        {"tool": "list_tasks", "outcome": "looked_up", "touched": [_ref("task", "1", "Rent")]},
        {
            "tool": "create_task",
            "outcome": "done",
            "touched": [_ref("task", "2", "Dentist")],
            "undo": {"handle": "trash:task:2", "how": "move it to the trash"},
        },
        {"tool": "trash_project", "outcome": "refused", "reason": "no such project"},
    ]
    turn = _turn(
        db_session,
        conversation,
        _ask(record),
        ToolCallRecord(tool="open_chess", arguments={"intent": "play"}, result="handed off"),
        ToolCallRecord(tool="list_agents", arguments={}, result="[]"),
        ToolCallRecord(tool="do_magic", arguments={}, error="Unknown tool 'do_magic'"),
    )
    assert (turn.status, turn.stop_reason, turn.run_id) == ("ended", "completed", "run-1")
    assert turn.ended_at is not None
    steps = db_session.execute(select(LedgerStep).order_by(LedgerStep.seq)).scalars().all()
    assert [(s.kind, s.app, s.outcome) for s in steps] == [
        ("ask", "tasks", "done"),
        ("open", "chess", "done"),
        ("local", None, "looked_up"),
        ("unknown", None, "not_run"),
    ]
    receipts = steps[0].receipts
    assert [(r.source, r.tool, r.outcome) for r in receipts] == [
        ("app", "list_tasks", "looked_up"),
        ("app", "create_task", "done"),
        ("app", "trash_project", "refused"),
    ]
    assert (receipts[1].undo_handle, receipts[1].undo_how) == (
        "trash:task:2",
        "move it to the trash",
    )
    assert receipts[2].reason == "no such project"
    assert all(not step.receipts for step in steps[1:])
    entities = db_session.execute(select(LedgerEntity)).scalars().all()
    assert {(e.app, e.kind, e.ref, e.title) for e in entities} == {
        ("tasks", "task", "1", "Rent"),
        ("tasks", "task", "2", "Dentist"),
    }


def test_an_app_that_ran_without_a_readable_record_gets_an_unknown_receipt(
    db_session: Session,
) -> None:
    conversation = _conversation(db_session)
    _turn(
        db_session,
        conversation,
        _ask(None),  # a v1 app answered
        _ask(None, result=None, error=f"tasks's agent {OUTCOME_UNKNOWN} this request"),
        _ask(None, result=None, error="tasks's agent is unavailable right now"),
        _ask([], result="nothing to do"),  # a v1.1 app saying nothing ran
    )
    steps = db_session.execute(select(LedgerStep).order_by(LedgerStep.seq)).scalars().all()
    assert [[(r.source, r.outcome) for r in s.receipts] for s in steps] == [
        [("conductor", "unknown")],
        [("conductor", "unknown")],
        [],
        [],
    ]


def test_a_confirm_step_files_its_receipts_under_the_app(db_session: Session) -> None:
    conversation = _conversation(db_session)
    _turn(
        db_session,
        conversation,
        ToolCallRecord(
            tool="confirm",
            arguments={"app": "tasks"},
            result="done",
            app_record=[{"tool": "trash_project", "outcome": "done", "touched": []}],
        ),
    )
    step = db_session.execute(select(LedgerStep)).scalar_one()
    assert (step.kind, step.app) == ("confirm", "tasks")
    assert [(r.app, r.outcome) for r in step.receipts] == [("tasks", "done")]


def test_a_job_keeps_its_latest_report_and_the_step_that_started_it(db_session: Session) -> None:
    conversation = _conversation(db_session)

    def queued(status: str, done: int) -> ToolCallRecord:
        return ToolCallRecord(
            tool="ask_music",
            arguments={"message": "grab it"},
            result="ok",
            app_record=[
                {
                    "tool": "queue_download",
                    "outcome": "done",
                    "touched": [{"app": "music", "kind": "download_job", "id": "112"}],
                    "job": {"ref": "112", "status": status, "done": done, "total": 400},
                }
            ],
        )

    _turn(db_session, conversation, queued("submitted", 0))
    first_step = db_session.execute(select(LedgerStep)).scalar_one()
    _turn(db_session, conversation, queued("working", 112))
    job = db_session.execute(select(LedgerJob)).scalar_one()
    assert (job.app, job.ref, job.status, job.done, job.total) == (
        "music",
        "112",
        "working",
        112,
        400,
    )
    assert job.started_by_step_id == first_step.id
    receipts = db_session.execute(select(LedgerReceipt)).scalars().all()
    assert [r.job_id for r in receipts] == [job.id, job.id]


def test_recent_entities_are_this_conversations_newest_first_titled_as_of_then(
    db_session: Session,
) -> None:
    mine, other = _conversation(db_session), _conversation(db_session)
    _turn(
        db_session,
        mine,
        _ask(
            [
                {
                    "tool": "create_task",
                    "outcome": "done",
                    "touched": [_ref("task", "1", "Old name"), _ref("task", "2", "Second")],
                }
            ]
        ),
    )
    _turn(
        db_session,
        other,
        _ask([{"tool": "x", "outcome": "done", "touched": [_ref("task", "9", "Theirs")]}]),
    )
    _turn(
        db_session,
        mine,
        _ask(
            [
                {"tool": "rename", "outcome": "refused", "touched": [_ref("task", "3", "Refused")]},
                {
                    "tool": "file_song",
                    "outcome": "done",
                    "untrusted": True,
                    "touched": [{"app": "music", "kind": "song", "id": "7", "title": "Odessa"}],
                },
                {
                    "tool": "update_task",
                    "outcome": "done",
                    "touched": [_ref("task", "1", "New name")],
                },
            ]
        ),
    )
    recent = ledger.recent_entities(db_session, mine.id)
    assert recent is not None
    assert recent == [
        Entity("tasks", "task", "1", "New name"),
        Entity("music", "song", "7", "Odessa", untrusted=True),
        Entity("tasks", "task", "2", "Second"),
    ]
    assert ledger.recent_entities(db_session, mine.id, limit=1) == recent[:1]
    # A conversation from before the ledger: the state block reads its records.
    assert ledger.recent_entities(db_session, _conversation(db_session).id) is None
    # The registry keeps the latest title; each touch keeps its own.
    entity = db_session.execute(select(LedgerEntity).where(LedgerEntity.ref == "1")).scalar_one()
    assert entity.title == "New name"
    titles = (
        db_session.execute(
            select(LedgerTouch.title)
            .where(LedgerTouch.entity_id == entity.id)
            .order_by(LedgerTouch.id)
        )
        .scalars()
        .all()
    )
    assert titles == ["Old name", "New name"]


def test_a_journal_failure_never_costs_the_turn(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = _conversation(db_session)
    turn = ledger.open_turn(db_session, conversation, _user(db_session, conversation, "hi"))

    def broken(*_: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(ledger, "_journal_step", broken)
    ledger.close_turn(db_session, turn, _run(_ask(None)))
    db_session.commit()
    # The savepoint rolled the half-written plan back; the turn and words stay.
    assert db_session.execute(select(LedgerStep)).first() is None
    kept = db_session.execute(select(LedgerTurn)).scalar_one()
    assert (kept.words, kept.status) == ("hi", "running")


def test_a_split_control_step_is_its_own_plan_with_the_turns_first_records(
    db_session: Session,
) -> None:
    # #75: a control step split off a longer plan ran first, as a plan of its
    # own; seq stays each record's place in the turn, as the live channel saw it.
    conversation = _conversation(db_session)
    control = TurnPlan(body={"steps": [{"id": "s2", "do": "undo", "target": "last"}]}, status="ran")
    rest = TurnPlan(
        body={"steps": [{"id": "s1", "do": "ask", "app": "tasks", "request": "x"}]},
        status="ran",
        first=control,
        first_steps=1,
    )
    undone = _ask([{"tool": "trash_task", "outcome": "done", "touched": [_ref("task", "2", "A")]}])
    asked = _ask([{"tool": "create_task", "outcome": "done", "touched": [_ref("task", "3", "B")]}])
    turn = ledger.open_turn(db_session, conversation, _user(db_session, conversation, "words"))
    ledger.close_turn(db_session, turn, _run(undone, asked, plan=rest))
    db_session.commit()
    plans = db_session.execute(select(LedgerPlan).order_by(LedgerPlan.id)).scalars().all()
    assert [(p.source, p.body) for p in plans] == [
        ("compiled", control.body),
        ("compiled", rest.body),
    ]
    assert [[(s.seq, s.receipts[0].tool) for s in p.steps] for p in plans] == [
        [(1, "trash_task")],
        [(2, "create_task")],
    ]


def test_a_failed_turn_journals_what_ran_under_its_plan(db_session: Session) -> None:
    # A failure stop carries its plan (#75): the write before the crash has a
    # receipt under the compiled plan, never a stray loop row.
    conversation = _conversation(db_session)
    wrote = _ask([{"tool": "create_task", "outcome": "done", "touched": [_ref("task", "4", "C")]}])
    crashed = ToolCallRecord(
        tool="ask_tasks", arguments={"message": "y"}, error="internal error: boom"
    )
    turn = ledger.open_turn(db_session, conversation, _user(db_session, conversation, "words"))
    plan = TurnPlan(body={"steps": []}, status="ran", note="stopped after s2")
    ledger.close_turn(db_session, turn, _run(wrote, crashed, stop="internal_error", plan=plan))
    db_session.commit()
    (row,) = db_session.execute(select(LedgerPlan)).scalars().all()
    assert (row.source, row.status, row.note) == ("compiled", "ran", "stopped after s2")
    assert [(s.outcome, [r.outcome for r in s.receipts]) for s in row.steps] == [
        ("done", ["done"]),
        ("failed", []),
    ]


def test_rows_from_before_75_still_read(db_session: Session) -> None:
    # A pre-#75 turn: the compiled plan fell back and the loop's plan acted.
    conversation = _conversation(db_session)
    message = _user(db_session, conversation, "add milk")
    old = LedgerTurn(
        conversation_id=conversation.id,
        user_message_id=message.id,
        kind="message",
        words="add milk",
        status="ended",
        stop_reason="completed",
    )
    db_session.add(old)
    db_session.flush()
    fell_back = LedgerPlan(
        turn_id=old.id,
        source="compiled",
        body={"steps": [{"id": "s1", "do": "look", "app": "tasks"}]},
        acted=False,
        status="fell_back",
        note="data_flow",
    )
    loop = LedgerPlan(turn_id=old.id, source="loop")
    db_session.add_all([fell_back, loop])
    db_session.flush()
    db_session.add(
        LedgerStep(
            plan_id=loop.id,
            seq=1,
            kind="ask",
            app="tasks",
            tool="ask_tasks",
            arguments={"message": "add milk"},
            outcome="done",
            result="Added milk.",
        )
    )
    db_session.commit()
    report = ledger.self_report(db_session, conversation.id)
    assert 'You were asked: "add milk"' in report
    assert "- asked tasks: it answered" in report
    assert ledger.paused_plan(db_session, conversation.id) is None


# --- the self-report says what conductor last said (#158) ---------------------------------


def _said(
    db: Session, conversation: Conversation, words: str, reply: str | None, *, hidden: bool = False
) -> LedgerTurn:
    turn = ledger.open_turn(db, conversation, _user(db, conversation, words), hidden=hidden)
    assert turn is not None
    message = ConversationMessage(
        conversation_id=conversation.id, role=ConversationRole.assistant, content=reply
    )
    db.add(message)
    db.flush()
    ledger.close_turn(db, turn, _run(), message)
    db.commit()
    return turn


def test_the_self_report_carries_the_last_reply_only(db_session: Session) -> None:
    conversation = _conversation(db_session)
    _said(db_session, conversation, "add milk", "Added “milk” to your tasks.")
    _said(db_session, conversation, "thanks!", "Any time.")
    report = ledger.self_report(db_session, conversation.id)
    assert '- you said: "Any time."' in report
    assert "Added" not in report  # an older reply isn't "what you just said"
    assert report.index('You were asked: "thanks!"') < report.index("you said")


def test_the_last_reply_is_clipped(db_session: Session) -> None:
    conversation = _conversation(db_session)
    _said(db_session, conversation, "tell me a story", "word " * 200)
    line = next(
        line
        for line in ledger.self_report(db_session, conversation.id).splitlines()
        if "you said" in line
    )
    assert line.endswith('…"') and len(line) < 400


def test_no_reply_line_without_a_reply(db_session: Session) -> None:
    conversation = _conversation(db_session)
    _said(db_session, conversation, "add milk", None)
    assert "you said" not in ledger.self_report(db_session, conversation.id)


def test_an_incognito_or_forgotten_last_reply_is_never_reported(db_session: Session) -> None:
    conversation = _conversation(db_session)
    _said(db_session, conversation, "add milk", "Added “milk”.")
    _said(db_session, conversation, "secret", "Your secret reply.", hidden=True)
    report = ledger.self_report(db_session, conversation.id)
    # Not the incognito reply, and no older one passed off as the last.
    assert "secret" not in report and "you said" not in report
    assert "An incognito turn" in report
    forgotten = _said(db_session, conversation, "my code is 1234", "Noted: 1234.")
    forgotten.forgotten_at = forgotten.ended_at
    db_session.commit()
    report = ledger.self_report(db_session, conversation.id)
    assert "1234" not in report and "you said" not in report
    assert "asked you to forget" in report


# --- the state block reads the ledger ------------------------------------------------------


def test_the_state_block_prefers_the_ledgers_entities() -> None:
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "x", "tool_calls": None},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                _ask(
                    [
                        {
                            "tool": "c",
                            "outcome": "done",
                            "touched": [_ref("task", "1", "From record")],
                        }
                    ]
                ).model_dump()
            ],
        },
    ]
    today = date(2026, 11, 6)
    assert "From record" in state_block.render(today, Now(), history)
    block = state_block.render(
        today, Now(), history, [Entity("music", "song", "7", "Odessa", True)]
    )
    assert 'music song 7: "Odessa" (outside text)' in block
    assert "From record" not in block
    assert "Recent entities" not in state_block.render(today, Now(), history, [])


def test_a_failed_entity_lookup_falls_back_to_the_records() -> None:
    history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                _ask(
                    [
                        {
                            "tool": "c",
                            "outcome": "done",
                            "touched": [_ref("task", "1", "From record")],
                        }
                    ]
                ).model_dump()
            ],
        }
    ]

    def broken() -> list[Entity]:
        raise RuntimeError("no ledger")

    opening = turn_context.build(
        date(2026, 11, 6), Now(), "hi", lambda: history, lambda: [], broken
    )
    assert "From record" in opening


# --- the routes journal ---------------------------------------------------------------------


def _create(client: TestClient) -> int:
    return int(client.post("/api/agent/conversations", json={}).json()["id"])


_TASKS = Fleet(
    apps=(
        FleetApp(
            name="tasks",
            title="Tasks",
            upstream="127.0.0.1:8100",
            agent=AgentSpec(description="Tasks.", api="/api/agent", examples=()),
        ),
    )
)


def _use(provider: ScriptedProvider) -> None:
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        provider, _TASKS, narrate=True
    )


def test_a_message_turn_and_a_note_are_on_the_ledger(
    client: TestClient, db_session: Session
) -> None:
    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch ask: an answer with no readable record."""
        return "Nobody's home."

    conversation = _create(client)
    plan = {
        "steps": [
            {"id": "s1", "do": "ask", "app": "tasks", "request": "who?", "destructive": False}
        ]
    }
    _use(ScriptedProvider([text_turn(json.dumps(plan)), text_turn("Nobody's home.")]))
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": "who's there?", "surface": "voice", "panel": "music"},
    )
    assert response.status_code == 200
    client.post(f"/api/agent/conversations/{conversation}/notes", json={"content": "filed a song"})
    db_session.expire_all()
    turns = db_session.execute(select(LedgerTurn).order_by(LedgerTurn.id)).scalars().all()
    assert [(t.kind, t.words, t.status) for t in turns] == [
        ("message", "who's there?", "ended"),
        ("note", "filed a song", "ended"),
    ]
    first = turns[0]
    assert (first.surface, first.panel, first.stop_reason) == ("voice", "music", "completed")
    assert first.assistant_message_id == response.json()["assistant_message"]["id"]
    assert first.run_id
    # One compiled plan, the one that acted: no `loop` rows since #75.
    assert [(p.source, p.status, p.acted) for p in first.plans] == [("compiled", "ran", True)]
    assert [(s.kind, s.tool) for p in first.plans for s in p.steps] == [("ask", "ask_tasks")]
    # The app answered with no record it could read: conductor's unknown receipt.
    (step,) = first.plans[0].steps
    assert [(r.source, r.outcome) for r in step.receipts] == [("conductor", "unknown")]
    assert turns[1].plans == []


def test_a_failed_run_still_ends_its_ledger_turn(client: TestClient, db_session: Session) -> None:
    conversation = _create(client)
    _use(ScriptedProvider([ProviderRequestError("llama-server request failed")]))
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "hi"}
    )
    assert response.status_code == 502
    db_session.expire_all()
    turn = db_session.execute(select(LedgerTurn)).scalar_one()
    assert (turn.status, turn.stop_reason, turn.assistant_message_id) == (
        "ended",
        "provider_error",
        None,
    )
    # The compile died: there was no plan, and none is made up.
    assert turn.plans == []


# --- the migration ------------------------------------------------------------------------


def test_the_migrations_build_the_models_schema_and_go_back_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        # No ini file: its logging section would disable the app's loggers for
        # every test after this one (``env.py`` only calls ``fileConfig`` with one).
        config = Config()
        config.set_main_option("script_location", str(BACKEND / "app" / "alembic"))
        command.upgrade(config, "head")
        engine = create_engine(url)
        tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert tables == set(Base.metadata.tables)
        for name in tables:
            columns = {c["name"] for c in inspect(engine).get_columns(name)}
            assert columns == set(Base.metadata.tables[name].columns.keys()), name
        command.downgrade(config, "cee4b4f848c9")
        assert not any(t.startswith("ledger_") for t in inspect(engine).get_table_names())
        engine.dispose()
    finally:
        get_settings.cache_clear()
