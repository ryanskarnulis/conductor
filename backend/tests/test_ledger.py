"""The ledger (#69): turns, plans, steps, receipts, jobs and entities."""

from __future__ import annotations

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
from app.ai.loop import AgentLoop, AgentRunResult, ToolCallRecord
from app.ai.provider import ProviderRequestError
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
    LedgerReceipt,
    LedgerStep,
    LedgerTouch,
    LedgerTurn,
)
from app.fleet.tools import OUTCOME_UNKNOWN
from app.main import app
from app.services import ledger
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn

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


def _run(*records: ToolCallRecord, stop: str = "completed") -> AgentRunResult:
    return AgentRunResult(
        reply="ok",
        stop_reason=stop,  # type: ignore[arg-type]
        tool_calls=list(records),
        iterations=1,
        messages=[],
        run_id="run-1",
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


def _use(provider: ScriptedProvider) -> None:
    app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(provider)


def test_a_message_turn_and_a_note_are_on_the_ledger(
    client: TestClient, db_session: Session
) -> None:
    conversation = _create(client)
    _use(ScriptedProvider([tool_calls_turn(("list_agents", {})), text_turn("Nobody's home.")]))
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
    assert [(s.kind, s.tool) for p in first.plans for s in p.steps] == [("local", "list_agents")]
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
