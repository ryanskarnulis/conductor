"""Barge-in (#77): what the person heard of a reply they spoke over.

The voice client reports, with the next turn, how much of the last spoken
reply played before the person cut in. The ledger keeps it on the
interrupted reply's turn, the state block says what they didn't hear, and the
self-report never claims they heard the rest. Incognito turns store and show
nothing; a forgotten turn lends nothing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app.ai import state_block
from app.ai.state_block import Heard, Now
from app.config import get_settings
from app.db.models import LedgerTurn
from app.schemas.conversations import HeardReport, MessageCreate
from app.services import ledger
from tests.scripted_provider import text_turn
from tests.test_memory import (  # noqa: F401  (fixtures)
    fleet,
    plan_turn,
    world,
)
from tests.test_timeline import Timeline, timeline  # noqa: F401  (fixture)

BACKEND = Path(__file__).resolve().parents[1]
REPLY = "Nothing is due today. Tomorrow you have the dentist at nine, so leave early."


def _chat(timeline: Timeline, words: str, reply: str = REPLY, **extra: Any) -> str:  # noqa: F811
    _, provider = timeline.say(
        words,
        plan_turn({"answer": "chat"}),
        text_turn(f'{{"about_you": false, "reply": "{reply}"}}'),
        surface="voice",
        **extra,
    )
    return str(provider.requests[0]["messages"][1]["content"])


def _heard(chars: int | None, total: int = len(REPLY)) -> dict[str, Any]:
    return {"played_ms": 1500, "total_ms": 6000, "heard_chars": chars, "total_chars": total}


def _turns(db: Session) -> list[LedgerTurn]:
    return list(db.execute(select(LedgerTurn).order_by(LedgerTurn.id)).scalars())


# --- the report on the request ---------------------------------------------------------------


def test_the_report_mirrors_the_voice_clients_heard_and_is_bounded() -> None:
    made = MessageCreate.model_validate({"content": "no, Saturday", "heard": _heard(20)})
    assert made.heard == HeardReport(
        played_ms=1500, total_ms=6000, heard_chars=20, total_chars=len(REPLY)
    )
    # Audio played with no known length: no estimate, as the client sends it.
    assert HeardReport(played_ms=900, total_ms=None, heard_chars=None, total_chars=10)
    # A rounded estimate past the end is the end, never a lost turn.
    assert HeardReport(played_ms=1, heard_chars=12, total_chars=10).heard_chars == 10
    for bad in (
        {**_heard(20), "played_ms": -1},
        {**_heard(20), "total_chars": 10**6},
        {**_heard(20), "words": "smuggled"},
    ):
        with pytest.raises(ValueError):
            HeardReport.model_validate(bad)


def test_a_bad_report_is_a_422_not_a_turn(
    client: TestClient,
    timeline: Timeline,  # noqa: F811
) -> None:
    response = client.post(
        f"/api/agent/conversations/{timeline.id}/messages",
        json={"content": "no", "heard": {"played_ms": "lots", "total_chars": 3}},
    )
    assert response.status_code == 422


# --- the ledger and the state block ---------------------------------------------------------


def test_the_interrupted_replys_turn_keeps_it_and_the_next_state_says_what_wasnt_heard(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    _chat(timeline, "what's on tomorrow?")
    opening = _chat(timeline, "wait, what time?", reply="Nine.", heard=_heard(22))
    first, second = _turns(db_session)
    assert first.heard == _heard(22)
    assert second.heard is None
    line = next(line for line in opening.splitlines() if line.startswith("Your last reply"))
    assert 'They heard up to "Nothing is due today."' in line
    assert 'not heard: "Tomorrow you have the dentist at nine, so leave early."' in line
    # The next reply played to the end: nothing is cut off any more.
    later = _chat(timeline, "thanks")
    assert state_block.CUT_OFF_PREFIX not in later


def test_a_report_that_doesnt_fit_the_last_reply_is_dropped(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    _chat(timeline, "what's on tomorrow?")
    opening = _chat(timeline, "hm?", heard=_heard(5, total=len(REPLY) + 40))
    assert all(turn.heard is None for turn in _turns(db_session))
    assert state_block.CUT_OFF_PREFIX not in opening


def test_with_no_reply_before_it_the_report_is_dropped(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    opening = _chat(timeline, "hello", heard=_heard(5))
    assert all(turn.heard is None for turn in _turns(db_session))
    assert state_block.CUT_OFF_PREFIX not in opening


def test_an_incognito_turn_stores_and_shows_nothing(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    _chat(timeline, "what's on tomorrow?")
    opening = _chat(timeline, "wait, what?", heard=_heard(22), incognito=True)
    assert all(turn.heard is None for turn in _turns(db_session))
    assert state_block.CUT_OFF_PREFIX not in opening


def test_an_incognito_reply_lends_nothing(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    _chat(timeline, "what's on tomorrow?", incognito=True)
    opening = _chat(timeline, "wait, what?", heard=_heard(22))
    assert all(turn.heard is None for turn in _turns(db_session))
    assert state_block.CUT_OFF_PREFIX not in opening


def test_a_forgotten_turn_lends_nothing_however_it_was_stored(db_session: Session) -> None:
    from app.services import conversations as conversations_service

    conversation = conversations_service.timeline(db_session)
    turn = LedgerTurn(conversation_id=conversation.id, words="x", status="ended")
    db_session.add(turn)
    db_session.flush()
    message = conversations_service.append_user_message(db_session, conversation, "x")
    turn.user_message_id = message.id
    turn.heard = _heard(3)
    turn.forgotten_at = datetime(2026, 11, 12, tzinfo=UTC)
    db_session.flush()
    assert ledger.last_heard(db_session, conversation.id) is None
    assert ledger.record_heard(db_session, conversation.id, _heard(3)) is False


def test_the_self_report_says_what_the_person_didnt_hear(
    timeline: Timeline,  # noqa: F811
    db_session: Session,
) -> None:
    _chat(timeline, "what's on tomorrow?")
    first = _turns(db_session)[0]
    assert ledger.record_heard(db_session, timeline.id, _heard(22))
    report = ledger.self_report(db_session, timeline.id)
    assert f'- you said: "{REPLY}"' in report
    assert "they heard only" in report and '"Nothing is due today."' in report
    assert first.heard == _heard(22)


# --- the line ---------------------------------------------------------------------------------


def test_the_cut_falls_at_a_word_and_counts_like_the_browser() -> None:
    # A half-heard word is not heard.
    said, unsaid = state_block.split_heard(Heard("Move it to Saturday morning", 14)) or ("", "")
    assert (said, unsaid) == ("Move it to", "Saturday morning")
    # An emoji is two of the browser's characters.
    reply = "🎵 Queued Honey. Say undo to take it back."
    said, unsaid = state_block.split_heard(Heard(reply, 16)) or ("", "")
    assert (said, unsaid) == ("🎵 Queued Honey.", "Say undo to take it back.")
    assert state_block.utf16_len(reply) == len(reply) + 1


def test_the_lines_for_none_all_and_unknown() -> None:
    assert "heard none of it" in state_block.cut_off_line(Heard(REPLY, 0))
    assert "heard all of it" in state_block.cut_off_line(Heard(REPLY, len(REPLY)))
    assert "unknown" in state_block.cut_off_line(Heard(REPLY, None))


def test_the_line_is_capped_and_the_block_stays_in_bounds() -> None:
    reply = "word " * 5000
    line = state_block.cut_off_line(Heard(reply, 9000))
    assert len(line) <= state_block.HEARD_MAX_CHARS
    block = state_block.render(date(2026, 11, 12), Now(surface="voice"), [], heard=Heard(reply, 9))
    assert line.split('"')[0] in block


# --- the migration ----------------------------------------------------------------------------


def test_the_heard_migration_is_additive_and_goes_back_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        config = Config()
        config.set_main_option("script_location", str(BACKEND / "app" / "alembic"))
        command.upgrade(config, "d4b8e2f61a97")
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO conversations (title) VALUES ('t')"))
            connection.execute(
                text(
                    "INSERT INTO ledger_turns (conversation_id, kind, words, status) "
                    "VALUES (1, 'message', 'hi', 'ended')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(text("SELECT heard FROM ledger_turns")).scalar() is None
        command.downgrade(config, "d4b8e2f61a97")
        turns = {c["name"] for c in inspect(engine).get_columns("ledger_turns")}
        assert "heard" not in turns and "words" in turns
        command.upgrade(config, "head")
        engine.dispose()
    finally:
        get_settings.cache_clear()
