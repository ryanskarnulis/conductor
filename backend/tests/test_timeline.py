"""One timeline and incognito (#82).

The timeline: every surface writes to one conversation; the planner's "this
conversation" is the turns since the last quiet gap, and anything older
reaches it only as cross-day recall, labelled with its day. Old conversations
are ``legacy``. Incognito: the words and the reply are never written; the
receipts are kept; memory, recall, the planner's context, search, facts,
logs, the trace and the capture all go without them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app import clock
from app.ai import incognito
from app.ai import trace as run_trace
from app.ai.context_capture import CallStamp
from app.ai.interpreter import Interpreter
from app.ai.providers.llamacpp import LlamaCppProvider
from app.api import routes_agent
from app.config import get_settings
from app.db.models import (
    ConversationMessage,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTurn,
    MemoryFact,
)
from app.fleet.manifests import Fleet
from app.main import app
from app.services import conversations as conversations_service
from app.services import memory as memory_service
from app.services.conversations import QUIET_GAP
from tests.eval_db import age
from tests.fake_world import FakeGlances, World
from tests.scripted_provider import ScriptedProvider, text_turn
from tests.test_memory import (  # noqa: F401  (fixtures)
    ask,
    fleet,
    plan_turn,
    recall,
    steps,
    world,
)

BACKEND = Path(__file__).resolve().parents[1]
ODESSA = "grab Odessa by Caribou"
SECRET = "grab Odessa by Caribou for the surprise party"


class Timeline:
    """The timeline through the real route, with a turn's extras (surface,
    incognito) and a tracer the test can read."""

    def __init__(
        self,
        client: TestClient,
        fleet: Fleet,  # noqa: F811
        db: Session,
        trace: Path,
    ) -> None:
        self.client = client
        self.fleet = fleet
        self.db = db
        self.trace = trace
        self.id = int(client.get("/api/agent/timeline").json()["id"])

    def say(self, words: str, *turns: Any, **extra: Any) -> tuple[dict[str, Any], ScriptedProvider]:
        provider = ScriptedProvider(list(turns))
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider,
            self.fleet,
            narrate=True,
            glance=FakeGlances(),
            tracer=run_trace.JsonlTracer(self.trace),
        )
        response = self.client.post(
            f"/api/agent/conversations/{self.id}/messages", json={"content": words, **extra}
        )
        assert response.status_code == 200, response.text
        return dict(response.json()), provider

    def later(self, by: timedelta = QUIET_GAP + timedelta(minutes=1)) -> None:
        age(self.db, by)


@pytest.fixture
def timeline(
    client: TestClient,
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
    db_session: Session,
    tmp_path: Path,
) -> Timeline:
    return Timeline(client, fleet, db_session, tmp_path / "trace.jsonl")


def _opening(provider: ScriptedProvider) -> str:
    return str(provider.requests[0]["messages"][1]["content"])


def _grab(timeline: Timeline, words: str = ODESSA, **extra: Any) -> dict[str, Any]:
    exchange, _ = timeline.say(
        words,
        plan_turn(steps(ask("music", "grab the song Odessa by Caribou"))),
        text_turn("Saved “Odessa” by Caribou."),
        **extra,
    )
    return exchange


def _chat(timeline: Timeline, words: str, **extra: Any) -> ScriptedProvider:
    _, provider = timeline.say(
        words,
        plan_turn({"answer": "chat"}),
        text_turn('{"about_you": false, "reply": "Ok."}'),
        **extra,
    )
    return provider


# --- the migration ---------------------------------------------------------------------------


def test_the_timeline_migration_makes_old_rows_legacy_and_goes_back_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        config = Config()
        config.set_main_option("script_location", str(BACKEND / "app" / "alembic"))
        command.upgrade(config, "f4c2a8d6e913")
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO conversations (title) VALUES ('weekend plans')"))
            connection.execute(
                text(
                    "INSERT INTO ledger_turns (conversation_id, kind, words, status) "
                    "VALUES (1, 'message', 'hi', 'ended')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT kind, carries, closed_at, result FROM conversations")
            ).one()
            assert tuple(row) == ("legacy", None, None, None)
            assert connection.execute(text("SELECT incognito FROM ledger_turns")).scalar() == 0
        command.downgrade(config, "f4c2a8d6e913")
        columns = {c["name"] for c in inspect(engine).get_columns("conversations")}
        assert "kind" not in columns and "carries" not in columns
        turns = {c["name"] for c in inspect(engine).get_columns("ledger_turns")}
        assert "incognito" not in turns
        command.upgrade(config, "head")
        engine.dispose()
    finally:
        get_settings.cache_clear()


# --- one timeline ----------------------------------------------------------------------------


def test_the_window_starts_after_the_last_quiet_gap(db_session: Session) -> None:
    timeline = conversations_service.timeline(db_session)
    now = datetime(2026, 11, 9, 20, 0, tzinfo=UTC)
    assert conversations_service.window_since(db_session, timeline.id, now=now) is None
    for hours_ago in (30, 29, 2, 1):
        db_session.add(
            ConversationMessage(
                conversation_id=timeline.id,
                role="user",
                content=f"{hours_ago}h ago",
                created_at=now - timedelta(hours=hours_ago),
            )
        )
    db_session.flush()
    since = conversations_service.window_since(db_session, timeline.id, now=now)
    assert since == now - timedelta(hours=2)
    turns = conversations_service.stored_turns(db_session, timeline.id, since=since)
    assert [turn["content"] for turn in turns] == ["2h ago", "1h ago"]
    # A gap since the newest message: the window starts now, empty.
    later = now + QUIET_GAP
    assert conversations_service.window_since(db_session, timeline.id, now=later) == later


def test_every_surface_writes_to_the_one_timeline_tagged(timeline: Timeline) -> None:
    _chat(timeline, "hello from the desk")
    _chat(timeline, "hello from the kitchen", surface="voice")
    detail = timeline.client.get(f"/api/agent/conversations/{timeline.id}").json()
    tags = [(m["role"], m["surface"]) for m in detail["messages"]]
    assert tags == [
        ("user", None),
        ("assistant", None),
        ("user", "voice"),
        ("assistant", "voice"),
    ]
    assert timeline.client.get("/api/agent/timeline").json()["id"] == timeline.id


def test_past_the_gap_the_earlier_turn_comes_only_as_a_dated_hit(timeline: Timeline) -> None:
    _grab(timeline)
    _chat(timeline, "thanks")
    within = _chat(timeline, "is that one downloaded yet?")
    assert "The record of earlier turns in this conversation" in _opening(within)
    timeline.later(timedelta(days=1))
    exchange, provider = timeline.say(
        "put that Caribou song in the house folder",
        plan_turn({"answer": "decline"}),
    )
    opening = _opening(provider)
    # Nothing of the earlier day is "this conversation"...
    assert "The record of earlier turns in this conversation" not in opening
    assert "The last exchange" not in opening
    # ...but the song is a lead from it, labelled with its day.
    yesterday = clock.local_day(clock.utc_now() - timedelta(days=1))
    assert "Odessa" in opening and f"{yesterday:%A} {yesterday.isoformat()}" in opening


def test_earlier_calls_reach_the_timeline_before_its_gap_only(timeline: Timeline) -> None:
    _grab(timeline)
    db = timeline.db
    since = conversations_service.window_since(db, timeline.id)
    assert conversations_service.earlier_calls(db, timeline.id, since=since) == []
    timeline.later()
    since = conversations_service.window_since(db, timeline.id)
    (call,) = conversations_service.earlier_calls(db, timeline.id, since=since)
    assert call.asked == ODESSA


def test_the_apps_threads_follow_the_window(timeline: Timeline, world: World) -> None:  # noqa: F811
    _grab(timeline)
    _grab(timeline)
    timeline.later()
    _grab(timeline)
    first, again, later = world.threads["music"]
    assert first == again  # one thread within the window
    assert later != first  # a fresh one after the quiet gap, as a new conversation had


def test_a_waiting_plan_lapses_over_a_quiet_gap(timeline: Timeline) -> None:
    timeline.say(
        "which song?",
        plan_turn(steps({"do": "ask_user", "question": "Which one?", "options": []})),
    )
    db = timeline.db
    assert conversations_service.window_since(db, timeline.id) is None
    from app.services import ledger

    assert ledger.paused_plan(db, timeline.id) is not None
    timeline.later()
    since = conversations_service.window_since(db, timeline.id)
    assert ledger.paused_plan(db, timeline.id, since=since) is None


# --- incognito -------------------------------------------------------------------------------


def test_an_incognito_turn_writes_no_words_and_keeps_its_receipts(timeline: Timeline) -> None:
    exchange = _grab(timeline, SECRET, incognito=True)
    # The person reads the reply now...
    assert exchange["user_message"]["content"] == SECRET
    assert exchange["assistant_message"]["content"] == "Saved “Odessa” by Caribou."
    assert exchange["assistant_message"]["incognito"] is True
    db = timeline.db
    db.expire_all()
    # ...but nothing of the words was written.
    stored = db.execute(select(ConversationMessage)).scalars().all()
    assert [m.content for m in stored] == [incognito.PLACEHOLDER, incognito.PLACEHOLDER]
    (turn,) = db.execute(select(LedgerTurn)).scalars().all()
    assert turn.incognito and turn.words == incognito.PLACEHOLDER
    assert turn.terms == [] and turn.topic_id is None
    everything = json.dumps(
        [m.tool_calls for m in stored]
        + [p.body for p in db.execute(select(LedgerPlan)).scalars()]
        + [(s.arguments, s.result) for s in db.execute(select(LedgerStep)).scalars()]
    )
    assert "surprise" not in everything and "grab the song" not in everything
    # The apps really changed things: the receipt stays.
    receipts = db.execute(select(LedgerReceipt)).scalars().all()
    assert receipts and receipts[0].outcome == "done"
    detail = timeline.client.get(f"/api/agent/conversations/{timeline.id}").json()
    assert [m["incognito"] for m in detail["messages"]] == [True, True]
    assert [m["content"] for m in detail["messages"]] == [incognito.PLACEHOLDER] * 2
    assert detail["messages"][1]["tool_calls"][0]["app_record"]


def test_incognito_words_never_reach_recall_the_plan_or_search(timeline: Timeline) -> None:
    _grab(timeline, SECRET, incognito=True)
    db = timeline.db
    # Not recall, not memory's matches.
    assert memory_service.recall(db, "surprise party").hits == ()
    next_turn = _chat(timeline, "what was that surprise again?")
    opening = _opening(next_turn)
    assert "surprise" not in opening.split("The person said")[0]
    assert "Odessa" not in opening  # not even its entity in State
    assert incognito.PLACEHOLDER in opening  # the turn is there, with nothing of it
    timeline.later(timedelta(days=1))
    later = _chat(timeline, "that surprise party song")
    assert "surprise party for" not in _opening(later)
    # Not search.
    found = timeline.client.get("/api/agent/timeline/search", params={"q": "surprise"}).json()
    assert [hit["said"] for hit in found] == [
        "that surprise party song",
        "what was that surprise again?",
    ]
    assert timeline.client.get("/api/agent/timeline/search", params={"q": "Odessa"}).json() == []


def test_what_you_just_said_reaches_the_speaker_unless_incognito(timeline: Timeline) -> None:
    # #158: a self answer can repeat the last reply, from the stored message.
    _grab(timeline)
    _, provider = timeline.say(
        "what did you just say?",
        plan_turn({"answer": "self"}),
        text_turn("That I saved “Odessa”."),
    )
    facts = str(provider.requests[1]["messages"][-1]["content"])
    assert '- you said: "Saved “Odessa” by Caribou."' in facts
    # An incognito reply was never kept, so it can't be repeated, and no
    # older reply stands in for it.
    _grab(timeline, SECRET, incognito=True)
    _, provider = timeline.say(
        "what did you just say?",
        plan_turn({"answer": "self"}),
        text_turn("I don't know."),
    )
    facts = str(provider.requests[1]["messages"][-1]["content"])
    assert "you said" not in facts and "surprise" not in facts
    assert "An incognito turn" in facts


def test_an_incognito_turn_remembers_no_fact_and_attaches_none(timeline: Timeline) -> None:
    exchange, _ = timeline.say(
        "remember I sort music on Sundays",
        plan_turn(steps(ask("memory", "remember I sort music on Sundays"))),
        text_turn("Nothing was saved."),
        incognito=True,
    )
    (call,) = exchange["assistant_message"]["tool_calls"]
    assert call["app_record"][0]["outcome"] == "refused"
    assert timeline.db.execute(select(MemoryFact)).scalars().all() == []
    # A saved fact never rides an incognito ask (#134).
    timeline.say(
        "remember I sort music on Sundays",
        plan_turn(steps(ask("memory", "remember I sort music on Sundays"))),
        text_turn("Got it."),
    )
    exchange, _ = timeline.say(
        "add a task for my music sorting day",
        plan_turn(steps(ask("tasks", "add a task for my music sorting day"))),
        text_turn("Added."),
        incognito=True,
    )
    (call,) = exchange["assistant_message"]["tool_calls"]
    assert "facts" not in call["arguments"]


def test_incognito_leaves_no_trace_capture_or_words_in_the_logs(timeline: Timeline) -> None:
    _chat(timeline, "a normal turn")
    with structlog.testing.capture_logs() as logs:
        _grab(timeline, SECRET, incognito=True)
    lines = timeline.trace.read_text().splitlines()
    assert len(lines) == 1 and "a normal turn" in lines[0]
    assert SECRET not in json.dumps(logs, default=str)


def test_the_log_redaction_blanks_wordy_fields_only_when_incognito() -> None:
    event = {"event": "x", "error": "the surprise party", "app": "music"}
    assert incognito.redact_log(None, "info", dict(event)) == event
    with incognito.turn(True):
        redacted = incognito.redact_log(None, "info", dict(event))
    assert redacted["error"] == incognito.PLACEHOLDER and redacted["app"] == "music"


def test_an_incognito_turn_is_never_captured() -> None:
    class Capture:
        def begin(self) -> CallStamp:
            return CallStamp(run_id="r", seq=0, started_at="now")

        def record(self, stamp: CallStamp, call: dict[str, Any]) -> None:
            raise AssertionError("captured")

    provider = LlamaCppProvider(
        "http://model",
        "m",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        capture=Capture(),
    )
    assert provider._begin_capture() is not None
    with incognito.turn(True):
        assert provider._begin_capture() is None


# --- search ----------------------------------------------------------------------------------


def test_search_runs_over_words_replies_and_receipt_titles(timeline: Timeline) -> None:
    _grab(timeline)
    _chat(timeline, "the garage shelves should be steel")
    client = timeline.client
    by_title = client.get("/api/agent/timeline/search", params={"q": "odessa"}).json()
    assert [hit["matched"] for hit in by_title] == [["words", "reply", "receipt"]]
    assert by_title[0]["titles"] == ["Odessa"]
    by_reply = client.get("/api/agent/timeline/search", params={"q": "ok"}).json()
    assert [hit["said"] for hit in by_reply] == ["the garage shelves should be steel"]
    assert by_reply[0]["matched"] == ["reply"]
    assert client.get("/api/agent/timeline/search", params={"q": "steel odessa"}).json() == []
    assert client.get("/api/agent/timeline/search", params={"q": "x"}).status_code == 422


def test_search_skips_forgotten_turns(timeline: Timeline) -> None:
    _chat(timeline, "the spare key is under the blue pot")
    turn = timeline.db.execute(select(LedgerTurn)).scalars().one()
    memory_service.unindex(timeline.db, turn.id)
    timeline.db.commit()
    assert timeline.client.get("/api/agent/timeline/search", params={"q": "blue pot"}).json() == []
