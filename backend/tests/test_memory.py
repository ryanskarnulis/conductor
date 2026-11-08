"""Memory over the ledger (#80): the index, quiet topics, recall and matches.

Pure pieces first (``app.ai.memory``), then the whole path through the real
route, compiler, interpreter, ledger and narrator over the session evals' fake
fleet, with the model scripted. Conversations stand in for days.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import dataflow, memory, narrator, state_block, turn_context
from app.ai.interpreter import Interpreter
from app.ai.runtime import ToolCallRecord
from app.ai.memory import Candidate, Found, Hit, Recalled
from app.ai.recall import EarlierCall
from app.ai.state_block import Entity, Match, Now
from app.api import routes_agent
from app.db.models import (
    Conversation,
    LedgerEntity,
    LedgerStep,
    LedgerTopic,
    LedgerTurn,
)
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.services import ledger
from app.services import memory as memory_service
from tests.fake_world import FakeGlances, World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn

TUESDAY = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)


# --- the words ------------------------------------------------------------------------------


def test_terms_keep_the_words_that_name_a_thing() -> None:
    assert memory.terms("What did we decide about the garage shelves last week?") == {
        "garage",
        "shelve",
    }
    assert memory.terms("that Caribou song") == {"caribou", "song"}
    assert memory.terms("Caribou's songs") == {"caribou", "song"}
    assert memory.terms("yes, the first one") == {"first"}
    assert memory.terms("ok") == set()


def test_merge_is_capped_and_the_same_however_it_was_built() -> None:
    once = memory.merge({}, ["a", "b", "b", "c"], cap=2)
    assert once == {"b": 2, "a": 1}  # ties broken by the word
    assert memory.merge(memory.merge({}, ["b", "a"], cap=2), ["c", "b"], cap=2) == once


def test_a_kind_word_alone_names_nothing_and_a_tie_names_nothing() -> None:
    terms = {"odessa": 1, "caribou": 1, "song": 1}
    assert memory.score({"song"}, terms, "song") == 0
    assert memory.score({"caribou", "song"}, terms, "song") == 2
    odessa = Found("music", "song", "1", "Odessa", True)
    sun = Found("music", "song", "2", "Sun", True)
    assert memory.best([Candidate(odessa, 2, TUESDAY), Candidate(sun, 1, TUESDAY)]) == odessa
    assert memory.best([Candidate(odessa, 2, TUESDAY), Candidate(sun, 2, TUESDAY)]) is None
    job = Found("music", "download_job", "3", "album download")
    assert memory.best([Candidate(job, 3, TUESDAY)]) is None


def test_the_record_cites_day_and_conversation_and_marks_what_it_resolved() -> None:
    odessa = Found("music", "song", "812", "Odessa", True)
    hit = Hit(7, 3, TUESDAY, "grab Odessa by Caribou", "Grab Odessa", found=(odessa,))
    result, entries = memory.record(
        Recalled("that Caribou song", (hit,), odessa), date(2026, 11, 6)
    )
    (entry,) = entries
    assert entry["outcome"] == "looked_up" and entry["source"] == "conductor"
    assert entry["fact"] == "grab Odessa by Caribou" and entry["from"] == "your words"
    assert (entry["turn"], entry["conversation"], entry["day"]) == (7, 3, "Tuesday 2026-11-03")
    assert entry["cues"][:2] == ["Tuesday", "2026-11-03"] and "3 days ago" in entry["cues"]
    assert entry["touched"] == [
        {
            "app": "music",
            "kind": "song",
            "id": "812",
            "title": "Odessa",
            "untrusted": True,
            "best": True,
        }
    ]
    assert result.startswith('Tuesday 2026-11-03, your words: "grab Odessa by Caribou"')
    assert '"Odessa" (outside text)' in result
    value = dataflow.from_recall(entries)
    assert value == dataflow.Value("“Odessa”", untrusted=True)
    assert memory.record(Recalled("x"), date(2026, 11, 6)) == ("Nothing on record matches.", [])
    assert dataflow.from_recall([]) is None


# --- the narrator's facts ------------------------------------------------------------------


def _recall_record(entries: list[dict[str, Any]], result: str = "x") -> ToolCallRecord:
    return ToolCallRecord(
        tool=memory.RECALL_TOOL, arguments={"query": "q"}, result=result, app_record=entries
    )


def _chat(reply: str) -> Any:
    def chat(messages: list[dict[str, Any]], max_tokens: int) -> Any:
        return text_turn(reply)

    return chat


def test_a_recall_that_found_nothing_says_so_and_never_guesses() -> None:
    record = _recall_record([], "Nothing on record matches.")
    (step,) = narrator.receipt_steps([record])
    assert step["result"] == "nothing_on_record"
    told = narrator.narrate(
        "what did we decide?", None, [record], surface=None, chat=_chat("Steel.")
    )
    assert told.path == "appended"
    assert "no record" in (told.reply or "")
    honest = narrator.narrate(
        "what did we decide?", None, [record], surface=None, chat=_chat("I don't have that.")
    )
    assert honest.reply == "I don't have that."


def test_a_recall_reply_says_when_and_names_what_it_resolved() -> None:
    odessa = Found("music", "song", "812", "Odessa", True)
    hit = Hit(7, 3, TUESDAY, "grab Odessa by Caribou", "Grab Odessa", found=(odessa,))
    _, entries = memory.record(Recalled("q", (hit,), odessa), date(2026, 11, 6))
    record = _recall_record(entries)
    (step,) = narrator.receipt_steps([record])
    assert step["result"] == "recalled"
    assert step["found"][0]["said"] == "grab Odessa by Caribou"
    assert step["found"][0]["items"] == [{"kind": "song", "title": "Odessa", "outside_text": True}]
    told = narrator.narrate(
        "that song?", None, [record], surface=None, chat=_chat("It was a song.")
    )
    assert "“Odessa”" in (told.reply or "") and "Tuesday 2026-11-03" in (told.reply or "")
    good = narrator.narrate(
        "that song?", None, [record], surface=None, chat=_chat("Tuesday you grabbed “Odessa”.")
    )
    assert good.path == "narrated"


# --- the state block and the opening -----------------------------------------------------------


def test_matches_render_as_leads_and_replace_the_word_matched_calls() -> None:
    match = Match(
        Entity("music", "song", "812", "Odessa", True), "Tuesday 2026-11-03", "grab Odessa"
    )
    block = state_block.render(date(2026, 11, 6), Now(), [], [], [], [match])
    assert "Matches for words in the message, from earlier conversations (as of then):" in block
    assert (
        '- music song 812: "Odessa" (outside text), Tuesday 2026-11-03, you said "grab Odessa"'
        in block
    )
    call = EarlierCall(
        TUESDAY,
        "grab Odessa by Caribou",
        ToolCallRecord(tool="ask_music", arguments={"message": "grab Odessa"}, result="Saved."),
    )
    words = "put that Caribou song in the house folder"
    both = turn_context.build(
        date(2026, 11, 6), Now(), words, lambda: [], lambda: [call], matches=lambda: [match]
    )
    assert "Matches for words" in both and "Possibly what the person" not in both
    alone = turn_context.build(
        date(2026, 11, 6), Now(), words, lambda: [], lambda: [call], matches=lambda: []
    )
    assert "Possibly what the person" in alone  # a v1 app's call, no registry entry

    def broken() -> list[Match]:
        raise RuntimeError("db gone")

    assert "Possibly what the person" in turn_context.build(
        date(2026, 11, 6), Now(), words, lambda: [], lambda: [call], matches=broken
    )


def test_the_opening_stays_bounded_with_matches() -> None:
    long = "x" * 400
    matches = [
        Match(Entity("music", "song", str(n) * 30, long, True), "Wednesday 2026-11-04", long)
        for n in range(5)
    ]
    block = state_block.render(date(2026, 11, 6), Now(), [], [], [], matches)
    matched = [line for line in block.splitlines() if line.startswith("- music song")]
    assert len(matched) == state_block.MAX_MATCHES
    assert sum(len(line) for line in matched) + 90 <= state_block.MATCHES_MAX_CHARS
    assert turn_context.MAX_CHARS <= 10_500


# --- through the route ----------------------------------------------------------------------


def plan_turn(body: dict[str, Any]) -> Any:
    return text_turn(json.dumps(body))


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]}


def ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


def recall(query: str) -> dict[str, Any]:
    return {"do": "recall", "query": query}


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


class Talk:
    """Conversations through the real route; ``new()`` is another day."""

    def __init__(self, client: TestClient, fleet: Fleet) -> None:
        self.client = client
        self.fleet = fleet
        self.conversation = 0
        self.new()

    def new(self) -> int:
        self.conversation = int(self.client.post("/api/agent/conversations", json={}).json()["id"])
        return self.conversation

    def say(self, words: str, *turns: Any) -> tuple[dict[str, Any], ScriptedProvider]:
        provider = ScriptedProvider(list(turns))
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider, self.fleet, narrate=True, glance=FakeGlances()
        )
        response = self.client.post(
            f"/api/agent/conversations/{self.conversation}/messages", json={"content": words}
        )
        assert response.status_code == 200, response.text
        return dict(response.json()["assistant_message"]), provider


@pytest.fixture
def talk(client: TestClient, fleet: Fleet, world: World) -> Generator[Talk, None, None]:
    yield Talk(client, fleet)


def _grab_odessa(talk: Talk) -> None:
    talk.say(
        "grab Odessa by Caribou",
        plan_turn(steps(ask("music", "grab the song Odessa by Caribou"))),
        text_turn("Saved “Odessa” by Caribou."),
    )


def test_that_caribou_song_days_later_is_a_match_in_state(talk: Talk, db_session: Session) -> None:
    _grab_odessa(talk)
    talk.new()
    _, provider = talk.say(
        "put that Caribou song in the house folder",
        plan_turn(steps(ask("music", "put Odessa by Caribou in the house folder"))),
        text_turn("Filed “Odessa” under house."),
    )
    opening = provider.requests[0]["messages"][1]["content"]
    assert "Matches for words in the message, from earlier conversations" in opening
    assert '"Odessa" (outside text)' in opening and 'you said "grab Odessa by Caribou"' in opening
    assert "Possibly what the person is pointing back at" not in opening
    # A fresh request that points back at nothing gets no matches.
    talk.new()
    _, fresh = talk.say(
        "move Caribou songs to house",
        plan_turn({"answer": "chat"}),
        text_turn('{"about_you": false, "reply": "Sure."}'),
    )
    assert "Matches for words" not in fresh.requests[0]["messages"][1]["content"]


def test_a_recall_resolves_that_song_and_passes_it_on_as_quoted_data(
    talk: Talk, world: World, db_session: Session
) -> None:
    _grab_odessa(talk)
    talk.new()
    reply, _ = talk.say(
        "add a task to play that Caribou song at the party",
        plan_turn(
            steps(recall("that Caribou song"), ask("tasks", "add a task to play {s1} at the party"))
        ),
        text_turn("Added “play Odessa at the party”, from Tuesday's “Odessa”."),
    )
    (task,) = [r for r in world.writes() if r.app == "tasks"]
    assert "odessa" in task.title.lower()
    sent = [call["arguments"] for call in reply["tool_calls"]]
    assert sent == [
        {"query": "that Caribou song"},
        {"message": "add a task to play “Odessa” at the party"},
    ]
    db_session.expire_all()
    step = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "recall")).scalar_one()
    (receipt,) = step.receipts
    assert (receipt.source, receipt.app, receipt.tool, receipt.outcome) == (
        "conductor",
        "conductor",
        "recall",
        "looked_up",
    )
    assert receipt.fact == "grab Odessa by Caribou"
    cited = db_session.get(LedgerTurn, receipt.cites_turn_id)
    assert cited is not None and cited.words == "grab Odessa by Caribou"
    assert [touch.title for touch in receipt.touched] == ["Odessa"]
    live = ledger.live_step(1, ToolCallRecord.model_validate(reply["tool_calls"][0]))
    assert live["kind"] == "recall" and live["receipts"][0]["fact"] == "grab Odessa by Caribou"


def test_an_outside_title_from_memory_still_never_steers_a_step(talk: Talk, world: World) -> None:
    _grab_odessa(talk)
    talk.new()
    # Fixture music doesn't confirm its own writes: outside text can't go there.
    talk.say(
        "put that Caribou song in the house folder",
        plan_turn(steps(recall("that Caribou song"), ask("music", "put {s1} in the house folder"))),
        text_turn("I didn't file it."),
    )
    assert not [r for r in world.writes() if r.kind == "file"]


def test_two_songs_that_fit_name_neither(talk: Talk, world: World, db_session: Session) -> None:
    _grab_odessa(talk)
    talk.say(
        "grab Kaili by Caribou",
        plan_turn(steps(ask("music", "grab the song Kaili by Caribou"))),
        text_turn("Saved “Kaili”."),
    )
    talk.new()
    talk.say(
        "add a task to play that Caribou song",
        plan_turn(steps(recall("that Caribou song"), ask("tasks", "add a task to play {s1}"))),
        text_turn("I couldn't tell which song."),
    )
    assert not [r for r in world.writes() if r.app == "tasks"]
    db_session.expire_all()
    note = db_session.execute(select(LedgerStep).where(LedgerStep.kind == "ask")).scalars().all()
    assert note[-1].outcome == "not_run"


def test_what_did_we_decide_is_cited_or_honestly_unknown(talk: Talk, db_session: Session) -> None:
    # Nothing on record yet: the honest answer, never a guess.
    reply, _ = talk.say(
        "what did we decide about the garage shelves last week?",
        plan_turn(steps(recall("garage shelves decision"))),
        text_turn("We went with the wooden ones."),
    )
    assert "no record" in reply["content"]
    # Monday: it's decided in the person's words, then a short follow-up.
    talk.new()
    for words in ("for the garage shelves, let's go with the steel rack", "and not wood"):
        talk.say(
            words, plan_turn({"answer": "chat"}), text_turn('{"about_you": false, "reply": "Ok."}')
        )
    talk.new()
    reply, provider = talk.say(
        "what did we decide about the garage shelves last week?",
        plan_turn(steps(recall("garage shelves decision"))),
        text_turn("You said to go with the steel rack."),
    )
    receipt = json.loads(
        provider.requests[1]["messages"][1]["content"].split("Receipts:\n")[1].split("\n\n")[0]
    )
    (step,) = receipt["steps"]
    said = [found["said"] for found in step["found"]]
    assert said[0] == "for the garage shelves, let's go with the steel rack"
    # "and not wood" never says "garage": its quiet topic brings it along.
    assert "and not wood" in said
    today = date.today()
    assert memory.day(today) in reply["content"] or "today" in reply["content"].lower()


def test_quiet_topics_are_built_by_code_and_follow_the_conversation(
    talk: Talk, db_session: Session
) -> None:
    chat = ('{"about_you": false, "reply": "Ok."}',)
    for words in (
        "the garage shelves should be steel",
        "yes",
        "and paint the fence green",
        "the fence needs two coats",
    ):
        talk.say(words, plan_turn({"answer": "chat"}), text_turn(*chat))
    talk.new()
    talk.say(
        "garage shelves: bolt them to the wall", plan_turn({"answer": "chat"}), text_turn(*chat)
    )
    db_session.expire_all()
    turns = db_session.execute(select(LedgerTurn).order_by(LedgerTurn.id)).scalars().all()
    topics = [turn.topic_id for turn in turns]
    shelves, yes, fence, coats, bolt = topics
    assert yes == shelves  # says nothing of its own: stays with the turn before
    assert fence != shelves and coats == fence  # a new subject, then more of it
    assert bolt == shelves  # another day, the same words: the same topic
    assert turns[0].terms == ["garage", "shelve", "steel"]
    assert db_session.execute(select(LedgerTopic)).scalars().all()  # never in any API
    paths = set(app.openapi()["paths"])
    assert not [path for path in paths if "topic" in path or "memory" in path]


def test_a_backlog_is_indexed_oldest_first_and_once(db_session: Session) -> None:
    conversation = Conversation(title="old")
    db_session.add(conversation)
    db_session.flush()
    for words in ("the garage shelves", "go with steel", "the fence"):
        db_session.add(LedgerTurn(conversation_id=conversation.id, words=words, status="ended"))
    db_session.add(LedgerTurn(conversation_id=conversation.id, words="running", status="running"))
    db_session.flush()
    assert memory_service.index_pending(db_session) == 3
    assert memory_service.index_pending(db_session) == 0
    turns = db_session.execute(select(LedgerTurn).order_by(LedgerTurn.id)).scalars().all()
    assert [t.terms is not None for t in turns] == [True, True, True, False]
    found = memory_service.recall(db_session, "garage shelves")
    # The direct match first; the short follow-ups ride along in its quiet topic.
    said = [hit.words for hit in found.hits]
    assert said[0] == "the garage shelves" and "go with steel" in said


def test_a_trashed_conversation_is_out_of_recall_and_matches(
    talk: Talk, client: TestClient, db_session: Session
) -> None:
    _grab_odessa(talk)
    first = talk.conversation
    assert memory_service.recall(db_session, "Caribou song").hits
    client.delete(f"/api/agent/conversations/{first}")
    db_session.expire_all()
    assert memory_service.recall(db_session, "Caribou song").hits == ()
    assert memory_service.matches(db_session, "that Caribou song", talk.new()) == []


def test_matches_leave_out_this_conversations_own_entities(talk: Talk, db_session: Session) -> None:
    _grab_odessa(talk)
    assert memory_service.matches(db_session, "that Caribou song", talk.conversation) == []
    elsewhere = memory_service.matches(db_session, "that Caribou song", talk.new())
    assert [m.entity.title for m in elsewhere] == ["Odessa"]
    entity = db_session.execute(select(LedgerEntity)).scalars().first()
    assert entity is not None and {"caribou", "odessa", "song"} <= set(entity.terms or {})


def test_the_self_report_says_what_was_recalled(talk: Talk, db_session: Session) -> None:
    _grab_odessa(talk)
    talk.say(
        "when did I grab that Caribou song?",
        plan_turn(steps(recall("Caribou song"))),
        text_turn("Today: “Odessa”."),
    )
    report = ledger.self_report(db_session, talk.conversation)
    assert "- looked back through what was said and done before: it answered" in report
    assert '  - recalled: "grab Odessa by Caribou"' in report


def test_recall_with_no_memory_reader_fails_the_step_honestly(fleet: Fleet, world: World) -> None:
    provider = ScriptedProvider([plan_turn(steps(recall("x"))), text_turn("I couldn't look.")])
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        result = conductor.run("what did we say?")
    assert result.stop_reason == "completed"
    assert result.tool_calls[0].error == "couldn't look back: memory isn't available here"
