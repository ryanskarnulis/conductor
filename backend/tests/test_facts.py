"""Memory you can see (#81): remember, forget, a list.

The pure pieces first (``app.ai.facts``: the person's words, the receipts),
then the built-in ``memory`` app through the real route, compiler,
interpreter, ledger and narrator over the session evals' fake fleet, the
model scripted; then the memory list's API and the migration.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app.ai import compiler, facts, run_record, state_block, turn_context
from app.ai import plan as plans
from app.ai import recall as recall_context
from app.ai.runtime import ToolCallRecord
from app.ai.state_block import Fact, Now
from app.config import get_settings
from app.db.models import (
    LedgerEntity,
    LedgerReceipt,
    LedgerStep,
    LedgerTopic,
    LedgerTurn,
    MemoryFact,
)
from app.fleet.manifests import Fleet
from app.services import conversations as conversations_service
from app.services import facts as facts_service
from app.services import ledger
from app.services import memory as memory_service
from tests.fake_world import World
from tests.scripted_provider import text_turn
from tests.test_memory import (  # noqa: F401  (fixtures)
    Talk,
    ask,
    fleet,
    plan_turn,
    recall,
    steps,
    talk,
    world,
)

BACKEND = Path(__file__).resolve().parents[1]
SUNDAYS = "I sort music on Sundays"


# --- the person's words ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_text", "kind", "text"),
    [
        ("Remember I sort music on Sundays", "remember", SUNDAYS),
        ("please remember that my sister is Ana.", "remember", "my sister is Ana"),
        ("don't forget the spare key is under the blue pot", "remember", "the spare key is under the blue pot"),
        ("go with the 30-inch brackets, remember that", "remember", "go with the 30-inch brackets"),
        ("keep in mind: focus time is 9 to 11", "remember", "focus time is 9 to 11"),
        ("forget that", "forget", ""),
        ("forget it", "forget", ""),
        ("forget that I sort music on Sundays", "forget", SUNDAYS),
        ("Forget about my sister", "forget", "my sister"),
        ("what do you know about me?", "list", "what do you know about me?"),
        ("do you remember what day I sort music?", "list", "do you remember what day I sort music?"),
        ("remember when I sort music?", "list", "remember when I sort music?"),
    ],
)  # fmt: skip
def test_parse_reads_the_leading_words(request_text: str, kind: str, text: str) -> None:
    assert facts.parse(request_text) == facts.Intent(kind, text)  # type: ignore[arg-type]


def test_a_fact_is_kept_in_the_persons_own_words() -> None:
    words = "Remember I Sort music on sundays, and add a task for it"
    # The planner's version is found in what they said: their spelling wins.
    assert facts.own_words("remember I sort music on Sundays", words) == "I Sort music on sundays"
    # A paraphrase isn't what they said: their own words after "remember" instead.
    assert (
        facts.own_words("remember that the person sorts music on Sundays", "remember I sort music")
        == "I sort music"
    )
    # Nothing in their words to keep (a resumed plan told only "yes").
    assert facts.own_words("remember the person sorts music", "yes") is None
    assert facts.own_words("remember that", "remember that") is None
    assert facts.literal("sort  MUSIC", "I sort music on Sundays") == "sort music"
    assert facts.same("I sort music.", "i sort  music")


def test_what_to_forget_comes_from_the_persons_words() -> None:
    # #134: told "forget that", the planner sometimes wrote its prompt's example.
    assert facts.forget_target(SUNDAYS, "forget that") == ""
    assert facts.forget_target(SUNDAYS, "forget that I sort music on Sundays") == SUNDAYS
    assert facts.forget_target("sort music", "please forget the Sundays music sorting") == (
        "sort music"
    )
    assert facts.forget_target("my sister is Ana", "forget my sister") == "my sister"
    # No words of theirs (a resumed plan), or words that aren't a forget: the plan's.
    assert facts.forget_target(SUNDAYS, "") == SUNDAYS
    assert facts.forget_target(SUNDAYS, "yes") == SUNDAYS


def test_the_receipts_are_v1_1_entries_with_no_undo_handle() -> None:
    saved = facts.Saved(4, SUNDAYS, datetime(2026, 11, 3), "", 9)
    result, (entry,) = facts.remembered(saved, already=False)
    assert entry["outcome"] == "done" and entry["tool"] == "remember"
    assert entry["touched"] == [{"app": "memory", "kind": "fact", "id": "4", "title": SUNDAYS}]
    assert "handle" not in entry["undo"] and entry["source_turn"] == 9
    assert SUNDAYS in result
    _, (gone,) = facts.forgot(saved)
    assert gone["outcome"] == "done" and gone["undo"] == {"none": facts.NO_UNDO_FORGET}
    _, (empty,) = facts.listed([], total=0)
    assert empty["outcome"] == "looked_up" and empty["reason"] == "nothing saved yet"


def test_memory_is_one_ask_enum_value_and_one_prompt_line(fleet: Fleet) -> None:  # noqa: F811
    schema = plans.schema(fleet)
    kinds = {
        k["properties"]["do"]["const"]: k
        for k in schema["anyOf"][0]["properties"]["steps"]["items"]["anyOf"]
    }
    assert "memory" in kinds["ask"]["properties"]["app"]["enum"]
    assert len(kinds) == 10  # the step kinds don't change
    prompt = compiler.planner_prompt(fleet)
    assert prompt.count("- memory (ask):") == 1 and '"forget that"' in prompt
    plan = plans.load(steps(ask("memory", "remember I sort music on Sundays")))
    assert plans.problems(plan, plans.Checks(fleet=fleet)) == []
    assert plans.unrunnable(plan) == []
    assert plans.ack(plan) is None  # memory answers at once: nothing to wait for


def test_state_lists_saved_facts_bounded() -> None:
    saved = [Fact("x" * 400, "Tuesday 2026-11-03") for _ in range(20)]
    block = state_block.render(date(2026, 11, 6), Now(), [], [], [], [], saved)
    assert "Facts the person asked you to remember (their words, as of when saved):" in block
    lines = [line for line in block.splitlines() if line.startswith('- "x')]
    assert len(lines) == state_block.MAX_FACTS
    assert sum(len(line) + 1 for line in lines) + 90 <= state_block.FACTS_MAX_CHARS
    assert len(block) <= state_block.MAX_CHARS

    def broken() -> list[Fact]:
        raise RuntimeError("db gone")

    opening = turn_context.build(
        date(2026, 11, 6), Now(), "hi", lambda: [], lambda: [], facts=broken
    )
    assert "Facts the person" not in opening and "[End of state]" in opening


# --- through the route ------------------------------------------------------------------------


def _remember(talk: Talk, words: str = f"remember {SUNDAYS}") -> dict[str, Any]:  # noqa: F811
    reply, _ = talk.say(
        words,
        plan_turn(steps(ask("memory", words))),
        text_turn(f"Got it: “{SUNDAYS}”."),
    )
    return reply


def _opening(provider: Any) -> str:
    return str(provider.requests[0]["messages"][1]["content"])


def test_remember_saves_the_words_with_a_receipt_and_state_shows_it_later(
    talk: Talk,  # noqa: F811
    world: World,  # noqa: F811
    db_session: Session,
) -> None:
    reply = _remember(talk)
    assert SUNDAYS in reply["content"]
    assert not world.writes()  # no app was asked
    (call,) = reply["tool_calls"]
    assert call["tool"] == "ask_memory" and call["app_record"][0]["outcome"] == "done"
    db_session.expire_all()
    (row,) = db_session.execute(select(MemoryFact)).scalars().all()
    assert row.text == SUNDAYS
    source = db_session.get(LedgerTurn, row.source_turn_id)
    assert source is not None and source.words == f"remember {SUNDAYS}"
    step = db_session.execute(select(LedgerStep)).scalar_one()
    assert (step.kind, step.app, step.outcome) == ("ask", "memory", "done")
    (receipt,) = step.receipts
    assert (receipt.app, receipt.tool, receipt.outcome) == ("memory", "remember", "done")
    assert receipt.fact == SUNDAYS and receipt.cites_turn_id == source.id
    assert receipt.undo_none == facts.NO_UNDO_REMEMBER and receipt.undo_handle is None
    assert not db_session.execute(select(LedgerEntity)).scalars().all()  # no registry entry
    # Another day: the fact is in State, and a plan can use it.
    talk.new()
    _, provider = talk.say(
        "what day do I sort music?",
        plan_turn({"answer": "chat"}),
        text_turn('{"about_you": true, "reply": "Sundays."}'),
    )
    opening = _opening(provider)
    assert "Facts the person asked you to remember (their words, as of when saved):" in opening
    assert f'- "{SUNDAYS}" (saved ' in opening
    speaker = provider.requests[1]["messages"][1]["content"]
    assert SUNDAYS in speaker  # "what do you know about me?" answered directly sees them too
    # Saying it twice keeps one fact.
    _remember(talk)
    db_session.expire_all()
    assert len(facts_service.live(db_session)) == 1


def test_a_question_lists_the_facts_with_where_each_came_from(
    talk: Talk,  # noqa: F811
) -> None:
    _remember(talk)
    talk.new()
    reply, provider = talk.say(
        "what do you know about me?",
        plan_turn(steps(ask("memory", "what do you know about me?"))),
        text_turn("You told me you sort music on Sundays."),
    )
    (call,) = reply["tool_calls"]
    assert call["app_record"][0]["touched"][0]["title"] == SUNDAYS
    assert (
        "you said it" in call["result"]
        and "in “remember I sort music on Sundays”" in (call["result"])
    )
    # Code holds the reply to the receipt: the fact is named, quoted.
    assert f"“{SUNDAYS}”" in reply["content"]


def test_forget_that_is_gone_from_recall_state_and_replays(
    talk: Talk,  # noqa: F811
    db_session: Session,
) -> None:
    _remember(talk)
    db_session.expire_all()
    assert memory_service.recall(db_session, "sort music").hits
    source_id = db_session.execute(select(MemoryFact.source_turn_id)).scalar_one()
    topic_id = db_session.get(LedgerTurn, source_id).topic_id  # type: ignore[union-attr]
    assert topic_id is not None
    reply, provider = talk.say(
        "forget that",
        plan_turn(steps(ask("memory", "forget that", destructive=True))),
        text_turn(f"Forgot “{SUNDAYS}”."),
    )
    # A forget is never gated: the person asked memory, and memory is code.
    (call,) = reply["tool_calls"]
    assert call["app_record"][0]["tool"] == "forget"
    assert call["app_record"][0]["outcome"] == "done"
    db_session.expire_all()
    assert facts_service.live(db_session) == []
    source = db_session.get(LedgerTurn, source_id)
    assert source is not None and source.forgotten_at is not None
    assert source.terms == [] and source.topic_id is None
    topic = db_session.get(LedgerTopic, topic_id)
    assert topic is not None and "music" not in (topic.terms or {})
    forgetter = db_session.execute(
        select(LedgerTurn).order_by(LedgerTurn.id.desc()).limit(1)
    ).scalar_one()
    assert forgetter.forgotten_at is not None and forgetter.terms == []
    receipts = db_session.execute(select(LedgerReceipt)).scalars().all()
    assert all(r.fact is None for r in receipts)  # no receipt quotes it any more
    # Gone from recall, matches and the state block; the replay keeps the
    # turns' shape and none of their words.
    assert memory_service.recall(db_session, "sort music Sundays").hits == ()
    assert memory_service.recall(db_session, "forget").hits == ()
    turns = conversations_service.stored_turns(db_session, talk.conversation)
    assert [t["content"] for t in turns] == [conversations_service.FORGOTTEN] * 4
    assert all(t["tool_calls"] is None for t in turns)
    report = ledger.self_report(db_session, talk.conversation)
    assert "sort music" not in report and "asked you to forget" in report
    _, after = talk.say(
        "what day do I sort music?",
        plan_turn(steps(recall("sort music"))),
        text_turn("I have no record of that."),
    )
    opening = _opening(after)
    assert "sort music on" not in opening and "Facts the person" not in opening
    talk.new()
    _, elsewhere = talk.say(
        "when do I sort music?",
        plan_turn({"answer": "chat"}),
        text_turn('{"about_you": true, "reply": "I don\'t know."}'),
    )
    assert "Sundays" not in _opening(elsewhere)
    assert "Sundays" not in elsewhere.requests[1]["messages"][1]["content"]


def test_forget_names_a_fact_by_its_words_and_never_guesses(
    talk: Talk,  # noqa: F811
    db_session: Session,
) -> None:
    _remember(talk)
    _remember(talk, "remember I water the plants on Sundays")
    talk.new()
    nothing_here, _ = talk.say(
        "forget that",
        plan_turn(steps(ask("memory", "forget that"))),
        text_turn("Nothing was saved here."),
    )
    assert nothing_here["tool_calls"][0]["app_record"][0]["outcome"] == "refused"
    tie, _ = talk.say(
        "forget the Sundays one",
        plan_turn(steps(ask("memory", "forget the Sundays one"))),
        text_turn("Two match."),
    )
    entry = tie["tool_calls"][0]["app_record"][0]
    assert entry["outcome"] == "refused" and "more than one" in entry["reason"]
    talk.say(
        "forget that I water the plants",
        plan_turn(steps(ask("memory", "forget that I water the plants"))),
        text_turn("Forgot “I water the plants on Sundays”."),
    )
    db_session.expire_all()
    assert [row.text for row in facts_service.live(db_session)] == [SUNDAYS]
    # "forget that" is the person's: a planner that names some other fact
    # (its prompt's example) forgets the one said here, never that one.
    _remember(talk, "remember the spare key is under the blue pot")
    talk.say(
        "forget that",
        plan_turn(steps(ask("memory", f"forget that {SUNDAYS}"))),
        text_turn("Forgot “the spare key is under the blue pot”."),
    )
    db_session.expire_all()
    assert [row.text for row in facts_service.live(db_session)] == [SUNDAYS]


def test_outside_text_never_becomes_a_fact(
    talk: Talk,  # noqa: F811
    db_session: Session,
) -> None:
    talk.say(
        "grab Odessa by Caribou",
        plan_turn(steps(ask("music", "grab the song Odessa by Caribou"))),
        text_turn("Saved “Odessa”."),
    )
    talk.new()
    talk.say(
        "remember that Caribou song",
        plan_turn(steps(recall("that Caribou song"), ask("memory", "remember {s1}"))),
        text_turn("I didn't save it."),
    )
    db_session.expire_all()
    assert facts_service.live(db_session) == []


def test_memory_with_no_app_fails_the_step_honestly(fleet: Fleet, world: World) -> None:  # noqa: F811
    from app.ai.interpreter import Interpreter
    from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
    from tests.fake_world import FakeGlances
    from tests.scripted_provider import ScriptedProvider

    provider = ScriptedProvider(
        [plan_turn(steps(ask("memory", "remember x"))), text_turn("I couldn't.")]
    )
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        result = conductor.run("remember x")
    assert result.tool_calls[0].error == "couldn't reach memory: it isn't available here"


# --- saved facts in another app's ask (#134) --------------------------------------------------

SORTING_DAY = "add a task for my music sorting day to clean the record player"


@pytest.mark.parametrize(
    ("request_text", "saved", "attached"),
    [
        # Real overlap: "music" and "sorting" are two of the fact's own terms.
        (SORTING_DAY, [SUNDAYS], [SUNDAYS]),
        ("add a task to buy music stands", [SUNDAYS], []),  # one shared word isn't enough
        ("add a task for Sunday", [SUNDAYS], []),  # nor a weekday
        ("what's due today?", [SUNDAYS], []),
        # Stems and verb forms still match: "dentists"/"dentist", "booked"/"book".
        (
            "remind me to book the dentists",
            ["Dr Okafor is my dentist, booked through the clinic"],
            ["Dr Okafor is my dentist, booked through the clinic"],
        ),
        # A fact the request already says isn't sent again.
        (f"add a task: {SUNDAYS.lower()}", [SUNDAYS], []),
        # Best first, at most two; a tie keeps newest first.
        (
            "add a task to sort the music by the blue record player on Sundays",
            ["I sort music on Sundays", "the record player is blue", "I sort music in folders"],
            ["I sort music on Sundays", "the record player is blue"],
        ),
        ("", [SUNDAYS], []),
    ],
)  # fmt: skip
def test_a_fact_goes_with_an_ask_only_on_real_overlap(
    request_text: str, saved: list[str], attached: list[str]
) -> None:
    assert facts.related(request_text, saved) == attached


def test_attached_facts_are_quoted_after_the_words_as_context() -> None:
    sent = facts.attach(SORTING_DAY, [SUNDAYS])
    # The person's words come first, untouched; the fact follows on its own line.
    assert sent == f'{SORTING_DAY}\n\n({facts.CONTEXT_LEAD} "{SUNDAYS}")'
    assert facts.attach(SORTING_DAY, []) == SORTING_DAY
    assert facts.attach("  ", [SUNDAYS]) == "  "  # nothing asked: nothing to add to
    odd = facts.attach("x", ['say "hi"\nto Ana', "y" * 500, "third"])
    assert "say 'hi' to Ana" in odd and "\nto" not in odd.split("\n\n", 1)[1]
    assert "y" * facts.MAX_ATTACHED_CHARS not in odd and "…" in odd
    assert "third" not in odd  # at most two


def test_the_record_lines_models_read_leave_the_facts_out() -> None:
    record = ToolCallRecord(
        tool="ask_tasks",
        arguments={"message": SORTING_DAY, facts.ARGUMENT: [SUNDAYS]},
        result="Created task",
    )
    assert run_record.shown(record.arguments) == {"message": SORTING_DAY}
    assert "Sundays" not in run_record.render([record])
    history: list[dict[str, Any]] = [
        {"role": "user", "content": SORTING_DAY},
        {"role": "assistant", "content": "ok", "tool_calls": [record.model_dump()]},
    ]
    history += [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "ok"}]
    lines = "\n".join(recall_context.Recall.of(history).record)
    assert SORTING_DAY in lines and "Sundays" not in lines
    hit = recall_context.EarlierCall(datetime(2026, 11, 3), SORTING_DAY, record)
    (found,) = recall_context.hits("that record player task again", [hit])
    assert "Sundays" not in found
    assert recall_context.hits("that Sundays one again", [hit]) == []  # nor searched


def _sorting_day(talk: Talk, request: str = SORTING_DAY) -> dict[str, Any]:  # noqa: F811
    reply, _ = talk.say(
        SORTING_DAY,
        plan_turn(steps(ask("tasks", request))),
        text_turn("Added “clean the record player” for Sunday."),
    )
    return reply


def test_an_ask_carries_the_fact_its_words_point_at(
    talk: Talk,  # noqa: F811
    world: World,  # noqa: F811
    db_session: Session,
) -> None:
    _remember(talk)
    talk.new()  # another day
    reply = _sorting_day(talk)
    # The app got the person's words, then the fact as quoted context.
    ((app, sent),) = [(app, sent) for app, sent in world.sent if app == "tasks"]
    assert sent == facts.attach(SORTING_DAY, [SUNDAYS])
    # The step's record and the ledger say which facts went.
    (call,) = reply["tool_calls"]
    assert call["arguments"] == {"message": SORTING_DAY, facts.ARGUMENT: [SUNDAYS]}
    db_session.expire_all()
    step = db_session.execute(select(LedgerStep).where(LedgerStep.app == "tasks")).scalar_one()
    assert step.arguments[facts.ARGUMENT] == [SUNDAYS]
    # The app did the interpreting: its task is due on a Sunday, titled with
    # the person's words and none of the context.
    ((task),) = [t for t in world.tasks.values() if "record player" in t.title]
    assert task.due is not None and task.due.weekday() == 6
    assert "Context" not in task.title and "Sundays" not in task.title


def test_an_ask_with_no_fact_to_point_at_goes_as_the_plan_wrote_it(
    talk: Talk,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    _remember(talk)
    reply = _sorting_day(talk, "add a task to clean the record player")
    assert reply["tool_calls"][0]["arguments"] == {
        "message": "add a task to clean the record player"
    }
    assert world.sent[-1] == ("tasks", "add a task to clean the record player")


def test_a_forgotten_fact_goes_with_no_later_ask(
    talk: Talk,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    _remember(talk)
    _sorting_day(talk)
    talk.say(
        "forget that I sort music on Sundays",
        plan_turn(steps(ask("memory", "forget that I sort music on Sundays"))),
        text_turn(f"Forgot “{SUNDAYS}”."),
    )
    reply = _sorting_day(talk)
    assert facts.ARGUMENT not in reply["tool_calls"][0]["arguments"]
    assert world.sent[-1] == ("tasks", SORTING_DAY)


def test_memory_that_cant_be_read_never_costs_the_ask(fleet: Fleet, world: World) -> None:  # noqa: F811
    from app.ai.interpreter import Interpreter
    from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
    from tests.fake_world import FakeGlances
    from tests.scripted_provider import ScriptedProvider

    def broken() -> list[str]:
        raise RuntimeError("db gone")

    provider = ScriptedProvider(
        [plan_turn(steps(ask("tasks", SORTING_DAY))), text_turn("Added it.")]
    )
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        result = conductor.run(SORTING_DAY, saved_facts=broken)
    assert result.tool_calls[0].arguments == {"message": SORTING_DAY}
    assert result.tool_calls[0].error is None


def test_parallel_reads_carry_their_facts_too(fleet: Fleet, world: World) -> None:  # noqa: F811
    from app.ai.interpreter import Interpreter
    from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
    from tests.fake_world import FakeGlances
    from tests.scripted_provider import ScriptedProvider

    provider = ScriptedProvider(
        [
            plan_turn(
                steps(
                    ask("tasks", "what's due on my music sorting day?"),
                    ask("music", "what songs still need sorting?"),
                )
            ),
            text_turn("Nothing's due; 3 songs wait."),
        ]
    )
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        result = conductor.run("what's due on my sorting day?", saved_facts=lambda: [SUNDAYS])
    tasks_call, music_call = result.tool_calls
    assert tasks_call.arguments[facts.ARGUMENT] == [SUNDAYS]
    assert facts.ARGUMENT not in music_call.arguments  # "sorting" alone isn't the fact


# --- the memory list's API ----------------------------------------------------------------


def test_the_list_shows_each_fact_and_where_it_came_from(
    talk: Talk,  # noqa: F811
    client: TestClient,
) -> None:
    _remember(talk)
    listed = client.get("/api/memory/facts")
    assert listed.status_code == 200, listed.text
    (fact,) = listed.json()
    assert fact["text"] == SUNDAYS
    source = fact["source"]
    assert source["words"] == f"remember {SUNDAYS}" and source["kind"] == "message"
    assert source["thread"] == {"id": talk.conversation, "title": f"remember {SUNDAYS}"}
    assert source["said_at"].endswith(("Z", "+00:00"))


def test_an_edit_is_the_persons_note_and_the_old_words_leave_memory(
    talk: Talk,  # noqa: F811
    client: TestClient,
    db_session: Session,
) -> None:
    _remember(talk)
    fact_id = client.get("/api/memory/facts").json()[0]["id"]
    old_source = db_session.execute(select(MemoryFact.source_turn_id)).scalar_one()
    changed = client.patch(
        f"/api/memory/facts/{fact_id}",
        json={"text": "I sort music on Saturdays", "conversation_id": talk.conversation},
    )
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert body["fact"]["text"] == "I sort music on Saturdays"
    assert body["fact"]["source"]["kind"] == "note"
    assert body["message"]["role"] == "user"
    assert body["message"]["content"] == "Changed a memory to “I sort music on Saturdays”"
    db_session.expire_all()
    assert db_session.get(LedgerTurn, old_source).forgotten_at is not None  # type: ignore[union-attr]
    # Indexed as the next turn closes: the new words are found, the old aren't.
    memory_service.index_pending(db_session)
    said = [hit.words for hit in memory_service.recall(db_session, "sort music").hits]
    assert said == ["Changed a memory to “I sort music on Saturdays”"]
    # Outside a conversation: no note, no source turn.
    alone = client.patch(f"/api/memory/facts/{fact_id}", json={"text": "I sort music daily"})
    assert alone.json()["message"] is None and alone.json()["fact"]["source"] is None
    assert client.patch(f"/api/memory/facts/{fact_id}", json={"text": "  "}).status_code == 422
    assert client.patch("/api/memory/facts/999", json={"text": "x"}).status_code == 404


def test_forget_from_the_list_is_a_note_that_is_itself_forgotten(
    talk: Talk,  # noqa: F811
    client: TestClient,
    db_session: Session,
) -> None:
    _remember(talk)
    fact_id = client.get("/api/memory/facts").json()[0]["id"]
    gone = client.delete(f"/api/memory/facts/{fact_id}?conversation_id={talk.conversation}")
    assert gone.status_code == 200, gone.text
    assert gone.json()["fact"] is None
    assert gone.json()["message"]["content"] == f"Forgot “{SUNDAYS}” in the memory list"
    assert client.get("/api/memory/facts").json() == []
    db_session.expire_all()
    memory_service.index_pending(db_session)
    assert memory_service.recall(db_session, "sort music Sundays").hits == ()
    assert client.delete(f"/api/memory/facts/{fact_id}").status_code == 404
    # The thread still shows what the person did, on screen.
    detail = client.get(f"/api/agent/conversations/{talk.conversation}").json()
    assert detail["messages"][-1]["content"] == f"Forgot “{SUNDAYS}” in the memory list"
    assert (
        client.delete("/api/memory/facts/1?conversation_id=999").status_code == 404
    )  # no such fact


# --- the migration -------------------------------------------------------------------------


def test_the_memory_facts_migration_goes_up_and_back_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        config = Config()
        config.set_main_option("script_location", str(BACKEND / "app" / "alembic"))
        command.upgrade(config, "head")
        engine = create_engine(url)
        assert "memory_facts" in inspect(engine).get_table_names()
        turns = {c["name"] for c in inspect(engine).get_columns("ledger_turns")}
        assert "forgotten_at" in turns
        command.downgrade(config, "e3a7c5d1f804")
        assert "memory_facts" not in inspect(engine).get_table_names()
        turns = {c["name"] for c in inspect(engine).get_columns("ledger_turns")}
        assert "forgotten_at" not in turns
        command.upgrade(config, "head")
        assert "memory_facts" in inspect(engine).get_table_names()
        engine.dispose()
    finally:
        get_settings.cache_clear()
