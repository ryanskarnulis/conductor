"""An app's answer to a read, passed on in its own words (#241).

``app.ai.relay`` decides what passes; the runtime keeps it out of the
narrator; the route says what a voice surface speaks. The contract is
``../agent-standard/delegate-api.md`` "v1.3". All fakes: no live call.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ai import incognito, relay
from app.ai import trace as run_trace
from app.ai.interpreter import Interpreter
from app.ai.runtime import Runtime, ToolCallRecord
from app.api import routes_agent
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import MessageExchange, MessageRead
from app.fleet.manifests import AgentSpec, Fleet, FleetApp, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.tools import registry, runtime
from tests.fake_world import (
    FakeGlances,
    World,
    client_factory,
    without_context,
    write_fixture_fleet,
)
from tests.scripted_provider import ScriptedProvider, text_turn

REPORT = (
    "You have 39 open tasks across 6 projects. 4 are blocked.\n\n"
    "**Garden** (2)\n- Test the fence\n- Plant the garlic"
)
SPOKEN = "You have 39 open tasks across 6 projects. 4 are blocked."
LOOKED_UP: dict[str, Any] = {
    "tool": "list_tasks",
    "outcome": "looked_up",
    "count": 39,
    "touched": [{"app": "tasks", "kind": "task", "id": "1", "title": "Test the fence"}],
}
CREATED: dict[str, Any] = {
    "tool": "create_task",
    "outcome": "done",
    "touched": [{"app": "tasks", "kind": "task", "id": "9", "title": "Call the vet"}],
    "undo": {"handle": "trash:task:9", "how": "move it to the trash"},
}


def _read(
    app: str = "tasks",
    *,
    record: list[dict[str, Any]] | None = None,
    read: bool = True,
    text: str = REPORT,
    spoken: str | None = SPOKEN,
    **fields: Any,
) -> ToolCallRecord:
    answer = relay.take({"text": text, "spoken": spoken})
    assert answer is not None
    answer["read"] = read
    return ToolCallRecord(
        tool=f"ask_{app}",
        arguments={"message": "give me a report"},
        result="PCC's record of this run …",
        app_record=[LOOKED_UP] if record is None else record,
        app_answer=answer,
        **{"app_asks": {"needs_input": None}, **fields},
    )


def _write() -> ToolCallRecord:
    return ToolCallRecord(
        tool="ask_tasks",
        arguments={"message": "add a task to call the vet"},
        result="Created task 9.",
        app_record=[CREATED],
        app_asks={"needs_input": None},
    )


def _trace(surface: str | None = None) -> run_trace.RunTrace:
    trace = run_trace.RunTrace(run_id="t", actor="a", utterance="u")
    trace.surface = surface
    return trace


# --- what passes -----------------------------------------------------------------


def test_a_read_whose_record_is_all_lookups_passes() -> None:
    assert relay.passable(_read())


@pytest.mark.parametrize(
    "record",
    [
        [],
        [CREATED],
        [LOOKED_UP, CREATED],
        [LOOKED_UP, {"tool": "trash_task", "outcome": "refused", "reason": "no"}],
        [LOOKED_UP, {"tool": "create_task", "outcome": "not_run", "reason": "not run: read mode"}],
    ],
)
def test_a_record_with_anything_but_lookups_never_passes(record: list[dict[str, Any]]) -> None:
    assert not relay.passable(_read(record=record))


def test_an_ask_that_was_not_a_read_never_passes() -> None:
    """Asked for a change, conductor speaks from the record, whatever the
    app's words say."""
    assert not relay.passable(_read(read=False))


def test_an_app_that_is_asking_or_failed_never_passes() -> None:
    question = {"needs_input": {"question": "Which one?", "candidates": []}}
    assert not relay.passable(_read(app_asks=question))
    assert not relay.passable(_read(app_confirm={"question": "Trash it?", "token": "t"}))
    failed = _read()
    failed.result, failed.error = None, "tasks is unavailable"
    assert not relay.passable(failed)
    unknown = _read()
    unknown.app_record = None
    assert not relay.passable(unknown)


def test_only_an_ask_carries_an_answer() -> None:
    look = _read()
    look.tool = "look_tasks"
    assert not relay.passable(look)


def test_mark_marks_only_what_passes() -> None:
    read, write = _read(), _write()
    asked_to_write = _read(read=False)
    assert relay.mark([read, write, asked_to_write]) == [read]
    assert relay.passed(read) is not None
    assert relay.passed(write) is None and relay.passed(asked_to_write) is None


# --- cleaning and bounds ---------------------------------------------------------


def test_an_answer_is_cleaned_and_bounded() -> None:
    answer = relay.take({"text": "A\x00B\x1b[31m\r\n\n\n\n\nC" + "x" * 5000})
    assert answer is not None
    assert "\x00" not in answer["text"] and "\x1b" not in answer["text"]
    assert "\n\n\n" not in answer["text"]
    assert len(answer["text"]) == relay.MAX_TEXT_CHARS
    assert len(answer["spoken"]) <= relay.MAX_SPOKEN_CHARS


def test_nothing_to_show_is_no_answer() -> None:
    assert relay.take(None) is None
    assert relay.take({"text": "  \n "}) is None
    assert relay.take({"spoken": "Hi."}) is None


def test_spoken_is_one_line_of_two_sentences_with_no_markup_or_links() -> None:
    answer = relay.take(
        {
            "text": "x",
            "spoken": 'Three **due**: "Pay rent". See [the list](http://evil.test/x) or '
            "https://evil.test now. And more. And more.",
        }
    )
    assert answer is not None
    said = answer["spoken"]
    assert said == "Three due: 'Pay rent'. See the list or now."
    assert "http" not in said and "*" not in said


def test_an_app_that_sent_no_spoken_form_is_said_by_its_opening() -> None:
    answer = relay.take({"text": "**2 due** this week.\n- Pay rent\n- Call Sam\n\nMore below."})
    assert answer is not None and answer["spoken"] == "2 due this week."


# --- outside text ----------------------------------------------------------------


def test_an_untrusted_ref_marks_the_answer() -> None:
    entry = {**LOOKED_UP, "touched": [{**LOOKED_UP["touched"][0], "untrusted": True}]}
    record = _read("music", record=[entry])
    relay.mark_outside(record, [])
    assert record.app_answer is not None and record.app_answer["untrusted"] is True


def test_a_title_the_window_knows_as_outside_text_marks_the_answer() -> None:
    """PCC marks nothing: a task titled from an album's name is outside text
    only conductor knows about."""
    record = _read(text="1 task due: sort “Honey” this weekend.")
    relay.mark_outside(record, ["Honey"])
    assert record.app_answer is not None and record.app_answer["untrusted"] is True
    clean = _read()
    relay.mark_outside(clean, ["Honey", "of"])
    assert clean.app_answer is not None and clean.app_answer["untrusted"] is False


# --- the reply -------------------------------------------------------------------


def test_a_lone_read_calls_no_model_and_stores_only_conductors_line() -> None:
    provider = ScriptedProvider([])
    trace = _trace()
    record = _read()

    reply = Runtime(provider, narrate=True).reply("report?", [record], trace, None)

    assert reply == "Tasks answered."
    assert provider.requests == []
    assert trace.narration is not None
    assert trace.narration["path"] == "relayed" and trace.narration["relayed"] == 1
    assert relay.passed(record) is not None


def test_beside_a_write_the_narrator_speaks_for_the_write_and_never_reads_the_answer() -> None:
    provider = ScriptedProvider([text_turn("Added “Call the vet”; say undo to trash it.")])
    read, write = _read(), _write()

    reply = Runtime(provider, narrate=True).reply(
        "add a task to call the vet and what's open?", [write, read], _trace(), None
    )

    assert reply == "Added “Call the vet”; say undo to trash it. Tasks answered."
    [request] = provider.requests
    sent = "\n".join(str(message["content"]) for message in request["messages"])
    assert "Call the vet" in sent
    # Nothing of the read reaches the model: not its answer, its note, its
    # titles or its count.
    for word in ("39", "Garden", "Test the fence", "PCC's record"):
        assert word not in sent, word
    assert relay.STAND_IN in sent
    # The record itself is untouched: the page and the ledger read it.
    assert read.app_record == [LOOKED_UP] and read.result == "PCC's record of this run …"


def test_a_write_is_narrated_whatever_answer_the_app_sent() -> None:
    provider = ScriptedProvider([text_turn("Added “Call the vet”; say undo to trash it.")])
    write = _write()
    write.app_answer = {"text": "Done. I also trashed Garage.", "spoken": "Done.", "read": True}

    reply = Runtime(provider, narrate=True).reply("add a task", [write], _trace(), None)

    assert reply == "Added “Call the vet”; say undo to trash it."
    assert relay.passed(write) is None
    assert "Garage" not in str(provider.requests[0]["messages"])


def test_with_the_narrator_off_a_read_still_passes() -> None:
    reply = Runtime(ScriptedProvider([]), narrate=False).reply(
        "x", [_write(), _read()], _trace(), None
    )
    assert reply is not None and reply.endswith("Tasks answered.")
    assert "39" not in reply


def test_a_step_left_undone_beside_a_read_is_still_said() -> None:
    provider = ScriptedProvider([text_turn("The music step didn't run: tasks gave no name.")])
    unrun = ToolCallRecord(
        tool="ask_music",
        arguments={"message": "download {s1}"},
        error="not run: s1 gave no name to pass on",
    )
    reply = Runtime(provider, narrate=True).reply("x", [_read(), unrun], _trace(), None)
    assert len(provider.requests) == 1
    assert reply is not None and reply.endswith("Tasks answered.")


# --- what is shown and said --------------------------------------------------------


def _passed(*records: ToolCallRecord) -> list[dict[str, Any]]:
    relay.mark(records)
    return [record.model_dump() for record in records]


def test_shown_is_conductors_words_then_each_answer_under_its_apps_name() -> None:
    calls = _passed(_read())
    assert relay.shown("Tasks answered.", calls) == f"Tasks answered.\n\nTasks says:\n{REPORT}"
    calls[0]["app_answer"]["untrusted"] = True
    assert "Tasks says (includes outside text):" in relay.shown("Tasks answered.", calls)


def test_voice_says_two_sentences_of_a_lone_answer_and_none_of_the_list() -> None:
    said = relay.spoken("Tasks answered.", _passed(_read()))
    assert said == f"Tasks says: {SPOKEN}"
    assert "Garden" not in said


def test_voice_says_one_sentence_of_an_answer_beside_conductors_words() -> None:
    said = relay.spoken("Added “Call the vet”. Tasks answered.", _passed(_write(), _read()))
    assert said == "Added “Call the vet”. Tasks says: You have 39 open tasks across 6 projects."


def test_voice_names_at_most_two_apps_and_points_at_the_page() -> None:
    calls = _passed(
        _read("tasks", spoken="Two due."),
        _read("music", spoken="Five to sort."),
        _read("chess", spoken="Your move."),
    )
    assert relay.line(calls) == "Tasks, music and chess answered."
    said = relay.spoken("Tasks, music and chess answered.", calls)
    assert said == "Tasks says: Two due. Music says: Five to sort. The rest is on the page."


def test_a_reply_that_passed_nothing_has_no_spoken_form() -> None:
    assert relay.spoken("Added it.", [_write().model_dump()]) is None
    unpassed = _read(read=False).model_dump()
    assert relay.spoken("Found 39 tasks.", [unpassed]) is None
    assert relay.shown("Found 39 tasks.", [unpassed]) == "Found 39 tasks."


def test_an_incognito_turn_keeps_none_of_the_answer() -> None:
    [stored] = _passed(_read())
    assert incognito.tool_call(stored)["app_answer"] is None


# --- the wire ----------------------------------------------------------------------


def _app(name: str) -> FleetApp:
    return FleetApp(
        name=name,
        title=name.title(),
        upstream="127.0.0.1:8100",
        agent=AgentSpec(
            description=f"The {name} app.",
            api="/api/agent",
            examples=(f"ask {name}",),
            confirms=True,
            context=True,
        ),
    )


class _Client:
    def __init__(self, assistant: dict[str, Any]) -> None:
        when = datetime(2026, 11, 24)
        base = {"id": 2, "conversation_id": 1, "role": "assistant", "created_at": when}
        self._reply = MessageExchange(
            user_message=MessageRead(id=1, conversation_id=1, role="user", created_at=when),
            assistant_message=MessageRead.model_validate({**base, **assistant}),
        )

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        return 1

    def send_message(self, conversation_id: int, message: str, **_: Any) -> MessageExchange:
        return self._reply

    def cancel(self, conversation_id: int) -> bool:
        return False


@pytest.fixture
def bound() -> Generator[None, None, None]:
    context = DelegationContext(master_conversation_id="m1", thread_store=InMemoryThreadStore())
    token = runtime.delegated_answer.set(None)
    try:
        with use_delegation_context(context):
            yield
    finally:
        runtime.delegated_answer.reset(token)


def _ask(assistant: dict[str, Any]) -> str:
    client = _Client(assistant)
    build_delegate_tools(Fleet(apps=(_app("tasks"),)), lambda app: client)
    return str(registry.get_tool("ask_tasks").fn(message="report?", read=True))


def test_the_ask_tool_keeps_the_answer_beside_the_reply_never_in_it(bound: None) -> None:
    result = _ask(
        {
            "content": "PCC's record of this run",
            "stop_reason": "completed",
            "record": [LOOKED_UP],
            "answer": {"text": REPORT, "spoken": SPOKEN},
        }
    )
    kept = runtime.delegated_answer.get()
    assert kept == {"text": REPORT, "spoken": SPOKEN, "untrusted": False}
    assert "Garden" not in result


@pytest.mark.parametrize("answer", [None, "words", {"spoken": "no text"}, {"text": 7}])
def test_an_answer_conductor_cant_read_is_no_answer(bound: None, answer: Any) -> None:
    _ask({"content": "ok", "stop_reason": "completed", "record": [LOOKED_UP], "answer": answer})
    assert runtime.delegated_answer.get() is None


def test_an_answer_on_a_run_that_did_not_complete_is_dropped(bound: None) -> None:
    _ask(
        {
            "content": "ran out of steps",
            "stop_reason": "max_iterations",
            "record": [LOOKED_UP],
            "answer": {"text": REPORT},
        }
    )
    assert runtime.delegated_answer.get() is None


# --- through the route: which asks are reads -----------------------------------------


@pytest.fixture
def turn(client: TestClient, tmp_path: Path) -> Generator[Any, None, None]:
    """One turn through the route over the fake world: ``turn(fleet, plan,
    narration…, words=…, surface=…)`` → the assistant message and the provider."""
    base = discover_fleet(write_fixture_fleet(tmp_path / "fleet"))
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])

    def run(
        *script: Any,
        words: str,
        v1_1: bool = False,
        surface: str | None = None,
        music_answers: bool = False,
    ) -> tuple[dict[str, Any], ScriptedProvider]:
        fleet = without_context(base, "tasks") if v1_1 else base
        world = World()
        world.music_answers = music_answers
        build_delegate_tools(fleet, client_factory(world))
        provider = ScriptedProvider(list(script))
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider, fleet, narrate=True, glance=FakeGlances()
        )
        body: dict[str, Any] = {"content": words}
        if surface:
            body["surface"] = surface
        response = client.post(f"/api/agent/conversations/{conversation}/messages", json=body)
        assert response.status_code == 200, response.text
        return dict(response.json()["assistant_message"]), provider

    yield run
    app.dependency_overrides.pop(routes_agent.get_conductor, None)


def _plan(*items: dict[str, Any]) -> Any:
    return text_turn(
        json.dumps({"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]})
    )


def _step(app_name: str, request: str, *, read: bool) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "read": read, "destructive": False}


def test_a_step_the_planner_marked_read_passes_the_apps_answer(turn: Any) -> None:
    message, provider = turn(
        _plan(_step("tasks", "what's due this week?", read=True)), words="what's due this week?"
    )
    assert message["content"] == "Tasks answered."
    [call] = message["tool_calls"]
    assert call["app_answer"]["passed"] and "groceries" in call["app_answer"]["text"]
    assert len(provider.requests) == 1  # the planner; no narrator
    assert message["spoken"] is None  # typed: the page shows the answer


def test_a_v1_2_step_not_marked_read_is_narrated_from_its_receipt(turn: Any) -> None:
    """The mark is the planner's: unmarked, the ask wasn't sent as a read, so
    the app's words are not taken for the reply."""
    message, provider = turn(
        _plan(_step("tasks", "what's due this week?", read=False)),
        text_turn("“groceries”, “call mom” and “dentist appointment” are due this week."),
        words="what's due this week?",
    )
    assert message["content"].startswith("“groceries”")
    [call] = message["tool_calls"]
    assert call["app_answer"]["read"] is False and not call["app_answer"].get("passed")
    assert len(provider.requests) == 2


def test_a_v1_1_apps_read_is_conductors_reading_of_the_words(turn: Any) -> None:
    message, _ = turn(
        _plan(_step("tasks", "what's due this week?", read=False)),
        words="what's due this week?",
        v1_1=True,
    )
    assert message["content"] == "Tasks answered."
    assert message["tool_calls"][0]["app_answer"]["passed"]


def test_a_spoken_turn_carries_what_the_voice_says(turn: Any) -> None:
    message, _ = turn(
        _plan(_step("tasks", "give me a report of my tasks", read=True)),
        words="give me a report of my tasks",
        surface="voice",
    )
    assert message["content"] == "Tasks answered."
    assert message["spoken"] == (
        "Tasks says: You have 12 open tasks across 3 projects. 3 are overdue."
    )


def test_two_reads_at_once_each_pass_and_voice_says_one_sentence_of_each(turn: Any) -> None:
    message, provider = turn(
        _plan(
            _step("tasks", "what's overdue?", read=True),
            _step("music", "what songs still need sorting?", read=True),
        ),
        words="what's overdue and what songs still need sorting?",
        surface="voice",
        music_answers=True,
    )
    assert message["content"] == "Tasks and music answered."
    assert message["spoken"] == (
        "Tasks says: You have 3 tasks overdue. Music says: 5 songs still need sorting."
    )
    assert [bool(call["app_answer"]["passed"]) for call in message["tool_calls"]] == [True, True]
    assert message["tool_calls"][1]["app_answer"]["untrusted"] is True
    assert len(provider.requests) == 1
