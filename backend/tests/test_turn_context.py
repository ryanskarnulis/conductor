"""The state block and recall that open every turn, in place of replayed chat (#64)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.ai import recall, state_block, turn_context
from app.ai.interpreter import Interpreter
from app.ai.runtime import ToolCallRecord
from app.ai.recall import EarlierCall, Recall
from app.ai.state_block import Now
from app.api import routes_agent
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet, discover_fleet
from app.main import app
from app.tools import registry
from tests.fake_world import write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn

THURSDAY = date(2026, 11, 5)


def _ask(app: str, message: str, result: str | None, **extra: Any) -> dict[str, Any]:
    return ToolCallRecord(
        tool=f"ask_{app}", arguments={"message": message}, result=result, **extra
    ).model_dump()


def _turns(*pairs: tuple[str, str | None, list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for said, reply, calls in pairs:
        history.append({"role": "user", "content": said, "tool_calls": None})
        history.append({"role": "assistant", "content": reply, "tool_calls": calls})
    return history


_ODESSA = {
    "tool": "save_song",
    "outcome": "done",
    "touched": [{"app": "music", "kind": "song", "id": "7", "title": "Odessa", "untrusted": True}],
    "undo": {"none": "files stay on disk"},
}


# --- the state block -------------------------------------------------------------------


def test_the_calendar_gives_every_weekday_one_date_after_today() -> None:
    friday = date(2026, 11, 6)
    lines = state_block.calendar_lines(friday)
    assert lines[0] == "Today: Friday 2026-11-06"
    upcoming = lines[1] + lines[2]
    assert upcoming.count("Friday") == 1 and "Friday 2026-11-13" in upcoming


def test_the_block_says_where_the_person_is() -> None:
    block = state_block.render(THURSDAY, Now(surface="voice", panel="music"), [])
    assert "your reply is spoken aloud: one or two short sentences, no markdown" in block
    assert "Open panel: music." in block
    assert "Surface" not in state_block.render(THURSDAY, Now(), [])


def test_a_question_the_last_reply_asked_is_waiting_on_the_person() -> None:
    history = _turns(
        ("wipe the kitchen project", "That trashes Kitchen and 3 tasks. Go ahead?", None)
    )
    block = state_block.render(THURSDAY, Now(), history)
    assert 'Waiting on the person: your last reply asked "That trashes Kitchen' in block


def test_recent_entities_and_running_jobs_come_from_the_apps_records() -> None:
    job = {
        "tool": "download_playlist",
        "outcome": "done",
        "touched": [{"app": "music", "kind": "download_job", "id": "112", "title": "Mix"}],
        "job": {"ref": "112", "status": "working", "done": 7, "total": 14},
    }
    finished = {**job, "job": {"ref": "9", "status": "completed", "done": 3, "total": 3}}
    history = _turns(
        ("grab Odessa", "Saved.", [_ask("music", "grab Odessa", "Saved.", app_record=[_ODESSA])]),
        (
            "grab the mix",
            "On it.",
            [_ask("music", "grab the mix", "ok", app_record=[job, finished])],
        ),
    )
    block = state_block.render(THURSDAY, Now(), history)
    assert "music download_playlist (job 112): working, 7 of 14" in block
    assert "job 9" not in block
    entities = block.split("Recent entities (newest first, as of then):\n")[1]
    assert entities.index("download_job 112") < entities.index("song 7")
    assert 'music song 7: "Odessa" (outside text)' in entities


# --- recall ----------------------------------------------------------------------------


def test_recall_keeps_the_record_the_requests_and_the_last_exchange() -> None:
    history = _turns(
        ("what's due today?", "Groceries.", [_ask("tasks", "what's due today?", "Groceries")]),
        ("add call mom", None, [_ask("tasks", "add call mom", None, error="down")]),
        ("push the dentist", "Moved it. Anything else?", [_ask("tasks", "push dentist", "Moved")]),
    )
    rendered = Recall.of(history).render()
    record = rendered.split("\n\n")[0]
    assert "Groceries" in record and "Moved" in record
    assert "down" not in record  # failed with nothing done: nothing to point back at
    assert '- "what\'s due today?"' in rendered and '- "add call mom"' in rendered
    assert 'The person said: "push the dentist"' in rendered
    assert 'What you said then (your words, not a record): "Moved it. Anything else?"' in rendered


def test_a_new_conversation_recalls_what_shares_the_persons_words() -> None:
    tuesday = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
    calls = [
        EarlierCall(
            tuesday,
            "grab Odessa by Caribou",
            ToolCallRecord.model_validate(
                _ask("music", "grab the song Odessa by Caribou", "Saved “Odessa” by Caribou.")
            ),
        ),
        EarlierCall(
            tuesday,
            "add water the plants",
            ToolCallRecord.model_validate(_ask("tasks", "add water the plants", "Added.")),
        ),
        EarlierCall(
            tuesday,
            "grab Sun by Caribou",
            ToolCallRecord.model_validate(_ask("music", "grab Sun by Caribou", None, error="x")),
        ),
    ]
    hits = recall.hits("put that Caribou song in the house folder", calls)
    assert len(hits) == 1
    assert hits[0].startswith('Tuesday 2026-11-03, asked "grab Odessa by Caribou"')
    assert "Odessa" in hits[0]
    assert recall.hits("put that in there", calls) == []  # no word that names a thing
    # A fresh request that points back at nothing gets no earlier conversation,
    # so a past write is never read as the state of things.
    assert recall.hits("move Caribou songs to the house folder", calls) == []


def _fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


def _plan_ask(app_name: str, request: str) -> Any:
    step = {"id": "s1", "do": "ask", "app": app_name, "request": request, "destructive": False}
    return text_turn(json.dumps({"steps": [step]}))


def _conversation(client: TestClient) -> int:
    return int(client.post("/api/agent/conversations", json={}).json()["id"])


def test_the_route_recalls_an_earlier_conversation(client: TestClient, tmp_path: Path) -> None:
    @registry.tool
    def ask_music(message: str) -> str:
        """Scratch music agent."""
        return "Saved “Odessa” by Caribou to the library."

    fleet = _fleet(tmp_path)

    def use(provider: ScriptedProvider) -> None:
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(provider, fleet)

    use(ScriptedProvider([_plan_ask("music", "grab Odessa by Caribou")]))
    first = _conversation(client)
    client.post(
        f"/api/agent/conversations/{first}/messages", json={"content": "grab Odessa by Caribou"}
    )

    later = ScriptedProvider([text_turn(json.dumps({"answer": "decline"}))])
    use(later)
    second = _conversation(client)
    response = client.post(
        f"/api/agent/conversations/{second}/messages",
        json={"content": "put that Caribou song in the house folder", "surface": "voice"},
    )
    assert response.status_code == 200
    opening = later.requests[0]["messages"][1]["content"]
    assert "Possibly what the person is pointing back at, from earlier conversations" in opening
    assert "Odessa" in opening
    assert "Surface: voice" in opening
    # The new conversation has no record of its own: only the earlier one's hit.
    assert "The record of earlier turns in this conversation" not in opening


# --- bounds ------------------------------------------------------------------------------


def _long_history(turns: int) -> list[dict[str, Any]]:
    words = "lorem ipsum dolor sit amet " * 80
    return _turns(
        *[
            (
                f"request {n} {words}",
                f"reply {n} {words}",
                [
                    _ask(
                        "tasks", f"ask {n} {words}", f"answer {n} {words}", app_record=[_ODESSA] * 5
                    )
                ],
            )
            for n in range(turns)
        ]
    )


def test_the_opening_is_bounded_however_long_the_history() -> None:
    history = _long_history(300)
    calls = [
        EarlierCall(
            datetime(2026, 11, 1, tzinfo=UTC) + timedelta(minutes=n),
            "lorem",
            ToolCallRecord.model_validate(_ask("tasks", "lorem ipsum", "lorem " * 400)),
        )
        for n in range(200)
    ]
    opening = turn_context.build(
        THURSDAY, Now(surface="voice", panel="music"), "lorem ipsum", lambda: history, lambda: calls
    )
    assert len(opening) <= turn_context.MAX_CHARS
    # ~4 characters a token: the opening stays under ~2.9k tokens (saved facts, #81).
    assert turn_context.MAX_CHARS <= 11_500
    short = turn_context.build(THURSDAY, Now(), "hi", lambda: _long_history(3), lambda: [])
    assert len(short) < len(opening)


def test_a_broken_history_leaves_the_calendar_and_never_the_turn() -> None:
    def broken() -> list[Any]:
        raise RuntimeError("db gone")

    opening = turn_context.build(THURSDAY, Now(), "hi", broken, broken)
    assert opening.startswith("[State at the start of this turn")
    assert "Today: Thursday 2026-11-05" in opening

    unreadable = [{"role": "assistant", "content": "x", "tool_calls": [{"nope": 1}]}]
    assert "Today:" in turn_context.build(THURSDAY, Now(), "hi", lambda: unreadable, lambda: [])


def test_a_turn_has_one_prompt_prefix_and_the_system_prompt_never_moves(
    tmp_path: Path,
) -> None:
    @registry.tool
    def ask_tasks(message: str) -> str:
        """Scratch tasks agent."""
        return f"noted {message}"

    provider = ScriptedProvider(
        [
            text_turn("not a plan"),  # the compiler's repair rides the same prefix
            _plan_ask("tasks", "note a and b"),
            _plan_ask("tasks", "again"),
        ]
    )
    conductor = Interpreter(provider, _fleet(tmp_path))
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        conductor.run("note two things", context="[State]\nToday: x")
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        conductor.run("and again", context="[State]\nToday: y")

    first_turn = provider.requests[:2]
    prefixes = {(r["messages"][0]["content"], r["messages"][1]["content"]) for r in first_turn}
    assert len(prefixes) <= 2  # the acceptance bound; in fact it is one
    assert len(prefixes) == 1
    assert first_turn[0]["messages"][1]["content"].startswith("[State]\nToday: x")
    systems = {r["messages"][0]["content"] for r in provider.requests}
    assert len(systems) == 1  # byte-identical across turns: the cache keeps it
    assert provider.requests[2]["messages"][1]["content"] == (
        "[State]\nToday: y\n\nThe person said: and again"
    )
