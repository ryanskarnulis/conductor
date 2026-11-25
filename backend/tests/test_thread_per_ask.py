"""A fresh app thread per ask (#262), through the real route and the fake world.

Code picks the thread, never a model: only an answer to the app's own
question (its confirm, its ``app_question``) goes to the thread that asked;
every other ask, reads included, opens a fresh one. A thread is deleted once
no question waits on it, and every thread of a window that is over goes too.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api import routes_agent
from app.db.models import DelegateThread
from app.fleet.manifests import Fleet
from app.fleet.thread_cleanup import SubagentThreadCleanup
from app.main import app
from tests import card_taps
from tests.fake_world import World, client_factory
from tests.scripted_provider import text_turn
from tests.test_memory import ask, fleet, plan_turn, steps, world  # noqa: F401  (fixtures)
from tests.test_timeline import Timeline, timeline  # noqa: F401  (fixtures)

REPORT = "give me a report of my tasks"


@pytest.fixture(autouse=True)
def _cleanup_over_the_fake_fleet(
    client: TestClient,
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
    session_factory: sessionmaker[Session],
) -> None:
    """Cleanup reaches the world's apps, so its deletes land in ``world.deleted``."""
    app.dependency_overrides[routes_agent.get_thread_cleanup] = lambda: SubagentThreadCleanup(
        fleet, session_factory, client_factory=client_factory(world)
    )


def _kept(db: Session) -> list[tuple[str, int]]:
    db.expire_all()
    rows = db.execute(select(DelegateThread).order_by(DelegateThread.id)).scalars()
    return [(row.app_name, row.subagent_conversation_id) for row in rows]


def _read(request: str) -> dict[str, Any]:
    return {**ask("tasks", request), "read": True}


def _passed(exchange: dict[str, Any]) -> list[dict[str, Any]]:
    calls = exchange["assistant_message"]["tool_calls"] or []
    return [c["app_answer"] for c in calls if (c.get("app_answer") or {}).get("passed")]


def test_two_reads_in_one_window_go_to_two_threads_and_neither_is_kept(
    timeline: Timeline,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    first, _ = timeline.say(REPORT, plan_turn(steps(_read(REPORT))))
    again, _ = timeline.say(REPORT, plan_turn(steps(_read(REPORT))))

    one, two = world.threads["tasks"]
    assert one != two
    # The second answer isn't read from the first one's history: both pass.
    assert len(_passed(first)) == 1 and len(_passed(again)) == 1
    assert world.deleted == [("tasks", one), ("tasks", two)]
    assert _kept(timeline.db) == []


def _caribou_queued(world: World) -> None:  # noqa: F811
    """S3's turn, sent straight to music: Honey queued, so music now refuses
    an artist-alone repeat and asks which album."""
    client_factory(world)(type("A", (), {"name": "music"})()).send_message(
        999, "download the new Caribou album"
    )


def test_an_answer_to_an_apps_question_goes_to_the_thread_that_asked(
    timeline: Timeline,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    _caribou_queued(world)
    plan = steps(
        ask("music", "download the other album by Caribou"),
        ask("tasks", "add a task to sort {s1} this weekend"),
    )
    timeline.say("do the same for their other album", plan_turn(plan))
    _, asked = world.threads["music"]
    # Its question waits on the thread that asked: kept, not deleted.
    assert _kept(timeline.db) == [("music", asked)]
    assert world.deleted == []

    timeline.say("Suddenly", text_turn("Queued “Suddenly”; added “sort Suddenly”."))

    assert world.threads["music"][1:] == [asked, asked]
    (task_thread,) = world.threads["tasks"]
    assert [w.title for w in world.writes()] == ["Honey", "Suddenly", "sort Suddenly"]
    # Answered, nothing waits on it any more.
    assert sorted(world.deleted) == sorted([("music", asked), ("tasks", task_thread)])
    assert _kept(timeline.db) == []


def test_a_confirm_card_keeps_its_thread_until_its_tap(
    timeline: Timeline,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    timeline.say(
        "trash the dentist task",
        plan_turn(steps(ask("tasks", "trash the dentist appointment task", destructive=True))),
    )
    (asked,) = world.threads["tasks"]
    assert _kept(timeline.db) == [("tasks", asked)]

    # An unrelated read meanwhile opens its own thread: the card's token
    # stays good app-side (a message on its thread would have voided it).
    timeline.say(REPORT, plan_turn(steps(_read(REPORT))))
    _, read = world.threads["tasks"]
    assert read != asked
    assert _kept(timeline.db) == [("tasks", asked)]
    assert world.deleted == [("tasks", read)]

    card_taps.tap(timeline.client, timeline.id)

    assert world.threads["tasks"][-1] == asked  # the token went where it was armed
    assert world.tasks[3].trashed
    assert world.deleted == [("tasks", read), ("tasks", asked)]
    assert _kept(timeline.db) == []


def test_a_quiet_gap_sweeps_the_threads_its_window_kept(
    timeline: Timeline,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    timeline.say(
        "trash the dentist task",
        plan_turn(steps(ask("tasks", "trash the dentist appointment task", destructive=True))),
    )
    (asked,) = world.threads["tasks"]
    timeline.later()  # the card lapses with its window

    timeline.say(REPORT, plan_turn(steps(_read(REPORT))))

    _, read = world.threads["tasks"]
    assert sorted(world.deleted) == sorted([("tasks", asked), ("tasks", read)])
    assert _kept(timeline.db) == []
    assert not world.tasks[3].trashed
