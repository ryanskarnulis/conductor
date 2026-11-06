"""The fake fleet reads requests the way a cooperative app would, and every
write leaves a receipt. CI, no GPU."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tests.fake_world import MusicApp, TasksApp, World, friday_targets, parse_date, weekend_targets
from app.fleet.delegate import DelegateConfirmRefused

THURSDAY = date(2026, 11, 5)


@pytest.fixture
def world() -> World:
    return World(THURSDAY)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("tomorrow", date(2026, 11, 6)),
        ("this weekend", date(2026, 11, 7)),
        ("on friday", date(2026, 11, 6)),
        ("next week", date(2026, 11, 9)),
        ("due 2026-12-01", date(2026, 12, 1)),
        ("Nov 20", date(2026, 11, 20)),
        ("no date here", None),
    ],
)
def test_parse_date(phrase: str, expected: date | None) -> None:
    assert parse_date(phrase, THURSDAY) == expected


def test_date_targets() -> None:
    assert weekend_targets(THURSDAY) == {date(2026, 11, 7), date(2026, 11, 8)}
    assert date(2026, 11, 6) in friday_targets(THURSDAY)


@pytest.mark.parametrize(
    ("message", "title"),
    [
        ("add a task to water the plants tomorrow", "water the plants"),
        ("Add a task called “sort Honey by Caribou” due this weekend", "sort Honey by Caribou"),
        (
            'add a task to sort the album "Honey" by Caribou this weekend',
            "sort the album Honey by Caribou",
        ),
        ("Please create a task: call the plumber on friday", "call the plumber"),
    ],
)
def test_tasks_create_reads_the_title(world: World, message: str, title: str) -> None:
    TasksApp(world).handle(1, message)
    assert [r.title for r in world.receipts] == [title]


def test_tasks_move_survives_an_stt_slip(world: World) -> None:
    reply = TasksApp(world).handle(1, "push the dennis thing to friday")
    assert world.tasks[3].due == date(2026, 11, 6)
    assert "dentist appointment" in reply.text


def test_tasks_reads_write_nothing(world: World) -> None:
    app = TasksApp(world)
    for message in ("what's due today?", "what's overdue?", "garage shelves"):
        app.handle(1, message)
    assert world.receipts == []
    assert "steel rack" in app.handle(1, "find the garage shelves").text


def test_tasks_undo_is_scoped_to_the_thread(world: World) -> None:
    app = TasksApp(world)
    app.handle(1, "add a task to water the plants tomorrow")
    assert "nothing" in app.handle(2, "undo that").text.lower()
    asked = app.handle(1, "undo that")
    assert asked.stop_reason == "confirm_required" and asked.confirm is not None
    app.handle(1, "yes", asked.confirm.token)
    assert [r.kind for r in world.receipts] == ["create", "trash"]


def test_tasks_wipe_project_asks_then_runs_on_the_token(world: World) -> None:
    app = TasksApp(world)
    asked = app.handle(1, "delete the Kitchen project")
    assert asked.confirm is not None and asked.text == asked.confirm.question
    assert world.receipts == [] and world.projects["Kitchen"] is False
    app.handle(1, "yes", asked.confirm.token)
    assert world.projects["Kitchen"] is True
    assert all(t.trashed for t in world.tasks.values() if t.project == "Kitchen")
    assert [(r.kind, r.title) for r in world.receipts] == [("trash", "Kitchen")]


def test_music_resolves_albums(world: World) -> None:
    app = MusicApp(world)
    assert "Honey" in app.handle(1, "download the new Caribou album").text
    assert "Suddenly" in app.handle(1, "download Caribou's other album").text
    assert [r.title for r in world.receipts] == ["Honey", "Suddenly"]


def test_music_asks_when_a_song_is_ambiguous(world: World) -> None:
    app = MusicApp(world)
    app.handle(1, "grab Odessa by Caribou")
    reply = app.handle(1, "put that Caribou song in the house folder")
    assert "?" in reply.text and len(world.receipts) == 1
    app.handle(1, "put Odessa in the house folder")
    assert world.receipts[-1].kind == "file"


def test_music_sort_stops_when_cancelled(world: World) -> None:
    app = MusicApp(world)

    def stop_after_two(_receipt: object) -> None:
        if len(world.writes(app="music")) >= 2:
            world.cancel_requested["music"] = True

    world.on_write = stop_after_two
    reply = app.handle(1, "sort my whole library")
    assert [r.kind for r in world.receipts] == ["file", "file", "cancel"]
    assert "Stopped after filing 2" in reply.text and world.unsorted()


def test_advance_finishes_jobs(world: World) -> None:
    MusicApp(world).handle(1, "grab this whole playlist https://example.com/playlist?list=1")
    world.advance()
    job = world.jobs[1]
    assert (job.state, job.done, job.failed) == ("finished", 38, 2)


def test_seeded_brief_counts(world: World) -> None:
    assert len(world.overdue()) == 3
    assert len(world.unsorted()) == 5
    assert {t.title for t in world.due_today()} == {"groceries", "call mom"}
    assert world.tasks[3].due not in friday_targets(THURSDAY)
    assert world.today - timedelta(days=7) < world.today


def test_tasks_another_message_voids_the_question(world: World) -> None:
    app = TasksApp(world)
    asked = app.handle(1, "delete the Kitchen project")
    assert asked.confirm is not None
    app.handle(1, "what's due today?")
    with pytest.raises(DelegateConfirmRefused):
        app.handle(1, "yes", asked.confirm.token)
    assert world.receipts == [] or all(r.kind != "trash" for r in world.receipts)
