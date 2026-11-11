"""The fake fleet reads requests the way a cooperative app would, and every
write leaves a receipt. CI, no GPU."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from tests.fake_world import (
    MusicApp,
    TasksApp,
    World,
    client_factory,
    friday_targets,
    parse_date,
    weekend_targets,
)
from app.fleet.delegate import DelegateConfirmRefused, RecordEntry

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


def test_tasks_takes_a_list_of_moves_one_per_line(world: World) -> None:
    TasksApp(world).handle(
        1,
        "- move clear out old bikes to monday\n- reschedule sweep the floor to friday",
    )
    assert [(r.kind, r.title) for r in world.receipts] == [
        ("update", "clear out old bikes"),
        ("update", "sweep the floor"),
    ]


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


def _music_record(world: World, message: str) -> list[RecordEntry]:
    music = client_factory(world)(_App("music"))
    record = music.send_message(1, message).assistant_message.record
    assert record is not None
    return list(record)


# The shape real music returns once music#96 lands: an album it resolved is an
# entity ref (the MusicBrainz release-group id, the album's title) beside the
# download job; one it can't resolve writes nothing and touches nothing, and
# its reply asks for a link or names the candidates. Nothing else about the
# artist's discography is knowledge a real music agent has.


def test_music_resolves_a_named_album_to_an_album_ref_beside_the_job(world: World) -> None:
    [entry] = _music_record(world, "download the new Caribou album")
    assert (entry.tool, entry.outcome) == ("queue_download", "done")
    album, job = entry.touched
    assert (album.app, album.kind, album.title) == ("music", "album", "Honey")
    assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", album.id), album.id
    assert album.id != job.id  # a release-group id, not the job's
    assert (job.kind, job.title) == ("download_job", "album download")
    assert entry.job is not None and entry.job.ref == job.id
    assert [r.title for r in world.receipts] == ["Honey"]


def test_the_album_id_is_stable_and_the_job_record_names_the_same_album(world: World) -> None:
    [entry] = _music_record(world, "download Suddenly by Caribou")
    album = entry.touched[0]
    assert album.title == "Suddenly"
    world.advance()
    ended = world.job_read("1")
    assert ended is not None
    assert [ref["id"] for ref in ended["record"][0]["touched"]] == [album.id]
    assert [ref["kind"] for ref in ended["record"][0]["touched"]] == ["album"]


def test_one_album_by_an_artist_resolves_without_naming_it(world: World) -> None:
    [entry] = _music_record(world, "download the Miles Davis album")
    assert entry.touched[0].title == "Kind of Blue"


@pytest.mark.parametrize(
    "message",
    [
        "download the Caribou album",
        "download the other album by Caribou",
        "download Caribou's other album",
        "download another Caribou album",
    ],
)
def test_music_cannot_resolve_one_album_so_it_writes_and_touches_nothing(
    world: World, message: str
) -> None:
    app = MusicApp(world)
    reply = app.handle(1, message)
    assert world.receipts == [] and world.jobs == {}
    # It names the candidates (all of them: it doesn't know which is "other").
    assert all(title in reply.text for title in ("Honey", "Suddenly", "Swim")) and "?" in reply.text
    assert _music_record(World(THURSDAY), message) == []


def test_music_does_not_know_what_other_means(world: World) -> None:
    app = MusicApp(world)
    app.handle(1, "download the new Caribou album")
    reply = app.handle(1, "download the other album by Caribou")
    assert [r.title for r in world.receipts] == ["Honey"]  # nothing queued for "other"
    assert "Suddenly" in reply.text and "Swim" in reply.text and "?" in reply.text


def test_music_with_no_artist_asks_for_a_link(world: World) -> None:
    reply = MusicApp(world).handle(1, "download the album with the red cover")
    assert "link" in reply.text and world.receipts == []


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


def test_replies_carry_v1_1_records_like_the_real_apps(world: World) -> None:
    # The narrator (#65) speaks from these, so they must say what was touched.
    tasks = client_factory(world)(_App("tasks"))
    made = tasks.send_message(1, "add a task to water the plants tomorrow").assistant_message
    assert made.record is not None
    [entry] = made.record
    assert (entry.tool, entry.outcome) == ("create_task", "done")
    assert [ref.title for ref in entry.touched] == ["water the plants"]
    assert entry.undo is not None and entry.undo.how == "move it to the trash"

    asked = tasks.send_message(1, "delete the Kitchen project").assistant_message
    assert asked.record is not None and [e.outcome for e in asked.record] == ["not_run"]

    music = client_factory(world)(_App("music"))
    saved = music.send_message(1, "download https://example.com/watch?v=1").assistant_message
    assert saved.record is not None
    [song] = saved.record[0].touched
    assert (song.title, song.untrusted) == ("Real Title", True)


class _App:
    def __init__(self, name: str) -> None:
        self.name = name
