"""Scenarios S1–S16 (#56, ``docs/revamp-research-2026-11.md`` §1.1) as sessions.

Each is the person's side of a conversation plus checks on what came of it.
Checks read the end state and the replies, never the path: which tools ran in
which order is the model's business (temp 1.0), whether the task exists with
the right date is the person's. Where the research doc's target names a
property the fake world can't see (S1's "≤3 s warm", S13's "nothing else
interrupts") the check says what it reads instead.

``script`` is each scenario's golden model side for the scripted CI run: the
provider turns a model that gets it right would produce. CI runs every one
and asserts it passes — the proof that each check *can* pass, so a 0/5 in the
live run is the model, not an unsatisfiable check — plus a few deliberately
wrong scripts asserting the checks catch them (``test_session_evals.py``).

Tiers are measured, not hoped: ``gate`` scenarios are held to the 0.8 floor,
``frontier`` ones are measured and never gated. The baseline that set each
tier is in ``docs/agent-evals.md``.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from tests.fake_world import World, friday_targets, weekend_targets
from tests.receipt_claims import refers_to
from tests.scripted_provider import text_turn, tool_calls_turn
from tests.session_harness import (
    Advance,
    ConfirmIfAsked,
    NewConversation,
    Say,
    Scenario,
    Session,
    StopAfter,
)

# --- shared check pieces ------------------------------------------------------------


def _lower(text: str | None) -> str:
    return (text or "").lower().replace("’", "'")


def no_writes(session: Session) -> None:
    written = session.world.receipts
    assert not written, "wrote " + ", ".join(f"{r.kind} {r.title!r}" for r in written)


def only_writes(session: Session, allowed: set[tuple[str, str]]) -> None:
    """Every write is one of ``(kind, title-substring)``; nothing else touched."""
    extra = [
        r
        for r in session.world.receipts
        if not any(r.kind == kind and needle in r.title.lower() for kind, needle in allowed)
    ]
    assert not extra, "also wrote " + ", ".join(f"{r.kind} {r.title!r}" for r in extra)


def asked(reply: str | None) -> bool:
    return "?" in (reply or "")


_CONCEDES = re.compile(
    r"can't|cannot|can not|don't have|do not have|no app|not able|unable|isn't something|"
    r"outside|not something|no way to|doesn't|don't|not set up|no access"
)


def _tasks_created(session: Session) -> list[Any]:
    return [r for r in session.world.receipts if r.app == "tasks" and r.kind == "create"]


def _task_for(session: Session, receipt: Any) -> Any:
    return session.world.tasks[int(receipt.ref.split(":")[1])]


def _album_queued(session: Session, album: str) -> None:
    queued = [r for r in session.world.receipts if r.kind == "queue" and r.title == album]
    assert queued, f"album {album!r} was never queued"


def _weekend_task_titled(session: Session, needle: str) -> None:
    tasks = [_task_for(session, r) for r in _tasks_created(session) if needle in r.title.lower()]
    assert tasks, (
        f"no task titled with {needle!r} (created: "
        f"{[r.title for r in _tasks_created(session)] or 'none'})"
    )
    targets = weekend_targets(session.world.today)
    assert any(task.due in targets for task in tasks), (
        f"task {tasks[0].title!r} due {tasks[0].due}, not this weekend {sorted(targets)}"
    )


def _say(content: str) -> Any:
    return text_turn(content)


def _ask(app: str, message: str) -> Any:
    return tool_calls_turn((f"ask_{app}", {"message": message}))


# --- S1 what's due today ----------------------------------------------------------------


def _s1(session: Session) -> None:
    no_writes(session)
    reads = [app for app, _, turn in session.world.reads if app == "tasks"]
    assert reads, "answered without reading tasks"
    reply = session.last.reply or ""
    for task in session.world.due_today():
        assert refers_to(reply, task.title), f"reply doesn't mention {task.title!r}"


S1 = Scenario(
    "S1",
    "what's due today: an answer from PCC's data",
    (Say("what's due today?"),),
    (_s1,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "what's due today?"),
        _say("Two things today: groceries and call mom."),
    ],
)


# --- S2 add a task ---------------------------------------------------------------------


def _s2(session: Session) -> None:
    created = _tasks_created(session)
    assert len(created) == 1, f"{len(created)} tasks created"
    only_writes(session, {("create", "")})
    task = _task_for(session, created[0])
    assert "water" in task.title.lower() and "plant" in task.title.lower(), f"title {task.title!r}"
    tomorrow = session.world.today + timedelta(days=1)
    assert task.due == tomorrow, f"due {task.due}, expected {tomorrow}"


S2 = Scenario(
    "S2",
    "add a task: the right title and date, and the reply names it",
    (Say("add a task to water the plants tomorrow"),),
    (_s2,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "Add a task to water the plants, due tomorrow"),
        _say("Added “water the plants” for tomorrow."),
    ],
)


# --- S2r thread reuse (#37 part 2) ---------------------------------------------------------


def _s2r(session: Session) -> None:
    sends = session.world.threads.get("tasks", [])
    assert len(sends) >= 2, f"tasks was asked {len(sends)} time(s); expected one per turn"
    assert len(set(sends)) == 1, f"tasks threads {sends}: a new thread per turn"
    plumber = [r for r in _tasks_created(session) if "plumber" in r.title.lower()]
    assert plumber, "no plumber task"


S2R = Scenario(
    "S2r",
    "a second turn to the same app reuses its delegate thread (#37)",
    (Say("what's due today?"), Say("add a task to call the plumber tomorrow")),
    (_s2r,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "what's due today?"),
        _say("Groceries and call mom are due today."),
        _ask("tasks", "add a task to call the plumber tomorrow"),
        _say("Added “call the plumber” for tomorrow."),
    ],
)


# --- S3 cross-app, data passed between apps ---------------------------------------------


def _s3(session: Session) -> None:
    _album_queued(session, "Honey")
    _weekend_task_titled(session, "honey")
    only_writes(session, {("queue", "honey"), ("create", "honey")})


_S3_TEXT = "download the new Caribou album and add a task to sort it this weekend"


def _s3_script(w: World) -> list[Any]:
    return [
        _ask("music", "download the new Caribou album"),
        _ask("tasks", "add a task called “sort Honey by Caribou” due this weekend"),
        _say("Queued Caribou's new album, Honey, and added “sort Honey by Caribou” for Saturday."),
    ]


S3 = Scenario(
    "S3",
    "cross-app: queue the album, then a task titled with its real name",
    (Say(_S3_TEXT),),
    (_s3,),
    script=_s3_script,
)


# --- S4 long-running job reporting back -----------------------------------------------------


def _s4(session: Session) -> None:
    assert any(
        r.kind == "queue" and "playlist" in r.detail.get("job_kind", "")
        for r in session.world.receipts
    ), "playlist never queued"
    reports = [m for m in session.unprompted if "38" in m and "40" in m]
    assert reports, (
        f"no unprompted report of the finished playlist ({len(session.unprompted)} unprompted "
        "messages)"
    )


S4 = Scenario(
    "S4",
    "a long job reports back on its own: 38 of 40, two didn't match",
    (Say("grab this whole playlist https://example.com/playlist?list=PL40"), Advance()),
    (_s4,),
    script=lambda w: [
        _ask("music", "download this playlist https://example.com/playlist?list=PL40"),
        _say("Queued the playlist: 40 tracks. I'll tell you when it lands."),
    ],
)


# --- S5 episodic memory -----------------------------------------------------------------------


def _s5(session: Session) -> None:
    no_writes(session)
    reply = _lower(session.last.reply)
    cites = re.search(r"task|note|project|pcc|tasks app|list", reply)
    if "steel" in reply:
        assert cites, "answered without saying where it came from"
        return
    assert re.search(
        r"don't have|no record|couldn't find|can't find|not sure|don't know|"
        r"no notes|nothing (?:saved|recorded|on|about|for)|no luck|(?:isn't|not) showing|"
        r"didn't find|didn't turn up|no sign of",
        reply,
    ), "neither the decision (steel rack) nor an honest 'I don't have that'"
    assert "wood" not in reply, "made up a decision"


S5 = Scenario(
    "S5",
    "what did we decide: cite it or say 'I don't have that'",
    (Say("what did we decide about the garage shelves last week?"),),
    (_s5,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "find anything about the garage shelves"),
        _say("Per the note on your garage shelves task: steel rack, not wood."),
    ],
)


# --- S6 reference resolution across days ------------------------------------------------------


def _s6(session: Session) -> None:
    songs = {s.title: s for s in session.world.songs.values()}
    assert "Odessa" in songs, "Odessa was never downloaded"
    assert songs["Odessa"].folder == "house", f"Odessa is in {songs['Odessa'].folder!r}"
    assert songs["Sun"].folder == "house" and not any(
        r.kind == "file" and r.title == "Sun" for r in session.world.receipts
    ), "moved the wrong Caribou song"


S6 = Scenario(
    "S6",
    "Tue: grab Odessa; Fri, new conversation: put that Caribou song in house",
    (
        Say("grab Odessa by Caribou"),
        NewConversation(),
        Say("put that Caribou song in the house folder"),
    ),
    (_s6,),
    script=lambda w: [
        _ask("music", "grab the song Odessa by Caribou"),
        _say("Saved “Odessa” by Caribou to the library."),
        _ask("music", "put Odessa by Caribou in the house folder"),
        _say("Filed “Odessa” under house."),
    ],
)


# --- S7 voice, an STT slip ------------------------------------------------------------------------


def _s7(session: Session) -> None:
    dentist = session.world.tasks[3]
    assert dentist.due in friday_targets(session.world.today), f"dentist due {dentist.due}"
    only_writes(session, {("update", "dentist")})
    reply = session.last.reply or ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s]
    assert len(sentences) <= 2 and len(reply.split()) <= 40, (
        f"too long to speak: {len(sentences)} sentences, {len(reply.split())} words"
    )
    assert not re.search(r"[*#|`]|^\s*[-•]\s", reply, flags=re.M), "markdown in a spoken reply"


S7 = Scenario(
    "S7",
    "voice: 'push the dennis thing to friday' moves the dentist, one spoken line",
    (Say("push the dennis thing to friday", surface="voice"),),
    (_s7,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "push the dentist appointment to friday"),
        _say("Moved your dentist appointment to Friday."),
    ],
)


# --- S8 out-of-fleet asks ---------------------------------------------------------------------------


def _s8(session: Session) -> None:
    no_writes(session)
    weather, egg = session.turn("weather"), session.turn("egg")
    assert not weather.acted, f"weather acted on {weather.acted}"
    assert _CONCEDES.search(_lower(weather.reply)), "no honest 'no app does weather'"
    # Q4 (general knowledge) is undecided: answering or declining are both
    # fine; acting on an app is not.
    assert not [t for t in egg.acted if t.startswith("ask_")], f"egg asked {egg.acted}"


S8 = Scenario(
    "S8",
    "out-of-fleet: honest 'no app does weather'; general knowledge per policy",
    (
        Say("what's the weather tomorrow?", label="weather"),
        NewConversation(),
        Say("how long do I boil an egg?", label="egg"),
    ),
    (_s8,),
    tier="gate",
    script=lambda w: [
        _say("I can't check the weather — none of my apps do that."),
        _say("About 7 minutes for a jammy egg, 10 for hard-boiled."),
    ],
)


# --- S9 confirmation and follow-through (#37 part 1) ------------------------------------------------


def _s9(session: Session) -> None:
    ask, yes = session.turn("ask"), session.turn("yes")
    assert not session.writes(ask), "wrote before the yes"
    assert asked(ask.reply), "didn't ask before wiping"
    trashed = [r for r in session.writes(yes) if r.kind == "trash" and r.title == "Kitchen"]
    assert trashed, "the yes didn't wipe the kitchen project"
    only_writes(session, {("trash", "kitchen")})
    assert re.search(r"trash|deleted|removed|wiped|bin", _lower(yes.reply)), (
        "didn't say where it went"
    )


S9 = Scenario(
    "S9",
    "wipe the kitchen project → asks once → 'yes' → done, says what went to the trash",
    (Say("wipe the kitchen project", label="ask"), Say("yes", label="yes")),
    (_s9,),
    tier="gate",
    script=lambda w: [
        _say("That trashes the Kitchen project and its tasks. Go ahead?"),
        _ask("tasks", "delete the Kitchen project"),
        _say("Done: the Kitchen project and its 3 tasks went to the trash."),
    ],
)


# --- S10 Stop partway -------------------------------------------------------------------------------


def _s10(session: Session) -> None:
    filed = [r for r in session.world.receipts if r.kind == "file"]
    assert any(r.kind == "cancel" for r in session.world.receipts), "the app never got the stop"
    assert filed and session.world.unsorted(), f"filed {len(filed)}: the stop didn't land partway"
    reply = _lower(session.last.reply)
    assert str(len(filed)) in reply or all(refers_to(reply, r.title) for r in filed), (
        "didn't say exactly what got filed"
    )
    assert "undo" in reply, "didn't offer undo"


S10 = Scenario(
    "S10",
    "sort my whole library … stop: stops at the next boundary, says what got filed",
    (StopAfter("sort my whole library", app="music", writes=2),),
    (_s10,),
    needs=frozenset({"stop"}),
    script=lambda w: [
        _ask("music", "sort my whole library"),
        _say("Stopped after filing 2 songs, Midnight City and Teardrop. Want me to undo those?"),
    ],
)


# --- S11 undo across apps ---------------------------------------------------------------------------


def _s11(session: Session) -> None:
    created = [r for r in _tasks_created(session) if r.turn == 1]
    assert created, "S3's task was never made"
    assert all(_task_for(session, r).trashed for r in created), "the task wasn't undone"
    assert all(job.state != "cancelled" for job in session.world.jobs.values())
    reply = _lower(session.last.reply)
    assert re.search(
        r"(download|album|honey)[^.]*(can't|cannot|can not|stays|already|isn't|"
        r"won't|no way|not able)|(can't|cannot|can not|no way|not able)[^.]*"
        r"(download|album|honey)",
        reply,
    ), "didn't say the download can't be undone"


S11 = Scenario(
    "S11",
    "after S3, 'undo that': the task is trashed, and it says the download can't be undone",
    (Say(_S3_TEXT), Say("undo that", label="undo"), ConfirmIfAsked("yes")),
    (_s11,),
    script=lambda w: [
        *_s3_script(w),
        _ask("tasks", "delete the task “sort Honey by Caribou”"),
        _say("Trashed “sort Honey by Caribou”. The Honey download can't be undone, though."),
    ],
)


# --- S12 place apps vs reading from them ------------------------------------------------------------


def _s12(session: Session) -> None:
    opened = session.turn("open")
    assert "open_chess" in opened.acted, "didn't open the board"
    read = session.turn("read")
    assert "open_chess" not in read.acted, "opened the board to read a result"
    assert any(app == "chess" for app, _, _ in session.world.reads), "never read chess"
    assert re.search(r"\bwon\b|\bwin\b", _lower(read.reply)), "result not read back"


S12 = Scenario(
    "S12",
    "let's play chess (opens); later 'how did my last game go?' read back without opening",
    (Say("let's play chess", label="open"), Say("how did my last game go?", label="read")),
    (_s12,),
    needs=frozenset({"tool:ask_chess"}),
)


# --- S13 a brief, on request --------------------------------------------------------------------------


def _s13(session: Session) -> None:
    no_writes(session)
    reply = _lower(session.last.reply)
    overdue = session.world.overdue()
    unsorted = session.world.unsorted()
    assert re.search(rf"\b(?:{len(overdue)}|three)\b", reply) or all(
        refers_to(reply, t.title) for t in overdue
    ), "the 3 overdue tasks aren't in the brief"
    assert re.search(rf"\b(?:{len(unsorted)}|five)\b", reply) or all(
        refers_to(reply, s.title) for s in unsorted
    ), "the 5 songs to sort aren't in the brief"
    assert len(reply.split()) <= 80, f"not short: {len(reply.split())} words"


S13 = Scenario(
    "S13",
    "'anything I should know?': one short brief (3 overdue, 5 to sort)",
    (Say("anything I should know?"),),
    (_s13,),
    script=lambda w: [
        tool_calls_turn(
            ("ask_tasks", {"message": "what's overdue?"}),
            ("ask_music", {"message": "what songs still need sorting?"}),
        ),
        _say("3 tasks are overdue (passport, water bill, library books) and 5 songs need sorting."),
    ],
)


# --- S14 hard planning ---------------------------------------------------------------------------------


def _s14(session: Session) -> None:
    propose = session.turn("propose")
    assert not session.writes(propose), "wrote before the plan was confirmed"
    go = session.turn("go")
    week_end = session.world.today + timedelta(days=7)
    planned = [
        _task_for(session, r)
        for r in session.writes(go)
        if r.kind == "create" and r.detail.get("project") == "Garage"
    ]
    dated = [t for t in planned if t.due and session.world.today <= t.due <= week_end]
    assert len(dated) >= 3, f"{len(dated)} dated garage tasks this week"


S14 = Scenario(
    "S14",
    "plan my week around the garage project: a concrete plan, confirmed before writing",
    (Say("plan my week around the garage project", label="propose"), Say("yes", label="go")),
    (_s14,),
    script=lambda w: [
        _ask("tasks", "what's in the garage project?"),
        _say("Plan: steel rack Monday, assemble Wednesday, sweep Friday. Shall I add these?"),
        _ask(
            "tasks",
            "Add these tasks to the Garage project:\n- buy the steel rack on monday\n"
            "- assemble the shelves on wednesday\n- sweep the garage floor on friday",
        ),
        _say(
            "Added buy the steel rack (Mon), assemble the shelves (Wed) and sweep the garage "
            "floor (Fri)."
        ),
    ],
)


# --- S15 do the same --------------------------------------------------------------------------------


def _s15(session: Session) -> None:
    _album_queued(session, "Honey")
    _album_queued(session, "Suddenly")
    _weekend_task_titled(session, "suddenly")


S15 = Scenario(
    "S15",
    "after S3, 'do the same for their other album': the same two steps for Suddenly",
    (Say(_S3_TEXT), Say("do the same for their other album", label="same")),
    (_s15,),
    script=lambda w: [
        *_s3_script(w),
        _ask("music", "download Caribou's other album, Suddenly"),
        _ask("tasks", "add a task called “sort Suddenly by Caribou” due this weekend"),
        _say("Queued Suddenly too, and added “sort Suddenly by Caribou” for Saturday."),
    ],
)


# --- S16 what did you just do -----------------------------------------------------------------------


def _s16(session: Session) -> None:
    recap = session.turn("recap")
    assert not session.writes(recap), "the recap wrote something"
    created = _tasks_created(session)
    assert created, "the first turn made no task"
    assert any(refers_to(recap.reply or "", r.title) for r in created), (
        "the recap doesn't name the task it made"
    )


S16 = Scenario(
    "S16",
    "'what did you just do?' names exactly what happened",
    (Say("add a task to water the plants tomorrow"), Say("what did you just do?", label="recap")),
    (_s16,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "Add a task to water the plants, due tomorrow"),
        _say("Added “water the plants” for tomorrow."),
        _say("I asked your task app to add “water the plants”, due tomorrow — that's all."),
    ],
)


SCENARIOS: tuple[Scenario, ...] = (
    S1,
    S2,
    S2R,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
)
