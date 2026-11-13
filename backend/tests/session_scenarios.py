"""Session scenarios: S1–S16 (#56, ``docs/revamp-research-2026-11.md`` §1.1), M1–M3 (#81, #134),
T1–T2 and S1v (#82: one timeline, incognito, a voice-tagged S1), TH1–TH2 (#139: side threads),
BI1–BI2 (#77: a reply spoken over, a spoken correction).

Each is the person's side of a conversation plus checks on what came of it.
Checks read the end state and the replies, never the path: which tools ran in
which order is the model's business (temp 1.0), whether the task exists with
the right date is the person's. Where the research doc's target names a
property the fake world can't see (S1's "≤3 s warm", S13's "nothing else
interrupts") the check says what it reads instead.

``script`` is each scenario's golden model side for the scripted CI run: the
provider turns a model that gets it right would produce, in the order the
compiled runtime calls it (#75): the planner's plan JSON, then the narrator's
or the speaker's words (a handoff, an app's question, a decline, a think's
reply and a Stop are code's and call no model). CI runs every one
and asserts it passes — the proof that each check *can* pass, so a 0/5 in the
live run is the model, not an unsatisfiable check — plus a few deliberately
wrong scripts asserting the checks catch them (``test_session_evals.py``).

Tiers are measured, not hoped: ``gate`` scenarios are held to the 0.8 floor,
``frontier`` ones are measured and never gated. The baseline that set each
tier is in ``docs/agent-evals.md``.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from tests.fake_world import PLAYLIST_MISSES, World, friday_targets, weekend_targets
from tests.receipt_claims import refers_to
from tests.scripted_provider import text_turn
from tests.session_harness import (
    Advance,
    CloseThread,
    ConfirmIfAsked,
    NewConversation,
    OpenThread,
    Say,
    Scenario,
    Search,
    Session,
    Settle,
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
    """A narrator's (or the plain speaker's) words."""
    return text_turn(content)


def _spoken(reply: str, *, about_you: bool = False) -> Any:
    """The speaker's structured answer to a ``knowledge`` or ``chat`` plan."""
    return text_turn(json.dumps({"about_you": about_you, "reply": reply}))


def plan(*steps: dict[str, Any]) -> Any:
    """The planner's plan: steps numbered s1, s2, ... in order."""
    body = {"steps": [{"id": f"s{n}", **step} for n, step in enumerate(steps, start=1)]}
    return text_turn(json.dumps(body))


def answer(kind: str) -> Any:
    """The planner's direct answer: knowledge, decline, self or chat."""
    return text_turn(json.dumps({"answer": kind}))


def ask(app: str, request: str, *, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app, "request": request, "destructive": destructive}


def look(app: str) -> dict[str, Any]:
    return {"do": "look", "app": app}


def _ask(app: str, request: str, *, destructive: bool = False) -> Any:
    """A one-step plan asking ``app``."""
    return plan(ask(app, request, destructive=destructive))


YES = plan({"do": "confirm", "answer": "yes"})


# --- S1 what's due today ----------------------------------------------------------------


def _s1(session: Session) -> None:
    no_writes(session)
    # A `look` (the glance, #73) is a read of tasks like an ask's list: the
    # world records both.
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
        plan(look("tasks")),
        _say("Two things due today: “Buy groceries” and “Call mom”."),
    ],
)


# S1 again, said by voice (#82): every timeline entry carries its surface, and
# the voice tag puts the spoken-reply line in State.
S1V = Scenario(
    "S1v",
    "what's due today, by voice: the same answer from PCC's data",
    (Say("what's due today?", surface="voice"),),
    (_s1,),
    # Gate since #82: 10/10 across two live runs (5/5 each).
    tier="gate",
    script=lambda w: [
        plan(look("tasks")),
        _say("Two things due today: “Buy groceries” and “Call mom”."),
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
    for word in ("lightbulbs", "plumber"):
        made = [r for r in _tasks_created(session) if word in r.title.lower()]
        assert made, f"no {word} task"


# Both turns are writes. A read like "what's due today?" is a `look` now (#73:
# the glance, no agent), so it would never open a delegate thread; two asks to
# tasks are what exercises thread reuse.
S2R = Scenario(
    "S2r",
    "a second turn to the same app reuses its delegate thread (#37)",
    (Say("add a task to buy lightbulbs"), Say("add a task to call the plumber tomorrow")),
    (_s2r,),
    tier="gate",
    script=lambda w: [
        _ask("tasks", "Add a task to buy lightbulbs"),
        _say("Added “buy lightbulbs”."),
        _ask("tasks", "Add a task to call the plumber, due tomorrow"),
        _say("Added “call the plumber” for tomorrow."),
    ],
)


# --- S3 cross-app, data passed between apps ---------------------------------------------


def _s3(session: Session) -> None:
    _album_queued(session, "Honey")
    # Music resolved one album, so its record names it by a release-group id
    # (music#96): the title the task carries comes from that ref.
    (queued,) = [r for r in session.world.receipts if r.kind == "queue"]
    assert queued.detail.get("album_id"), "music queued a download that names no album"
    _weekend_task_titled(session, "honey")
    only_writes(session, {("queue", "honey"), ("create", "honey")})


_S3_TEXT = "download the new Caribou album and add a task to sort it this weekend"


S3_PLAN = plan(
    ask("music", "download the new Caribou album"),
    ask("tasks", "add a task to sort {s1} this weekend"),
)


def _s3_script(w: World) -> list[Any]:
    return [
        S3_PLAN,
        _say("Queued “Honey” by Caribou and added “sort Honey” for the weekend."),
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
    assert len(session.unprompted) == 1, f"{len(session.unprompted)} unprompted messages, not one"
    missed = [title for title in PLAYLIST_MISSES if not refers_to(reports[0], title)]
    assert not missed, f"the report doesn't name what didn't match: {missed}"


S4 = Scenario(
    "S4",
    "a long job reports back on its own: one note, 38 of 40, the two misses named",
    (Say("grab this whole playlist https://example.com/playlist?list=PL40"), Advance()),
    (_s4,),
    tier="gate",
    script=lambda w: [
        _ask("music", "download this playlist https://example.com/playlist?list=PL40"),
        _say("Queued your playlist: 40 tracks. I'll post a note when it's done."),
        # The job's note (#76), narrated from the app's final report.
        _say(
            "Your playlist is in: 38 of 40 downloaded; “Lost Signal” and “Paper Moons” "
            "couldn't be found."
        ),
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
        plan({"do": "recall", "query": "garage shelves decision"}),
        _say("I have no record of a decision about the garage shelves."),
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
    # Gate since #80: the entity registry carries "Odessa" across conversations
    # (19/20 live, compiler on).
    tier="gate",
    script=lambda w: [
        _ask("music", "grab the song Odessa by Caribou"),
        _say("Saved “Odessa” by Caribou."),
        # The new conversation's State matches "that Caribou song" to Odessa (#80).
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
        answer("decline"),
        answer("knowledge"),
        _spoken("About 7 minutes for a jammy egg, 10 for hard-boiled."),
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
        _ask("tasks", "delete the Kitchen project", destructive=True),
        YES,
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
    # The Stop lands mid-delegate, so the turn ends there: no narrator after.
    script=lambda w: [_ask("music", "sort my whole library")],
)


# --- S11 undo across apps ---------------------------------------------------------------------------


def _s11(session: Session) -> None:
    created = [r for r in _tasks_created(session) if r.turn == 1]
    assert created, "S3's task was never made"
    assert all(_task_for(session, r).trashed for r in created), "the task wasn't undone"
    assert all(job.state != "cancelled" for job in session.world.jobs.values())
    reply = _lower(session.last.reply)
    # #185: the download is left running on purpose, so saying it still runs
    # says why it wasn't undone.
    assert re.search(
        r"(download|album|honey)[^.]*(can't|cannot|can not|could not|couldn't|stays|already|"
        r"isn't|won't|no way|not able|still (?:running|downloading|going|queued))"
        r"|(can't|cannot|can not|could not|couldn't|no way|not able)[^.]*"
        r"(download|album|honey)",
        reply,
    ), "didn't say the download can't be undone or is still running"


S11 = Scenario(
    "S11",
    "after S3, 'undo that': the task is trashed, and it says the download can't be undone",
    (Say(_S3_TEXT), Say("undo that", label="undo"), ConfirmIfAsked("yes")),
    (_s11,),
    # The undo is tasks' own question, passed through; the yes is narrated (#74).
    script=lambda w: [
        *_s3_script(w),
        plan({"do": "undo", "target": "last"}),
        YES,
        _say("Trashed “sort Honey”. The Honey download can't be undone, and it's still running."),
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
    script=lambda w: [
        plan({"do": "open", "app": "chess", "intent": "let's play chess"}),
        plan(look("chess")),
        _say("You won your last game."),
    ],
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
        plan(look("tasks"), look("music")),
        _say("3 tasks are overdue and 5 songs need sorting."),
    ],
)


# --- S14 hard planning ---------------------------------------------------------------------------------


def _s14(session: Session) -> None:
    propose = session.turn("propose")
    assert not session.writes(propose), "wrote before the plan was confirmed"
    go = session.turn("go")
    week_end = session.world.today + timedelta(days=7)
    # New garage tasks or existing ones put on days this week (#78: the 26B
    # often schedules the garage tasks already there): either is the plan.
    planned = {
        task.id: task
        for task in (
            _task_for(session, r) for r in session.writes(go) if r.kind in ("create", "update")
        )
        if task.project == "Garage"
    }
    dated = [t for t in planned.values() if t.due and session.world.today <= t.due <= week_end]
    assert len(dated) >= 3, f"{len(dated)} dated garage tasks this week"


S14 = Scenario(
    "S14",
    "plan my week around the garage project: a concrete plan, confirmed before writing",
    # A think (#78) answers at once and posts its proposal later: the person
    # waits for it before saying yes.
    (
        Say("plan my week around the garage project", label="propose"),
        Settle(),
        Say("yes", label="go"),
    ),
    (_s14,),
    script=lambda w: [
        plan({"do": "think", "problem": "plan my week around the garage project", "inputs": []}),
        # The 26B's proposal, when the person waits (the watcher's thinker).
        text_turn(
            json.dumps(
                {
                    "summary": "3 garage tasks, Monday to Friday",
                    "steps": [
                        {
                            "id": "s1",
                            **ask(
                                "tasks",
                                "Add these tasks to the Garage project:\n- buy the steel rack "
                                "on monday\n- assemble the shelves on wednesday\n- sweep the "
                                "garage floor on friday",
                            ),
                        }
                    ],
                }
            )
        ),
        YES,
        _say(
            "Added buy the steel rack (Mon), assemble the shelves (Wed) and sweep the garage "
            "floor (Fri)."
        ),
    ],
)


# --- S15 do the same --------------------------------------------------------------------------------


def _s15(session: Session) -> None:
    # Real music can't tell which album is "the other" (music#96): it names the
    # candidates and writes nothing, so the task that would carry the album's
    # name has none to carry. Honey from S3 stays the only album, the only task.
    _album_queued(session, "Honey")
    same = session.turn("same")
    assert not session.writes(same), (
        "wrote "
        + ", ".join(f"{r.kind} {r.title!r}" for r in session.writes(same))
        + " though music couldn't say which album"
    )
    reply = _lower(same.reply)
    assert re.search(r"suddenly|swim|which|link|\?", reply), (
        "didn't name the albums to pick from or ask which one"
    )


S15 = Scenario(
    "S15",
    "after S3, 'do the same for their other album': music can't say which, so it names the "
    "candidates and nothing more is written",
    (Say(_S3_TEXT), Say("do the same for their other album", label="same")),
    (_s15,),
    script=lambda w: [
        *_s3_script(w),
        plan(
            ask("music", "download the other album by Caribou"),
            ask("tasks", "add a task to sort {s1} this weekend"),
        ),
        # Music refuses the repeat and asks which album (music#102): its
        # question passes through in its own words, so no narrator turn (#168).
    ],
)


def _s15_picked(session: Session) -> None:
    # #168: S15, then the person names one of the albums music listed. That
    # one is queued, and the work that waited on its name (the weekend task)
    # goes ahead: nothing else is written, and nothing more is queued.
    _s15(session)
    picked = session.turn("pick")
    queued = [r.title for r in session.writes(picked) if r.kind == "queue"]
    assert queued == ["Suddenly"], f"queued {queued or 'nothing'} for the pick, not Suddenly"
    _weekend_task_titled(session, "suddenly")
    only_writes(
        session,
        {("queue", "honey"), ("create", "honey"), ("queue", "suddenly"), ("create", "suddenly")},
    )


S15_PICKED = Scenario(
    "S15p",
    "S15, then 'Suddenly': the album music listed is queued and its task is made",
    (
        Say(_S3_TEXT),
        Say("do the same for their other album", label="same"),
        Say("Suddenly", label="pick"),
    ),
    (_s15_picked,),
    script=lambda w: [
        *(S15.script(w) if S15.script else []),
        # Music's question paused the plan: "Suddenly" resumes it, the
        # answer goes back to music and the task waits on its name.
        text_turn(json.dumps({"about": "the waiting question", "resume": "Suddenly"})),
        _say("Queued “Suddenly” by Caribou and added “sort Suddenly” for the weekend."),
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
        answer("self"),
        _say("I asked your task app to add “water the plants”, due tomorrow — that's all."),
    ],
)


# --- memory you can see (#81) ------------------------------------------------------------------
#
# Conductor's own memory app: its writes are world receipts too (the harness
# files them, ``memory_writes``), so the generic claim checks hold its replies.


def _memory(session: Session, kind: str, label: str) -> Any:
    turn = session.turn(label)
    found = [r for r in session.writes(turn) if r.app == "memory" and r.kind == kind]
    assert found, f"turn {label!r} {'saved' if kind == 'create' else 'forgot'} nothing"
    return found[0]


def _only_memory_writes(session: Session) -> None:
    others = [r for r in session.world.receipts if r.app != "memory"]
    assert not others, "also wrote " + ", ".join(f"{r.kind} {r.title!r}" for r in others)


def _remember_used(session: Session) -> None:
    _only_memory_writes(session)
    saved = _memory(session, "create", "remember")
    assert "sunday" in saved.title.lower(), f"saved {saved.title!r}, not the person's words"
    assert "sunday" in _lower(session.turn("later").reply), "the later reply doesn't use it"


REMEMBER_USED = Scenario(
    "M1",
    "'remember I sort music on Sundays'; another day, 'what day do I sort music?' uses it",
    (
        Say("remember I sort music on Sundays", label="remember"),
        NewConversation(),
        Say("what day do I sort music?", label="later"),
    ),
    (_remember_used,),
    tier="gate",
    # Memory you can see: the fact is in the new conversation's State.
    script=lambda w: [
        _ask("memory", "remember I sort music on Sundays"),
        _say("Got it: “I sort music on Sundays”."),
        answer("chat"),
        _spoken("You sort music on Sundays.", about_you=True),
    ],
)


def _forget_gone(session: Session) -> None:
    _only_memory_writes(session)
    saved = _memory(session, "create", "remember")
    forgot = _memory(session, "trash", "forget")
    assert forgot.ref == saved.ref, f"forgot {forgot.title!r}, not {saved.title!r}"
    after = _lower(session.turn("after").reply)
    assert "blue pot" not in after and "the pot" not in after, "still knew where the key is"


FORGET_GONE = Scenario(
    "M2",
    "remember where the spare key is, 'forget that'; another day it's gone from recall and State",
    (
        Say("remember the spare key is under the blue pot", label="remember"),
        Say("forget that", label="forget"),
        NewConversation(),
        Say("where is the spare key?", label="after"),
    ),
    (_forget_gone,),
    tier="gate",
    script=lambda w: [
        _ask("memory", "remember the spare key is under the blue pot"),
        _say("Got it: “the spare key is under the blue pot”."),
        _ask("memory", "forget that"),
        _say("Forgot “the spare key is under the blue pot”."),
        plan({"do": "recall", "query": "spare key"}),
        _say("I have no record of where the spare key is."),
    ],
)


# A saved fact in another app's ask (#134): code sends it along, the app reads it.
_SORTING_DAY = "add a task for my music sorting day to clean the record player"


def _fact_reaches_the_app(session: Session) -> None:
    saved = _memory(session, "create", "remember")
    used = session.turn("used")
    sent = [message for app, message in session.world.sent if app == "tasks"]
    assert sent, "tasks was never asked"
    assert any(saved.title in message for message in sent), (
        f"tasks never got the fact {saved.title!r}: {sent[-1]!r}"
    )
    made = [r for r in session.writes(used) if r.app == "tasks" and r.kind == "create"]
    assert any("record player" in r.title.lower() for r in made), (
        f"no record player task (made: {[r.title for r in made] or 'none'})"
    )
    only_writes(session, {("create", "record player"), ("create", "sunday")})


FACT_IN_ASK = Scenario(
    "M3",
    "remember a fact; another day an ask to tasks that needs it carries it to the app",
    (
        Say("remember I sort music on Sundays", label="remember"),
        NewConversation(),
        Say(_SORTING_DAY, label="used"),
    ),
    (_fact_reaches_the_app,),
    tier="gate",
    script=lambda w: [
        _ask("memory", "remember I sort music on Sundays"),
        _say("Got it: “I sort music on Sundays”."),
        _ask("tasks", _SORTING_DAY),
        _say("Added “my music sorting day to clean the record player” for Sunday."),
    ],
)


# --- T1 the timeline across a quiet gap (#82) --------------------------------------------


def _timeline_recall(session: Session) -> None:
    only_writes(session, {("create", "shel"), ("create", "steel")})
    later = _lower(session.turn("later").reply)
    # The decision is a day old, past the quiet gap: it is no longer "this
    # conversation", and said in words only, so recall is the one way back.
    assert "steel" in later, "the earlier day's decision (steel rack) wasn't recalled"
    assert "wood" not in later, "made up a decision"


TIMELINE_RECALL = Scenario(
    "T1",
    "one timeline: a decision said yesterday, past the quiet gap, comes back through recall",
    (
        Say("for the garage shelves, let's go with the steel rack", label="decide"),
        NewConversation(),
        Say("what did we decide about the garage shelves?", label="later"),
    ),
    (_timeline_recall,),
    # Gate since #82: 10/10 across two live runs (5/5 each).
    tier="gate",
    script=lambda w: [
        answer("chat"),
        _spoken("Steel rack it is."),
        plan({"do": "recall", "query": "garage shelves decision"}),
        _say("Yesterday you said to go with the steel rack."),
    ],
)


# --- T2 incognito (#82) ----------------------------------------------------------------------

_GATE_CODE = "4417"


def _incognito_kept_nothing(session: Session) -> None:
    secret, later = session.turn("secret"), session.turn("later")
    assert secret.incognito, "the secret wasn't sent incognito"
    saved = [r for r in session.world.receipts if r.app == "memory"]
    assert not saved, "an incognito turn saved a fact: " + ", ".join(r.title for r in saved)
    # Not the plan: nothing the later turn sent anywhere carries the words.
    sent = json.dumps([call.get("arguments") for call in later.tools])
    assert _GATE_CODE not in sent, f"the later plan carried the incognito words: {sent}"
    # Not recall: a recall the later turn ran found nothing of it.
    recalled = json.dumps([call.get("app_record") for call in later.tools])
    assert _GATE_CODE not in recalled, "recall found the incognito words"
    assert _GATE_CODE not in (later.reply or ""), "the later reply knew the incognito words"
    # Not search.
    assert session.searches["code"] == [], f"search found it: {session.searches['code']}"


INCOGNITO = Scenario(
    "T2",
    "incognito: 'remember the gate code' keeps nothing; later the plan, recall and search "
    "don't know it",
    (
        Say(f"remember the gate code is {_GATE_CODE}", label="secret", incognito=True),
        Say("what's the gate code?", label="later"),
        Search(_GATE_CODE, label="code"),
    ),
    (_incognito_kept_nothing,),
    # Gate since #82: 10/10 across two live runs (5/5 each).
    tier="gate",
    script=lambda w: [
        _ask("memory", f"remember the gate code is {_GATE_CODE}"),
        _say("Nothing was saved: incognito is on, so nothing from this turn is remembered."),
        plan({"do": "recall", "query": "gate code"}),
        _say("I have no record of the gate code."),
    ],
)


# --- TH1–TH2 side threads (#139) -------------------------------------------------------------

_MEASURE = "add a task to measure the garage wall"
_BRACKETS = "add a task to order the shelf brackets"


def _result(session: Session, label: str) -> str:
    """The thread's result note: posted into the timeline, as a note."""
    thread = session.threads.get(label)
    assert thread is not None and thread.note, f"side thread {label!r} never closed"
    note = thread.note
    assert note.get("stop_reason") == "thread_note", f"the result is a {note.get('stop_reason')}"
    assert any(m["id"] == note["id"] for m in thread.timeline), "the result isn't in the timeline"
    return str(note.get("content") or "")


def _thread_flowed_back(session: Session) -> None:
    only_writes(session, {("create", "measure"), ("create", "bracket")})
    thread = session.threads["garage"]
    carried = [str(c.get("title") or c.get("text") or "") for c in thread.carries]
    assert any("measure" in c.lower() for c in carried), (
        f"the thread didn't carry the timeline's task in: {carried}"
    )
    inside = session.writes(session.turn("inside"))
    brackets = [r for r in inside if r.kind == "create" and "bracket" in r.title.lower()]
    assert brackets, "nothing was added inside the thread"
    text = _result(session, "garage")
    assert "garage shelves" in _lower(text), f"the result doesn't name its thread: {text!r}"
    assert refers_to(text, brackets[0].title), f"the result doesn't name {brackets[0].title!r}"


THREAD_FLOWS_BACK = Scenario(
    "TH1",
    "side thread: start one carrying the timeline's task, add a task in it, close it, and its "
    "result shows in the timeline",
    (
        Say(_MEASURE, label="before"),
        OpenThread("Garage shelves", label="garage"),
        Say(_BRACKETS, label="inside"),
        CloseThread("garage"),
    ),
    (_thread_flowed_back,),
    needs=frozenset({"threads"}),
    # Unmeasured: added with #139, not yet run live (the batch's eval gate runs it).
    tier="frontier",
    script=lambda w: [
        _ask("tasks", _MEASURE),
        _say("Added “measure the garage wall”."),
        _ask("tasks", _BRACKETS),
        _say("Added “order the shelf brackets”."),
        # The result's narrator, from the thread's receipts.
        _say("Side thread “Garage shelves”: added “order the shelf brackets”."),
    ],
)

_GIFT = "add a task to wrap the surprise gift"
_CANDLES = "add a task to buy candles"


def _thread_kept_incognito(session: Session) -> None:
    secret = session.turn("secret")
    assert secret.incognito, "the secret wasn't sent incognito"
    # The app still saw the incognito request: the write happened.
    assert any("gift" in r.title.lower() for r in session.writes(secret)), "the gift wasn't added"
    text = _lower(_result(session, "party"))
    assert "candles" in text, f"the result doesn't name the kept task: {text!r}"
    for word in ("gift", "surprise", "wrap"):
        assert word not in text, f"the result carried the incognito turn's {word!r}: {text!r}"


THREAD_KEEPS_INCOGNITO = Scenario(
    "TH2",
    "side thread: an incognito turn inside it never rides back in its result",
    (
        OpenThread("Party", label="party"),
        Say(_GIFT, label="secret", incognito=True),
        Say(_CANDLES, label="kept"),
        CloseThread("party"),
    ),
    (_thread_kept_incognito,),
    needs=frozenset({"threads"}),
    # Unmeasured: added with #139, not yet run live (the batch's eval gate runs it).
    tier="frontier",
    script=lambda w: [
        _ask("tasks", _GIFT),
        _say("Added “wrap the surprise gift”."),
        _ask("tasks", _CANDLES),
        _say("Added “buy candles”."),
        _say("Side thread “Party”: added “buy candles”."),
    ],
)


# --- barge-in (#77) ------------------------------------------------------------------------------
#
# Said over the reply: the turn carries the voice client's heard report for it
# (``Say.cut``, the share that played). Real audio (does the speaker trip the
# mic?) is the person's to judge; these read what conductor made of it.


def _repeated_what_was_missed(session: Session) -> None:
    no_writes(session)
    again = session.turn("again")
    due = [task.title for task in session.world.due_today()]
    assert due, "the world has nothing due today"
    assert any(refers_to(again.reply or "", title) for title in due), (
        f"didn't say again what was cut off (due today: {due})"
    )


INTERRUPTED = Scenario(
    "BI1",
    "voice: 'what's due today?', cut a quarter in, 'sorry, what was that?' says it again",
    (
        Say("what's due today?", surface="voice", label="due"),
        Say("sorry, what was that?", surface="voice", cut=0.25, label="again"),
    ),
    (_repeated_what_was_missed,),
    # Unmeasured: added with #77, not yet run live (the batch's eval gate runs it).
    tier="frontier",
    script=lambda w: [
        plan(look("tasks")),
        _say("Groceries and call mom are due today, and three things are overdue."),
        answer("self"),
        _say("I said groceries and call mom are due today, and three things are overdue."),
    ],
)


def _corrected_not_dropped(session: Session) -> None:
    assert not session.world.projects.get("Kitchen"), "the Kitchen project went to the trash"
    cabinets = [t for t in session.world.tasks.values() if t.title == "paint cabinets"]
    assert cabinets and cabinets[0].trashed, "the correction was dropped: paint cabinets stayed"
    only_writes(session, {("trash", "paint cabinets")})


CORRECTED = Scenario(
    "BI2",
    "voice: 'wipe the kitchen project', then over the question 'no, just the paint cabinets "
    "task': only that task goes",
    (
        Say("wipe the kitchen project", surface="voice", label="ask"),
        Say("no, just the paint cabinets task", surface="voice", cut=0.3, label="fix"),
        ConfirmIfAsked("yes"),
    ),
    (_corrected_not_dropped,),
    # Unmeasured: added with #77, not yet run live (the batch's eval gate runs it).
    tier="frontier",
    # Tasks asks each time (its own question, passed through); the amend
    # resends the request with the change, and the yes is narrated.
    script=lambda w: [
        _ask("tasks", "delete the Kitchen project", destructive=True),
        plan({"do": "amend", "target": "pending", "change": "just the paint cabinets task"}),
        YES,
        _say("Trashed “paint cabinets”."),
    ],
)


SCENARIOS: tuple[Scenario, ...] = (
    S1,
    S1V,
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
    S15_PICKED,
    S16,
    REMEMBER_USED,
    FORGET_GONE,
    FACT_IN_ASK,
    TIMELINE_RECALL,
    INCOGNITO,
    THREAD_FLOWS_BACK,
    THREAD_KEEPS_INCOGNITO,
    INTERRUPTED,
    CORRECTED,
)
