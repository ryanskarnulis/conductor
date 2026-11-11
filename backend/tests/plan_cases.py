"""The plan-level eval cases (#70): spike D1's bench, promoted.

Every routing golden re-labelled as a plan, the S1–S16 scenarios, D1's old
held-out set (the ``mixed`` group since #142: it was tuned against), and the
long-plan cases (6–10 steps) added when the 5-step limit went (decided
2026-11-06). Each case lists the plan shapes that count as correct; a sample
passes when the compiled plan is valid and matches one of them. The held-out
set lives apart, in ``held_out_cases.py``.

Run by ``test_plan_evals.py`` (opt-in, the real model). Pure data and
matchers: no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from itertools import permutations
from typing import Any

from pydantic import BaseModel

from app.ai import facts
from app.ai import plan as plans
from app.ai import recall
from app.ai.runtime import ToolCallRecord
from app.ai.state_block import calendar_lines


@dataclass(frozen=True)
class M:
    """One expected step: its kind(s), app, words, and flags."""

    do: str | tuple[str, ...]
    app: str | None = None
    words: tuple[str, ...] = ()
    ref: bool | None = None
    ref_or_words: tuple[str, ...] = ()
    destructive: bool | None = None
    answer: str | None = None
    target_has: str | None = None
    # Saved facts code must send with this ask (#134): the request keeps the
    # words that point at them, and the interpreter attaches them.
    attaches: tuple[str, ...] = ()

    def matches(self, step: BaseModel) -> bool:
        kinds = (self.do,) if isinstance(self.do, str) else self.do
        if step.do not in kinds:  # type: ignore[attr-defined]
            return False
        if self.app is not None and getattr(step, "app", None) != self.app:
            return False
        text = plans.step_text(step).lower()
        if any(word not in text for word in self.words):
            return False
        has_ref = "{s" in text
        if self.ref is not None and has_ref != self.ref:
            return False
        if self.ref_or_words and not (has_ref or any(w in text for w in self.ref_or_words)):
            return False
        if self.destructive is not None and getattr(step, "destructive", None) != self.destructive:
            return False
        if self.answer is not None and getattr(step, "answer", None) != self.answer:
            return False
        if self.target_has is not None and self.target_has not in getattr(step, "target", ""):
            return False
        if self.attaches:
            request = plans.REF.sub(" ", str(getattr(step, "request", "")))
            if facts.related(request, self.attaches) != list(self.attaches):
                return False
        return True


# v1's "say" values for a direct answer, as the shipped schema names them.
_ANSWERS = {"capabilities": "self", "status": "self"}


@dataclass(frozen=True)
class Shape:
    """One correct plan: exact steps (in order unless ``ordered`` is off), or an answer."""

    steps: tuple[M, ...] = ()
    say: str = "result"
    ordered: bool = True

    def matches(self, plan: plans.Plan) -> bool:
        if isinstance(plan, plans.Direct):
            return not self.steps and plan.answer == _ANSWERS.get(self.say, self.say)
        if not isinstance(plan, plans.StepPlan) or self.say != "result":
            return False
        if len(plan.steps) != len(self.steps):
            return False
        if self.ordered:
            return all(m.matches(s) for m, s in zip(self.steps, plan.steps, strict=True))
        return any(
            all(m.matches(s) for m, s in zip(order, plan.steps, strict=True))
            for order in permutations(self.steps)
        )


@dataclass(frozen=True)
class Cover:
    """A long plan's requirement: the steps for one app, together, cover every word.

    Long requests can compile one request per thing or several things per
    request; both are right, so long cases check coverage, not exact shape.
    """

    app: str | None
    kinds: tuple[str, ...]
    words: tuple[str, ...] = ()
    destructive: bool | None = None

    def matches(self, plan: plans.Plan) -> bool:
        if not isinstance(plan, plans.StepPlan):
            return False
        mine = [
            step
            for step in plan.steps
            if step.do in self.kinds
            and (self.app is None or getattr(step, "app", None) == self.app)
        ]
        if not mine:
            return False
        text = " ".join(plans.step_text(step).lower() for step in mine)
        if any(word not in text for word in self.words):
            return False
        if self.destructive is not None and not any(
            getattr(step, "destructive", None) == self.destructive for step in mine
        ):
            return False
        return True


@dataclass(frozen=True)
class Long:
    """A long-plan expectation: every cover met, the last step's kind, a step floor."""

    covers: tuple[Cover, ...]
    last: tuple[str, ...] | None = None
    min_steps: int = 1

    def matches(self, plan: plans.Plan) -> bool:
        if not isinstance(plan, plans.StepPlan) or len(plan.steps) < self.min_steps:
            return False
        if self.last is not None and plan.steps[-1].do not in self.last:
            return False
        return all(cover.matches(plan) for cover in self.covers)


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    expect: tuple[Shape | Long, ...]
    group: str
    state: str = ""
    refs: frozenset[str] = field(default_factory=frozenset)

    @property
    def confirmable(self) -> bool:
        return "pending" in self.refs


_TODAY = date(2026, 11, 5)
_HEADER = "[State at the start of this turn — written by conductor, not the person]"


def state(**parts: str) -> str:
    """A state block in the shipped block's shape, with the case's lines."""
    lines = [_HEADER, *calendar_lines(_TODAY)]
    now = parts.get("now", "")
    if "voice" in now:
        lines.append(
            "Surface: voice (your reply is spoken aloud: one or two short sentences, "
            "no markdown, no lists)."
        )
    if parts.get("jobs"):
        lines += ["Running jobs (as of each one's last report):", f"- {parts['jobs']}"]
    if parts.get("waiting"):
        lines.append(f"Waiting on the person: {parts['waiting']}")
    if parts.get("last"):
        lines.append(f"Last plan: {parts['last']}")
    if parts.get("recent"):
        lines += ["Recent entities (newest first, as of then):", f"- {parts['recent']}"]
    if parts.get("matches"):
        lines.append(f"Matches for words in the message: {parts['matches']}")
    if parts.get("facts"):
        lines.append("Facts the person asked you to remember (their words, as of when saved):")
        lines += [f'- "{fact}" (saved Tuesday 2026-11-03)' for fact in parts["facts"].split("|")]
    lines.append("[End of state]")
    return "\n".join(lines)


def empty_state() -> str:
    return state()


def _call(tool: str, arguments: dict[str, Any], result: str) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, arguments=arguments, result=result)


def before(
    words: str,
    *,
    earlier: Sequence[tuple[str, ToolCallRecord]] = (),
    history: Sequence[dict[str, Any]] = (),
) -> str:
    """An empty state block, then "Before this turn:" as the shipped recall
    renders it: ``earlier`` other conversations' calls (asked, record) from
    Tuesday, and this conversation's ``history`` (#127's leak)."""
    tuesday = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
    hits = recall.hits(
        words, [recall.EarlierCall(tuesday, asked, record) for asked, record in earlier]
    )
    section = recall.section(recall.Recall.of(list(history)), hits)
    return state() + (f"\n\n{section}" if section else "")


def one(*steps: M, say: str = "result", ordered: bool = True) -> Shape:
    return Shape(steps=steps, say=say, ordered=ordered)


def none(say: str) -> Shape:
    return Shape(steps=(), say=say)


READ_TASKS = M(("look", "ask"), "tasks")
READ_MUSIC = M(("look", "ask"), "music")

_WEEK = "what am I working on this week?"
_DUE = "what's due today?"
_PLAYLIST = "download this playlist https://www.youtube.com/playlist?list=PL9"
_PLAYLIST_NOW = "is that playlist done downloading yet?"
_CHESS_NOW = "how is that chess game going?"

_LAST_S3 = (
    's1 ask music "download the new Caribou album" -> queued album "Honey" '
    "(music:album:55, job music:job:20; a download has no undo); "
    's2 ask tasks "add a task to sort {s1} this weekend" -> created task "Sort Honey" '
    "(pcc:task:301, due Sat; undo: trash)"
)
_PENDING = (
    'tasks asked "Trash the Kitchen project and its 12 tasks?" '
    "(pending plan p7, step s1, ref pending:p7)"
)

CASES: tuple[Case, ...] = (
    # --- the routing goldens, as plans -------------------------------------------
    Case("tasks-due", "what's due today?", (one(READ_TASKS),), "golden"),
    Case(
        "tasks-create",
        "add a task to water the plants tomorrow",
        (one(M("ask", "tasks", ("plants",), destructive=False)),),
        "golden",
    ),
    Case("tasks-week", "what am I working on this week?", (one(READ_TASKS),), "golden"),
    Case("chess-play", "let's play chess", (one(M("open", "chess")),), "golden"),
    Case("chess-move", "play e4", (one(M("open", "chess")),), "golden"),
    Case(
        "chess-status",
        "how is our chess game going?",
        (one(M("open", "chess")), one(M("look", "chess"))),
        "golden",
    ),
    Case(
        "chess-analysis",
        "analyze the current position on the board",
        (one(M("open", "chess")),),
        "golden",
    ),
    Case("chess-reset", "reset the chess game", (one(M("open", "chess")),), "golden"),
    Case("chess-resign", "resign the game", (one(M("open", "chess")),), "golden"),
    Case(
        "music-download-link",
        "download this song https://example.com/watch?v=abc",
        (one(M("ask", "music", ("example.com",))),),
        "golden",
    ),
    Case(
        "music-download-named",
        "grab the song Odessa by Caribou",
        (one(M("ask", "music", ("odessa",))),),
        "golden",
    ),
    Case("music-save", "save this track to the music folder", (one(M("ask", "music")),), "golden"),
    Case("music-sort", "sort my music", (one(M("ask", "music")),), "golden"),
    Case("music-sort-ask", "what songs still need sorting?", (one(READ_MUSIC),), "golden"),
    Case("music-organize", "organize my library into folders", (one(M("ask", "music")),), "golden"),
    Case("refuse-lights", "turn off the living room lights", (none("decline"),), "golden"),
    Case("refuse-weather", "what's the weather tomorrow?", (none("decline"),), "golden"),
    # No fleet app plays music (#36): routing to music is right (its own agent
    # says so one hop later); a plain decline is no longer required, or wrong.
    Case(
        "refuse-playback",
        "play some jazz music",
        (one(M("ask", "music")), none("decline")),
        "golden",
    ),
    Case(
        "confirm-delete-task",
        "delete the groceries task",
        (one(M("ask", "tasks", ("groceries",), destructive=True)),),
        "golden",
    ),
    Case(
        "confirm-wipe-project",
        "wipe the whole kitchen project",
        (one(M("ask", "tasks", ("kitchen",), destructive=True)),),
        "golden",
    ),
    Case("local-capabilities", "what can you do?", (none("capabilities"),), "golden"),
    # --- cross-app -----------------------------------------------------------------
    Case(
        "s3-album-task",
        "download the new Caribou album and add a task to sort it this weekend",
        (one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref=True)),),
        "cross",
    ),
    Case(
        "cross-named-task",
        "grab the song Odessa by Caribou and add a task to put it on the party playlist tomorrow",
        (one(M("ask", "music", ("odessa",)), M("ask", "tasks", ref_or_words=("odessa",))),),
        "cross",
    ),
    Case(
        "cross-playlist-task",
        "download this playlist https://www.youtube.com/playlist?list=PL9 and remind me "
        "Sunday to sort it",
        (one(M("ask", "music", ("playlist",)), M("ask", "tasks", ref_or_words=("playlist",))),),
        "cross",
    ),
    Case(
        "cross-two-reads",
        "what's due today, and how are my downloads going?",
        (one(READ_TASKS, READ_MUSIC, ordered=False),),
        "cross",
    ),
    Case(
        "cross-three-reads",
        "what's due today, how's my chess game, and is the playlist done yet?",
        (one(READ_TASKS, READ_MUSIC, M("look", "chess"), ordered=False),),
        "cross",
    ),
    Case(
        "cross-task-then-chess",
        "add a task to study the Sicilian this weekend, then let's play",
        (one(M("ask", "tasks", ("sicilian",)), M("open", "chess")),),
        "cross",
    ),
    Case(
        "s15-same-again",
        "do the same for their other album",
        (
            one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref=True)),
            one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref_or_words=("sort",))),
            one(M("ask_user")),
            one(M("ask_user"), M("ask", "music"), M("ask", "tasks", ref=True)),
        ),
        "cross",
        state(last=_LAST_S3, recent='album "Honey" by Caribou (music:album:55)'),
    ),
    # --- references and resolution -------------------------------------------------
    Case(
        "s6-that-song",
        "put that Caribou song in the house folder",
        (one(M("ask", "music", ("odessa",))),),
        "ref",
        state(recent='music:track:812 song "Odessa" by Caribou (saved Tue, in Unsorted)'),
    ),
    Case(
        "ref-task-from-song",
        "add a task to make a party playlist with that Caribou song",
        (
            one(M("ask", "tasks", ("odessa",))),
            one(M("recall", words=("caribou",)), M("ask", "tasks", ref=True)),
        ),
        "ref",
        state(recent='music:track:812 song "Odessa" by Caribou (saved Tue, in Unsorted)'),
    ),
    Case(
        "s7-dennis",
        "push the dennis thing to friday",
        (one(M("ask", "tasks", ("dentist", "friday"))),),
        "ref",
        state(
            now="Thursday 2026-11-05, 07:55, kitchen speaker (voice, no screen)",
            matches='tasks: "Dentist appointment" (pcc:task:44, due Wed)',
        ),
    ),
    Case(
        "ref-ambiguous",
        "push the dentist thing to friday",
        (one(M("ask_user")), one(M("ask_user"), M("ask", "tasks", ref=True))),
        "ref",
        state(
            matches='tasks: "Dentist appointment" (pcc:task:44, due Wed); '
            '"Call the dentist about the bill" (pcc:task:61, due Thu)'
        ),
    ),
    # --- control: confirm, cancel, amend, undo -----------------------------------------
    Case(
        "s9-yes",
        "yes",
        (one(M("confirm", answer="yes")),),
        "control",
        state(waiting=_PENDING),
        frozenset({"pending", "pending:p7", "p7"}),
    ),
    Case(
        "s9-no",
        "no, leave it",
        (one(M("confirm", answer="no")),),
        "control",
        state(waiting=_PENDING),
        frozenset({"pending", "pending:p7", "p7"}),
    ),
    Case(
        "s10-stop",
        "actually, stop",
        (one(M("cancel", target_has="31")),),
        "control",
        state(jobs="music:job:31 sorting the whole library (running, 112 of 400 filed)"),
        frozenset({"music:job:31", "job:31", "31"}),
    ),
    Case(
        "amend-job",
        "actually just grab the first ten songs",
        (one(M("amend", target_has="17")),),
        "control",
        state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
        frozenset({"music:job:17", "job:17", "17"}),
    ),
    Case(
        "s11-undo",
        "undo that",
        (one(M("undo")),),
        "control",
        state(last=_LAST_S3),
        frozenset({"pcc:task:301", "music:album:55", "last"}),
    ),
    # --- memory ----------------------------------------------------------------------
    Case(
        "s5-garage",
        "what did we decide about the garage shelves last week?",
        (one(M("recall", words=("garage",))), one(M("recall", words=("garage",)), READ_TASKS)),
        "memory",
    ),
    Case(
        "memory-caribou",
        "when did I last download something by Caribou?",
        (one(M("recall", words=("caribou",))), one(M("ask", "music", ("caribou",)))),
        "memory",
    ),
    # Memory you can see (#81): asks to conductor's own memory app.
    Case(
        "memory-remember",
        "remember I sort music on Sundays",
        (one(M("ask", "memory", ("sort music", "sunday"))),),
        "memory",
    ),
    Case(
        "memory-about-me",
        "what do you know about me?",
        (one(M("ask", "memory")), none("self")),
        "memory",
        state(facts="I sort music on Sundays|focus time is 9 to 11 on weekdays"),
    ),
    Case(
        "memory-forget-that",
        "forget that",
        (one(M("ask", "memory", ("forget",))),),
        "memory",
        state(facts="the spare key is under the blue pot"),
    ),
    Case(
        "memory-forget-named",
        "forget that I sort music on Sundays",
        (one(M("ask", "memory", ("forget", "sunday"))),),
        "memory",
        state(facts="I sort music on Sundays|focus time is 9 to 11 on weekdays"),
    ),
    # A saved fact in another app's ask (#134): the planner passes the words
    # through and code sends the fact along, so the request must keep the
    # words that point at it (the session scenario M3 checks the app got it).
    Case(
        "memory-used",
        "add a task for my music sorting day to clean the record player",
        (
            one(
                M(
                    "ask",
                    "tasks",
                    ("record player",),
                    destructive=False,
                    attaches=("I sort music on Sundays",),
                )
            ),
            one(M("ask", "tasks", ("record player", "sunday"), destructive=False)),
        ),
        "memory",
        state(facts="I sort music on Sundays"),
    ),
    # --- general knowledge and declines ------------------------------------------------
    Case("knowledge-egg", "how long do I boil an egg?", (none("knowledge"),), "knowledge"),
    Case("knowledge-capital", "what's the capital of Portugal?", (none("knowledge"),), "knowledge"),
    Case("decline-news", "what's in the news today?", (none("decline"),), "knowledge"),
    # --- hard planning -------------------------------------------------------------------
    Case(
        "s14-plan-week",
        "plan my week around the garage project",
        (one(M("think")), one(READ_TASKS, M("think"))),
        "hard",
    ),
    # --- self-report and chat -----------------------------------------------------------
    Case(
        "status-now",
        "what are you working on right now?",
        (none("status"), one(M("look", "music"))),
        "self",
        state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
    ),
    Case("status-last", "what did you just do?", (none("status"),), "self", state(last=_LAST_S3)),
    Case("chat-thanks", "thanks!", (none("chat"),), "self"),
    # --- a question about now, over a recalled answer (#127, #122's leak) --------------
    Case(
        "stale-week-recalled",
        _WEEK,
        (one(READ_TASKS),),
        "stale",
        before(
            _WEEK,
            earlier=[(_WEEK, _call("ask_tasks", {"message": _WEEK}, "Nothing's due today."))],
        ),
    ),
    Case(
        "stale-due-same-thread",
        _DUE,
        (one(READ_TASKS),),
        "stale",
        before(
            _DUE,
            history=[
                {"role": "user", "content": _DUE},
                {
                    "role": "assistant",
                    "content": "Nothing's due today.",
                    "tool_calls": [
                        _call("ask_tasks", {"message": _DUE}, "No tasks due today.").model_dump()
                    ],
                },
            ],
        ),
    ),
    # #141: the same question earlier in the thread, with another exchange since.
    Case(
        "stale-due-earlier-in-thread",
        _DUE,
        (one(READ_TASKS),),
        "stale",
        before(
            _DUE,
            history=[
                {"role": "user", "content": _DUE},
                {
                    "role": "assistant",
                    "content": "Nothing's due today.",
                    "tool_calls": [
                        _call("ask_tasks", {"message": _DUE}, "No tasks due today.").model_dump()
                    ],
                },
                {"role": "user", "content": "add a task to call mom tomorrow"},
                {
                    "role": "assistant",
                    "content": "Added “call mom” for tomorrow.",
                    "tool_calls": [
                        _call(
                            "ask_tasks",
                            {"message": "add a task to call mom tomorrow"},
                            "Created task “call mom”, due tomorrow.",
                        ).model_dump()
                    ],
                },
            ],
        ),
    ),
    Case(
        "stale-playlist-recalled",
        _PLAYLIST_NOW,
        (one(READ_MUSIC),),
        "stale",
        before(
            _PLAYLIST_NOW,
            earlier=[
                (
                    _PLAYLIST,
                    _call(
                        "ask_music",
                        {"message": _PLAYLIST},
                        "Queued 40 songs from the playlist; it's downloading now.",
                    ),
                )
            ],
        ),
    ),
    Case(
        "stale-chess-recalled",
        _CHESS_NOW,
        (one(M("look", "chess")), one(M("open", "chess"))),
        "stale",
        before(
            _CHESS_NOW,
            earlier=[
                (
                    "how is our chess game going?",
                    _call("look_chess", {}, "Your move, move 12; material is even."),
                )
            ],
        ),
    ),
)


# Spike D1's held-out set, written after v1's misses were read. Every PR has
# run it and decided on it since 2026-11-06, so it is an ordinary group now
# (#142); the held-out set is ``held_out_cases.py``.
MIXED: tuple[Case, ...] = (
    Case(
        "ho-swim-task",
        "download Swim by Caribou and add a task to sort it Monday",
        (one(M("ask", "music", ("swim",)), M("ask", "tasks", ref_or_words=("swim",))),),
        "mixed",
    ),
    Case(
        "ho-lyrics-task",
        "find the song that goes 'I can't do without you' and add a task to put it on the "
        "party playlist",
        (one(M("ask", "music", ("without you",)), M("ask", "tasks", ref=True)),),
        "mixed",
    ),
    Case(
        "ho-two-reads",
        "how many songs are waiting to be sorted, and what's overdue?",
        (one(READ_MUSIC, READ_TASKS, ordered=False),),
        "mixed",
    ),
    Case(
        "ho-cancel",
        "cancel that download",
        (one(M("cancel", target_has="17")),),
        "mixed",
        state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
        frozenset({"music:job:17", "job:17", "17"}),
    ),
    Case(
        "ho-amend",
        "make it the first twenty instead",
        (one(M("amend", target_has="17")),),
        "mixed",
        state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
        frozenset({"music:job:17", "job:17", "17"}),
    ),
    Case(
        "ho-recall",
        "what do you remember about the fence project?",
        (one(M("recall", words=("fence",))), one(M("recall", words=("fence",)), READ_TASKS)),
        "mixed",
    ),
    Case("ho-knowledge", "how many teaspoons are in a tablespoon?", (none("knowledge"),), "mixed"),
    Case("ho-decline-score", "what's the score of the Leafs game?", (none("decline"),), "mixed"),
    Case(
        "ho-delete-project",
        "delete the fence project",
        (one(M("ask", "tasks", ("fence",), destructive=True)),),
        "mixed",
    ),
    Case(
        "ho-chess-last",
        "how did my last chess game go?",
        (one(M("look", "chess")), one(M("open", "chess"))),
        "mixed",
    ),
)


# Long plans (decided 2026-11-06): requests that need six to ten steps when
# compiled one thing per step. D1 never measured past five.
LONG: tuple[Case, ...] = (
    Case(
        "long-three-albums",
        "download the new albums from Bonobo, Caribou and Four Tet, add a task for each to "
        "sort it on Saturday, and then let's play chess",
        (
            Long(
                (
                    Cover("music", ("ask",), ("bonobo", "caribou", "four tet")),
                    Cover("tasks", ("ask",), ("saturday",)),
                ),
                last=("open",),
                min_steps=3,
            ),
        ),
        "long",
    ),
    Case(
        "long-rundown",
        "give me a rundown: what's due today, what's due tomorrow, what's overdue, how many "
        "songs need sorting, what's downloading, and how's my chess game?",
        (
            Long(
                (
                    Cover("tasks", ("look", "ask")),
                    Cover("music", ("look", "ask")),
                    Cover("chess", ("look", "open")),
                ),
                min_steps=3,
            ),
        ),
        "long",
    ),
    Case(
        "long-errands",
        "add tasks to call the plumber Monday, renew the car registration Tuesday and book a "
        "haircut Friday, download the playlist https://www.youtube.com/playlist?list=PL7, sort "
        "my music, and then open the chess board",
        (
            Long(
                (
                    Cover("tasks", ("ask",), ("plumber", "registration", "haircut")),
                    Cover("music", ("ask",), ("playlist", "sort")),
                ),
                last=("open",),
                min_steps=3,
            ),
        ),
        "long",
    ),
    Case(
        "long-cleanup",
        "I'm cleaning house: trash the groceries task, mark fix the drawer done, move paint "
        "cabinets to Sunday, sort my music, and stop that playlist download",
        (
            Long(
                (
                    Cover("tasks", ("ask",), ("groceries", "drawer", "cabinets"), destructive=True),
                    Cover("music", ("ask",), ("sort",)),
                    Cover(None, ("cancel", "ask"), ()),
                ),
                min_steps=3,
            ),
        ),
        "long",
        state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
        frozenset({"music:job:17", "job:17", "17"}),
    ),
    Case(
        "long-swim-chain",
        "download Swim by Caribou, add a task to put it on the party playlist, another to tell "
        "Sam about it, and remind me Sunday to rate it",
        (
            Long(
                (
                    Cover("music", ("ask",), ("swim",)),
                    Cover("tasks", ("ask",), ("party", "sam", "sunday")),
                ),
                min_steps=2,
            ),
        ),
        "long",
    ),
    Case(
        "long-fence-weekend",
        "look back at what we said about the fence project, check what's overdue and what's "
        "due this weekend, and then plan my weekend around all of it",
        (
            Long(
                (
                    Cover(None, ("recall",), ("fence",)),
                    Cover("tasks", ("look", "ask")),
                    Cover(None, ("think",)),
                ),
                last=("think",),
                min_steps=3,
            ),
        ),
        "long",
    ),
    Case(
        "long-mixed-dentist",
        "push the dentist thing to Friday, add a task to pick up the dry cleaning, grab Odessa "
        "by Caribou, and then let's play chess",
        (
            Long(
                (
                    Cover(None, ("ask_user", "ask"), ("dentist",)),
                    Cover("tasks", ("ask",), ("dry cleaning",)),
                    Cover("music", ("ask",), ("odessa",)),
                ),
                last=("open",),
                min_steps=4,
            ),
        ),
        "long",
        state(
            matches='tasks: "Dentist appointment" (pcc:task:44, due Wed); '
            '"Call the dentist about the bill" (pcc:task:61, due Thu)'
        ),
    ),
    Case(
        "long-week-setup",
        "set up my week: add a task for each weekday morning to stretch, Monday through Friday, "
        "download the Morning Mix playlist https://www.youtube.com/playlist?list=PL3 to go with "
        "it, and sort the library afterwards",
        (
            Long(
                (
                    Cover("tasks", ("ask",), ("stretch",)),
                    Cover("music", ("ask",), ("morning mix", "sort")),
                ),
                min_steps=2,
            ),
        ),
        "long",
    ),
)
