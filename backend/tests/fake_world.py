"""A fake fleet whose apps really change state, for conductor's session evals.

The routing gate stubs each app with one canned sentence: enough to see where a
request went, not what came of it. A session eval (#63) judges the end state —
was the task created, with the right title and date; is the album queued; did
"undo that" trash the task — so the apps behind conductor's delegate tools have
to *do* things. This module is that world: a tasks app (PCC's shape: projects,
tasks, due dates, trash) and a music app (a catalog to download from, a library
with folders, download jobs), each behind a ``DelegateClientLike`` that reads
conductor's free-text request the way a cooperative app agent would.

**The apps are deterministic, not model-driven.** Conductor's choices are the
thing under eval; an app agent sampling at temp 1.0 behind it would add a
second source of noise the verdict could not attribute. Each app reads its
message with plain rules — generous ones, because an app the size of PCC's
agent understands far more than a regex does — and replies saying exactly
what it did, with titles and ISO dates, the way the real apps' agents do. What
the rules cannot read they answer with a question and change nothing, which is
what a real app does with an ambiguous request too (S6 leans on exactly that:
"that Caribou song" is ambiguous to music, so conductor has to carry "Odessa").

**Every write leaves a receipt** (``World.receipts``): app, kind, the thing
touched and when. Receipts are the ground truth the checks read — the end
state, and what a reply may claim (``receipt_claims.py``). They are this
world's version of the ledger invariant "every app write has a receipt row"
(#69): the fake apps cannot write without one.

Runtime-agnostic on purpose: the world is reached only through the delegate
client seam (``app.fleet.tools.ClientFactory``), which today's loop and the
compiled-plan runtime both sit on top of.
"""

from __future__ import annotations

import difflib
import copy
import json
import re
import textwrap
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from app import clock
from app.ai.facts import CONTEXT_LEAD
from app.fleet.delegate import (
    ConfirmRequired,
    DelegateConfirmRefused,
    EntityRef,
    JobRef,
    MessageExchange,
    MessageRead,
    NeedsInputRead,
    RecordEntry,
    ToolCallRead,
    Undo,
)

# --- dates -----------------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def upcoming(today: date, weekday: int) -> date:
    """The next ``weekday`` strictly after today (a "friday" said on a Friday
    means next week's)."""
    ahead = (weekday - today.weekday()) % 7
    return today + timedelta(days=ahead or 7)


def weekend_targets(today: date) -> frozenset[date]:
    """Days "this weekend" may land on: the coming Saturday and Sunday, or
    today/tomorrow when today is already the weekend."""
    if today.weekday() == 5:
        return frozenset({today, today + timedelta(days=1)})
    if today.weekday() == 6:
        return frozenset({today})
    saturday = upcoming(today, 5)
    return frozenset({saturday, saturday + timedelta(days=1)})


def friday_targets(today: date) -> frozenset[date]:
    """ "Friday" read either way a person might mean it: the coming Friday, and
    — said early in the week — the one after ("next Friday")."""
    first = upcoming(today, 4)
    return frozenset({first, first + timedelta(days=7)})


_DATE_PATTERNS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str], date], date]], ...] = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m, t: date(*map(int, m.groups()))),
    (re.compile(r"\btoday\b|\btonight\b"), lambda m, t: t),
    (re.compile(r"\btomorrow\b"), lambda m, t: t + timedelta(days=1)),
    (re.compile(r"\b(?:this |the )?weekend\b"), lambda m, t: min(weekend_targets(t))),
    (
        re.compile(r"\bnext (" + "|".join(_WEEKDAYS) + r")\b"),
        lambda m, t: upcoming(t, _WEEKDAYS.index(m.group(1))),
    ),
    (re.compile(r"\bnext week\b"), lambda m, t: upcoming(t, 0)),
    (
        re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b"),
        lambda m, t: upcoming(t, _WEEKDAYS.index(m.group(1))),
    ),
    (
        re.compile(r"\b(" + "|".join(m[:3] for m in _MONTHS) + r")[a-z]*\.? (\d{1,2})\b"),
        lambda m, t: _month_day(t, m.group(1), int(m.group(2))),
    ),
)


def _month_day(today: date, month: str, day: int) -> date:
    candidate = date(today.year, [m[:3] for m in _MONTHS].index(month[:3]) + 1, day)
    return (
        candidate
        if candidate >= today - timedelta(days=60)
        else candidate.replace(year=today.year + 1)
    )


def parse_date(text: str, today: date) -> date | None:
    """The first date phrase in ``text``, read the way a task app reads one."""
    lowered = text.lower()
    found: list[tuple[int, date]] = []
    for pattern, resolve in _DATE_PATTERNS:
        match = pattern.search(lowered)
        if match:
            try:
                found.append((match.start(), resolve(match, today)))
            except ValueError:
                continue
    return min(found)[1] if found else None


_DATE_PHRASE = re.compile(
    r"\s*\b(?:(?:due|on|by|for|this|next|until)\s+)*(?:"
    r"\d{4}-\d{2}-\d{2}|today|tonight|tomorrow|weekend|week|"
    + "|".join(_WEEKDAYS)
    + r"|(?:"
    + "|".join(m[:3] for m in _MONTHS)
    + r")[a-z]*\.? \d{1,2})\b"
)


# --- conductor's context after a request (#134) ---------------------------------

_CONTEXT_MARK = f"\n\n({CONTEXT_LEAD} "
_PLURAL_DAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")s\b")


def split_context(message: str) -> tuple[str, str]:
    """The request and the saved facts conductor sent after it, which an
    app's agent reads as context, never as part of what to do."""
    request, mark, context = message.partition(_CONTEXT_MARK)
    return (request, context.rstrip(")")) if mark else (message, "")


# What conductor's ``amend`` appends to a waiting step's request (#74, #77).
_CHANGE = re.compile(r"\s*\(change:\s*(?P<change>.*)\)\s*$", re.DOTALL)
_DAY_WORD = re.compile(
    r"\b(?:on\s+|to\s+|for\s+)?(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b",
    re.IGNORECASE,
)


def split_change(message: str) -> tuple[str, str]:
    """The request and the person's change to it, which an app's agent reads
    as overriding what it contradicts: "delete the Kitchen project (change:
    no, just the paint cabinets task)" is the task alone."""
    found = _CHANGE.search(message)
    return (message[: found.start()], found.group("change")) if found else (message, "")


def _context_day(context: str, today: date) -> date | None:
    """A weekday the context names as a habit ("on Sundays"): the next one."""
    found = _PLURAL_DAY.search(context.lower())
    return upcoming(today, _WEEKDAYS.index(found.group(1))) if found else None


# --- fuzzy matching -------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an the to for of in on at by and or my me i you it that this thing please
    can could would will task tasks todo add create new put move push due date
    with from into up out is be are was do does did just all whole one".""".split()
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def keywords(text: str) -> list[str]:
    return [word for word in words(text) if len(word) >= 3 and word not in _STOPWORDS]


def _token_score(needle: str, title: str) -> float:
    best = max(
        (difflib.SequenceMatcher(None, needle, token).ratio() for token in words(title)),
        default=0.0,
    )
    return best if best >= 0.75 else 0.0


def best_match(text: str, titles: dict[int, str]) -> int | None:
    """The id whose title the message's keywords hit hardest, or None. Ties are
    ambiguity, and ambiguity is a question, never a guess."""
    scores = {
        key: sum(_token_score(needle, title) for needle in keywords(text))
        for key, title in titles.items()
    }
    top = max(scores.values(), default=0.0)
    if top <= 0:
        return None
    winners = [key for key, score in scores.items() if score == top]
    return winners[0] if len(winners) == 1 else None


def mentions(text: str, title: str) -> bool:
    """Does ``text`` name ``title``: every one of its keywords present (one
    missing allowed for long titles), case- and punctuation-blind."""
    have = set(words(text))
    need = keywords(title) or words(title)
    missing = sum(1 for word in need if word not in have)
    return missing == 0 or (len(need) >= 4 and missing <= 1)


# --- state ------------------------------------------------------------------------


@dataclass
class Receipt:
    """One app write, as the world recorded it. ``kind`` is the write's family
    (``create``, ``update``, ``complete``, ``trash``, ``queue``, ``file``,
    ``cancel``); ``title`` the name of what it touched; ``turn`` the session
    turn it happened in (set by the harness before each turn)."""

    app: str
    kind: str
    ref: str
    title: str
    turn: int
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: int
    title: str
    project: str | None = None
    due: date | None = None
    notes: str = ""
    done: bool = False
    trashed: bool = False


@dataclass
class Song:
    id: int
    title: str
    artist: str
    album: str
    genre: str
    folder: str | None = None
    liked: bool = False


@dataclass
class Album:
    title: str
    artist: str
    year: int
    tracks: tuple[str, ...]
    genre: str

    @property
    def mbid(self) -> str:
        """A MusicBrainz release-group id: what real music names an album by
        (music#96). Stable per album, and not a job's id."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"musicbrainz.test/release-group/{self.artist}/{self.title}"
            )
        )


@dataclass
class Job:
    id: int
    kind: str  # "album" | "playlist" | "song"
    title: str
    total: int
    done: int = 0
    failed: int = 0
    state: str = "queued"  # queued | finished | cancelled
    # The tracks that never matched, by title (outside text), once it finished.
    misses: tuple[str, ...] = ()
    # The album the job fetches, when one was resolved (a MusicBrainz id).
    album_id: str | None = None


# The two playlist tracks that never match (S4): titles from outside.
PLAYLIST_MISSES = ("Lost Signal", "Paper Moons")
# Music's limits on a refused album lookup (music ``albums.MAX_CANDIDATES``,
# ``run_record._MAX_FIELD_CHARS``): at most three albums listed, and the
# record's ``reason`` cut to 160 characters, so a long list can lose its end.
MUSIC_MAX_CANDIDATES = 3
# How real music's note asks after a refusal that lists albums (music#107).
MUSIC_CANDIDATES_QUESTION = "Which one do you want, or do you have a link?"
MUSIC_FIELD_CHARS = 160


def _music_clip(text: str) -> str:
    """Music's ``run_record.clip``: whitespace flattened, cut with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= MUSIC_FIELD_CHARS else text[: MUSIC_FIELD_CHARS - 1] + "…"


# The jobs contract's words for the world's job states (delegate-api "Jobs").
_JOB_STATUS = {"queued": "working", "finished": "completed", "cancelled": "canceled"}


@dataclass
class ChessGame:
    result: str
    summary: str


class World:
    """Every app's state, the receipts of every write, and the threads used.

    ``today`` anchors the seeded dates (the prompt reads the household's date,
    so the world must too). ``turn`` is advanced by the harness so receipts
    know which turn wrote them. ``on_write`` is the Stop seam: called after
    every receipt, it lets a trajectory or a Stop scenario act mid-delegate.
    """

    def __init__(self, today: date | None = None) -> None:
        self.today = today or clock.local_today()
        self.turn = 0
        self.receipts: list[Receipt] = []
        self.reads: list[tuple[str, str, int]] = []  # (app, what, turn)
        self.threads: dict[str, list[int]] = {}  # app → conversation id per send
        # Every message an app received, as sent: (app, message), in order.
        self.sent: list[tuple[str, str]] = []
        # Every message's delegate v1.2 fields (#202): (app, context, mode).
        self.sent_v12: list[tuple[str, dict[str, Any] | None, str | None]] = []
        self.cancel_requested: dict[str, bool] = {}
        self.on_write: Callable[[Receipt], None] | None = None
        self.tasks: dict[int, Task] = {}
        self.projects: dict[str, bool] = {}  # name → trashed
        self.songs: dict[int, Song] = {}
        self.catalog: list[Album] = []
        self.jobs: dict[int, Job] = {}
        self.chess = ChessGame(
            result="won",
            summary="You won your last game as white in 34 moves, with the Italian.",
        )
        self._lock = threading.Lock()
        self._seed()

    # -- seeding --

    def _seed(self) -> None:
        t = self.today
        dentist_day = upcoming(t, 1)  # a Tuesday: never a Friday target
        seeded = [
            Task(1, "groceries", "Household", t),
            Task(2, "call mom", "Household", t),
            Task(3, "dentist appointment", "Household", dentist_day),
            Task(4, "renew passport", "Household", t - timedelta(days=4)),
            Task(5, "pay water bill", "Household", t - timedelta(days=2)),
            Task(6, "return library books", "Household", t - timedelta(days=1)),
            Task(7, "buy new sink", "Kitchen", t + timedelta(days=10)),
            Task(8, "paint cabinets", "Kitchen", None),
            Task(9, "fix the drawer", "Kitchen", None),
            Task(
                10,
                "garage shelves",
                "Garage",
                None,
                notes=(
                    f"Decided {(t - timedelta(days=7)).isoformat()}: go with the steel rack, "
                    "not wood — it holds the paint cans."
                ),
            ),
            Task(11, "clear out old bikes", "Garage", None),
            Task(12, "sweep the floor", "Garage", None),
        ]
        self.tasks = {task.id: task for task in seeded}
        self.projects = {"Household": False, "Kitchen": False, "Garage": False}
        self.catalog = [
            Album(
                "Honey",
                "Caribou",
                2024,
                ("Volume", "Honey", "Come Find Me", "Broke My Heart"),
                "electronic",
            ),
            Album(
                "Suddenly",
                "Caribou",
                2020,
                ("Sister", "You and I", "Sunny's Time", "Home"),
                "electronic",
            ),
            Album("Swim", "Caribou", 2010, ("Odessa", "Sun", "Kaili", "Found Out"), "house"),
            Album("Kind of Blue", "Miles Davis", 1959, ("So What", "Freddie Freeloader"), "jazz"),
        ]
        library = [
            Song(1, "Sun", "Caribou", "Swim", "house", "house"),
            Song(2, "So What", "Miles Davis", "Kind of Blue", "jazz", "jazz"),
            Song(3, "Midnight City", "M83", "Hurry Up, We're Dreaming", "electronic", liked=True),
            Song(4, "Teardrop", "Massive Attack", "Mezzanine", "trip hop", liked=True),
            Song(5, "Glue", "Bicep", "Bicep", "house", liked=True),
            Song(6, "Rhubarb", "Aphex Twin", "Selected Ambient Works", "ambient", liked=True),
            Song(7, "Windowlicker", "Aphex Twin", "Windowlicker", "electronic", liked=True),
        ]
        self.songs = {song.id: song for song in library}

    # -- the write path: every write leaves a receipt --

    def record(self, app: str, kind: str, ref: str, title: str, **detail: Any) -> Receipt:
        receipt = Receipt(app, kind, ref, title, self.turn, dict(detail))
        with self._lock:
            self.receipts.append(receipt)
        if self.on_write is not None:
            self.on_write(receipt)
        return receipt

    # What a read-mode message must leave as it found (#202).
    _WRITABLE = ("tasks", "projects", "songs", "catalog", "jobs")

    def writable(self) -> dict[str, Any]:
        """A copy of everything an app write can change."""
        return {name: copy.deepcopy(getattr(self, name)) for name in self._WRITABLE}

    def restore(self, saved: dict[str, Any], receipts: int) -> None:
        """Put back :meth:`writable`'s copy and drop the receipts after ``receipts``."""
        for name, value in saved.items():
            setattr(self, name, value)
        del self.receipts[receipts:]

    def read(self, app: str, what: str) -> None:
        self.reads.append((app, what, self.turn))

    def glance(self, app: str) -> dict[str, Any]:
        """What the app's glance endpoint would answer from the world as it is
        now (agent-standard#21), recorded as a read of that app: a look is a
        read, so a check that wants "the app was read" sees it. Shapes follow
        the apps' own: PCC's due_today refs, overdue count, focus and next_due;
        music's jobs and unsorted count; chess's game and last game."""
        as_of = datetime.now(UTC).isoformat()
        if app == "tasks":
            self.read("tasks", "glance")
            due = self.due_today()
            upcoming_due = sorted(
                {t.due for t in self.live_tasks() if not t.done and t.due and t.due > self.today}
            )
            pinned = due + self.overdue()
            focus = pinned[0] if pinned else None
            return {
                "as_of": as_of,
                "due_today": [_ref_json(_task_ref(task)) for task in due[:5]],
                "overdue": len(self.overdue()),
                "focus": _ref_json(_task_ref(focus)) if focus else None,
                "next_due": upcoming_due[0].isoformat() if upcoming_due else None,
            }
        if app == "music":
            self.read("music", "glance")
            return {
                "as_of": as_of,
                "jobs": [
                    {
                        **_ref_json(
                            EntityRef(app="music", kind="download_job", id=str(j.id), title=j.title)
                        ),
                        "status": j.state,
                        "done": j.done,
                        "total": j.total,
                        "failed": j.failed,
                    }
                    for j in list(self.jobs.values())[-5:]
                ],
                "queued": sum(1 for j in self.jobs.values() if j.state == "queued"),
                "unsorted": len(self.unsorted()),
            }
        if app == "chess":
            self.read("chess", "glance")
            return {
                "as_of": as_of,
                "game": None,
                "last_game": {"result": self.chess.result, "summary": self.chess.summary},
            }
        raise KeyError(f"no glance for {app}")

    def glance_reader(self) -> Callable[[Any], dict[str, Any]]:
        """This world as a ``GlanceReader`` for the interpreter: no HTTP, ever."""
        return lambda fleet_app: self.glance(fleet_app.name)

    # -- queries the checks use --

    def live_tasks(self) -> list[Task]:
        return [task for task in self.tasks.values() if not task.trashed]

    def find_tasks(self, predicate: Callable[[Task], bool]) -> list[Task]:
        return [task for task in self.live_tasks() if predicate(task)]

    def writes(self, *, turn: int | None = None, app: str | None = None) -> list[Receipt]:
        return [
            receipt
            for receipt in self.receipts
            if (turn is None or receipt.turn == turn) and (app is None or receipt.app == app)
        ]

    def overdue(self) -> list[Task]:
        return self.find_tasks(lambda t: not t.done and t.due is not None and t.due < self.today)

    def due_today(self) -> list[Task]:
        return self.find_tasks(lambda t: not t.done and t.due == self.today)

    def unsorted(self) -> list[Song]:
        return [song for song in self.songs.values() if song.folder is None]

    def advance(self) -> None:
        """Time passes: every queued job finishes. A playlist lands 38 of 40
        (two tracks never match), which S4's report has to say."""
        for job in self.jobs.values():
            if job.state == "queued":
                job.misses = PLAYLIST_MISSES if job.kind == "playlist" else ()
                job.done = job.total - len(job.misses)
                job.failed = job.total - job.done
                job.state = "finished"

    # -- the jobs contract (delegate-api "Jobs", #76) --

    def job_read(self, ref: str) -> dict[str, Any] | None:
        """``GET …/jobs/{ref}`` as music answers it, or ``None`` (a 404). Not
        a read of the app the checks count: it's conductor's watcher, not the
        model, asking."""
        job = self.jobs.get(int(ref)) if ref.isdigit() else None
        if job is None:
            return None
        status = _JOB_STATUS[job.state]
        record = None
        if job.state != "queued":
            record = [
                {
                    "tool": f"download_{job.kind}",
                    "outcome": "done",
                    "touched": [_ref_json(_job_ref(job))],
                    "count": job.done,
                    "undo": {"none": "a download can't be undone"},
                }
            ]
            if job.misses:
                record.append(
                    {
                        "tool": "download_track",
                        "outcome": "refused",
                        "reason": "no match found",
                        "touched": [
                            {
                                "app": "music",
                                "kind": "song",
                                "id": f"{job.id}-miss-{n}",
                                "title": title,
                                "untrusted": True,
                            }
                            for n, title in enumerate(job.misses, start=1)
                        ],
                    }
                )
        return {
            "ref": str(job.id),
            "status": status,
            "done": job.done,
            "total": job.total,
            "title": job.title,
            "untrusted": True,
            "reason": f"{job.failed} tracks couldn't be downloaded" if job.failed else None,
            "record": record,
        }

    def held_album_ids(self) -> set[str]:
        """The albums music already has (music#102's ``jobs.held_album_ids``):
        a queued or finished album job's; a cancelled one holds nothing."""
        return {
            job.album_id for job in self.jobs.values() if job.album_id and job.state != "cancelled"
        }

    def job_cancel(self, ref: str) -> dict[str, Any] | None:
        """``POST …/jobs/{ref}/cancel``: a queued job stops at once, with what
        it had (nothing yet), and the cancel is a write with a receipt."""
        job = self.jobs.get(int(ref)) if ref.isdigit() else None
        if job is None:
            return None
        live = job.state == "queued"
        if live:
            job.state = "cancelled"
            self.record("music", "cancel", f"job:{job.id}", job.title)
        read = self.job_read(ref)
        assert read is not None
        return {**read, "cancel_requested": live}


# --- the apps ----------------------------------------------------------------------


@dataclass
class AppReply:
    text: str
    tools: list[ToolCallRead] = field(default_factory=list)
    stop_reason: str = "completed"
    confirm: ConfirmRequired | None = None
    # What a read found, as the real apps' v1.1 records name it. Writes need
    # none: the client builds their entries from the receipts they left.
    found: list[RecordEntry] = field(default_factory=list)
    # A question the real app asks by code (delegate v1.2's ``needs_input``,
    # #202): PCC's ``ask_user``, music's ``ask_folder`` and its album refusal.
    # A question its model words alone ("What should the task be called?") is
    # not one.
    asks: bool = False
    # The question ``needs_input`` carries when it isn't the text's last one
    # (music's fixed candidates question, music#110).
    question: str | None = None


def _tool(tool_name: str, **arguments: Any) -> ToolCallRead:
    return ToolCallRead(tool=tool_name, arguments=arguments, result="ok")


# --- the apps' typed records (delegate-api v1.1) --------------------------------------
#
# Shaped as PCC's and music's ``app/ai/receipts.py`` build them: a task or
# project is named by its own title; a song's title comes from outside (web
# titles, file tags), so it is ``untrusted``; a download job is named by the
# request. Writes are read back off the receipts each call left.


def _task_ref(task: Task) -> EntityRef:
    return EntityRef(app="tasks", kind="task", id=str(task.id), title=task.title)


def _song_ref(song: Song) -> EntityRef:
    return EntityRef(app="music", kind="song", id=str(song.id), title=song.title, untrusted=True)


def _ref_json(ref: EntityRef) -> dict[str, Any]:
    return ref.model_dump(exclude_none=True)


def _job_ref(job: Job) -> EntityRef:
    """What a job is, as its final report names it: the album it resolved to
    (by its release-group id), else the job's own id."""
    ident = job.album_id or str(job.id)
    return EntityRef(app="music", kind=job.kind, id=ident, title=job.title, untrusted=True)


def _looked_up(tool: str, refs: list[EntityRef]) -> RecordEntry:
    return RecordEntry(tool=tool, outcome="looked_up", touched=refs, count=len(refs))


# Each write family as (tool, undo): the inverse the real app offers, or why
# there is none.
_TASK_WRITES: dict[str, tuple[str, Undo]] = {
    "create": ("create_task", Undo(how="move it to the trash")),
    "update": ("update_task", Undo(none="an edit isn't versioned; change it back by hand")),
    "complete": ("complete_task", Undo(how="reopen it")),
    "trash": ("trash_task", Undo(how="restore it from the trash")),
}


def _write_entry(receipt: Receipt) -> RecordEntry:
    if receipt.kind == "cancel":
        return RecordEntry(tool="cancel", outcome="done", count=receipt.detail.get("filed"))
    kind, ident = receipt.ref.split(":", 1)
    if receipt.app == "tasks":
        tool, undo = _TASK_WRITES[receipt.kind]
        if kind == "project":
            tool, undo = "trash_project", Undo(how="restore it and its tasks from the trash")
        if undo.how:
            undo = Undo(handle=f"{tool}:{ident}", how=undo.how)
        ref = EntityRef(app="tasks", kind=kind, id=ident, title=receipt.title)
        return RecordEntry(tool=tool, outcome="done", touched=[ref], undo=undo)
    if receipt.kind == "file":
        ref = EntityRef(app="music", kind="song", id=ident, title=receipt.title, untrusted=True)
        undo = Undo(handle="undo_last_filing", how="file the songs again")
        return RecordEntry(tool="file_song", outcome="done", touched=[ref], undo=undo)
    no_undo = Undo(none="a download can't be undone")
    if kind == "song":
        ref = EntityRef(app="music", kind="song", id=ident, title=receipt.title, untrusted=True)
        return RecordEntry(tool="download_song", outcome="done", touched=[ref], undo=no_undo)
    job_kind = str(receipt.detail.get("job_kind", "album"))
    # An album music resolved is an entity ref beside its job (music#96): its
    # MusicBrainz release-group id and title, outside text like any title it
    # didn't write. The job is bookkeeping, named by the kind of request.
    subject = str(receipt.detail.get("album_id") or ident)
    refs = [
        EntityRef(app="music", kind=job_kind, id=subject, title=receipt.title, untrusted=True),
        EntityRef(app="music", kind="download_job", id=ident, title=f"{job_kind} download"),
    ]
    # A queued download is a job (delegate-api "Jobs"): accepted now, done later.
    job = JobRef(ref=ident, status="submitted", done=0, total=receipt.detail.get("total"))
    return RecordEntry(tool="queue_download", outcome="done", touched=refs, undo=no_undo, job=job)


def _needs_input(reply: AppReply, written: list[Receipt]) -> NeedsInputRead | None:
    """The fake apps' typed question (#202), as the real apps build it: only
    from a code-asked question, in a call that wrote nothing."""
    if written or not reply.asks:
        return None
    asked = [s for s in re.split(r"(?<=[.?!])\s+", reply.text.strip()) if s.endswith("?")]
    question = reply.question or (asked[-1].strip('"') if asked else reply.text)
    candidates = [
        ref.model_dump(exclude_none=True)
        for entry in reply.found
        if entry.outcome == "refused"
        for ref in getattr(entry, "candidates", None) or []
    ]
    return NeedsInputRead(question=question, candidates=candidates)


def _record(reply: AppReply, written: list[Receipt]) -> list[RecordEntry]:
    if reply.confirm is not None:
        return [RecordEntry(tool=reply.confirm.tool, outcome="not_run", reason="needs a yes")]
    return [*reply.found, *(_write_entry(receipt) for receipt in written)]


_MOVE_WORDS = r"\b(push|move|reschedule|postpone|bump|shift|change the due)\b"
# A line of a list that puts an existing task on a day (a think's proposal, #78).
_LIST_MOVE_WORDS = (
    r"\b(push|move|reschedule|postpone|bump|shift|change the due|schedule|set the due)\b"
)


class TasksApp:
    """PCC's shape: read, create, move, complete and trash tasks and projects.

    Trashing is gated as PCC gates it for a delegate (PCC#503): it arms a
    single-use token and asks; only that thread's next message, carrying the
    token, runs it. Any other message voids the question.
    """

    name = "tasks"

    def __init__(self, world: World) -> None:
        self.world = world
        self._last_created: dict[int, list[int]] = {}  # thread → task ids it created
        self._armed: dict[int, tuple[str, Callable[[], AppReply]]] = {}  # thread → (token, run)

    def handle(self, thread: int, message: str, confirm_token: str | None = None) -> AppReply:
        armed = self._armed.pop(thread, None)  # the next message settles it
        if confirm_token is not None:
            if armed is None or armed[0] != confirm_token:
                raise DelegateConfirmRefused("409: confirm token unknown", kind="unknown")
            return armed[1]()
        message, context = split_context(message)
        message = self._changed(message)
        text = message.lower()
        lines = [line.strip(" -*\t") for line in message.splitlines() if line.strip(" -*\t")]
        if len(lines) > 1 and all(re.search(_LIST_MOVE_WORDS, line.lower()) for line in lines):
            # A list of moves, one per line (a think's proposal, #78), as PCC's agent takes it.
            return self._moves(lines)
        leads_with_create = re.match(
            r"\W*(?:please\s+|can you\s+|could you\s+)?(?:add|create|make|remind me)\b", text
        ) or re.search(r"^\s*(?:[-*]|\d+[.)])\s", message, flags=re.M)
        if leads_with_create:
            return self._create(thread, message, context)
        if re.search(r"\b(undo|revert|take (?:that|it) back)\b", text):
            return self._undo(thread)
        if re.search(r"\b(delete|trash|remove|wipe|bin|get rid of)\b", text):
            return self._trash(thread, message)
        if re.search(_MOVE_WORDS, text):
            return self._move(message)
        if re.search(r"\b(mark|complete|finish|tick off|check off)\b", text) and "done" in text:
            return self._complete(message)
        if re.search(r"\b(add|create|make|new task|remind me)\b", text) and not (
            re.search(r"\?\s*$", text) or text.startswith(("what", "which", "how", "is ", "are "))
        ):
            return self._create(thread, message, context)
        return self._read(message)

    # reads

    _NOT_SEARCH = frozenset(
        """today tomorrow week weekend last overdue late search find show any anything
        regarding related about notes note decide decided decision what whats plan
        planning around list everything""".split()
    )

    def _read(self, message: str) -> AppReply:
        text = message.lower()
        world = self.world
        terms = [k for k in keywords(message) if k not in self._NOT_SEARCH]
        hits = [
            task
            for task in world.live_tasks()
            if any(_token_score(k, f"{task.title} {task.project} {task.notes}") for k in terms)
        ]
        if hits:
            found, what = hits, "matching"
        elif "overdue" in text or "late" in text:
            found, what = world.overdue(), "overdue"
        elif "today" in text:
            found, what = world.due_today(), "due today"
        elif "week" in text:
            end = world.today + timedelta(days=7)
            found = world.find_tasks(lambda t: t.due is not None and world.today <= t.due <= end)
            what = "due this week"
        else:
            found = world.due_today() + world.overdue()
            what = "due today or overdue"
        world.read("tasks", what)
        if not found:
            return AppReply(f"No tasks {what}.", [_tool("list_tasks", filter=what)])
        lines = [f"{len(found)} task{'s' if len(found) != 1 else ''} {what}:"]
        for task in found:
            due = f", due {task.due.isoformat()}" if task.due else ""
            notes = f" — note: {task.notes}" if task.notes else ""
            lines.append(f"- {task.title} (#{task.id}, {task.project}{due}){notes}")
        return AppReply(
            "\n".join(lines),
            [_tool("list_tasks", filter=what)],
            found=[_looked_up("list_tasks", [_task_ref(task) for task in found])],
        )

    # writes

    def _create(self, thread: int, message: str, context: str = "") -> AppReply:
        items = [
            line.strip(" -*\t")
            for line in message.splitlines()
            if re.match(r"\s*(?:[-*]|\d+[.)])\s", line)
        ]
        items = [re.sub(r"^\d+[.)]\s*", "", item) for item in items] or [message]
        project = self._project_in(message)
        made: list[Task] = []
        for item in items:
            title = self._title(item)
            if not title:
                continue
            # A request with no date of its own takes the day conductor's
            # context names ("I sort music on Sundays"), as an agent would.
            due = parse_date(item, self.world.today) or _context_day(context, self.world.today)
            task = Task(max(self.world.tasks) + 1, title, project or "Inbox", due)
            self.world.tasks[task.id] = task
            self.world.record(
                "tasks",
                "create",
                f"task:{task.id}",
                title,
                due=due.isoformat() if due else None,
                project=task.project,
            )
            made.append(task)
        if not made:
            return AppReply("What should the task be called?")
        self._last_created[thread] = [task.id for task in made]
        lines = [
            f'Created task "{task.title}" (#{task.id}) in {task.project}'
            + (f", due {task.due.isoformat()}" if task.due else ", no due date")
            + "."
            for task in made
        ]
        return AppReply(" ".join(lines), [_tool("create_task", title=t.title) for t in made])

    @staticmethod
    def _title(text: str) -> str:
        quoted = re.search(
            r"(?:called|titled|named|task|to-?do|reminder)\s*:?\s*[\"“'‘]([^\"”'’]{3,})[\"”'’]",
            text,
            flags=re.IGNORECASE,
        )
        if quoted:
            return quoted.group(1).strip()
        title = re.sub(
            r"^.*?\b(?:add|create|make|remind me)\b(?:\s+(?:a|an|the|new|another))*"
            r"(?:\s+(?:task|todo|to-do|reminder))?:?(?:\s+(?:to|called|named|titled|for|that says))?\s+",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        title = re.sub(r"\s+(?:to|in|under) (?:the |my )?\w+ project\b.*$", "", title, flags=re.I)
        title = re.sub(r"\s*,?\s*(?:and )?(?:make it |set it )?(?:due|for)\s*$", "", title)
        title = _DATE_PHRASE.sub("", title).replace('"', "").replace("“", "").replace("”", "")
        title = re.sub(r"\s*,?\s*(?:and )?(?:make it |set it )?(?:due|for)\s*$", "", title)
        return title.strip(" .,!;:-").strip()

    def _project_in(self, message: str) -> str | None:
        for name, trashed in self.world.projects.items():
            if not trashed and re.search(rf"\b{name.lower()}\b", message.lower()):
                return name
        return None

    def _changed(self, message: str) -> str:
        """The request with the person's change applied (#77), as PCC's agent
        reads one: a day in the change is the day; a task it names, when the
        request was about a project, is the task ("only the paint cabinets
        task within the Kitchen project")."""
        request, change = split_change(message)
        if not change:
            return message
        day = parse_date(change, self.world.today)
        if day is not None:
            when = f"to {day.isoformat()}"
            replaced = _DAY_WORD.sub(when, request)
            request = replaced if replaced != request else f"{request} {when}"
        named = best_match(change, {t.id: t.title for t in self.world.live_tasks()})
        one_task = re.search(r"\btask\b", change, re.IGNORECASE) or "project" not in change.lower()
        if named is not None and "project" in request.lower() and one_task:
            verb = re.match(r"\W*(\w+)", request)
            title = self.world.tasks[named].title
            request = f"{verb.group(1) if verb else 'delete'} the {title} task"
        return request

    def _move(self, message: str) -> AppReply:
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        due = parse_date(message, self.world.today)
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.", asks=True)
        if due is None:
            return AppReply(f'When should "{self.world.tasks[target].title}" move to?')
        task = self.world.tasks[target]
        old = task.due
        task.due = due
        self.world.record(
            "tasks",
            "update",
            f"task:{task.id}",
            task.title,
            due=due.isoformat(),
            was=old.isoformat() if old else None,
        )
        return AppReply(
            f'Moved "{task.title}" (#{task.id}) to {due.isoformat()} '
            f"(was {old.isoformat() if old else 'unscheduled'}).",
            [_tool("update_task", task_id=task.id, due_date=due.isoformat())],
        )

    def _moves(self, lines: list[str]) -> AppReply:
        """Several moves in one request, one per line: each is its own move."""
        replies = [self._move(line) for line in lines]
        return AppReply(
            " ".join(reply.text for reply in replies),
            [tool for reply in replies for tool in reply.tools],
        )

    def _complete(self, message: str) -> AppReply:
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.", asks=True)
        task = self.world.tasks[target]
        task.done = True
        self.world.record("tasks", "complete", f"task:{task.id}", task.title)
        return AppReply(
            f'Marked "{task.title}" (#{task.id}) done.', [_tool("complete_task", task_id=task.id)]
        )

    def _gate(self, thread: int, tool: str, question: str, run: Callable[[], AppReply]) -> AppReply:
        token = f"tok-{thread}-{len(self.world.receipts)}-{tool}"
        self._armed[thread] = (token, run)
        return AppReply(
            question,
            [ToolCallRead(tool=tool, arguments={}, error="not run: needs confirmation")],
            stop_reason="confirm_required",
            confirm=ConfirmRequired(token=token, tool=tool, question=question),
        )

    def _trash(self, thread: int, message: str) -> AppReply:
        project = self._project_in(message) if "project" in message.lower() else None
        if project is not None:
            count = len([t for t in self.world.live_tasks() if t.project == project])
            return self._gate(
                thread,
                "trash_project",
                f"Trash {project} and its {count} tasks?",
                lambda: self._trash_project(project),
            )
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.", asks=True)
        title = self.world.tasks[target].title
        return self._gate(
            thread,
            "trash_task",
            f'Trash "{title}"?',
            lambda: AppReply(self._trash_task(target), [_tool("delete_task", task_id=target)]),
        )

    def _trash_project(self, project: str) -> AppReply:
        moved = [t for t in self.world.live_tasks() if t.project == project]
        for task in moved:
            task.trashed = True
        self.world.projects[project] = True
        self.world.record(
            "tasks",
            "trash",
            f"project:{project}",
            project,
            tasks=[task.title for task in moved],
        )
        names = ", ".join(task.title for task in moved) or "no tasks"
        return AppReply(
            f"Moved the {project} project to the trash, with its {len(moved)} tasks "
            f"({names}). Restore from the trash within 30 days.",
            [_tool("delete_project", name=project)],
        )

    def _trash_task(self, task_id: int) -> str:
        task = self.world.tasks[task_id]
        task.trashed = True
        self.world.record("tasks", "trash", f"task:{task.id}", task.title)
        return f'Moved "{task.title}" (#{task.id}) to the trash.'

    def _undo(self, thread: int) -> AppReply:
        made = [i for i in self._last_created.get(thread, []) if not self.world.tasks[i].trashed]
        if not made:
            return AppReply("There's nothing I just did in this conversation to undo.")

        def run() -> AppReply:
            lines = [self._trash_task(task_id) for task_id in made]
            self._last_created[thread] = []
            return AppReply(" ".join(lines), [_tool("delete_task", task_id=i) for i in made])

        names = ", ".join(f'"{self.world.tasks[i].title}"' for i in made)
        return self._gate(thread, "trash_task", f"Trash {names}?", run)


class MusicApp:
    """Music's shape: download songs, albums and playlists; sort the library
    into folders; report on download jobs."""

    name = "music"

    def __init__(self, world: World) -> None:
        self.world = world

    def handle(self, thread: int, message: str, context: dict[str, Any] | None = None) -> AppReply:
        message, _ = split_context(message)
        text = message.lower()
        if re.match(r"\s*(?:please\s+)?(?:download|queue|fetch|rip)\b", text):
            # A request that opens with a download is one, whatever the title
            # it carries says ("download the album “sort Honey”").
            return self._download(message, context)
        if re.search(r"\b(how are|status|progress|how's|how is)\b.*\bdownload", text):
            return self._status()
        if re.search(r"\b(need|needs|still|left|unsorted)\b.*\bsort", text) or re.search(
            r"\bwhat (?:songs|tracks)\b.*\bsort", text
        ):
            return self._unsorted()
        if re.search(r"\bfolder\b", text) and re.search(r"\b(put|move|file|sort|add)\b", text):
            return self._file_one(message)
        if re.search(r"\b(sort|organi[sz]e)\b", text):
            return self._sort_all()
        if re.search(r"\b(cancel|stop|scrap|undo|remove|delete|pull)\b", text) and re.search(
            r"\b(download|album|playlist|queue)", text
        ):
            # Downloads land in seconds: by the time anyone asks, it's on disk.
            self.world.read("music", "jobs")
            return AppReply(
                "That download already finished — I can't un-download it. "
                "The files are in the library if you want to delete them yourself."
            )
        if re.search(r"\b(download|grab|get|queue|fetch|save|rip)\b", text):
            return self._download(message, context)
        self.world.read("music", "library")
        return AppReply(
            f"The library has {len(self.world.songs)} songs; {len(self.world.unsorted())} "
            "still need sorting. I can download songs, albums and playlists, and sort."
        )

    def _status(self) -> AppReply:
        self.world.read("music", "jobs")
        if not self.world.jobs:
            return AppReply("No downloads queued.")
        lines = [
            f'Job {job.id}: {job.kind} "{job.title}" — {job.state}, {job.done} of {job.total} '
            f"downloaded" + (f", {job.failed} didn't match" if job.failed else "")
            for job in self.world.jobs.values()
        ]
        # As real music reads its jobs (music#100, music#106): an album job
        # touches its album, then the job, and both titles are outside text
        # (MusicBrainz's), so the entry is flagged untrusted.
        refs: list[EntityRef] = []
        for j in self.world.jobs.values():
            album = j.kind == "album"
            if album and j.album_id:
                refs.append(
                    EntityRef(
                        app="music", kind="album", id=j.album_id, title=j.title, untrusted=True
                    )
                )
            refs.append(
                EntityRef(
                    app="music", kind="download_job", id=str(j.id), title=j.title, untrusted=album
                )
            )
        read = _looked_up("list_jobs", refs)
        read.untrusted = any(ref.untrusted for ref in refs)
        return AppReply("\n".join(lines), [_tool("list_jobs")], found=[read])

    def _unsorted(self) -> AppReply:
        self.world.read("music", "unsorted")
        songs = self.world.unsorted()
        names = "; ".join(f"{s.title} by {s.artist}" for s in songs)
        return AppReply(
            f"{len(songs)} songs still need sorting: {names}.",
            [_tool("list_unsorted")],
            found=[_looked_up("list_unsorted", [_song_ref(song) for song in songs])],
        )

    def _resolve_song(self, message: str) -> list[Song]:
        text = message.lower()
        by_title = [
            s
            for s in self.world.songs.values()
            if re.search(rf"\b{re.escape(s.title.lower())}\b", text)
        ]
        if by_title:
            return by_title
        return [s for s in self.world.songs.values() if s.artist.lower() in text]

    def _file_one(self, message: str) -> AppReply:
        folder_match = re.search(r"\b(?:the |my |a )?([a-z][a-z \-]*?) folder\b", message.lower())
        folder = folder_match.group(1).split()[-1] if folder_match else None
        songs = self._resolve_song(message)
        if not folder:
            return AppReply("Which folder should it go in?", asks=True)
        if len(songs) != 1:
            if not songs:
                return AppReply("I can't find that song in the library.")
            names = " or ".join(f'"{s.title}"' for s in songs)
            return AppReply(f"Which one do you mean — {names}? Nothing moved yet.")
        song = songs[0]
        song.folder = folder
        self.world.record("music", "file", f"song:{song.id}", song.title, folder=folder)
        return AppReply(
            f'Filed "{song.title}" by {song.artist} under {folder}.',
            [_tool("file_song", song_id=song.id, folder=folder)],
        )

    def _sort_all(self) -> AppReply:
        self.world.cancel_requested["music"] = False
        filed: list[Song] = []
        tools: list[ToolCallRead] = []
        for song in list(self.world.unsorted()):
            if self.world.cancel_requested.get("music"):
                break
            song.folder = song.genre
            filed.append(song)
            tools.append(_tool("file_song", song_id=song.id, folder=song.genre))
            self.world.record("music", "file", f"song:{song.id}", song.title, folder=song.genre)
        left = len(self.world.unsorted())
        names = "; ".join(f'"{s.title}" → {s.folder}' for s in filed)
        if self.world.cancel_requested.get("music"):
            self.world.record("music", "cancel", "sort", "sort", filed=len(filed))
            return AppReply(
                f"Stopped after filing {len(filed)} songs ({names}); {left} left unsorted. "
                "Each move can be undone.",
                tools,
                stop_reason="cancelled",
            )
        return AppReply(f"Sorted {len(filed)} songs: {names}.", tools)

    def _download(self, message: str, context: dict[str, Any] | None = None) -> AppReply:
        text = message.lower()
        if re.search(r"https?://\S*(?:playlist|list=)", text) or "playlist" in text:
            return self._queue_job("playlist", "your playlist", 40)
        if re.search(r"https?://", text):
            return self._add_song(Song(0, "Real Title", "Unknown Artist", "Singles", "unsorted"))
        for album in self.world.catalog:
            for track in album.tracks:
                if (
                    re.search(rf"\b{re.escape(track.lower())}\b", text)
                    and (album.artist.lower() in text or "album" not in text)
                    and track.lower() != album.title.lower()
                ):
                    return self._add_song(Song(0, track, album.artist, album.title, album.genre))
        artist_albums = [a for a in self.world.catalog if a.artist.lower() in text]
        # A quoted title (conductor passes names on quoted) is matched whole,
        # as music's lookup matches a title; bare words match a title in them.
        quoted = {q.strip().lower() for q in re.findall(r"[“\"]([^”\"]+)[”\"]", message)}
        named = [
            a
            for a in self.world.catalog
            if (a.title.lower() in quoted)
            or (not quoted and re.search(rf"\b{re.escape(a.title.lower())}\b", text))
        ]
        if quoted and not named and not artist_albums:
            return AppReply("I couldn't find that. Give me a link, or the artist and title.")
        # A v1.2 caller's album ref (#196): "their other album" is music's
        # model passing that album's id as ``other_than`` (music#111).
        mine = [
            ref
            for ref in (context or {}).get("refs") or []
            if ref.get("app") == "music" and ref.get("kind") == "album"
        ]
        # The person's pick from music's own list (#196): music queues that
        # album by id (music#112's ``album_id``), no title read.
        picked = next((r for r in mine if r.get("picked")), None)
        if picked is not None:
            chosen = next((a for a in self.world.catalog if a.mbid == picked.get("id")), None)
            if chosen is None:
                return AppReply("No album with that id, so nothing was queued.")
            return self._queue_job(
                "album",
                chosen.title,
                len(chosen.tracks),
                artist=chosen.artist,
                album_id=chosen.mbid,
            )
        # A title it was given still names its album, as music's
        # ``download_album`` takes ``album`` over ``other_than``: the person's
        # answer to its question ("Suddenly") is that.
        if mine and len(named) != 1 and re.search(r"\b(other|another|different)\b", text):
            return self._other_than(mine[0], artist_albums)
        # Real music's ``download_album(artist, album=None)`` (music#96, #102):
        # a title it was given names its album, held or not; the artist alone
        # is their newest studio album ("the new album" and "the other album"
        # are the same call), unless music already has that one, and then it
        # refuses and lists the others, for the person to pick.
        if len(named) == 1:
            album = named[0]
        elif artist_albums:
            album = max(artist_albums, key=lambda a: a.year)
            if album.mbid in self.world.held_album_ids():
                return self._album_held(album, artist_albums)
        else:
            return AppReply("I couldn't find that. Give me a link, or the artist and title.")
        return self._queue_job(
            "album", album.title, len(album.tracks), artist=album.artist, album_id=album.mbid
        )

    def _album_held(self, album: Album, artist_albums: list[Album]) -> AppReply:
        """Music's refusal for an artist-alone ask whose newest album is held
        (music#102, music#107): nothing queued; the artist's other studio
        albums not held, newest first, at most three, as ``candidates``."""
        held = self.world.held_album_ids()
        others = sorted(
            (a for a in artist_albums if a.mbid not in held), key=lambda a: a.year, reverse=True
        )[:MUSIC_MAX_CANDIDATES]
        have = f"{album.title} ({album.year}) by {album.artist}"
        if others:
            many = "album" if len(others) == 1 else "albums"
            reason = (
                f"{have} was already asked for, so nothing was queued. "
                f"MusicBrainz lists {len(others)} other {many} by {album.artist}."
            )
        else:
            reason = (
                f"{have} was already asked for, so nothing was queued. "
                f"MusicBrainz lists no other studio album by {album.artist}."
            )
        return self._album_refusal(album.artist, reason, others)

    def _other_than(self, ref: dict[str, Any], artist_albums: list[Album]) -> AppReply:
        """Music's ``download_album(other_than=…)`` (music#111, conductor#145
        decision 3): the one studio album of that album's artist left that
        isn't it and isn't held is queued; several are a refusal with them as
        candidates; none is a plain no."""
        that = next((a for a in self.world.catalog if a.mbid == ref.get("id")), None)
        if that is not None:
            artist = that.artist
        elif artist_albums:
            artist = artist_albums[0].artist
        else:
            return AppReply("I couldn't find that. Give me a link, or the artist and title.")
        held = self.world.held_album_ids()
        left = sorted(
            (
                a
                for a in self.world.catalog
                if a.artist == artist and a.mbid != ref.get("id") and a.mbid not in held
            ),
            key=lambda a: a.year,
            reverse=True,
        )
        if len(left) == 1:
            album = left[0]
            return self._queue_job(
                "album", album.title, len(album.tracks), artist=album.artist, album_id=album.mbid
            )
        if left:
            reason = (
                f"{artist} has {len(left)} other albums music doesn't have, so nothing was queued."
            )
        else:
            reason = (
                f"{artist} has no other studio album music doesn't have, so nothing was queued."
            )
        return self._album_refusal(artist, reason, left[:MUSIC_MAX_CANDIDATES])

    def _album_refusal(self, artist: str, reason: str, others: list[Album]) -> AppReply:
        """A refused ``download_album`` as real music answers an agent caller
        (music#97, music#107): its record, with a short ``reason`` and the
        albums offered as ``candidates`` (outside text), then code's note,
        which ends on its question when there is something to pick."""
        self.world.read("music", "albums")
        candidates = [
            EntityRef(app="music", kind="album", id=a.mbid, title=a.title, untrusted=True)
            for a in others
        ]
        note = reason[:1].upper() + reason[1:]
        if candidates:
            note += " Albums listed: " + "; ".join(f"“{a.title}”" for a in others) + "."
            note += f" {MUSIC_CANDIDATES_QUESTION}"
        args = json.dumps({"artist": artist}, separators=(",", ":"))
        text = (
            "Music's record of this run (built by the app from its tool results; nothing "
            'outside "Done" was saved or changed):\nDone:\n- nothing\nRefused:\n'
            f"- #1 download_album {args} -> {_music_clip(reason)}\n\n"
            f"The app's note (built from the record, not the model's words): \"{note}\""
        )
        call = ToolCallRead(tool="download_album", arguments={"artist": artist}, error=reason)
        refused = RecordEntry(
            tool="download_album",
            outcome="refused",
            reason=_music_clip(reason),
            untrusted=bool(candidates),
            candidates=candidates,
        )
        return AppReply(
            text,
            [call],
            found=[refused],
            asks=bool(candidates),
            question=MUSIC_CANDIDATES_QUESTION if candidates else None,
        )

    def _queue_job(self, kind: str, title: str, total: int, **detail: Any) -> AppReply:
        job = Job(len(self.world.jobs) + 1, kind, title, total, album_id=detail.get("album_id"))
        self.world.jobs[job.id] = job
        self.world.record(
            "music", "queue", f"job:{job.id}", title, job_kind=kind, total=total, **detail
        )
        by = f" by {detail['artist']}" if "artist" in detail else ""
        return AppReply(
            f'Queued the {kind} "{title}"{by}: {total} tracks, job {job.id}. '
            "I'll have it in a few minutes.",
            [_tool("queue_download", kind=kind, title=title)],
        )

    def _add_song(self, song: Song) -> AppReply:
        song.id = max(self.world.songs) + 1
        self.world.songs[song.id] = song
        self.world.record(
            "music", "queue", f"song:{song.id}", song.title, job_kind="song", artist=song.artist
        )
        return AppReply(
            f'Saved "{song.title}" by {song.artist} to the library (unsorted).',
            [_tool("download_song", title=song.title)],
        )


# --- the delegate seam -----------------------------------------------------------


class FakeDelegateClient:
    """``DelegateClientLike`` over the world's apps.

    ``cancel`` mirrors the apps' ``…/cancel`` (PCC and music expose one; #66
    passes conductor's Stop on to it): it raises the app's cancel flag, which
    a long write loop (music's sort) checks between steps.
    """

    def __init__(self, app: TasksApp | MusicApp | None, app_name: str, world: World) -> None:
        self._app = app
        self._app_name = app_name
        self._world = world

    def __enter__(self) -> FakeDelegateClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        used = self._world.threads.setdefault(f"{self._app_name}:created", [])
        thread_id = len(used) + 1
        used.append(thread_id)
        return thread_id

    def send_message(
        self,
        conversation_id: int,
        message: str,
        *,
        confirm_token: str | None = None,
        context: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> MessageExchange:
        self._world.threads.setdefault(self._app_name, []).append(conversation_id)
        self._world.sent.append((self._app_name, message))
        self._world.sent_v12.append((self._app_name, context, mode))
        before = len(self._world.receipts)
        saved = self._world.writable() if mode == "read" else None
        if self._app is None:
            reply = AppReply("Done — nothing else to report.")
        elif isinstance(self._app, TasksApp):
            reply = self._app.handle(conversation_id, message, confirm_token)
        else:
            reply = self._app.handle(conversation_id, message, context)
        if saved is not None and len(self._world.receipts) > before:
            # Read mode (#202): the real apps stop a write before their service
            # layer, so here it never happened: the world is put back and the
            # write is ``not_run``, as their records say.
            undone = self._world.receipts[before:]
            self._world.restore(saved, before)
            reply = AppReply(
                "Nothing was changed: this message was read-only.",
                found=[
                    *reply.found,
                    *(
                        RecordEntry(
                            tool=_write_entry(r).tool,
                            outcome="not_run",
                            reason="not run: read mode",
                        )
                        for r in undone
                    ),
                ],
            )
        now = datetime.now(UTC)
        return MessageExchange(
            user_message=MessageRead(
                id=1,
                conversation_id=conversation_id,
                role="user",
                content=message,
                created_at=now,
            ),
            assistant_message=MessageRead(
                id=2,
                conversation_id=conversation_id,
                role="assistant",
                content=reply.text,
                tool_calls=reply.tools or None,
                stop_reason=reply.stop_reason,
                created_at=now,
                confirm_required=reply.confirm,
                record=_record(reply, self._world.receipts[before:]),
                needs_input=_needs_input(reply, self._world.receipts[before:]),
            ),
        )

    def cancel(self, conversation_id: int) -> bool:
        self._world.cancel_requested[self._app_name] = True
        return True

    def delete_conversation(self, conversation_id: int) -> None:
        return None


def client_factory(world: World) -> Callable[[Any], FakeDelegateClient]:
    """A ``ClientFactory`` over ``world``: one app instance per app for the
    whole session, so app-side thread memory ("undo that") survives turns."""
    apps: dict[str, TasksApp | MusicApp] = {"tasks": TasksApp(world), "music": MusicApp(world)}

    def factory(fleet_app: Any) -> FakeDelegateClient:
        return FakeDelegateClient(apps.get(fleet_app.name), fleet_app.name, world)

    return factory


def titles(receipts: Iterable[Receipt]) -> list[str]:
    return [receipt.title for receipt in receipts]


# --- glances -----------------------------------------------------------------------


class FakeGlances:
    """A ``GlanceReader`` (``app.fleet.glance``) over canned glances: what each
    app's endpoint would answer, or the failure it would give. Records every
    app it was asked for, so a test can see a look read and an ask not made."""

    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, Any]] = {
            "tasks": {
                "as_of": "2026-11-07T09:00:00Z",
                "due_today": [
                    {"app": "tasks", "kind": "task", "id": "41", "title": "Pay rent"},
                    {"app": "tasks", "kind": "task", "id": "42", "title": "Water the plants"},
                ],
                "overdue": 2,
                "focus": {"app": "tasks", "kind": "task", "id": "44", "title": "Call Sam"},
                "next_due": "2026-11-08",
            },
            "chess": {"as_of": "2026-11-07T09:00:00Z", "last_game": {"result": "you won"}},
        }
        self.failures: dict[str, Exception] = {}
        self.asked: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, fleet_app: Any) -> dict[str, Any]:
        with self._lock:
            self.asked.append(fleet_app.name)
        if fleet_app.name in self.failures:
            raise self.failures[fleet_app.name]
        return self.payloads[fleet_app.name]


# --- a fleet to discover ---------------------------------------------------------

# The fleet the scripted run discovers: the three apps the scenarios use, in
# the shapes the real manifests have (tasks and music delegate, chess opens).
FIXTURE_MANIFESTS = {
    "project-command-center": """\
        name: tasks
        title: Project Command Center
        upstream: 127.0.0.1:8100
        agent:
          description: "Manages projects and tasks."
          api: /api/agent
          confirms: true
          examples: ["what's due today"]
        glance:
          holds: "Today's and overdue tasks, the Focus plan, and the next due date."
        """,
    "music": """\
        name: music
        title: Music
        upstream: 127.0.0.1:8500
        agent:
          description: "Downloads songs and sorts the library."
          api: /api/agent
          examples: ["download this song"]
        glance:
          holds: "Running and recent download jobs with progress, and how many songs wait to be sorted."
        """,
    "chess": """\
        name: chess
        title: Chess
        upstream: 127.0.0.1:8000
        open:
          description: "Opens the chess board."
          path: /
          intent_param: intent
          examples: ["let's play chess"]
        glance:
          holds: "Where the current game stands (whose move, move count) and the last finished game's result; it doesn't analyze positions."
        """,
}


def write_fixture_fleet(root: Path) -> Path:
    for directory, manifest in FIXTURE_MANIFESTS.items():
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "app.yaml").write_text(textwrap.dedent(manifest), encoding="utf-8")
    return root
