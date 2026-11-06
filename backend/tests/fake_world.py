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
import re
import textwrap
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.fleet.delegate import MessageExchange, MessageRead, ToolCallRead

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


@dataclass
class Job:
    id: int
    kind: str  # "album" | "playlist" | "song"
    title: str
    total: int
    done: int = 0
    failed: int = 0
    state: str = "queued"  # queued | finished | cancelled


@dataclass
class ChessGame:
    result: str
    summary: str


class World:
    """Every app's state, the receipts of every write, and the threads used.

    ``today`` anchors the seeded dates (the loop's prompt reads the real date,
    so the world must too). ``turn`` is advanced by the harness so receipts
    know which turn wrote them. ``on_write`` is the Stop seam: called after
    every receipt, it lets a trajectory or a Stop scenario act mid-delegate.
    """

    def __init__(self, today: date | None = None) -> None:
        self.today = today or date.today()
        self.turn = 0
        self.receipts: list[Receipt] = []
        self.reads: list[tuple[str, str, int]] = []  # (app, what, turn)
        self.threads: dict[str, list[int]] = {}  # app → conversation id per send
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

    def read(self, app: str, what: str) -> None:
        self.reads.append((app, what, self.turn))

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
                job.done = job.total - (2 if job.kind == "playlist" else 0)
                job.failed = job.total - job.done
                job.state = "finished"


# --- the apps ----------------------------------------------------------------------


@dataclass
class AppReply:
    text: str
    tools: list[ToolCallRead] = field(default_factory=list)


def _tool(tool_name: str, **arguments: Any) -> ToolCallRead:
    return ToolCallRead(tool=tool_name, arguments=arguments, result="ok")


class TasksApp:
    """PCC's shape: read, create, move, complete and trash tasks and projects."""

    name = "tasks"

    def __init__(self, world: World) -> None:
        self.world = world
        self._last_created: dict[int, list[int]] = {}  # thread → task ids it created

    def handle(self, thread: int, message: str) -> AppReply:
        text = message.lower()
        leads_with_create = re.match(
            r"\W*(?:please\s+|can you\s+|could you\s+)?(?:add|create|make|remind me)\b", text
        ) or re.search(r"^\s*(?:[-*]|\d+[.)])\s", message, flags=re.M)
        if leads_with_create:
            return self._create(thread, message)
        if re.search(r"\b(undo|revert|take (?:that|it) back)\b", text):
            return self._undo(thread)
        if re.search(r"\b(delete|trash|remove|wipe|bin|get rid of)\b", text):
            return self._trash(message)
        if re.search(r"\b(push|move|reschedule|postpone|bump|shift|change the due)\b", text):
            return self._move(message)
        if re.search(r"\b(mark|complete|finish|tick off|check off)\b", text) and "done" in text:
            return self._complete(message)
        if re.search(r"\b(add|create|make|new task|remind me)\b", text) and not (
            re.search(r"\?\s*$", text) or text.startswith(("what", "which", "how", "is ", "are "))
        ):
            return self._create(thread, message)
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
        return AppReply("\n".join(lines), [_tool("list_tasks", filter=what)])

    # writes

    def _create(self, thread: int, message: str) -> AppReply:
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
            due = parse_date(item, self.world.today)
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

    def _move(self, message: str) -> AppReply:
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        due = parse_date(message, self.world.today)
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.")
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

    def _complete(self, message: str) -> AppReply:
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.")
        task = self.world.tasks[target]
        task.done = True
        self.world.record("tasks", "complete", f"task:{task.id}", task.title)
        return AppReply(
            f'Marked "{task.title}" (#{task.id}) done.', [_tool("complete_task", task_id=task.id)]
        )

    def _trash(self, message: str) -> AppReply:
        project = self._project_in(message) if "project" in message.lower() else None
        if project is not None:
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
        target = best_match(message, {t.id: t.title for t in self.world.live_tasks()})
        if target is None:
            return AppReply("Which task do you mean? I couldn't tell from that.")
        return AppReply(self._trash_task(target), [_tool("delete_task", task_id=target)])

    def _trash_task(self, task_id: int) -> str:
        task = self.world.tasks[task_id]
        task.trashed = True
        self.world.record("tasks", "trash", f"task:{task.id}", task.title)
        return f'Moved "{task.title}" (#{task.id}) to the trash.'

    def _undo(self, thread: int) -> AppReply:
        made = [i for i in self._last_created.get(thread, []) if not self.world.tasks[i].trashed]
        if not made:
            return AppReply("There's nothing I just did in this conversation to undo.")
        lines = [self._trash_task(task_id) for task_id in made]
        self._last_created[thread] = []
        return AppReply(" ".join(lines), [_tool("delete_task", task_id=i) for i in made])


class MusicApp:
    """Music's shape: download songs, albums and playlists; sort the library
    into folders; report on download jobs."""

    name = "music"

    def __init__(self, world: World) -> None:
        self.world = world

    def handle(self, thread: int, message: str) -> AppReply:
        text = message.lower()
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
            return self._download(message)
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
        return AppReply("\n".join(lines), [_tool("list_jobs")])

    def _unsorted(self) -> AppReply:
        self.world.read("music", "unsorted")
        songs = self.world.unsorted()
        names = "; ".join(f"{s.title} by {s.artist}" for s in songs)
        return AppReply(
            f"{len(songs)} songs still need sorting: {names}.", [_tool("list_unsorted")]
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
            return AppReply("Which folder should it go in?")
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
            )
        return AppReply(f"Sorted {len(filed)} songs: {names}.", tools)

    def _download(self, message: str) -> AppReply:
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
        named = [
            a for a in self.world.catalog if re.search(rf"\b{re.escape(a.title.lower())}\b", text)
        ]
        if named:
            album = named[0]
        elif artist_albums and re.search(r"\b(new|newest|latest|recent)\b", text):
            album = max(artist_albums, key=lambda a: a.year)
        elif artist_albums and re.search(r"\b(other|another)\b", text):
            queued = {job.title for job in self.world.jobs.values()}
            rest = [a for a in artist_albums if a.title not in queued and a.year >= 2015]
            if len(rest) != 1:
                titles = ", ".join(a.title for a in artist_albums)
                return AppReply(f"Which album? {artist_albums[0].artist} has: {titles}.")
            album = rest[0]
        elif artist_albums:
            titles = ", ".join(f"{a.title} ({a.year})" for a in artist_albums)
            return AppReply(f"Which one? {artist_albums[0].artist} has: {titles}.")
        else:
            return AppReply("I couldn't find that. Give me a link, or the artist and title.")
        return self._queue_job("album", album.title, len(album.tracks), artist=album.artist)

    def _queue_job(self, kind: str, title: str, total: int, **detail: Any) -> AppReply:
        job = Job(len(self.world.jobs) + 1, kind, title, total)
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

    def send_message(self, conversation_id: int, message: str) -> MessageExchange:
        self._world.threads.setdefault(self._app_name, []).append(conversation_id)
        if self._app is None:
            reply = AppReply("Done — nothing else to report.")
        else:
            reply = self._app.handle(conversation_id, message)
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
                stop_reason="completed",
                created_at=now,
            ),
        )

    def cancel(self, conversation_id: int) -> None:
        self._world.cancel_requested[self._app_name] = True

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
          examples: ["what's due today"]
        """,
    "music": """\
        name: music
        title: Music
        upstream: 127.0.0.1:8500
        agent:
          description: "Downloads songs and sorts the library."
          api: /api/agent
          examples: ["download this song"]
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
        """,
}


def write_fixture_fleet(root: Path) -> Path:
    for directory, manifest in FIXTURE_MANIFESTS.items():
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "app.yaml").write_text(textwrap.dedent(manifest), encoding="utf-8")
    return root
