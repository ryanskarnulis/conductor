"""The state block: what conductor is shown about now, every turn (#64).

Ported from PCC's ``app/ai/state_block.py`` (#412): code does the date math
and the bookkeeping, the model reads the answer. Built once at the start of a
turn and put at the head of the turn's user message, after the system prompt
and tool schemas (which stay byte-identical across turns, so the server's
prompt cache keeps them), and never stored: the conversation keeps what the
person typed, and the next turn gets a fresh block.

It holds, each line only when there is something to say:

- the calendar: today, tomorrow and the next six days with weekdays, next
  Monday and the month's end;
- the surface and device the person is on (``voice`` → one or two short
  spoken sentences, no markdown) and the app panel they have open;
- the jobs an app reported as still running, as of their last report;
- what is waiting on the person: a question the last reply asked, or an
  app's question that a yes answers with ``confirm`` (#68);
- a reply the person spoke over (#77): how much of it they heard, and the
  start of what they didn't, as data;
- recent entities: what this conversation's calls touched, newest first,
  from the ledger (#69) when the caller passes them, else from the apps' own
  records on the stored turns (delegate-api v1.1). Outside text is quoted and
  marked, never an instruction;
- matches (#80): things from earlier conversations the entity registry finds
  for the person's words when they point back at something ("that Caribou
  song"), each with the day and the person's words when it was last touched.
  Leads, as of then: the same quoting rules;
- facts you saved (#81): what the person asked conductor to remember, in
  their own words, quoted, the ones that share words with the message first,
  then the newest (capped).

Pure: no session, no I/O.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from app.ai import confirmation
from app.ai.runtime import ToolCallRecord

Surface = Literal["desktop", "phone", "voice", "room_speaker"]

MAX_ENTITIES = 10
MAX_JOBS = 5
MAX_TITLE_CHARS = 60
MAX_QUESTION_CHARS = 300
MAX_MATCHES = 3
MAX_SAID_CHARS = 100
# The matches' share of the block: three lines and their heading.
MATCHES_MAX_CHARS = 900
# Facts the person saved (#81): how many, each one's length, and their share.
MAX_FACTS = 8
MAX_FACT_CHARS = 120
FACTS_MAX_CHARS = MAX_FACTS * (MAX_FACT_CHARS + 40) + 90
# A reply the person spoke over (#77): the end of what they heard, the start
# of what they didn't, and the line's share.
HEARD_TAIL_CHARS = 80
UNHEARD_CHARS = 240
HEARD_MAX_CHARS = HEARD_TAIL_CHARS + UNHEARD_CHARS + 180
# The block is never longer than this (asserted in the tests).
MAX_CHARS = 2400 + MATCHES_MAX_CHARS + FACTS_MAX_CHARS + HEARD_MAX_CHARS

_HEADER = "[State at the start of this turn — written by conductor, not the person]"
MATCHES_HEADING = "Matches for words in the message, from earlier conversations (as of then):"
_FOOTER = "[End of state]"

_SURFACE_LINES: dict[str, str] = {
    "desktop": "Surface: desktop (the person reads your reply on a screen).",
    "phone": "Surface: phone (a small screen: keep it short).",
    "voice": (
        "Surface: voice (your reply is spoken aloud: one or two short sentences, "
        "no markdown, no lists)."
    ),
    "room_speaker": (
        "Surface: room speaker (your reply is spoken aloud to the room: one or two "
        "short sentences, no markdown, no lists)."
    ),
}

_FINISHED_JOB = frozenset({"completed", "failed", "canceled"})


@dataclass(frozen=True)
class Entity:
    """One recently touched thing, with its title as of the receipt that touched it."""

    app: str
    kind: str
    id: str
    title: str | None = None
    untrusted: bool = False


@dataclass(frozen=True)
class Match:
    """An entity from an earlier conversation that shares the person's words
    (#80), with the day and the person's words on the last turn that touched it."""

    entity: Entity
    day: str
    said: str = ""


@dataclass(frozen=True)
class Fact:
    """Something the person asked conductor to remember (#81), in their words,
    with the day it was saved."""

    text: str
    day: str


@dataclass(frozen=True)
class Heard:
    """The last reply, cut off when the person spoke over it (#77): its text
    and how many of its characters (UTF-16 units, as the browser counts)
    played first; ``None`` when the browser couldn't tell."""

    reply: str
    heard_chars: int | None


@dataclass(frozen=True)
class Now:
    """What the client told conductor about where the person is."""

    surface: Surface | None = None
    panel: str | None = None


def day(when: date) -> str:
    """``Thursday 2026-11-05``: the weekday is what date math needs."""
    return f"{calendar.day_name[when.weekday()]} {when.isoformat()}"


def calendar_lines(today: date) -> list[str]:
    """Today, tomorrow, the six days after it (so every weekday name has
    exactly one date after today, PCC#484), next Monday and the month's end."""
    tomorrow = today + timedelta(days=1)
    after = [tomorrow + timedelta(days=offset) for offset in range(1, 7)]
    next_monday = today + timedelta(days=7 - today.weekday())
    month_end = today.replace(day=calendar.monthrange(today.year, today.month)[1])
    return [
        f"Today: {day(today)}",
        f"Tomorrow: {day(tomorrow)}",
        "Coming days: " + ", ".join(day(when) for when in after),
        f"Next Monday: {day(next_monday)}",
        f"End of this month: {day(month_end)}",
    ]


def render(
    today: date,
    now: Now,
    history: Sequence[dict[str, Any]],
    entities: Sequence[Entity] | None = None,
    jobs: Sequence[str] | None = None,
    matches: Sequence[Match] = (),
    facts: Sequence[Fact] = (),
    heard: Heard | None = None,
) -> str:
    """The block for a turn starting ``today`` on a conversation with
    ``history`` (stored turns, oldest first, as :func:`app.ai.recall.Recall.of`
    takes them). ``entities``, newest first, are the ledger's; ``None`` reads
    them from ``history``'s records instead. ``jobs`` are the ledger's running
    jobs as the watcher last read them (#76); ``None`` reads each job's last
    report from the records. ``matches`` are the entity registry's leads from
    earlier conversations (#80). ``facts`` are the ones the person saved (#81),
    most relevant first. ``heard`` is the last reply when the person spoke
    over it (#77)."""
    lines = [_HEADER, *calendar_lines(today)]
    if now.surface is not None:
        lines.append(_SURFACE_LINES[now.surface])
    if now.panel:
        lines.append(f"Open panel: {_clip(now.panel)}.")
    records = _records(history)
    jobs = list(jobs[-MAX_JOBS:]) if jobs is not None else _running_jobs(records)
    if jobs:
        lines.append("Running jobs (as of each one's last report):")
        lines += [f"- {job}" for job in jobs]
    asked = confirmation.pending(history)
    question = _waiting(history)
    if asked is not None:
        lines.append(
            f'Waiting on the person: {asked.app} asked "{_clip_question(asked.question)}" '
            f'If their message says yes to it, call confirm(app="{asked.app}"); '
            "if not, don't — the question lapses."
        )
    elif question:
        lines.append(f'Waiting on the person: your last reply asked "{question}"')
    if heard is not None:
        lines.append(cut_off_line(heard))
    recent = list(entities[:MAX_ENTITIES]) if entities is not None else _entities(records)
    if recent:
        lines.append("Recent entities (newest first, as of then):")
        lines += [f"- {_entity(entity)}" for entity in recent]
    if matches:
        lines.append(MATCHES_HEADING)
        lines += [f"- {_match(match)}" for match in matches[:MAX_MATCHES]]
    if facts:
        lines.append("Facts the person asked you to remember (their words, as of when saved):")
        lines += [f"- {_fact(fact)}" for fact in facts[:MAX_FACTS]]
    lines.append(_FOOTER)
    return "\n".join(lines)


CUT_OFF_PREFIX = "Your last reply was cut off: the person spoke over it"


def utf16_index(text: str, units: int) -> int:
    """The index in ``text`` where ``units`` UTF-16 code units (the browser's
    characters) end; past the end is the end."""
    seen = 0
    for index, char in enumerate(text):
        if seen >= units:
            return index
        seen += 2 if ord(char) > 0xFFFF else 1
    return len(text)


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_heard(heard: Heard) -> tuple[str, str] | None:
    """What was heard and what wasn't, flattened, cut at a word: the cut
    is an estimate from playback time, so a half-heard word counts as not
    heard. ``None`` when how much played is unknown."""
    if heard.heard_chars is None:
        return None
    at = utf16_index(heard.reply, heard.heard_chars)
    if 0 < at < len(heard.reply) and not heard.reply[at].isspace():
        space = max(heard.reply.rfind(" ", 0, at), heard.reply.rfind("\n", 0, at))
        at = space + 1 if space >= 0 else 0
    return " ".join(heard.reply[:at].split()), " ".join(heard.reply[at:].split())


def cut_off_line(heard: Heard) -> str:
    """The state block's line for a reply the person spoke over (#77): how
    much of it they heard, and what they didn't, quoted as data (capped)."""
    parts = split_heard(heard)
    if parts is None:
        return f"{CUT_OFF_PREFIX} partway; how much of it they heard is unknown."
    said, unsaid = parts
    if not unsaid:
        return f"{CUT_OFF_PREFIX} at its very end; they heard all of it."
    if len(unsaid) > UNHEARD_CHARS:
        unsaid = unsaid[: UNHEARD_CHARS - 1] + "…"
    if not said:
        return (
            f"{CUT_OFF_PREFIX} before any of it played; they heard none of it "
            f'(your words, not heard: "{unsaid}").'
        )
    if len(said) > HEARD_TAIL_CHARS:
        said = "…" + said[-(HEARD_TAIL_CHARS - 1) :]
    return (
        f'{CUT_OFF_PREFIX}. They heard up to "{said}" and not the rest '
        f'(your words, not heard: "{unsaid}").'
    )


def _records(history: Sequence[dict[str, Any]]) -> list[ToolCallRecord]:
    return [
        ToolCallRecord.model_validate(stored)
        for message in history
        if message["role"] == "assistant"
        for stored in message.get("tool_calls") or []
    ]


def _running_jobs(records: Sequence[ToolCallRecord]) -> list[str]:
    """Each job's latest report, if it hadn't finished."""
    latest: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        app = confirmation.record_app(record.tool, record.arguments) or record.tool
        for entry in record.app_record or []:
            job = entry.get("job")
            if isinstance(job, dict) and job.get("ref") is not None:
                latest[f"{app}:{job['ref']}"] = (app, entry, job)
    lines = []
    for app, entry, job in latest.values():
        if job.get("status") in _FINISHED_JOB:
            continue
        total = job.get("total")
        progress = (
            f"{job.get('done', 0)} of {total}"
            if total is not None
            else f"{job.get('done', 0)} so far"
        )
        lines.append(
            f"{app} {entry.get('tool')} (job {job['ref']}): {job.get('status')}, {progress}"
        )
    return lines[-MAX_JOBS:]


def _waiting(history: Sequence[dict[str, Any]]) -> str | None:
    """The question the last reply asked, if the person hasn't answered yet."""
    if not history or history[-1]["role"] != "assistant":
        return None
    reply = " ".join(str(history[-1].get("content") or "").split())
    if "?" not in reply:
        return None
    return _clip_question(reply[: reply.rfind("?") + 1])


def _clip_question(question: str) -> str:
    question = " ".join(question.split())
    return question if len(question) <= MAX_QUESTION_CHARS else "…" + question[-MAX_QUESTION_CHARS:]


def _entities(records: Sequence[ToolCallRecord]) -> list[Entity]:
    seen: set[tuple[str, str, str]] = set()
    lines: list[Entity] = []
    for record in reversed(records):
        for entry in reversed(record.app_record or []):
            if entry.get("outcome") not in ("done", "looked_up"):
                continue
            for ref in entry.get("touched") or []:
                key = (str(ref.get("app")), str(ref.get("kind")), str(ref.get("id")))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    Entity(
                        app=key[0],
                        kind=key[1],
                        id=key[2],
                        title=ref.get("title"),
                        untrusted=bool(entry.get("untrusted") or ref.get("untrusted")),
                    )
                )
                if len(lines) == MAX_ENTITIES:
                    return lines
    return lines


def _entity(entity: Entity) -> str:
    head = f"{entity.app} {entity.kind} {entity.id}"
    title = entity.title
    if not title:
        return head
    if entity.untrusted:
        return f'{head}: "{_clip(str(title))}" (outside text)'
    return f"{head}: {_clip(str(title))}"


def _match(match: Match) -> str:
    line = f"{_entity(match.entity)}, {match.day}"
    said = " ".join(match.said.split())
    if said:
        if len(said) > MAX_SAID_CHARS:
            said = said[: MAX_SAID_CHARS - 1] + "…"
        line += f', you said "{said}"'
    return line


def _fact(fact: Fact) -> str:
    text = " ".join(fact.text.split())
    if len(text) > MAX_FACT_CHARS:
        text = text[: MAX_FACT_CHARS - 1] + "…"
    return f'"{text}" (saved {fact.day})'


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_TITLE_CHARS else flat[: MAX_TITLE_CHARS - 1] + "…"
