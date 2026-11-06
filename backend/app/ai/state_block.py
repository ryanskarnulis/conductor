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
- what is waiting on the person: a question the last reply asked;
- recent entities: what this conversation's calls touched, from the apps'
  own records (delegate-api v1.1), newest first. Outside text is quoted and
  marked, never an instruction.

Conductor has no plans yet; "the last plan" joins the block when compiled
plans do (#68). Pure: no session, no I/O.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from app.ai.loop import ToolCallRecord

Surface = Literal["desktop", "phone", "voice", "room_speaker"]

MAX_ENTITIES = 10
MAX_JOBS = 5
MAX_TITLE_CHARS = 60
MAX_QUESTION_CHARS = 300
# The block is never longer than this (asserted in the tests).
MAX_CHARS = 2400

_HEADER = "[State at the start of this turn — written by conductor, not the person]"
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


def render(today: date, now: Now, history: Sequence[dict[str, Any]]) -> str:
    """The block for a turn starting ``today`` on a conversation with
    ``history`` (stored turns, oldest first, as :func:`app.ai.recall.Recall.of`
    takes them)."""
    lines = [_HEADER, *calendar_lines(today)]
    if now.surface is not None:
        lines.append(_SURFACE_LINES[now.surface])
    if now.panel:
        lines.append(f"Open panel: {_clip(now.panel)}.")
    records = _records(history)
    jobs = _running_jobs(records)
    if jobs:
        lines.append("Running jobs (as of each one's last report):")
        lines += [f"- {job}" for job in jobs]
    question = _waiting(history)
    if question:
        lines.append(f'Waiting on the person: your last reply asked "{question}"')
    entities = _entities(records)
    if entities:
        lines.append("Recent entities (newest first, as of then):")
        lines += [f"- {entity}" for entity in entities]
    lines.append(_FOOTER)
    return "\n".join(lines)


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
        app = record.tool.removeprefix("ask_")
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
    question = reply[: reply.rfind("?") + 1]
    return question if len(question) <= MAX_QUESTION_CHARS else "…" + question[-MAX_QUESTION_CHARS:]


def _entities(records: Sequence[ToolCallRecord]) -> list[str]:
    seen: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    for record in reversed(records):
        for entry in reversed(record.app_record or []):
            if entry.get("outcome") not in ("done", "looked_up"):
                continue
            for ref in entry.get("touched") or []:
                key = (str(ref.get("app")), str(ref.get("kind")), str(ref.get("id")))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(_entity(ref, untrusted=bool(entry.get("untrusted"))))
                if len(lines) == MAX_ENTITIES:
                    return lines
    return lines


def _entity(ref: dict[str, Any], *, untrusted: bool) -> str:
    head = f"{ref.get('app')} {ref.get('kind')} {ref.get('id')}"
    title = ref.get("title")
    if not title:
        return head
    if untrusted or ref.get("untrusted"):
        return f'{head}: "{_clip(str(title))}" (outside text)'
    return f"{head}: {_clip(str(title))}"


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_TITLE_CHARS else flat[: MAX_TITLE_CHARS - 1] + "…"
