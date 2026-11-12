"""Side threads (#139): a small conversation started from the timeline, closed back into it.

A side thread is a ``conversations`` row with ``kind = thread``. The person
starts one from the page (a UI control: no step kind, no planner change), and
it is a conversation like any other for its turns: its own stored turns, its
own apps' subagent threads (``DelegateThread`` is keyed by conversation),
its own live channel. Several may be open at once.

**What it carries in** (``carries``): the entities and saved facts from the
timeline's current window (since its quiet gap), as quoted data, never raw
turns. Incognito and forgotten turns lend nothing. What a thread carries is
re-checked on every read: an entity whose turn was forgotten since, or a fact
since forgotten or edited away, no longer rides (an edited fact rides as it
reads now).

**Inside a thread** "this conversation" is the whole thread (it is small and
deliberate: no quiet gap). Its state block's recent entities are its own,
then what it carries; its facts are what it carries, then the saved facts as
anywhere else (:func:`turn_entities`, :func:`turn_facts`).

**Closing** (:func:`close`) posts the thread's one result note into the
timeline (:func:`app.services.ledger.note_thread`), narrated from the
thread's receipts (:func:`ending`, :mod:`app.ai.thread_note`). The close is
claimed once (``closed_at``), so a double tap never posts twice. A closed
thread takes no more turns; a job it left running notes into the timeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from app.ai import facts, memory, think, thread_note
from app.ai.memory import day
from app.ai.state_block import MAX_ENTITIES, MAX_FACTS, Entity, Fact
from app.ai.thread_note import ThreadEnd
from app.db.models import (
    Conversation,
    ConversationKind,
    ConversationMessage,
    LedgerEntity,
    LedgerJob,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTouch,
    LedgerTurn,
    MemoryFact,
)
from app.fleet.jobs import FINISHED
from app.services import conversations as conversations_service
from app.services import facts as facts_service
from app.services import ledger
from app.services.common import active

logger = structlog.get_logger(__name__)

# A thread's name, as the person typed it.
MAX_TITLE_CHARS = 60
# How many open threads the page lists, at most.
MAX_OPEN = 20
# How far back a thread looks for what it carries, in touches (bounded, like
# the state block's own recent entities).
TOUCHES_SCANNED = 200

# Receipts whose things a thread carries in, and whose work its result reports.
_CARRIED_OUTCOMES = ("done", "looked_up")
_RESULT_OUTCOMES = ("done", "looked_up", "unknown")
# Conductor's own receipts that are no app's work: a recall, a think.
_NOT_APP_WORK = (facts.APP, think.APP)


def is_thread(conversation: Conversation) -> bool:
    return conversation.kind == ConversationKind.thread.value


def is_closed(conversation: Conversation) -> bool:
    return conversation.closed_at is not None


def open_threads(db: Session) -> Sequence[Conversation]:
    """The side threads still open, oldest first."""
    return (
        db.execute(
            active(Conversation)
            .where(
                Conversation.kind == ConversationKind.thread.value,
                Conversation.closed_at.is_(None),
            )
            .order_by(Conversation.id)
            .limit(MAX_OPEN)
        )
        .scalars()
        .all()
    )


def get(db: Session, thread_id: int) -> Conversation | None:
    """A live side thread, open or closed."""
    return db.execute(
        active(Conversation).where(
            Conversation.id == thread_id, Conversation.kind == ConversationKind.thread.value
        )
    ).scalar_one_or_none()


def start(db: Session, *, title: str | None = None, hidden: bool = False) -> Conversation:
    """Open a side thread carrying what the timeline's current window touched
    and saved. An incognito start (``hidden``) keeps no name: the name is the
    person's words."""
    timeline = conversations_service.timeline(db)
    since = conversations_service.window_since(db, timeline.id)
    name = None if hidden else " ".join((title or "").split())[:MAX_TITLE_CHARS] or None
    thread = Conversation(
        title=name or thread_note.DEFAULT_TITLE,
        kind=ConversationKind.thread.value,
        carries=window_carries(db, timeline.id, since),
    )
    db.add(thread)
    db.flush()
    db.refresh(thread)
    logger.info("thread_started", thread_id=thread.id, carries=len(thread.carries or []))
    return thread


# --- what a thread carries ------------------------------------------------------------------


def window_carries(db: Session, timeline_id: int, since: datetime | None) -> list[dict[str, Any]]:
    """The entities the timeline's window touched (newest first) and the facts
    saved in it, as a thread keeps them: each with the turn it came from.
    Incognito and forgotten turns lend nothing; the memory app's receipts
    are facts, never entities."""
    window = (LedgerTurn.created_at >= since,) if since is not None else ()
    rows = db.execute(
        select(LedgerTouch, LedgerEntity, LedgerTurn.id)
        .join(LedgerEntity, LedgerEntity.id == LedgerTouch.entity_id)
        .join(LedgerReceipt, LedgerReceipt.id == LedgerTouch.receipt_id)
        .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
        .where(
            LedgerTurn.conversation_id == timeline_id,
            LedgerTurn.incognito.is_(False),
            LedgerTurn.forgotten_at.is_(None),
            LedgerPlan.acted.is_(True),
            LedgerReceipt.outcome.in_(_CARRIED_OUTCOMES),
            LedgerReceipt.app.not_in(_NOT_APP_WORK),
            LedgerEntity.app != facts.APP,
            *window,
        )
        .order_by(LedgerReceipt.id.desc(), LedgerTouch.id)
        .limit(TOUCHES_SCANNED)
    ).all()
    carried: list[dict[str, Any]] = []
    seen: set[int] = set()
    for touch, entity, turn_id in rows:
        if entity.id in seen or str(entity.kind).endswith("job"):
            continue
        seen.add(entity.id)
        carried.append(
            {
                "type": "entity",
                "app": entity.app,
                "kind": entity.kind,
                "id": entity.ref,
                "title": touch.title,
                "untrusted": bool(touch.untrusted),
                "turn_id": turn_id,
            }
        )
        if len(carried) == MAX_ENTITIES:
            break
    saved = db.execute(
        select(MemoryFact, LedgerTurn.id)
        .join(LedgerTurn, LedgerTurn.id == MemoryFact.source_turn_id)
        .where(
            MemoryFact.deleted_at.is_(None),
            LedgerTurn.conversation_id == timeline_id,
            LedgerTurn.incognito.is_(False),
            LedgerTurn.forgotten_at.is_(None),
            *window,
        )
        .order_by(MemoryFact.id.desc())
        .limit(MAX_FACTS)
    ).all()
    carried += [
        {"type": "fact", "fact_id": row.id, "text": row.text, "turn_id": turn_id}
        for row, turn_id in saved
    ]
    return carried


@dataclass(frozen=True)
class Carried:
    """What a thread carries now: its entities and facts, re-checked."""

    entities: list[Entity]
    facts: list[Fact]


def carried(db: Session, thread: Conversation) -> Carried:
    """The thread's carries as they stand now: an entity whose turn was since
    forgotten (or is incognito) drops out, and so does a fact since forgotten;
    a fact edited since rides in its new words."""
    items = [item for item in thread.carries or [] if isinstance(item, dict)]
    turn_ids = {int(item["turn_id"]) for item in items if isinstance(item.get("turn_id"), int)}
    gone = (
        set(
            db.execute(
                select(LedgerTurn.id).where(
                    LedgerTurn.id.in_(turn_ids),
                    (LedgerTurn.forgotten_at.is_not(None)) | (LedgerTurn.incognito.is_(True)),
                )
            )
            .scalars()
            .all()
        )
        if turn_ids
        else set()
    )
    fact_ids = [int(i["fact_id"]) for i in items if i.get("type") == "fact" and "fact_id" in i]
    live = (
        {
            row.id: row
            for row in db.execute(
                select(MemoryFact).where(
                    MemoryFact.id.in_(fact_ids), MemoryFact.deleted_at.is_(None)
                )
            ).scalars()
        }
        if fact_ids
        else {}
    )
    entities = [
        Entity(
            app=str(item.get("app")),
            kind=str(item.get("kind")),
            id=str(item.get("id")),
            title=item.get("title"),
            untrusted=bool(item.get("untrusted")),
        )
        for item in items
        if item.get("type") == "entity" and item.get("turn_id") not in gone
    ]
    kept = [live[i] for i in fact_ids if i in live]
    return Carried(entities, [Fact(row.text, day(_aware(row.updated_at))) for row in kept])


def turn_entities(db: Session, thread: Conversation) -> list[Entity] | None:
    """A thread turn's recent entities: its own, newest first, then what it
    carries; ``None`` (the state block reads the stored turns) when it has
    neither."""
    own = ledger.recent_entities(db, thread.id)
    brought = carried(db, thread).entities
    if not brought:
        return own
    keys = {(e.app, e.kind, e.id) for e in own or []}
    merged = list(own or []) + [e for e in brought if (e.app, e.kind, e.id) not in keys]
    return merged[:MAX_ENTITIES]


def turn_facts(db: Session, thread: Conversation, words: str) -> list[Fact]:
    """A thread turn's facts: what it carries, then the saved facts as on any
    turn, once each, capped."""
    merged: list[Fact] = []
    seen: set[str] = set()
    for fact in [*carried(db, thread).facts, *facts_service.for_state(db, words)]:
        key = " ".join(fact.text.lower().split())
        if key not in seen:
            seen.add(key)
            merged.append(fact)
    return merged[:MAX_FACTS]


# --- closing --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ending:
    """What a thread did, for its result note, and the receipts it rests on."""

    end: ThreadEnd
    receipts: list[LedgerReceipt]


def ending(db: Session, thread: Conversation) -> Ending:
    """The thread's work as its result reports it: the apps' receipts on its
    turns, oldest first, never an incognito or forgotten turn's, a write
    since undone, an undo's own receipt, a recall or the memory app's facts;
    plus its jobs still running."""
    receipts = list(
        db.execute(
            select(LedgerReceipt)
            .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
            .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
            .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
            .where(
                LedgerTurn.conversation_id == thread.id,
                LedgerTurn.kind == "message",
                LedgerTurn.incognito.is_(False),
                LedgerTurn.forgotten_at.is_(None),
                LedgerPlan.acted.is_(True),
                LedgerReceipt.outcome.in_(_RESULT_OUTCOMES),
                LedgerReceipt.undone_by_step_id.is_(None),
                LedgerReceipt.app.not_in(_NOT_APP_WORK),
                LedgerReceipt.tool.is_distinct_from(memory.RECALL_TOOL),
            )
            .order_by(LedgerReceipt.id)
        )
        .scalars()
        .all()
    )
    undoers = {
        step
        for step in db.execute(
            select(LedgerReceipt.undone_by_step_id).where(
                LedgerReceipt.undone_by_step_id.in_({r.step_id for r in receipts})
            )
        ).scalars()
        if step is not None
    }
    receipts = [r for r in receipts if r.step_id not in undoers]
    entries = [
        {
            "app": r.app,
            "tool": r.tool,
            "outcome": r.outcome,
            "count": r.count,
            "untrusted": r.untrusted,
            "touched": [
                {
                    "app": t.entity.app,
                    "kind": t.entity.kind,
                    "id": t.entity.ref,
                    "title": t.title,
                    "untrusted": t.untrusted,
                }
                for t in r.touched
            ],
        }
        for r in receipts
    ]
    jobs = db.execute(
        select(LedgerJob)
        .join(LedgerStep, LedgerStep.id == LedgerJob.started_by_step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
        .where(
            LedgerTurn.conversation_id == thread.id,
            LedgerTurn.incognito.is_(False),
            LedgerTurn.forgotten_at.is_(None),
            LedgerJob.noted_at.is_(None),
            LedgerJob.status.not_in(FINISHED),
        )
        .order_by(LedgerJob.id)
    ).scalars()
    running = [(job.app, job.title, bool(job.untrusted)) for job in jobs]
    # A failed ask leaves a step and no receipt (#189): name the app, so a
    # thread whose every turn failed doesn't read as one that found nothing.
    failed = list(
        dict.fromkeys(
            db.execute(
                select(LedgerStep.app)
                .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
                .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
                .where(
                    LedgerTurn.conversation_id == thread.id,
                    LedgerTurn.incognito.is_(False),
                    LedgerTurn.forgotten_at.is_(None),
                    LedgerStep.outcome == "failed",
                    LedgerStep.app.is_not(None),
                    LedgerStep.app.not_in(_NOT_APP_WORK),
                )
                .order_by(LedgerStep.id)
            ).scalars()
        )
    )
    end = ThreadEnd(
        title=thread.title or thread_note.DEFAULT_TITLE,
        entries=entries,
        running=running,
        failed=[app for app in failed if app],
    )
    return Ending(end, receipts)


def close(
    db: Session,
    thread: Conversation,
    text: str,
    receipts: Sequence[LedgerReceipt],
    *,
    now: datetime,
) -> ConversationMessage | None:
    """Claim the close (only if still open), record the result and post it
    into the timeline, in the caller's commit. ``None``, with nothing
    written, when the thread had already closed."""
    claimed = db.execute(
        update(Conversation)
        .where(Conversation.id == thread.id, Conversation.closed_at.is_(None))
        .values(closed_at=now, result=text)
    )
    if not cast("CursorResult[Any]", claimed).rowcount:
        return None
    db.refresh(thread)
    timeline = conversations_service.timeline(db)
    message = ledger.note_thread(
        db, timeline, thread, text, receipts, stop_reason=thread_note.THREAD_NOTE, now=now
    )
    logger.info("thread_closed", thread_id=thread.id, message_id=message.id)
    return message


@dataclass(frozen=True)
class NoteThread:
    """The side thread a timeline note came from."""

    id: int
    title: str | None


def note_threads(db: Session, message_ids: Sequence[int]) -> dict[int, NoteThread]:
    """Each thread result note's thread, by message id."""
    if not message_ids:
        return {}
    rows = db.execute(
        select(LedgerTurn.assistant_message_id, LedgerStep.arguments)
        .join(LedgerPlan, LedgerPlan.turn_id == LedgerTurn.id)
        .join(LedgerStep, LedgerStep.plan_id == LedgerPlan.id)
        .where(
            LedgerTurn.kind == ledger.THREAD_TURN,
            LedgerTurn.assistant_message_id.in_(message_ids),
            LedgerStep.kind == ledger.THREAD_STEP,
        )
    ).all()
    ids = {
        message_id: int(arguments["thread_id"])
        for message_id, arguments in rows
        if message_id is not None and isinstance((arguments or {}).get("thread_id"), int)
    }
    titles: dict[int, str | None] = {
        row.id: row.title
        for row in db.execute(
            select(Conversation.id, Conversation.title).where(Conversation.id.in_(ids.values()))
        )
    }
    return {message: NoteThread(thread, titles.get(thread)) for message, thread in ids.items()}


def _aware(value: datetime) -> datetime:
    return conversations_service._aware(value)
