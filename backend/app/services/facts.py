"""Memory you can see (#81): the saved facts, and the ``memory`` app that keeps them.

The only write path for ``memory_facts``. A fact is the person's own words
(:mod:`app.ai.facts` finds them), with the turn it came from. Code only: no
model writes or rewrites one.

- :func:`handle` is the built-in ``memory`` app a plan's ``ask(app="memory")``
  reaches (remember, forget, or a question about the list); it answers with
  v1.1 receipts like any app.
- :func:`forget` soft-deletes a fact and takes the turn it was said in out of
  memory (:func:`app.services.memory.unindex`), and the turn that forgot it
  too, since its words name it: gone from recall, the state block and every
  model replay, not just hidden. The visible transcript keeps what was said.
- :func:`edit` is the person's change in the memory list: the new words, with
  the note turn that recorded it as the source; the old source turn is
  forgotten, so recall never finds the old version.
- :func:`for_state` is the state block's "facts you saved", bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import facts
from app.ai.facts import Saved
from app.ai.memory import day
from app.ai.state_block import MAX_FACTS, Fact
from app.db.models import Conversation, LedgerTurn, MemoryFact, utcnow
from app.services import memory as memory_index

logger = structlog.get_logger(__name__)

# The facts any one read looks at: bounded, however long the list grows.
SCANNED = 200


def live(db: Session, *, limit: int = SCANNED) -> list[MemoryFact]:
    """Live facts, newest first."""
    return list(
        db.execute(
            select(MemoryFact)
            .where(MemoryFact.deleted_at.is_(None))
            .order_by(MemoryFact.updated_at.desc(), MemoryFact.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def get(db: Session, fact_id: int) -> MemoryFact | None:
    return db.execute(
        select(MemoryFact).where(MemoryFact.id == fact_id, MemoryFact.deleted_at.is_(None))
    ).scalar_one_or_none()


def for_state(db: Session, words: str) -> list[Fact]:
    """The state block's facts: those that share words with what the person
    just said first, then the newest, capped."""
    said = facts.words_of(words)
    rows = live(db)
    ranked = sorted(
        enumerate(rows), key=lambda item: (-len(said & facts.words_of(item[1].text)), item[0])
    )
    return [Fact(row.text, day(_aware(row.updated_at))) for _, row in ranked[:MAX_FACTS]]


def self_lines(db: Session) -> str:
    """What the person saved, for "what do you know about me?" answered directly."""
    rows = live(db, limit=MAX_FACTS)
    if not rows:
        return "The person hasn't asked you to remember anything."
    lines = [f'- "{row.text}" (saved {day(_aware(row.updated_at))})' for row in rows]
    return "What the person asked you to remember, in their words:\n" + "\n".join(lines)


# --- the memory app ------------------------------------------------------------------------


def handle(
    db: Session,
    request: str,
    words: str,
    *,
    conversation_id: int,
    turn_id: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    """One ``ask(app="memory")``: its result line and v1.1 receipts. ``words``
    is what the person said this turn ("" when a resumed plan was told only
    "yes"); ``turn_id`` is that turn, the new fact's source. Runs in a
    savepoint: a failure changes nothing and raises."""
    intent = facts.parse(request)
    with db.begin_nested():
        if intent.kind == "remember":
            return _remember(db, request, words, turn_id)
        if intent.kind == "forget":
            return _forget(db, intent.text, conversation_id, turn_id)
        return _list(db, intent.text)


def _remember(
    db: Session, request: str, words: str, turn_id: int | None
) -> tuple[str, list[dict[str, Any]]]:
    text = facts.own_words(request, words)
    if text is None:
        return facts.refused(
            facts.REMEMBER, "what to remember has to be in your own words, after “remember”"
        )
    if len(text) > facts.MAX_FACT_CHARS:
        return facts.refused(
            facts.REMEMBER, f"that's longer than {facts.MAX_FACT_CHARS} characters; say it shorter"
        )
    existing = next((row for row in live(db) if facts.same(row.text, text)), None)
    if existing is not None:
        return facts.remembered(_saved(existing), already=True)
    row = MemoryFact(text=text, source_turn_id=turn_id)
    db.add(row)
    db.flush()
    logger.info("memory_fact_saved", fact_id=row.id, turn_id=turn_id)
    return facts.remembered(_saved(row), already=False)


def _forget(
    db: Session, target: str, conversation_id: int, turn_id: int | None
) -> tuple[str, list[dict[str, Any]]]:
    if facts.everything(target):
        return facts.refused(
            facts.FORGET, "I forget one saved fact at a time; the memory list can clear them"
        )
    rows = live(db)
    wanted = facts.words_of(target)
    if not wanted:
        # "forget that": the newest fact saved in this conversation.
        here = _saved_here(db, rows, conversation_id)
        if here is None:
            return facts.refused(
                facts.FORGET, "nothing was saved in this conversation; say which fact to forget"
            )
        chosen = here
    else:
        scored = sorted(
            ((len(wanted & facts.words_of(row.text)), row) for row in rows),
            key=lambda item: -item[0],
        )
        best = [row for points, row in scored if points and points == scored[0][0]]
        if not best:
            return facts.refused(facts.FORGET, f"nothing saved matches “{target}”")
        if len(best) > 1:
            named = "; ".join(f"“{row.text}”" for row in best[:3])
            return facts.refused(facts.FORGET, f"more than one saved fact matches: {named}")
        chosen = best[0]
    saved = _saved(chosen)
    forget(db, chosen, by_turn_id=turn_id)
    return facts.forgot(saved)


def _saved_here(db: Session, rows: list[MemoryFact], conversation_id: int) -> MemoryFact | None:
    sources = [row.source_turn_id for row in rows if row.source_turn_id is not None]
    if not sources:
        return None
    here = set(
        db.execute(
            select(LedgerTurn.id).where(
                LedgerTurn.id.in_(sources), LedgerTurn.conversation_id == conversation_id
            )
        ).scalars()
    )
    candidates = [row for row in rows if row.source_turn_id in here]
    return max(candidates, key=lambda row: row.id, default=None)


def _list(db: Session, question: str) -> tuple[str, list[dict[str, Any]]]:
    rows = live(db)
    wanted = facts.words_of(question)
    matching = [row for row in rows if wanted & facts.words_of(row.text)] if wanted else []
    chosen = matching or rows
    shown = [_saved(row, db=db) for row in chosen[: facts.MAX_LISTED]]
    return facts.listed(shown, total=len(chosen))


# --- forgetting and editing ------------------------------------------------------------------


def forget(db: Session, row: MemoryFact, *, by_turn_id: int | None) -> None:
    """Forget a fact: soft-deleted, and the turn it was said in (and the one
    that forgot it, whose words name it) out of memory."""
    row.deleted_at = utcnow()
    for turn_id in (row.source_turn_id, by_turn_id):
        if turn_id is not None:
            memory_index.unindex(db, turn_id)
    db.flush()
    logger.info("memory_fact_forgotten", fact_id=row.id, by_turn_id=by_turn_id)


def edit(db: Session, row: MemoryFact, text: str, *, note_turn_id: int | None) -> None:
    """The person changed a fact in the memory list: their new words, sourced
    to the note turn that recorded the change. The old words' turn leaves
    memory, so recall never finds the old version."""
    old = row.source_turn_id
    row.text = text
    row.source_turn_id = note_turn_id
    row.updated_at = utcnow()
    if old is not None and old != note_turn_id:
        memory_index.unindex(db, old)
    db.flush()
    logger.info("memory_fact_edited", fact_id=row.id, note_turn_id=note_turn_id)


# --- where a fact came from ------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """The turn a fact came from, as the memory list shows it."""

    turn_id: int
    kind: str
    # The person's words then; ``None`` once that turn was forgotten.
    words: str | None
    said_at: datetime
    surface: str | None
    thread_id: int | None
    thread_title: str | None


def source(db: Session, row: MemoryFact) -> Source | None:
    if row.source_turn_id is None:
        return None
    turn = db.get(LedgerTurn, row.source_turn_id)
    if turn is None:
        return None
    conversation = db.get(Conversation, turn.conversation_id)
    live_thread = conversation is not None and conversation.deleted_at is None
    return Source(
        turn_id=turn.id,
        kind=turn.kind,
        words=turn.words if turn.forgotten_at is None else None,
        said_at=_aware(turn.created_at),
        surface=turn.surface,
        thread_id=conversation.id if live_thread and conversation else None,
        thread_title=conversation.title if live_thread and conversation else None,
    )


def _saved(row: MemoryFact, *, db: Session | None = None) -> Saved:
    """A fact as the receipts show it; with ``db``, where it came from too."""
    line = ""
    if db is not None:
        found = source(db, row)
        line = (
            facts.source_line(found.said_at, kind=found.kind, thread=found.thread_title)
            if found is not None
            else facts.source_line(_aware(row.updated_at), edited=True)
        )
    return Saved(row.id, row.text, _aware(row.updated_at), line, row.source_turn_id)


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; conductor writes UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
