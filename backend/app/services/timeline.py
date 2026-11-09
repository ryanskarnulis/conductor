"""The timeline's page reads (#82): each entry's tags, and search.

**Tags.** Every timeline entry shows the surface it came from and whether it
was incognito, from its ledger turn (:func:`message_tags`).

**Search** (:func:`search`) runs over what the person said, conductor's
replies and the titles the apps' receipts touched, newest first. It is the
person's own view of their record, so it reads the visible words; incognito
turns (whose words were never written) and forgotten ones (#81) are never
hits. Plain case-insensitive matching on every word of the query: no model,
no topics. Bounded: it reads a fixed window of the newest turns.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.db.models import (
    ConversationMessage,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTouch,
    LedgerTurn,
)

# The newest turns one search reads, and the hits it returns at most.
SEARCH_SCANNED = 2000
MAX_HITS = 30
# The query's own bounds (the route enforces them too).
MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 200
# How much of what was said and of the reply a hit carries.
SAID_CHARS = 200
REPLY_CHARS = 280
MAX_TITLES = 3


@dataclass(frozen=True)
class Tag:
    surface: str | None
    incognito: bool


def message_tags(db: Session, message_ids: Iterable[int]) -> dict[int, Tag]:
    """Each message's turn tags: its surface and whether it was incognito."""
    ids = list(message_ids)
    if not ids:
        return {}
    rows = db.execute(
        select(
            LedgerTurn.user_message_id,
            LedgerTurn.assistant_message_id,
            LedgerTurn.surface,
            LedgerTurn.incognito,
        ).where((LedgerTurn.user_message_id.in_(ids)) | (LedgerTurn.assistant_message_id.in_(ids)))
    ).all()
    tags: dict[int, Tag] = {}
    for user_id, assistant_id, surface, hidden in rows:
        for message_id in (user_id, assistant_id):
            if message_id is not None:
                tags[message_id] = Tag(surface, bool(hidden))
    return tags


@dataclass(frozen=True)
class Hit:
    turn_id: int
    message_id: int | None
    at: datetime
    surface: str | None
    kind: str
    said: str | None
    reply: str | None
    titles: list[str]
    matched: list[str]


def search(db: Session, conversation_id: int, query: str, *, limit: int = MAX_HITS) -> list[Hit]:
    """The timeline's turns whose words, reply or receipt titles hold every
    word of ``query``, newest first."""
    wanted = [word for word in query.lower().split() if word]
    if not wanted:
        return []
    said_message = aliased(ConversationMessage)
    reply_message = aliased(ConversationMessage)
    rows = db.execute(
        select(LedgerTurn, said_message.content, reply_message.content)
        .outerjoin(said_message, said_message.id == LedgerTurn.user_message_id)
        .outerjoin(reply_message, reply_message.id == LedgerTurn.assistant_message_id)
        .where(
            LedgerTurn.conversation_id == conversation_id,
            LedgerTurn.incognito.is_(False),
            LedgerTurn.forgotten_at.is_(None),
        )
        .order_by(LedgerTurn.id.desc())
        .limit(SEARCH_SCANNED)
    ).all()
    titles = _titles(db, [turn.id for turn, _, _ in rows])
    hits: list[Hit] = []
    for turn, said_text, reply_text in rows:
        # A job's turn has code's words, not the person's: only its note is shown.
        said = None if turn.kind == "job" else (said_text or turn.words)
        reply = reply_text
        own = titles.get(turn.id, [])
        fields = {
            "words": (said or "").lower(),
            "reply": (reply or "").lower(),
            "receipt": " ".join(own).lower(),
        }
        everything = " ".join(fields.values())
        if not all(word in everything for word in wanted):
            continue
        matched = [name for name, text in fields.items() if any(w in text for w in wanted)]
        shown = [title for title in own if any(w in title.lower() for w in wanted)]
        hits.append(
            Hit(
                turn_id=turn.id,
                message_id=turn.user_message_id or turn.assistant_message_id,
                at=turn.created_at,
                surface=turn.surface,
                kind=turn.kind,
                said=_clip(said, SAID_CHARS),
                reply=_clip(reply, REPLY_CHARS),
                titles=shown[:MAX_TITLES],
                matched=matched,
            )
        )
        if len(hits) == limit:
            break
    return hits


def _titles(db: Session, turn_ids: list[int]) -> dict[int, list[str]]:
    """The titles each turn's receipts touched, in receipt order, once each."""
    if not turn_ids:
        return {}
    rows = db.execute(
        select(LedgerPlan.turn_id, LedgerTouch.title)
        .join(LedgerReceipt, LedgerReceipt.id == LedgerTouch.receipt_id)
        .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .where(LedgerPlan.turn_id.in_(turn_ids), LedgerTouch.title.is_not(None))
        .order_by(LedgerReceipt.id, LedgerTouch.id)
    ).all()
    found: dict[int, list[str]] = defaultdict(list)
    for turn_id, title in rows:
        if title and title not in found[turn_id]:
            found[turn_id].append(title)
    return found


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
