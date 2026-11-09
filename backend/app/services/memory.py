"""Memory over the ledger (#80): the index, quiet topics, recall and matches.

Everything here is code over the ledger's rows; no model call writes memory.

**The index.** Once a turn ends, :func:`index_pending` files the words recall
matches it on (:func:`app.ai.memory.terms` of the person's words and the
titles its receipts touched) on ``ledger_turns.terms``, adds them to each
entity it touched (``ledger_entities.terms``: the registry that resolves "that
Caribou song" days later), and puts the turn in a quiet topic. Turns are
indexed oldest first, so a backlog (rows from before this revision, a job's
note turn) is filed in the order it happened, a batch each time a turn closes.

**Quiet topics.** A turn joins the topic it shares the most words or entities
with (at least :data:`JOIN_SCORE`), else stays with the turn before it in the
same conversation when it says little of its own ("yes", "the first one") or
still shares a word with it, else starts a topic. Deterministic, bounded, and
never shown: the person never sees or manages a topic. Recall reads a matching
topic's turns beside the turns that match outright, so "go with steel, then"
is found under "the garage shelves" though it never says "garage".

**Recall** (:func:`recall`) scores ended turns across every live conversation
and day, cites the best few by day and conversation, and resolves the query to
one entity when one clearly answers it. **Matches** (:func:`matches`) are the
state block's leads from earlier conversations for words the person just
used, only when they point back at something.

**Forgetting** (#81, :func:`unindex`): a turn whose saved fact the person
forgot leaves the index for good: its words come off its topic and off the
entities it touched, and recall, matches and the replays skip it.

Bounded: every read scans a fixed window, so a turn's cost never grows with the
ledger.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai import memory
from app.ai.memory import Candidate, Found, Hit, Recalled
from app.ai.recall import points_back
from app.ai.state_block import Entity, Match
from app.db.models import (
    Conversation,
    LedgerEntity,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTopic,
    LedgerTouch,
    LedgerTurn,
    utcnow,
)

logger = structlog.get_logger(__name__)

# Turns indexed per call: a closing turn catches up this many at most.
INDEX_BATCH = 200
# Words kept per topic and per entity, most frequent first.
TOPIC_TERMS = 40
ENTITY_TERMS = 30
# Shared words (an entity counts double) for a turn to join an existing topic.
JOIN_SCORE = 2
# A turn whose receipt touched more than this many things was a list read: its
# words describe the list, not each thing in it.
NAMED_PER_RECEIPT = 3
# The windows each read scans.
TOPICS_SCANNED = 200
TURNS_SCANNED = 500
ENTITIES_SCANNED = 500
MAX_MATCHES = 3
# Best-scored entities checked for a last turn in another conversation.
MATCHES_LOOKED_AT = 20

_SEEN = ("done", "looked_up")


# --- the index ---------------------------------------------------------------------------


def index_pending(db: Session, *, limit: int = INDEX_BATCH) -> int:
    """Index ended turns code hasn't yet, oldest first; how many it did."""
    turns = (
        db.execute(
            select(LedgerTurn)
            .where(LedgerTurn.status == "ended", LedgerTurn.terms.is_(None))
            .order_by(LedgerTurn.id)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    for turn in turns:
        _index(db, turn)
    if turns:
        db.flush()
    return len(turns)


@dataclass(frozen=True)
class _Touched:
    entity: LedgerEntity
    title: str | None
    untrusted: bool
    # How many things its receipt touched: more than a few is a list read.
    crowd: int


def _touched(db: Session, turn_ids: Sequence[int]) -> dict[int, list[_Touched]]:
    """What each turn's acted plans changed or found, in receipt order."""
    if not turn_ids:
        return {}
    rows = db.execute(
        select(LedgerPlan.turn_id, LedgerTouch, LedgerEntity)
        .join(LedgerEntity, LedgerEntity.id == LedgerTouch.entity_id)
        .join(LedgerReceipt, LedgerReceipt.id == LedgerTouch.receipt_id)
        .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .where(
            LedgerPlan.turn_id.in_(turn_ids),
            LedgerPlan.acted.is_(True),
            LedgerReceipt.outcome.in_(_SEEN),
        )
        .order_by(LedgerReceipt.id, LedgerTouch.id)
    ).all()
    # Every touch of these receipts is in ``rows``: count them here, not over the table.
    crowd: dict[int, int] = defaultdict(int)
    for _, touch, _ in rows:
        crowd[touch.receipt_id] += 1
    found: dict[int, list[_Touched]] = defaultdict(list)
    for turn_id, touch, entity in rows:
        found[turn_id].append(
            _Touched(
                entity, touch.title, touch.untrusted or entity.untrusted, crowd[touch.receipt_id]
            )
        )
    return found


def _index(db: Session, turn: LedgerTurn) -> None:
    touched = _touched(db, [turn.id]).get(turn.id, [])
    # A job's note turn has code's words ("music job 20 finished"), not the person's.
    said = memory.terms(turn.words) if turn.kind != "job" else set()
    titles: set[str] = set()
    for item in touched:
        titles |= memory.terms(item.title or "")
    words = said | titles
    turn.terms = sorted(words)
    entity_ids = sorted({item.entity.id for item in touched})
    turn.topic_id = _topic_for(db, turn, words, entity_ids)
    for item in touched:
        own = memory.terms(item.title or "") | {memory.stem(item.entity.kind.lower())}
        context = said if item.crowd <= NAMED_PER_RECEIPT else set()
        item.entity.terms = memory.merge(item.entity.terms, own | context, cap=ENTITY_TERMS)


def _topic_for(db: Session, turn: LedgerTurn, words: set[str], entity_ids: list[int]) -> int | None:
    previous = db.execute(
        select(LedgerTurn)
        .where(
            LedgerTurn.conversation_id == turn.conversation_id,
            LedgerTurn.id < turn.id,
            LedgerTurn.terms.is_not(None),
        )
        .order_by(LedgerTurn.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    before = previous.topic_id if previous is not None else None
    if not words and not entity_ids:
        return _grow(db, before, words)
    scores: dict[int, int] = {}
    topics = (
        db.execute(
            select(LedgerTopic)
            .order_by(LedgerTopic.updated_at.desc(), LedgerTopic.id.desc())
            .limit(TOPICS_SCANNED)
        )
        .scalars()
        .all()
    )
    for topic in topics:
        shared = len(words & set(topic.terms or {}))
        if shared:
            scores[topic.id] = shared
    if entity_ids:
        together = db.execute(
            select(LedgerTurn.topic_id, func.count(func.distinct(LedgerTouch.entity_id)))
            .join(LedgerPlan, LedgerPlan.turn_id == LedgerTurn.id)
            .join(LedgerStep, LedgerStep.plan_id == LedgerPlan.id)
            .join(LedgerReceipt, LedgerReceipt.step_id == LedgerStep.id)
            .join(LedgerTouch, LedgerTouch.receipt_id == LedgerReceipt.id)
            .where(
                LedgerTouch.entity_id.in_(entity_ids),
                LedgerTurn.topic_id.is_not(None),
                LedgerTurn.id != turn.id,
            )
            .group_by(LedgerTurn.topic_id)
        ).all()
        for topic_id, shared in together:
            scores[topic_id] = scores.get(topic_id, 0) + 2 * int(shared)
    best = max(scores.items(), key=lambda item: (item[1], item[0]), default=None)
    if best is not None and best[1] >= JOIN_SCORE:
        return _grow(db, best[0], words)
    if before is not None and (len(words) <= 2 or scores.get(before, 0) >= 1):
        return _grow(db, before, words)
    return _grow(db, None, words, start=True)


def _grow(db: Session, topic_id: int | None, words: set[str], *, start: bool = False) -> int | None:
    """Add a turn's words to its topic (or start one); the topic's id."""
    if topic_id is None and not start:
        return None
    topic = db.get(LedgerTopic, topic_id) if topic_id is not None else None
    if topic is None:
        topic = LedgerTopic(terms={}, turns=0)
        db.add(topic)
    topic.terms = memory.merge(topic.terms, words, cap=TOPIC_TERMS)
    topic.turns += 1
    db.flush()
    return topic.id


def unindex(db: Session, turn_id: int) -> None:
    """Take a turn out of memory (#81): marked forgotten, so it is never
    indexed, recalled or replayed again; the words it added come off its
    topic, and the person's words come off the entities it touched (their own
    titles stay: those are the app's). Memory's receipts that quote it lose
    the quote."""
    turn = db.get(LedgerTurn, turn_id)
    if turn is None or turn.forgotten_at is not None:
        return
    turn.forgotten_at = utcnow()
    if turn.terms:
        if turn.topic_id is not None:
            topic = db.get(LedgerTopic, turn.topic_id)
            if topic is not None:
                topic.terms = _less(topic.terms, turn.terms)
                topic.turns = max(0, topic.turns - 1)
        said = memory.terms(turn.words) if turn.kind != "job" else set()
        for item in _touched(db, [turn.id]).get(turn.id, []):
            if item.crowd > NAMED_PER_RECEIPT:
                continue
            own = memory.terms(item.title or "") | {memory.stem(item.entity.kind.lower())}
            item.entity.terms = _less(item.entity.terms, said - own) or None
    turn.terms = []
    turn.topic_id = None
    quoting = db.execute(
        select(LedgerReceipt).where(LedgerReceipt.cites_turn_id == turn.id)
    ).scalars()
    for receipt in quoting:
        receipt.fact = None
    db.flush()


def _less(counts: dict[str, int] | None, words: Sequence[str] | set[str]) -> dict[str, int]:
    """``counts`` with one less of each of ``words``; a word at zero goes."""
    left = dict(counts or {})
    for word in words:
        if word in left:
            left[word] -= 1
            if left[word] <= 0:
                del left[word]
    return left


# --- recall --------------------------------------------------------------------------------


def recall(db: Session, query: str, *, limit: int = memory.MAX_HITS) -> Recalled:
    """The turns that best answer ``query``, across every live conversation and
    day, cited by day and conversation, and the one entity they clearly point
    at, if any. Reads only what is indexed (a turn is indexed as it closes)."""
    wanted = memory.terms(query)
    if not wanted:
        return Recalled(query)
    scope = _topics_in_scope(db, wanted)
    live = (
        select(LedgerTurn, Conversation.title)
        .join(Conversation, Conversation.id == LedgerTurn.conversation_id)
        .where(
            LedgerTurn.status == "ended",
            LedgerTurn.terms.is_not(None),
            Conversation.deleted_at.is_(None),
        )
    )
    recent = db.execute(live.order_by(LedgerTurn.id.desc()).limit(TURNS_SCANNED)).all()
    topical = (
        db.execute(
            live.where(LedgerTurn.topic_id.in_(scope))
            .order_by(LedgerTurn.id.desc())
            .limit(TURNS_SCANNED)
        ).all()
        if scope
        else []
    )
    scored: dict[int, tuple[int, LedgerTurn, str | None]] = {}
    for turn, title in [*recent, *topical]:
        if turn.id in scored or not turn.terms:
            continue
        shared = len(wanted & set(turn.terms))
        points = 2 * shared + (1 if turn.topic_id in scope else 0)
        if points:
            scored[turn.id] = (points, turn, title)
    ranked = sorted(scored.values(), key=lambda item: (-item[0], -item[1].id))[:limit]
    if not ranked:
        return Recalled(query)
    touched = _touched(db, [turn.id for _, turn, _ in ranked])
    hits: list[Hit] = []
    candidates: dict[tuple[str, str, str], Candidate] = {}
    for _, turn, title in ranked:
        items = touched.get(turn.id, [])
        matching = [
            item for item in items if memory.score(wanted, item.entity.terms, item.entity.kind) > 0
        ]
        shown = matching or (items if len(items) <= NAMED_PER_RECEIPT else [])
        found: list[Found] = []
        for item in shown:
            entity = item.entity
            one = Found(entity.app, entity.kind, entity.ref, item.title, item.untrusted)
            if one not in found:
                found.append(one)
            key = (entity.app, entity.kind, entity.ref)
            points = memory.score(wanted, entity.terms, entity.kind)
            if key not in candidates and points:
                candidates[key] = Candidate(one, points, _aware(entity.last_seen_at))
        hits.append(
            Hit(
                turn_id=turn.id,
                conversation_id=turn.conversation_id,
                when=_aware(turn.created_at),
                words=turn.words,
                conversation=title,
                kind=turn.kind,
                found=tuple(found[:5]),
            )
        )
    best = memory.best(list(candidates.values()))
    logger.info("memory_recalled", hits=len(hits), resolved=best is not None, topics=len(scope))
    return Recalled(query, tuple(hits), best)


def _topics_in_scope(db: Session, wanted: set[str]) -> set[int]:
    """The topics that share at least half the query's words (and at least
    one), the best three."""
    floor = max(1, (len(wanted) + 1) // 2)
    topics = (
        db.execute(
            select(LedgerTopic)
            .order_by(LedgerTopic.updated_at.desc(), LedgerTopic.id.desc())
            .limit(TOPICS_SCANNED)
        )
        .scalars()
        .all()
    )
    scored = sorted(
        ((len(wanted & set(topic.terms or {})), topic.id) for topic in topics),
        reverse=True,
    )
    return {topic_id for shared, topic_id in scored[:3] if shared >= floor}


# --- matches for the state block ----------------------------------------------------------


def matches(
    db: Session, words: str, conversation_id: int, *, limit: int = MAX_MATCHES
) -> list[Match]:
    """Leads for a reference to something from another conversation ("that
    Caribou song"), only when the words point back at something. This
    conversation's own entities are its recent entities, not matches."""
    if not points_back(words):
        return []
    wanted = memory.terms(words)
    if not wanted:
        return []
    entities = (
        db.execute(
            select(LedgerEntity)
            .where(LedgerEntity.terms.is_not(None))
            .order_by(LedgerEntity.last_seen_at.desc(), LedgerEntity.id.desc())
            .limit(ENTITIES_SCANNED)
        )
        .scalars()
        .all()
    )
    scored = [
        (points, entity)
        for entity in entities
        if (points := memory.score(wanted, entity.terms, entity.kind)) > 0
        and entity.title
        and not entity.kind.endswith("job")
    ]
    scored.sort(key=lambda item: (-item[0], -_aware(item[1].last_seen_at).timestamp()))
    found: list[Match] = []
    for _, entity in scored[:MATCHES_LOOKED_AT]:
        last = _last_turn(db, entity.id)
        if last is None or last.conversation_id == conversation_id:
            continue
        found.append(
            Match(
                Entity(entity.app, entity.kind, entity.ref, entity.title, entity.untrusted),
                memory.day(_aware(last.created_at)),
                last.words if last.kind != "job" else "",
            )
        )
        if len(found) == limit:
            break
    return found


def _last_turn(db: Session, entity_id: int) -> LedgerTurn | None:
    """The newest live turn that touched this entity."""
    return db.execute(
        select(LedgerTurn)
        .join(LedgerPlan, LedgerPlan.turn_id == LedgerTurn.id)
        .join(LedgerStep, LedgerStep.plan_id == LedgerPlan.id)
        .join(LedgerReceipt, LedgerReceipt.step_id == LedgerStep.id)
        .join(LedgerTouch, LedgerTouch.receipt_id == LedgerReceipt.id)
        .join(Conversation, Conversation.id == LedgerTurn.conversation_id)
        .where(
            LedgerTouch.entity_id == entity_id,
            LedgerTurn.forgotten_at.is_(None),
            LedgerPlan.acted.is_(True),
            LedgerReceipt.outcome.in_(_SEEN),
            Conversation.deleted_at.is_(None),
        )
        .order_by(LedgerTurn.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; the ledger writes UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
