"""Turns by meaning (#147): the stored vectors, and cosines against a query.

Memory (:mod:`app.services.memory`) ranks turns by the words they share with a
query and, through this module, by meaning: each indexed turn is embedded once
when it closes (:func:`app.services.memory.embed_pending`) and its vector kept
in ``ledger_turn_vectors``; a query is embedded on demand and compared with a
dot product in pure Python (the vectors are L2-normalised).

**Optional, and never in a turn's way.** The service is off
(``CONDUCTOR_EMBEDDINGS``), down, slow or erroring: every call here answers
``None`` and memory uses word overlap alone, with a structlog line. After a
failure the service is left alone for :data:`RETRY_S` (no turn pays a timeout
twice a minute); the turns it couldn't embed are embedded by a later close or
the backfill (``scripts/embed_turns.py``).

**What is sent.** Only indexed turns' words and touched titles, never an
incognito or forgotten turn's (:func:`app.services.memory.embed_pending`
selects them out), and no query while an incognito turn runs. Forgetting a
turn deletes its vector (:func:`delete`).

**Code only.** A cosine ranks; it never writes memory or decides what a step
passes on (``{sN}`` stays the word-matched entity, :func:`app.ai.memory.best`).
"""

from __future__ import annotations

import operator
import sys
import time
from array import array
from collections.abc import Sequence
from functools import lru_cache

import httpx
import structlog
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import incognito, memory
from app.ai.embeddings import (
    QUERY_TIMEOUT,
    Embedded,
    Embedder,
    Embeds,
    EmbeddingsUnavailable,
    query_text,
)
from app.config import get_settings
from app.db.models import Conversation, LedgerTurn, LedgerTurnVector

logger = structlog.get_logger(__name__)

# The newest embedded turns a query is compared with: bounded like every read.
VECTORS_SCANNED = 2000
# After a failure, how long the service is left alone (the backfill ignores it).
RETRY_S = 60.0

_down_until = 0.0


def _reset() -> None:
    """Forget a failure (tests)."""
    global _down_until
    _down_until = 0.0


@lru_cache(maxsize=4)
def _client(base_url: str) -> Embedder:
    return Embedder(base_url)


def embedder() -> Embeds | None:
    """The service's client, or ``None`` when it is switched off."""
    settings = get_settings()
    if not settings.conductor_embeddings:
        return None
    return _client(settings.embeddings_base_url)


def resting() -> bool:
    """Whether a recent failure says to leave the service alone for now."""
    return time.monotonic() < _down_until


def embed(
    texts: Sequence[str], *, timeout: httpx.Timeout, patient: bool = False
) -> Embedded | None:
    """Vectors for ``texts``, or ``None`` when the service is off, resting or
    fails (which rests it, unless ``patient``: the backfill's own retries)."""
    global _down_until
    client = embedder()
    if client is None or not texts or (resting() and not patient):
        return None
    try:
        return client.embed(texts, timeout=timeout)
    except EmbeddingsUnavailable as exc:
        if not patient:
            _down_until = time.monotonic() + RETRY_S
        logger.info("memory_embeddings_unavailable", reason=str(exc), texts=len(texts))
        return None


# --- the store -------------------------------------------------------------------------------


def pack(vector: Sequence[float]) -> bytes:
    values = array("f", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def unpack(blob: bytes) -> array[float]:
    values = array("f")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def store(db: Session, turn_id: int, model: str, vector: Sequence[float]) -> None:
    row = db.get(LedgerTurnVector, turn_id)
    if row is None:
        db.add(LedgerTurnVector(turn_id=turn_id, model=model, vector=pack(vector)))
    else:
        row.model, row.vector = model, pack(vector)


def delete(db: Session, turn_id: int) -> None:
    """A forgotten turn's vector goes with it (#81)."""
    db.execute(sql_delete(LedgerTurnVector).where(LedgerTurnVector.turn_id == turn_id))


def drop_other_models(db: Session, model: str) -> int:
    """Delete the vectors another model made: they don't compare with this one's."""
    result = db.execute(sql_delete(LedgerTurnVector).where(LedgerTurnVector.model != model))
    return int(getattr(result, "rowcount", 0) or 0)


# --- searching -------------------------------------------------------------------------------


def cosines(db: Session, query: str) -> dict[int, float] | None:
    """Each live embedded turn's cosine with ``query``, by turn id, over the
    newest :data:`VECTORS_SCANNED`; ``None`` when meaning can't be used (off,
    incognito, too few turns embedded, the service failed, another model)."""
    if incognito.active() or not query.strip() or embedder() is None or resting():
        return None
    rows = db.execute(
        select(LedgerTurnVector.turn_id, LedgerTurnVector.model, LedgerTurnVector.vector)
        .join(LedgerTurn, LedgerTurn.id == LedgerTurnVector.turn_id)
        .join(Conversation, Conversation.id == LedgerTurn.conversation_id)
        .where(
            LedgerTurn.forgotten_at.is_(None),
            LedgerTurn.incognito.is_(False),
            Conversation.deleted_at.is_(None),
        )
        .order_by(LedgerTurnVector.turn_id.desc())
        .limit(VECTORS_SCANNED)
    ).all()
    if len(rows) < memory.MIN_VECTOR_CORPUS:
        return None
    embedded = embed([query_text(query)], timeout=QUERY_TIMEOUT)
    if embedded is None:
        return None
    wanted = embedded.vectors[0]
    found = {
        turn_id: sum(map(operator.mul, wanted, unpack(blob)))
        for turn_id, model, blob in rows
        if model == embedded.model and len(blob) == 4 * len(wanted)
    }
    if len(found) < memory.MIN_VECTOR_CORPUS:
        logger.info(
            "memory_embeddings_stale", model=embedded.model, stored=len(rows), usable=len(found)
        )
        return None
    return found
