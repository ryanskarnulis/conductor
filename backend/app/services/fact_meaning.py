"""Saved facts by meaning (#204, #145's decision 4): which fact an app's ask
means when it shares no two words with any (:func:`app.ai.facts.related`
finds nothing).

The request is embedded as a query and each saved fact, and the fixed
:data:`app.ai.facts.BACKGROUND`, as documents, in one call to the embeddings
service (#147, through :func:`app.services.turn_vectors.embed`, so off, down,
slow or resting means no fact by meaning, never a waiting ask). Fact and
background vectors are kept in process, by text, for the model that made them;
a fact edited or forgotten is simply never asked for again.

**What is sent.** Saved facts' words and the ask's request, never during an
incognito turn (the route passes nothing then; this checks again).

**Code only.** A cosine picks at most one fact to send as quoted context
(:func:`app.ai.facts.nearest`); it never rewrites the request or saves anything.
"""

from __future__ import annotations

import operator
import statistics
from collections.abc import Sequence

import structlog

from app.ai import facts, incognito
from app.ai.embeddings import INDEX_TIMEOUT, QUERY_TIMEOUT, document_text, query_text
from app.config import get_settings
from app.services import turn_vectors

logger = structlog.get_logger(__name__)

# Documents kept per model; past it the cache starts over.
CACHE_MAX = 512
# More uncached documents than this in one call (a cold start: the background
# and every fact) get the indexing timeout instead of a query's.
COLD = 4

_cache: dict[str, list[float]] = {}
_model: str | None = None


def _reset() -> None:
    """Forget every vector (tests)."""
    global _model
    _cache.clear()
    _model = None


def _vectors(
    request: str, documents: Sequence[str]
) -> tuple[list[float], list[list[float]]] | None:
    """The request's query vector and each document's, or ``None``."""
    global _model
    missing = [text for text in dict.fromkeys(documents) if text not in _cache]
    texts = [query_text(request)] + [document_text(text) for text in missing]
    timeout = INDEX_TIMEOUT if len(missing) > COLD else QUERY_TIMEOUT
    embedded = turn_vectors.embed(texts, timeout=timeout)
    if embedded is None:
        return None
    if embedded.model != _model:
        stale = len(_cache) > 0
        _cache.clear()
        _model = embedded.model
        if stale:  # what was cached came from another model: embed it all again
            return _vectors(request, documents)
    if len(_cache) + len(missing) > CACHE_MAX:
        _cache.clear()
    _cache.update(zip(missing, embedded.vectors[1:], strict=True))
    ready = [_cache.get(text) for text in documents]
    vectors = [vector for vector in ready if vector is not None]
    if len(vectors) != len(documents):
        return None  # the cache started over under this call: words alone this time
    return embedded.vectors[0], vectors


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(map(operator.mul, a, b)))


def nearest(request: str, saved: Sequence[str]) -> str | None:
    """The one saved fact ``request`` clearly means, or ``None``. ``saved`` is
    newest first; a fact the request already says is never chosen."""
    if not get_settings().conductor_facts_by_meaning or incognito.active() or not request.strip():
        return None
    candidates = [
        fact for fact in saved[: facts.MEANING_SCANNED] if not facts.literal(fact, request)
    ]
    if not candidates:
        return None
    found = _vectors(request, [*candidates, *facts.BACKGROUND])
    if found is None:
        return None
    query, vectors = found
    cosines = [_cosine(query, vector) for vector in vectors]
    baseline = statistics.median(cosines[len(candidates) :])
    index = facts.nearest(cosines[: len(candidates)], baseline)
    logger.info(
        "facts_by_meaning",
        facts=len(candidates),
        chosen=index is not None,
        lift=round(max(cosines[: len(candidates)]) - baseline, 3),
    )
    return None if index is None else candidates[index]
