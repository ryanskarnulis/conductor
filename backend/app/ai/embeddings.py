"""Embeddings client for the shared workspace service (``../embeddings``, #147).

EmbeddingGemma 2 behind a llama.cpp server on 8600, speaking the OpenAI
``/v1/embeddings`` shape: one POST, a list of strings in, a list of vectors
out (768-dim, L2-normalised, so a dot product is the cosine). The server adds
no prompt prefix; the model wants an asymmetric pair, so the client builds
them: :func:`query_text` for what is searched for, :func:`document_text` for
a turn. Plain httpx, no SDK. Same as PCC's ``app/ai/embeddings.py``.

The service is optional: any failure (refused, slow, a non-200, a body that
doesn't parse) is one :class:`EmbeddingsUnavailable`, which memory turns into
word overlap alone (:mod:`app.services.turn_vectors`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

# A query a turn waits on: the service answers in ~10 ms warm, so 0.3 s is
# slack for a busy box and still a cost nobody notices when it is down.
QUERY_TIMEOUT = httpx.Timeout(0.3, connect=0.2)
# Embedding the turns that just closed, on the response's path: a few short texts.
INDEX_TIMEOUT = httpx.Timeout(1.0, connect=0.2)
# The backfill, off any turn's path: a cold service and a big batch.
BATCH_TIMEOUT = httpx.Timeout(120.0, connect=2.0)
# Texts per request in the backfill.
BATCH_SIZE = 16


def query_text(query: str) -> str:
    """Words to search for, as EmbeddingGemma's retrieval query."""
    return f"task: search result | query: {' '.join(query.split())}"


def document_text(text: str) -> str:
    """A turn as EmbeddingGemma's retrieval document (it has no title)."""
    return f"title: none | text: {' '.join(text.split())}"


class EmbeddingsUnavailable(Exception):
    """No usable vectors: the service is down, slow, or answered nonsense."""


@dataclass(frozen=True)
class Embedded:
    """Vectors in input order, and the model the server says made them."""

    model: str
    vectors: list[list[float]]


class Embeds(Protocol):
    """What memory needs of an embedder (tests inject a fake)."""

    def embed(self, texts: Sequence[str], *, timeout: httpx.Timeout) -> Embedded: ...


class Embedder:
    """``POST {base_url}/embeddings`` for a batch of texts."""

    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._client = client if client is not None else httpx.Client()

    def embed(self, texts: Sequence[str], *, timeout: httpx.Timeout) -> Embedded:
        if not texts:
            raise EmbeddingsUnavailable("nothing to embed")
        try:
            response = self._client.post(self._url, json={"input": list(texts)}, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            data = sorted(body["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in data]
            model = str(body["model"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingsUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingsUnavailable(f"asked for {len(texts)} vectors, got {len(vectors)}")
        return Embedded(model=model, vectors=vectors)
