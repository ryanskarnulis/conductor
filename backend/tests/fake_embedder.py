"""A deterministic stand-in for the embeddings service (#147).

Each text's memory words, folded through ``same`` (a paraphrase table:
"carport" means "garage"), are hashed into :data:`DIMS` dimensions over one
shared baseline dimension, then normalised. The baseline makes every pair of
texts score above zero, so cosines are compressed like the real model's and
only a shared meaning lifts a turn clear of the median. It records every text
it was sent, so tests can say what never left the process.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence

import httpx
import pytest

from app.ai import memory
from app.ai.embeddings import Embedded, EmbeddingsUnavailable
from app.config import get_settings
from app.services import turn_vectors

DIMS = 1031
BASELINE = 1.0


class FakeEmbedder:
    def __init__(self, same: Mapping[str, str] | None = None, model: str = "fake-embed") -> None:
        self.same = dict(same or {})
        self.model = model
        self.sent: list[str] = []
        self.calls = 0
        # Set to make every call fail, as a stopped or slow service does.
        self.down: str | None = None

    def embed(self, texts: Sequence[str], *, timeout: httpx.Timeout) -> Embedded:
        self.calls += 1
        self.sent.extend(texts)
        if self.down is not None:
            raise EmbeddingsUnavailable(self.down)
        return Embedded(self.model, [self.vector(text) for text in texts])

    def vector(self, text: str) -> list[float]:
        body = text.split(" | ", 1)[-1].split(": ", 1)[-1]
        values = [0.0] * DIMS
        values[0] = BASELINE
        for word in memory.terms(body):
            concept = self.same.get(word, word)
            values[1 + zlib.crc32(concept.encode()) % (DIMS - 1)] += 1.0
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def said(self, needle: str) -> bool:
        """Whether any text sent to it held ``needle``."""
        return any(needle.lower() in text.lower() for text in self.sent)


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeEmbedder) -> FakeEmbedder:
    """Switch meaning on, served by ``fake`` (the switch still turns it off)."""
    monkeypatch.setattr(get_settings(), "conductor_embeddings", True)
    monkeypatch.setattr(
        turn_vectors, "embedder", lambda: fake if get_settings().conductor_embeddings else None
    )
    turn_vectors._reset()
    return fake
