"""Paired sampling seeds for the live evals (#53, STANDARD.md §6).

An A/B of a prompt or manifest change compares two arms on one llama-server.
Sample *j* of a scenario sends the sampling seed ``base + j`` in **both** arms
(``evalstats.sample_seed``), so most pairs draw the same stream and the
verdict lives in the pairs the change flipped (``evalstats.mcnemar_exact``).

``CONDUCTOR_AGENT_EVALS_SEED=<base>`` turns it on; unset, nothing here touches
the run — the gate samples unseeded exactly as before. The seed reaches
llama-server through the provider's public ``client=`` argument: a transport
that adds ``"seed"`` to each chat-completion body while a sample is running.
No app code knows about it, so the production provider never sends one.

Pairing is approximate — llama.cpp is not bit-reproducible across batch sizes
or slots — and never worse than unpaired. It does not replace alternating the
arms' blocks (``scripts/eval_campaign.sh``).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app.ai.providers.llamacpp import LlamaCppProvider
from app.config import get_settings


def seed_base() -> int | None:
    raw = os.environ.get("CONDUCTOR_AGENT_EVALS_SEED")
    return int(raw) if raw not in (None, "") else None


class _Current:
    """The seed of the sample in flight (the harness runs one at a time; the
    request runs on TestClient's worker thread, so a contextvar would not
    reach it)."""

    seed: int | None = None


current = _Current()


class SeedingTransport(httpx.BaseTransport):
    """Adds ``"seed": current.seed`` to JSON POST bodies while one is set."""

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        seed = current.seed
        if seed is not None and request.method == "POST":
            request.read()
            body: dict[str, Any] = json.loads(request.content)
            body["seed"] = seed
            content = json.dumps(body).encode()
            headers = httpx.Headers(request.headers)
            headers["content-length"] = str(len(content))
            request = httpx.Request(
                request.method, request.url, headers=headers, content=content,
                extensions=request.extensions,
            )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def seeded_provider(transport: httpx.BaseTransport | None = None) -> LlamaCppProvider:
    """``provider_from_settings()``, with the seeding transport."""
    settings = get_settings()
    timeout = httpx.Timeout(settings.llamacpp_timeout_seconds, connect=10.0)
    return LlamaCppProvider(
        settings.llamacpp_base_url,
        settings.llamacpp_model,
        timeout_seconds=settings.llamacpp_timeout_seconds,
        client=httpx.Client(timeout=timeout, transport=SeedingTransport(transport)),
    )
