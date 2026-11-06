"""The paired-seed seam (``eval_seed``): a seed reaches llama-server only while a
sample sets one, and nothing else in the request changes. CI, no GPU."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx
import pytest

from tests import eval_seed


@pytest.fixture
def bodies() -> Generator[list[dict[str, Any]], None, None]:
    sent: list[dict[str, Any]] = []
    yield sent
    eval_seed.current.seed = None


def _provider(bodies: list[dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        message = {"role": "assistant", "content": "ok"}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    return eval_seed.seeded_provider(httpx.MockTransport(handler))


def test_no_seed_unless_a_sample_sets_one(bodies: list[dict[str, Any]]) -> None:
    provider = _provider(bodies)
    provider.chat([{"role": "user", "content": "hi"}])
    eval_seed.current.seed = 410
    provider.chat([{"role": "user", "content": "hi"}])
    assert "seed" not in bodies[0]
    assert bodies[1]["seed"] == 410
    assert {k: v for k, v in bodies[1].items() if k != "seed"} == bodies[0]


def test_seed_base_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDUCTOR_AGENT_EVALS_SEED", raising=False)
    assert eval_seed.seed_base() is None
    monkeypatch.setenv("CONDUCTOR_AGENT_EVALS_SEED", "410")
    assert eval_seed.seed_base() == 410
