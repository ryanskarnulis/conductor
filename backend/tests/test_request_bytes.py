"""The bytes of a planner, a narrator and a think request, pinned (agent-standard#32).

Each call goes out through the settings-backed factory and the code that
makes it (:class:`Runtime`, the thinker's ``_call``) over a faked server, and
``request.content`` is compared with the bytes conductor sent before sampling
and caps came from model profiles (``app/ai/profiles/``). A diff here is a
model-facing change: it runs the eval gate (``CLAUDE.md``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.ai import compiler, narrator
from app.ai import trace as run_trace
from app.ai.providers.llamacpp import LlamaCppProvider, provider_from_settings
from app.ai.runtime import Runtime
from app.config import get_settings
from app.fleet.manifests import Fleet
from app.services.thinker import Thinker, think_provider_from_settings

_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You plan."},
    {"role": "user", "content": "add a task to pay rent"},
]
_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "plan", "strict": True, "schema": {"type": "object"}},
}

_HEAD = (
    b'"messages":[{"role":"system","content":"You plan."},'
    b'{"role":"user","content":"add a task to pay rent"}],'
    b'"temperature":1.0,"top_p":0.95,"top_k":64,'
)
_SCHEMA = (
    b'"response_format":{"type":"json_schema","json_schema":'
    b'{"name":"plan","strict":true,"schema":{"type":"object"}}}'
)
PLANNER = (
    b'{"model":"gemma-4-12b",'
    + _HEAD
    + b'"chat_template_kwargs":{"enable_thinking":false},"max_tokens":1500,'
    + _SCHEMA
    + b"}"
)
NARRATOR = (
    b'{"model":"gemma-4-12b",'
    + _HEAD
    + b'"chat_template_kwargs":{"enable_thinking":false},"max_tokens":300}'
)
THINK = (
    b'{"model":"gemma-4-26b-a4b",'
    + _HEAD
    + b'"chat_template_kwargs":{"enable_thinking":true},"max_tokens":6000,'
    + _SCHEMA.replace(b'"name":"plan"', b'"name":"proposal"')
    + b',"stream":true}'
)


@pytest.fixture
def default_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """The settings as shipped: no ``.env``, none of the knobs set."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "LLAMACPP_MODEL",
        "CONDUCTOR_MAX_TOKENS",
        "CONDUCTOR_PLANNER_TEMPERATURE",
        "CONDUCTOR_THINK_MODEL",
        "CONDUCTOR_CONTEXT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _over(provider: LlamaCppProvider, response: httpx.Response) -> list[bytes]:
    """Point ``provider`` at a fake server; the bodies it receives."""
    bodies: list[bytes] = []

    def server(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return response

    provider._client.close()
    provider._client = httpx.Client(transport=httpx.MockTransport(server))
    return bodies


_ANSWER = httpx.Response(
    200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
)


def test_a_planner_and_a_narrator_call_send_the_pinned_bytes(default_settings: None) -> None:
    with provider_from_settings() as provider:
        bodies = _over(provider, _ANSWER)
        rt = Runtime(provider, max_tokens=get_settings().conductor_max_tokens)
        trace = run_trace.RunTrace(run_id="pinned", actor="test", utterance="pinned")
        rt.chat(
            _MESSAGES,
            trace,
            role="compiler",
            max_tokens=compiler.MAX_TOKENS,
            response_format=_FORMAT,
        )
        rt.chat(_MESSAGES, trace, role="narrator", max_tokens=narrator.MAX_TOKENS)
    assert bodies == [PLANNER, NARRATOR]


def test_a_think_call_sends_the_pinned_bytes(
    default_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.ai.think.response_format",
        lambda fleet: {**_FORMAT, "json_schema": {**_FORMAT["json_schema"], "name": "proposal"}},
    )
    provider = think_provider_from_settings()
    stream = httpx.Response(
        200,
        content=b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n",
    )
    bodies = _over(provider, stream)
    thinker = Thinker(
        cast("sessionmaker[Session]", None), Fleet(apps=()), provider=lambda: provider
    )
    try:
        assert thinker._call(provider, _MESSAGES, "think-pinned").content == "ok"
    finally:
        provider.close()
    assert bodies == [THINK]
