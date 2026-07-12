"""llama.cpp provider: chat completions with tools, spoken over plain httpx.

Conductor's client for the shared workspace ``../llama-swap/`` stack — the same
gemma-4-12b every app agent uses. OpenAI wire format over plain ``httpx`` (no
SDK), with every response validated by Pydantic wire models at the boundary:
malformed server output or tool-call arguments that aren't a JSON object raise
typed errors (from ``app/ai/provider.py``); nothing is best-effort parsed.

Gemma quirks handled here (matching PCC and chess on the same server):
chain-of-thought arrives in a separate ``reasoning_content`` field and is never
treated as answer text nor echoed back into history, and thinking is toggled per
request via ``chat_template_kwargs``. Conductor keeps thinking OFF by default —
routing turns must be fast — and opts in per call only when a turn genuinely
needs analysis (the model profile's guidance).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

from app.ai.provider import (
    ChatResult,
    ProviderRequestError,
    ProviderResponseError,
    ToolCall,
    ToolCallArgumentsError,
    ToolSpec,
    Usage,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)

# The negotiated gemma-4 sampling set. Canonical source is
# ../agent-standard/model-profile.md (model knowledge, not app knowledge);
# ../llama-swap/config.yaml carries the same values as server-side defaults, but
# the provider always sets them per request so a server-config drift never
# changes conductor's behavior.
_TEMPERATURE = 1.0
_TOP_P = 0.95
_TOP_K = 64


class _WireFunction(BaseModel):
    name: str
    arguments: str | None = None


class _WireToolCall(BaseModel):
    id: str = ""
    function: _WireFunction


class _WireMessage(BaseModel):
    content: str | None = None
    # Gemma's chain-of-thought channel: validated so unexpected shapes fail
    # loudly, but never surfaced as answer text and never sent back in history.
    reasoning_content: str | None = None
    tool_calls: list[_WireToolCall] = []


class _WireChoice(BaseModel):
    message: _WireMessage
    finish_reason: str | None = None


class _WireCompletion(BaseModel):
    choices: list[_WireChoice] = Field(min_length=1)
    usage: Usage | None = None


class LlamaCppProvider:
    """Chat-completions client for the shared llama-server (behind llama-swap).

    Synchronous by design, matching the sync tool registry; FastAPI runs sync
    callers in worker threads. ``client`` is injectable for tests
    (``httpx.MockTransport``); otherwise the provider owns one.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # One generous read timeout rather than a special-cased first request:
        # a cold load through llama-swap is ~100 s before the first byte, and
        # warm calls never get near it.
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaCppProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One completion turn, optionally offering tools."""
        payload = self._payload(
            messages, tools=tools, enable_thinking=enable_thinking, max_tokens=max_tokens
        )
        return self._result(self._post(payload))

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
            # llama-server accepts these OpenAI extensions as plain body fields
            # (no SDK extra_body indirection needed).
            "top_k": _TOP_K,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.to_wire() for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any]) -> _WireCompletion:
        log = logger.bind(llm_call_id=uuid.uuid4().hex[:8], model=self._model)
        log.info(
            "llm_request",
            messages=len(payload["messages"]),
            tools=len(payload.get("tools", ())),
        )
        started = time.monotonic()
        try:
            response = self._client.post(f"{self._base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            log.error("llm_request_failed", error=str(exc))
            raise ProviderRequestError(f"llama-server request failed: {exc}") from exc
        duration_ms = round((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            log.error("llm_request_failed", status=response.status_code, duration_ms=duration_ms)
            raise ProviderRequestError(
                f"llama-server returned {response.status_code}: {response.text[:500]}"
            )
        try:
            completion = _WireCompletion.model_validate_json(response.text)
        except ValidationError as exc:
            log.error("llm_response_invalid", duration_ms=duration_ms, error=str(exc))
            raise ProviderResponseError(
                f"llama-server response failed wire validation: {exc}"
            ) from exc
        choice = completion.choices[0]
        log.info(
            "llm_response",
            duration_ms=duration_ms,
            finish_reason=choice.finish_reason,
            tool_calls=len(choice.message.tool_calls),
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else None,
            completion_tokens=completion.usage.completion_tokens if completion.usage else None,
        )
        return completion

    @staticmethod
    def _result(completion: _WireCompletion) -> ChatResult:
        choice = completion.choices[0]
        calls: list[ToolCall] = []
        for wire_call in choice.message.tool_calls:
            raw = wire_call.function.arguments
            if raw is None or not raw.strip():
                arguments: Any = {}
            else:
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ToolCallArgumentsError(
                        wire_call.function.name, f"arguments are not valid JSON ({exc.msg})"
                    ) from exc
            if not isinstance(arguments, dict):
                raise ToolCallArgumentsError(
                    wire_call.function.name, "arguments are not a JSON object"
                )
            calls.append(
                ToolCall(id=wire_call.id, name=wire_call.function.name, arguments=arguments)
            )
        # reasoning_content is deliberately dropped here: it never becomes
        # answer text and never re-enters history.
        return ChatResult(
            content=choice.message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            usage=completion.usage,
        )


def provider_from_settings() -> LlamaCppProvider:
    """The provider as configured (``LLAMACPP_*`` env / ``.env``)."""
    settings = get_settings()
    return LlamaCppProvider(
        settings.llamacpp_base_url,
        settings.llamacpp_model,
        timeout_seconds=settings.llamacpp_timeout_seconds,
    )
