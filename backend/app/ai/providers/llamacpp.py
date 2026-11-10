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
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

from app.ai import incognito
from app.ai.context_capture import CallStamp, ContextCapture, context_capture_from_settings
from app.ai.provider import (
    ChatResult,
    ProviderRequestError,
    ProviderResponseError,
    Sampling,
    SamplingRole,
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

# Per-role profiles (#150). Every role starts on the profile above, so request
# bytes are the same as before roles existed; a role moves only on evidence
# (``docs/agent-evals.md``). The planner's temperature is the one knob
# (``CONDUCTOR_PLANNER_TEMPERATURE``) so its A/B can be rerun.
DEFAULT_SAMPLING = Sampling(_TEMPERATURE, _TOP_P, _TOP_K)
ROLE_SAMPLING: Mapping[SamplingRole, Sampling] = {
    "planner": DEFAULT_SAMPLING,
    "narrator": DEFAULT_SAMPLING,
    "speaker": DEFAULT_SAMPLING,
}

# The rendered-template fetch is diagnostics beside a call that already ran:
# short, and never retried.
_TEMPLATE_TIMEOUT_SECONDS = 10.0


@dataclass
class _Wire:
    """How far one captured call got: the request once built, the response
    once it arrived."""

    request: httpx.Request | None = None
    response: httpx.Response | None = None


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


class _WireUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # OpenAI's spelling of the cache hit, ``{"cached_tokens": n}``. Untyped:
    # a server that shapes it differently must not fail the call (#152).
    prompt_tokens_details: Any = None


class _WireCompletion(BaseModel):
    choices: list[_WireChoice] = Field(min_length=1)
    usage: _WireUsage | None = None
    # llama-server's own per-call timings (``cache_n``, ``prompt_n``,
    # ``prompt_ms``…); untyped for the same reason.
    timings: Any = None


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
        capture: ContextCapture | None = None,
        sampling: Sampling | None = None,
        role_sampling: Mapping[SamplingRole, Sampling] | None = None,
        stream: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # The model's sampling profile; the 12B's unless a caller brings its
        # own (the 26B's ``think``, #78). ``role_sampling`` overrides it for
        # the roles it names (#150); a call without a role uses the default.
        self._sampling = sampling or DEFAULT_SAMPLING
        self._role_sampling = dict(role_sampling or {})
        # Streamed calls (the ``think`` lane, #78) can be cut off between
        # chunks: closing a blocked non-streamed read from another thread
        # doesn't unblock it, and llama-swap holds the GPU for a request
        # until it ends, so a turn would wait out the whole thought.
        self._stream = stream
        self._abort = threading.Event()
        # Context capture (#62): ``None``, the default, sends one request per
        # call and records nothing.
        self._capture = capture
        # llama-swap's root: ``/upstream/<model>/apply-template`` hangs off it,
        # not off the OpenAI ``/v1`` prefix.
        self._root = self._base_url.removesuffix("/v1")
        # One generous read timeout rather than a special-cased first request:
        # a cold load through llama-swap is ~100 s before the first byte, and
        # warm calls never get near it.
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0))

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        self._client.close()

    def abort(self) -> None:
        """Cut off a call in flight from another thread (a ``think`` yielding
        the GPU, #78). A streamed call stops reading at its next chunk and
        closes the connection, so llama-swap ends the request and llama-server
        stops generating for a client that left; the call raises. The provider
        is spent."""
        self._abort.set()

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
        response_format: dict[str, Any] | None = None,
        role: SamplingRole | None = None,
    ) -> ChatResult:
        """One completion turn, optionally offering tools, or decoding under a
        ``response_format`` (``json_schema``: llama-server's grammar, #70),
        sampled as ``role``'s profile (#150)."""
        payload = self._payload(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            sampling=self._role_sampling.get(role, self._sampling) if role else self._sampling,
        )
        if response_format is not None:
            payload["response_format"] = response_format
        if self._stream:
            return self._result(self._post_stream(payload))
        stamp = self._begin_capture()
        if stamp is None:
            return self._result(self._post(payload, None))
        started = time.monotonic()
        wire = _Wire()
        error: Exception | None = None
        try:
            # Parsed inside the capture so a bad tool-call argument is recorded
            # beside the bytes behind it.
            return self._result(self._post(payload, wire))
        except Exception as exc:
            error = exc
            raise
        finally:
            self._capture_call(stamp, wire, error, started)

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None,
        enable_thinking: bool,
        max_tokens: int | None,
        sampling: Sampling,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            # llama-server accepts these OpenAI extensions as plain body fields
            # (no SDK extra_body indirection needed).
            "top_k": sampling.top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.to_wire() for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any], wire: _Wire | None) -> _WireCompletion:
        log = logger.bind(llm_call_id=uuid.uuid4().hex[:8], model=self._model)
        log.info(
            "llm_request",
            messages=len(payload["messages"]),
            tools=len(payload.get("tools", ())),
        )
        started = time.monotonic()
        try:
            request = self._client.build_request(
                "POST", f"{self._base_url}/chat/completions", json=payload
            )
            if wire is not None:
                wire.request = request
            response = self._client.send(request)
            if wire is not None:
                wire.response = response
        except (httpx.HTTPError, RuntimeError) as exc:
            # A RuntimeError is a client closed under the call (``abort``).
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
        usage = _usage(completion)
        log.info(
            "llm_response",
            duration_ms=duration_ms,
            finish_reason=choice.finish_reason,
            tool_calls=len(choice.message.tool_calls),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            cached_tokens=usage.cached_tokens if usage else None,
            prefill_ms=usage.prefill_ms if usage else None,
        )
        return completion

    def _post_stream(self, payload: dict[str, Any]) -> _WireCompletion:
        """One streamed call, read chunk by chunk until done or aborted; the
        answer text and the finish reason assembled into one completion.
        Reasoning deltas are read (they keep the stream moving while the
        model thinks) and dropped, as in a plain call."""
        log = logger.bind(llm_call_id=uuid.uuid4().hex[:8], model=self._model)
        log.info("llm_request", messages=len(payload["messages"]), tools=0, stream=True)
        started = time.monotonic()
        content: list[str] = []
        finish: str | None = None
        usage: _WireUsage | None = None
        timings: Any = None
        try:
            with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json={**payload, "stream": True}
            ) as response:
                if response.status_code != 200:
                    response.read()
                    log.error("llm_request_failed", status=response.status_code)
                    raise ProviderRequestError(
                        f"llama-server returned {response.status_code}: {response.text[:500]}"
                    )
                for line in response.iter_lines():
                    if self._abort.is_set():
                        raise ProviderRequestError("llama-server request cut off: aborted")
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if isinstance(chunk.get("usage"), dict):
                        usage = _WireUsage.model_validate(chunk["usage"])
                    if chunk.get("timings") is not None:
                        timings = chunk["timings"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if isinstance(delta.get("content"), str):
                            content.append(delta["content"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except (httpx.HTTPError, RuntimeError) as exc:
            log.error("llm_request_failed", error=str(exc))
            raise ProviderRequestError(f"llama-server request failed: {exc}") from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            log.error("llm_response_invalid", error=str(exc))
            raise ProviderResponseError(f"llama-server stream failed validation: {exc}") from exc
        log.info(
            "llm_response",
            duration_ms=round((time.monotonic() - started) * 1000),
            finish_reason=finish,
            tool_calls=0,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        message = _WireMessage(content="".join(content) or None)
        return _WireCompletion(
            choices=[_WireChoice(message=message, finish_reason=finish)],
            usage=usage,
            timings=timings,
        )

    def _begin_capture(self) -> CallStamp | None:
        # An incognito turn's calls (#82) are never captured: they hold its words.
        if self._capture is None or incognito.active():
            return None
        try:
            return self._capture.begin()
        except Exception as exc:
            logger.warning("context_capture_failed", error=str(exc), error_type=type(exc).__name__)
            return None

    def _capture_call(
        self, stamp: CallStamp, wire: _Wire, error: Exception | None, started: float
    ) -> None:
        """Hand one finished call to the capture. Never raises."""
        assert self._capture is not None
        try:
            request, response = wire.request, wire.response
            self._capture.record(
                stamp,
                {
                    "url": None if request is None else str(request.url),
                    "ms": round((time.monotonic() - started) * 1000),
                    "request": None if request is None else _body_text(request.content),
                    "status_code": None if response is None else response.status_code,
                    "response": None if response is None else _body_text(response.content),
                    "error": "" if error is None else f"{type(error).__name__}: {error}",
                    # A server that never answered cannot render either.
                    "template": None
                    if request is None or response is None
                    else self._render_template(request.content),
                },
            )
        except Exception as exc:
            logger.warning("context_capture_failed", error=str(exc), error_type=type(exc).__name__)

    def _render_template(self, body: bytes) -> dict[str, str]:
        """The prompt llama-server's chat template renders from ``body``.

        Asked of the server, not re-rendered here: the text the model tokenized
        is whatever its own template makes of these bytes. llama-swap routes it
        only under ``/upstream/<model>/``.
        """
        try:
            response = self._client.post(
                f"{self._root}/upstream/{self._model}/apply-template",
                content=body,
                headers={"content-type": "application/json"},
                timeout=_TEMPLATE_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:500]}"}
            prompt = response.json().get("prompt")
            if not isinstance(prompt, str):
                return {"error": "apply-template answered without a prompt string"}
            return {"prompt": prompt}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

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
            usage=_usage(completion),
        )


def provider_from_settings() -> LlamaCppProvider:
    """The provider as configured (``LLAMACPP_*`` env / ``.env``)."""
    settings = get_settings()
    return LlamaCppProvider(
        settings.llamacpp_base_url,
        settings.llamacpp_model,
        timeout_seconds=settings.llamacpp_timeout_seconds,
        capture=context_capture_from_settings(),
        role_sampling=role_sampling_from_settings(),
    )


def role_sampling_from_settings() -> Mapping[SamplingRole, Sampling]:
    """:data:`ROLE_SAMPLING`, with the planner's temperature as configured."""
    temperature = get_settings().conductor_planner_temperature
    if temperature is None:
        return ROLE_SAMPLING
    planner = ROLE_SAMPLING["planner"]
    return {**ROLE_SAMPLING, "planner": Sampling(temperature, planner.top_p, planner.top_k)}


def _number(value: Any) -> float | None:
    """A JSON number, or ``None`` for anything else (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _usage(completion: _WireCompletion) -> Usage | None:
    """The server's counts for one call (#152): OpenAI's ``usage`` plus the
    prompt-cache hit and prefill time from llama-server's ``timings``. Any
    field the server leaves out stays ``None``; none can fail a call."""
    wire = completion.usage
    if wire is None:
        return None
    timings = completion.timings if isinstance(completion.timings, dict) else {}
    details = wire.prompt_tokens_details if isinstance(wire.prompt_tokens_details, dict) else {}
    cached = _number(timings.get("cache_n"))
    if cached is None:
        cached = _number(details.get("cached_tokens"))
    return Usage(
        prompt_tokens=wire.prompt_tokens,
        completion_tokens=wire.completion_tokens,
        total_tokens=wire.total_tokens,
        cached_tokens=None if cached is None else int(cached),
        prefill_ms=_number(timings.get("prompt_ms")),
    )


def _body_text(body: bytes) -> str:
    """A body as text, byte for byte when it is UTF-8 (JSON always is)."""
    return body.decode("utf-8", errors="replace")
