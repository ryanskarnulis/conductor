"""llama.cpp provider: chat completions with tools, spoken over plain httpx.

Conductor's client for the shared workspace ``../llama-swap/`` stack — the same
gemma-4-12b every app agent uses. OpenAI wire format over plain ``httpx`` (no
SDK). This file is the request side: payload, HTTP flow, logging, context
capture. Responses are read by the fleet's vendored chat wire
(``app/ai/agentkit/chat.py``): validated at the boundary, with malformed server
output or tool-call arguments that aren't a JSON object raising typed errors;
nothing is best-effort parsed.

Gemma quirks handled here (matching PCC and chess on the same server):
chain-of-thought arrives in a separate ``reasoning_content`` field and is never
treated as answer text nor echoed back into history, and thinking is toggled per
request via ``chat_template_kwargs``. Conductor keeps thinking OFF by default —
routing turns must be fast — and opts in per call only when a turn genuinely
needs analysis (the model profile's guidance).

What changes with the model — its sampling, the kwarg that toggles thinking,
each role's temperature and cap — is the model's profile
(:mod:`app.ai.model_profile`), picked by the model name. The provider sets the
profile's sampling on every request, so a server-config drift
(``../llama-swap/config.yaml`` carries the same values as defaults) never
changes conductor's behavior.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from app.ai import incognito
from app.ai.agentkit.chat import (
    StreamedTurn,
    WireCompletion,
    bad_status,
    body_text,
    chat_result,
    parse_completion,
    request_failed,
)
from app.ai.context_capture import CallStamp, ContextCapture, context_capture_from_settings
from app.ai.model_profile import PLANNER, ModelProfile, load_profile
from app.ai.provider import (
    ChatResult,
    ProviderRequestError,
    ProviderResponseError,
    SamplingRole,
    ToolSpec,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)

# The rendered-template fetch is diagnostics beside a call that already ran:
# short, and never retried.
_TEMPLATE_TIMEOUT_SECONDS = 10.0


@dataclass
class _Wire:
    """How far one captured call got: the request once built, the response
    once it arrived."""

    request: httpx.Request | None = None
    response: httpx.Response | None = None


class LlamaCppProvider:
    """Chat-completions client for the shared llama-server (behind llama-swap).

    Synchronous by design, matching the sync tool registry; FastAPI runs sync
    callers in worker threads. ``client`` is injectable for tests
    (``httpx.MockTransport``); otherwise the provider owns one.

    ``profile`` is the model's; ``None``, the default, loads the one named for
    ``model``, or the default profile with a warning when none is.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        capture: ContextCapture | None = None,
        profile: ModelProfile | None = None,
        stream: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # A role's call samples and is capped as its phase of the profile
        # says (#150); a call without a role uses the profile's sampling.
        self._profile = profile if profile is not None else load_profile(model)
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

    @property
    def profile(self) -> ModelProfile:
        """The model's profile: its sampling, thinking kwarg and phases."""
        return self._profile

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
        sampled and capped as ``role``'s phase of the profile (#150): its own
        temperature if it has one, and never more than its ``max_tokens``."""
        temperature: float | int | None = None
        if role is not None:
            phase = self._profile.phase(role)
            temperature = phase.temperature
            max_tokens = (
                phase.max_tokens if max_tokens is None else min(max_tokens, phase.max_tokens)
            )
        payload = self._payload(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format is not None:
            payload["response_format"] = response_format
        if self._stream:
            return chat_result(self._post_stream(payload))
        stamp = self._begin_capture()
        if stamp is None:
            return chat_result(self._post(payload, None))
        started = time.monotonic()
        wire = _Wire()
        error: Exception | None = None
        try:
            # Parsed inside the capture so a bad tool-call argument is recorded
            # beside the bytes behind it.
            return chat_result(self._post(payload, wire))
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
        temperature: float | int | None,
    ) -> dict[str, Any]:
        # The profile's sampling: only the keys it sets, so an absent one is
        # the server's default. A role's own temperature goes where the
        # profile's would, so the key order never moves.
        sampling = dict(self._profile.sampling)
        if temperature is not None:
            sampling.pop("temperature", None)
            sampling = {"temperature": temperature, **sampling}
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            # llama-server accepts ``top_k`` and ``chat_template_kwargs``, both
            # OpenAI extensions, as plain body fields (no SDK extra_body).
            **sampling,
        }
        kwarg = self._profile.thinking_kwarg
        if kwarg:
            # A model with no thinking toggle gets no template kwargs at all.
            payload["chat_template_kwargs"] = {kwarg: enable_thinking}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.to_wire() for tool in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any], wire: _Wire | None) -> WireCompletion:
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
            raise request_failed(exc) from exc
        duration_ms = round((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            log.error("llm_request_failed", status=response.status_code, duration_ms=duration_ms)
            raise bad_status(response.status_code, response.text)
        try:
            completion = parse_completion(response.text)
        except ProviderResponseError as exc:
            log.error("llm_response_invalid", duration_ms=duration_ms, error=_cause(exc))
            raise
        choice = completion.choices[0]
        usage = completion.read_usage()
        server = completion.stamp()
        log.info(
            "llm_response",
            duration_ms=duration_ms,
            finish_reason=choice.finish_reason,
            tool_calls=len(choice.message.tool_calls),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            cached_tokens=usage.cached_tokens if usage else None,
            # Beside the counts it explains: none without them, as before.
            prefill_ms=server.prefill_ms if usage and server else None,
        )
        return completion

    def _post_stream(self, payload: dict[str, Any]) -> WireCompletion:
        """One streamed call, read chunk by chunk until done or aborted; the
        answer text and the finish reason assembled into one completion.
        Reasoning deltas are read (they keep the stream moving while the
        model thinks) and dropped, as in a plain call."""
        log = logger.bind(llm_call_id=uuid.uuid4().hex[:8], model=self._model)
        log.info("llm_request", messages=len(payload["messages"]), tools=0, stream=True)
        started = time.monotonic()
        turn = StreamedTurn()
        try:
            with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json={**payload, "stream": True}
            ) as response:
                if response.status_code != 200:
                    response.read()
                    log.error("llm_request_failed", status=response.status_code)
                    raise bad_status(response.status_code, response.text)
                for line in response.iter_lines():
                    if self._abort.is_set():
                        raise ProviderRequestError("llama-server request cut off: aborted")
                    turn.feed(line)
                    if turn.done:
                        break
            completion = turn.completion()
        except (httpx.HTTPError, RuntimeError) as exc:
            log.error("llm_request_failed", error=str(exc))
            raise request_failed(exc) from exc
        except ProviderResponseError as exc:
            log.error("llm_response_invalid", error=_cause(exc))
            raise
        choice = completion.choices[0]
        usage = completion.read_usage()
        log.info(
            "llm_response",
            duration_ms=round((time.monotonic() - started) * 1000),
            finish_reason=choice.finish_reason,
            tool_calls=len(choice.message.tool_calls),
            completion_tokens=usage.completion_tokens if usage else None,
        )
        return completion

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
                    "request": None if request is None else body_text(request.content),
                    "status_code": None if response is None else response.status_code,
                    "response": None if response is None else body_text(response.content),
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


def provider_from_settings() -> LlamaCppProvider:
    """The provider as configured (``LLAMACPP_*`` env / ``.env``)."""
    settings = get_settings()
    return LlamaCppProvider(
        settings.llamacpp_base_url,
        settings.llamacpp_model,
        timeout_seconds=settings.llamacpp_timeout_seconds,
        capture=context_capture_from_settings(),
        profile=profile_from_settings(),
    )


def profile_from_settings() -> ModelProfile:
    """The configured model's profile, with the planner's temperature as
    configured (``CONDUCTOR_PLANNER_TEMPERATURE``, the A/B's knob, #150)."""
    settings = get_settings()
    profile = load_profile(settings.llamacpp_model)
    temperature = settings.conductor_planner_temperature
    if temperature is None:
        return profile
    return profile.with_phase(PLANNER, temperature=temperature)


def _cause(exc: ProviderResponseError) -> str:
    """What failed validation, as the log has always carried it: the
    underlying error's text when there is one."""
    return str(exc.__cause__ or exc)
