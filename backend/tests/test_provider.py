"""LlamaCppProvider wire parsing against an httpx.MockTransport (no GPU).

Every case drives the real ``chat()`` path — payload build, HTTP round trip,
Pydantic wire validation, result assembly — with a mocked transport standing in
for llama-server, so parsing and error mapping are asserted at the wire shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.ai.provider import (
    ProviderRequestError,
    ProviderResponseError,
    Sampling,
    ToolCallArgumentsError,
    ToolSpec,
)
from app.ai.providers.llamacpp import LlamaCppProvider

_Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: _Handler) -> tuple[LlamaCppProvider, list[httpx.Request]]:
    """A provider wired to a recording MockTransport that runs ``handler``."""
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(recording))
    return LlamaCppProvider("http://test/v1", "gemma-4-12b", client=client), requests


def _response(
    message: dict[str, Any], *, finish_reason: str = "stop", status: int = 200
) -> httpx.Response:
    body = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    return httpx.Response(status, json=body)


def _ok(message: dict[str, Any], **kw: Any) -> _Handler:
    return lambda _request: _response(message, **kw)


def test_happy_text_turn() -> None:
    provider, _ = _provider(_ok({"content": "all set"}))
    result = provider.chat([{"role": "user", "content": "hi"}])
    assert result.content == "all set"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage is not None and result.usage.total_tokens == 3


def test_tool_call_turn_parses_json_string_arguments() -> None:
    message = {
        "content": None,
        "tool_calls": [
            {"id": "call_1", "function": {"name": "do_it", "arguments": '{"a": 1, "b": "x"}'}}
        ],
    }
    provider, _ = _provider(_ok(message, finish_reason="tool_calls"))
    result = provider.chat([{"role": "user", "content": "go"}])
    assert result.content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "do_it"
    assert call.arguments == {"a": 1, "b": "x"}


def test_empty_tool_call_arguments_default_to_empty_dict() -> None:
    message = {
        "content": None,
        "tool_calls": [{"id": "c", "function": {"name": "noargs", "arguments": ""}}],
    }
    provider, _ = _provider(_ok(message, finish_reason="tool_calls"))
    result = provider.chat([{"role": "user", "content": "go"}])
    assert result.tool_calls[0].arguments == {}


def test_malformed_tool_call_arguments_raise_typed_error() -> None:
    message = {
        "content": None,
        "tool_calls": [{"id": "c", "function": {"name": "do_it", "arguments": "{not json"}}],
    }
    provider, _ = _provider(_ok(message, finish_reason="tool_calls"))
    with pytest.raises(ToolCallArgumentsError) as exc_info:
        provider.chat([{"role": "user", "content": "go"}])
    assert exc_info.value.tool_name == "do_it"


def test_non_object_tool_call_arguments_raise_typed_error() -> None:
    message = {
        "content": None,
        "tool_calls": [{"id": "c", "function": {"name": "do_it", "arguments": "[1, 2]"}}],
    }
    provider, _ = _provider(_ok(message, finish_reason="tool_calls"))
    with pytest.raises(ToolCallArgumentsError, match="not a JSON object"):
        provider.chat([{"role": "user", "content": "go"}])


def test_reasoning_content_is_dropped() -> None:
    message = {"content": "the answer", "reasoning_content": "secret chain of thought"}
    provider, _ = _provider(_ok(message))
    result = provider.chat([{"role": "user", "content": "think"}])
    assert result.content == "the answer"
    # There is nowhere for reasoning to live on the result, and it never
    # round-trips into the next request's history.
    assert "secret" not in json.dumps(result.to_message())


def test_connection_failure_becomes_provider_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, _ = _provider(handler)
    with pytest.raises(ProviderRequestError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_timeout_becomes_provider_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider, _ = _provider(handler)
    with pytest.raises(ProviderRequestError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_non_200_status_becomes_provider_request_error() -> None:
    provider, _ = _provider(lambda _r: httpx.Response(503, text="unavailable"))
    with pytest.raises(ProviderRequestError, match="503"):
        provider.chat([{"role": "user", "content": "hi"}])


def test_invalid_wire_body_becomes_provider_response_error() -> None:
    provider, _ = _provider(lambda _r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderResponseError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_request_payload_carries_sampling_params_and_thinking_flag() -> None:
    provider, requests = _provider(_ok({"content": "ok"}))
    spec = ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})
    provider.chat(
        [{"role": "user", "content": "hi"}],
        tools=[spec],
        enable_thinking=True,
        max_tokens=128,
    )
    payload = json.loads(requests[0].content)
    # Model-profile sampling set (../agent-standard/model-profile.md).
    assert payload["model"] == "gemma-4-12b"
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 128
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "t"


def test_thinking_defaults_off_and_optional_fields_omitted() -> None:
    provider, requests = _provider(_ok({"content": "ok"}))
    provider.chat([{"role": "user", "content": "hi"}])
    payload = json.loads(requests[0].content)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "max_tokens" not in payload


# --- per-role sampling (#150) ----------------------------------------------------------


def _sent(provider: LlamaCppProvider, requests: list[httpx.Request], **kw: Any) -> dict[str, Any]:
    requests.clear()
    provider.chat([{"role": "user", "content": "hi"}], **kw)
    sent: dict[str, Any] = json.loads(requests[0].content)
    return sent


def test_every_role_sends_the_same_bytes_as_no_role_by_default() -> None:
    provider, requests = _provider(_ok({"content": "ok"}))
    plain = _sent(provider, requests)
    for role in ("planner", "narrator", "speaker"):
        assert _sent(provider, requests, role=role) == plain
    assert (plain["temperature"], plain["top_p"], plain["top_k"]) == (1.0, 0.95, 64)


def test_a_role_with_its_own_sampling_changes_only_that_role() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response({"content": "ok"})

    provider = LlamaCppProvider(
        "http://test/v1",
        "gemma-4-12b",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        role_sampling={"planner": Sampling(temperature=0.3, top_p=0.9, top_k=40)},
    )
    planner = _sent(provider, requests, role="planner")
    assert (planner["temperature"], planner["top_p"], planner["top_k"]) == (0.3, 0.9, 40)
    for kw in ({"role": "narrator"}, {"role": "speaker"}, {}):
        other = _sent(provider, requests, **kw)
        assert (other["temperature"], other["top_p"], other["top_k"]) == (1.0, 0.95, 64)


def test_a_provider_sampling_is_every_roles_unless_the_role_names_its_own() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response({"content": "ok"})

    provider = LlamaCppProvider(
        "http://test/v1",
        "gemma-4-26b-a4b",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sampling=Sampling(temperature=0.5, top_p=0.9, top_k=40),
    )
    assert _sent(provider, requests, role="narrator")["temperature"] == 0.5


# --- the prompt cache (#152) -----------------------------------------------------------

# What llama-server answers (b9935): OpenAI's usage plus its own timings.
_USAGE = {
    "prompt_tokens": 19,
    "completion_tokens": 8,
    "total_tokens": 27,
    "prompt_tokens_details": {"cached_tokens": 7},
}
_TIMINGS = {"cache_n": 7, "prompt_n": 12, "prompt_ms": 105.135, "predicted_n": 8}


def _answering(**extra: Any) -> LlamaCppProvider:
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], **extra}
    provider, _ = _provider(lambda _request: httpx.Response(200, json=body))
    return provider


def test_a_call_carries_cached_tokens_and_prefill_time() -> None:
    result = _answering(usage=_USAGE, timings=_TIMINGS).chat([{"role": "user", "content": "hi"}])
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (19, 8)
    assert result.usage.cached_tokens == 7
    assert result.usage.prefill_ms == 105.135


def test_a_total_miss_is_a_measured_zero() -> None:
    timings = {"cache_n": 0, "prompt_n": 19, "prompt_ms": 155.8}
    result = _answering(usage=_USAGE, timings=timings).chat([{"role": "user", "content": "hi"}])
    assert result.usage is not None
    assert result.usage.cached_tokens == 0


@pytest.mark.parametrize(
    ("extra", "cached", "prefill"),
    [
        # OpenAI's field alone still counts the hit; there is no prefill time.
        ({"usage": _USAGE}, 7, None),
        # Timings alone: no usage, so nothing to attach them to.
        ({"timings": _TIMINGS}, None, None),
        # A server that answers neither: unknown, not zero.
        ({"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}, None, None),
        # Wrong shapes and nulls never fail the call.
        ({"usage": {**_USAGE, "prompt_tokens_details": "x"}, "timings": "x"}, None, None),
        ({"usage": _USAGE, "timings": {"cache_n": None, "prompt_ms": "fast"}}, 7, None),
        ({"usage": _USAGE, "timings": {"cache_n": True, "prompt_ms": False}}, 7, None),
    ],
)
def test_missing_or_odd_cache_fields_never_break_a_call(
    extra: dict[str, Any], cached: int | None, prefill: float | None
) -> None:
    result = _answering(**extra).chat([{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    if "usage" not in extra:
        assert result.usage is None
    else:
        assert result.usage is not None
        assert (result.usage.cached_tokens, result.usage.prefill_ms) == (cached, prefill)


def test_a_streamed_call_carries_the_cache_figures_from_its_last_chunk() -> None:
    chunks = _sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": _USAGE, "timings": _TIMINGS},
    )
    provider, _ = _streaming(lambda _p: iter(chunks))
    result = provider.chat([{"role": "user", "content": "hi"}])
    assert result.usage is not None
    assert (result.usage.cached_tokens, result.usage.prefill_ms) == (7, 105.135)


# --- the think lane's streamed calls (#78) ---------------------------------------------


def _sse(*chunks: dict[str, Any]) -> list[bytes]:
    return [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks] + [b"data: [DONE]\n\n"]


def _streaming(
    lines: Callable[[LlamaCppProvider], Any],
) -> tuple[LlamaCppProvider, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    holder: dict[str, LlamaCppProvider] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=lines(holder["provider"]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = LlamaCppProvider(
        "http://test/v1",
        "gemma-4-26b-a4b",
        client=client,
        sampling=Sampling(temperature=0.5, top_p=0.9, top_k=40),
        stream=True,
    )
    holder["provider"] = provider
    return provider, requests


def test_a_streamed_call_assembles_the_answer_and_drops_the_thought() -> None:
    chunks = _sse(
        {"choices": [{"delta": {"reasoning_content": "let me think"}}]},
        {"choices": [{"delta": {"content": '{"summary": '}}]},
        {"choices": [{"delta": {"content": '"x"}'}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}},
    )
    provider, requests = _streaming(lambda _p: iter(chunks))
    result = provider.chat([{"role": "user", "content": "hi"}], enable_thinking=True)
    assert result.content == '{"summary": "x"}'
    assert result.finish_reason == "stop"
    assert result.usage is not None and result.usage.completion_tokens == 3
    payload = json.loads(requests[0].content)
    assert payload["stream"] is True
    assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (0.5, 0.9, 40)
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_abort_cuts_a_streamed_call_off_between_chunks() -> None:
    def lines(provider: LlamaCppProvider) -> Any:
        yield b'data: {"choices": [{"delta": {"reasoning_content": "hmm"}}]}\n\n'
        provider.abort()  # a turn needs the GPU, from another thread
        yield b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    provider, _ = _streaming(lines)
    with pytest.raises(ProviderRequestError, match="cut off"):
        provider.chat([{"role": "user", "content": "hi"}])
