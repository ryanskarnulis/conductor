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
