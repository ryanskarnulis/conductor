"""Delegate REST client against httpx.MockTransport — no live calls.

Every endpoint, the actor header, the timeout config, and each typed error the
contract defines.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.fleet.delegate import (
    CONDUCTOR_ACTOR,
    DelegateClient,
    DelegateProtocolError,
    DelegateRateLimited,
    DelegateThreadGone,
    DelegateUnavailable,
)

_BASE = "http://app.test/api/agent"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> DelegateClient:
    transport = httpx.MockTransport(handler)
    return DelegateClient(_BASE, client=httpx.Client(transport=transport))


def _assistant_exchange() -> dict[str, object]:
    return {
        "user_message": {
            "id": 1,
            "conversation_id": 12,
            "role": "user",
            "content": "what's due today",
            "tool_calls": None,
            "stop_reason": None,
            "created_at": "2026-07-11T00:00:00Z",
        },
        "assistant_message": {
            "id": 2,
            "conversation_id": 12,
            "role": "assistant",
            "content": "one thing: pay rent",
            "tool_calls": [
                {"tool": "list_tasks", "arguments": {"limit": 5}, "result": "[...]", "error": None}
            ],
            "stop_reason": "completed",
            "created_at": "2026-07-11T00:00:01Z",
        },
    }


def test_create_conversation_returns_id_and_sends_actor_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["actor"] = request.headers.get("X-Agent-Actor")
        seen["body"] = json.loads(request.content or b"{}")
        return httpx.Response(201, json={"id": 12, "title": None})

    client = _client(handler)
    assert client.create_conversation() == 12
    assert seen["method"] == "POST"
    assert seen["url"] == f"{_BASE}/conversations"
    assert seen["actor"] == CONDUCTOR_ACTOR == "agent:conductor"


def test_create_conversation_forwards_title() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 3})

    _client(handler).create_conversation(title="Weekly triage")
    assert seen["body"] == {"title": "Weekly triage"}


def test_send_message_posts_content_and_parses_exchange() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["actor"] = request.headers.get("X-Agent-Actor")
        return httpx.Response(200, json=_assistant_exchange())

    exchange = _client(handler).send_message(12, "what's due today")
    assert seen["url"] == f"{_BASE}/conversations/12/messages"
    assert seen["body"] == {"content": "what's due today"}
    assert seen["actor"] == "agent:conductor"
    assert exchange.assistant_message.content == "one thing: pay rent"
    assert exchange.assistant_message.stop_reason == "completed"
    assert exchange.assistant_message.tool_calls is not None
    assert exchange.assistant_message.tool_calls[0].tool == "list_tasks"
    assert exchange.assistant_message.tool_calls[0].error is None


def test_get_conversation_parses_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "id": 12,
                "title": "Weekly triage",
                "created_at": "2026-07-11T00:00:00Z",
                "updated_at": "2026-07-11T00:00:00Z",
                "messages": [_assistant_exchange()["assistant_message"]],
            },
        )

    detail = _client(handler).get_conversation(12)
    assert detail.id == 12
    assert detail.title == "Weekly triage"
    assert len(detail.messages) == 1


def test_delete_conversation_ok_on_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.headers.get("X-Agent-Actor") == "agent:conductor"
        return httpx.Response(204)

    _client(handler).delete_conversation(12)  # returns None, no raise


def test_404_raises_thread_gone() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    with pytest.raises(DelegateThreadGone):
        _client(handler).send_message(99, "hi")


def test_429_raises_rate_limited_and_captures_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "slow down"})

    with pytest.raises(DelegateRateLimited) as excinfo:
        _client(handler).send_message(12, "hi")
    assert excinfo.value.retry_after == 7


def test_429_without_retry_after_has_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "slow down"})

    with pytest.raises(DelegateRateLimited) as excinfo:
        _client(handler).send_message(12, "hi")
    assert excinfo.value.retry_after is None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_raises_unavailable(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="boom")

    with pytest.raises(DelegateUnavailable):
        _client(handler).send_message(12, "hi")


def test_connect_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(DelegateUnavailable):
        _client(handler).create_conversation()


def test_timeout_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    with pytest.raises(DelegateUnavailable):
        _client(handler).send_message(12, "hi")


def test_unparseable_body_raises_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # 200 but the body isn't a valid MessageExchange.
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(DelegateProtocolError):
        _client(handler).send_message(12, "hi")


def test_non_json_body_raises_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(DelegateProtocolError):
        _client(handler).send_message(12, "hi")


def test_default_client_timeouts_follow_the_latency_profile() -> None:
    # A long read timeout (cold model load ~100s) but a short connect timeout.
    client = DelegateClient(_BASE)
    try:
        timeout = client._client.timeout
        assert timeout.read == 300.0
        assert timeout.connect == 5.0
    finally:
        client.close()
