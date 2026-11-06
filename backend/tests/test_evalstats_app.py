"""Conductor's drift guards for the vendored eval gate.

``tests/evalstats.py`` and its spec ``test_evalstats.py`` are vendored from
agent-standard and import nothing of the app. Conductor's harness classifies
on the HTTP response alone (a failed turn carries no stop reason), so what is
pinned here is the route's status for a provider death, the provider's
error text that ``provider_failure_kind`` reads, and the loop's stop
vocabulary against the gate's infra stops (#52).
"""

from __future__ import annotations

from typing import get_args

import httpx
import pytest

from app.ai.loop import FAILURE_STATUS, _StopReason
from app.ai.provider import ProviderError
from app.ai.providers.llamacpp import LlamaCppProvider
from tests.evalstats import (
    INFRA_STATUSES,
    INFRA_STOPS,
    Outcome,
    classify,
    provider_failure_kind,
)

# STANDARD.md §3's vocabulary, the same strings in every app.
_STANDARD_STOPS = {
    "completed",
    "max_iterations",
    "correction_limit",
    "truncated",
    "no_progress",
    "tool_call_limit",
    "cancelled",
    "provider_error",
    "timed_out",
    "internal_error",
}


def test_the_loop_speaks_the_standard_stop_vocabulary() -> None:
    assert set(get_args(_StopReason)) == _STANDARD_STOPS


def test_every_infra_stop_is_a_loop_stop_answered_with_an_infra_status() -> None:
    assert INFRA_STOPS <= set(get_args(_StopReason))
    for stop in INFRA_STOPS:
        assert FAILURE_STATUS[stop] in INFRA_STATUSES


def test_internal_error_is_not_infrastructure() -> None:
    # A tool that raised is a real miss, never retried away.
    assert "internal_error" not in INFRA_STOPS
    assert FAILURE_STATUS["internal_error"] not in INFRA_STATUSES


def test_the_routes_provider_failure_status_is_an_infra_status() -> None:
    # The loop's provider_error stop, which routes_agent.post_message answers with.
    assert FAILURE_STATUS["provider_error"] == 502
    assert 502 in INFRA_STATUSES


def _death(handler: httpx.MockTransport) -> str:
    provider = LlamaCppProvider(
        "http://llm.test/v1", "gemma-4-12b", client=httpx.Client(transport=handler)
    )
    with pytest.raises(ProviderError) as caught:
        provider.chat([{"role": "user", "content": "hi"}])
    # The route's 502 detail wraps the provider's message.
    return f"agent run failed: {caught.value}"


def _outcome(detail: str) -> Outcome:
    return classify(
        status_code=502,
        stop_reason=None,
        error=None,
        provider_failure=provider_failure_kind(detail),
    )


def test_a_4xx_from_the_server_is_a_rejection() -> None:
    detail = _death(httpx.MockTransport(lambda _r: httpx.Response(400, text="ctx overrun")))
    assert _outcome(detail) is Outcome.PROVIDER_REJECTED


def test_a_5xx_from_the_server_is_retried() -> None:
    detail = _death(httpx.MockTransport(lambda _r: httpx.Response(503, text="Loading model")))
    assert _outcome(detail) is Outcome.INFRA


def test_a_dead_socket_is_retried() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert _outcome(_death(httpx.MockTransport(refuse))) is Outcome.INFRA


def test_an_unparseable_body_is_a_rejection() -> None:
    detail = _death(httpx.MockTransport(lambda _r: httpx.Response(200, json={"nope": 1})))
    assert _outcome(detail) is Outcome.PROVIDER_REJECTED
