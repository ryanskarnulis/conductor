"""Conductor's drift guards for the vendored eval gate.

``tests/evalstats.py`` and its spec ``test_evalstats.py`` are vendored from
agent-standard and import nothing of the app. Conductor's harness classifies
on the HTTP response alone (a failed turn carries no stop reason), so what is
pinned here is the route's status for a provider death and the provider's
error text that ``provider_failure_kind`` reads.
"""

from __future__ import annotations

import httpx
import pytest

from app.ai.provider import ProviderError
from app.ai.providers.llamacpp import LlamaCppProvider
from tests.evalstats import INFRA_STATUSES, Outcome, classify, provider_failure_kind


def test_the_routes_provider_failure_status_is_an_infra_status() -> None:
    # routes_agent.post_message maps every ProviderError to 502.
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
