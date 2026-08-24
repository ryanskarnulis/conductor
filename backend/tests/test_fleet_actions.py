"""The fleet action proxy: a page in conductor's UI acting on an app.

Against httpx.MockTransport — no live app, no network. What matters here is not
the forwarding (that is httpx's job) but the boundary: which apps are reachable,
which paths are, and what a browser can and cannot make conductor do.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import routes_fleet
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.main import app as fastapi_app

_MUSIC = FleetApp(
    name="music",
    title="Music",
    upstream="127.0.0.1:8500",
    agent=AgentSpec(
        description="Downloads and sorts music.",
        api="/api/agent",
        examples=(),
        actions="/api/sorting",
    ),
)
_CHESS = FleetApp(
    name="chess",
    title="Chess",
    upstream="127.0.0.1:8000",
    agent=AgentSpec(description="Plays chess.", api="/api/agent", examples=()),
)


@pytest.fixture
def upstream(client: TestClient) -> Generator[list[httpx.Request], None, None]:
    """Record what the proxy forwards, and answer it with a plain 200."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    with _wired(handler):
        yield seen


def _wired(handler: Callable[[httpx.Request], httpx.Response]) -> "_Wiring":
    return _Wiring(handler)


class _Wiring:
    """Fleet + mock upstream bound to the app for the duration of a test."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler

    def __enter__(self) -> None:
        fastapi_app.state.fleet = Fleet(apps=(_MUSIC, _CHESS))
        transport = httpx.MockTransport(self._handler)
        client = httpx.AsyncClient(transport=transport)
        fastapi_app.dependency_overrides[routes_fleet.get_action_client] = lambda: client

    def __exit__(self, *exc: object) -> None:
        fastapi_app.dependency_overrides.pop(routes_fleet.get_action_client, None)
        fastapi_app.state.fleet = None


def test_a_declared_prefix_is_forwarded(client: TestClient, upstream: list[httpx.Request]) -> None:
    response = client.post(
        "/api/fleet/music/actions/filings", json={"artist": "Zeds Dead", "genre": "Dubstep"}
    )

    assert response.status_code == 200
    assert str(upstream[0].url) == "http://127.0.0.1:8500/api/sorting/filings"
    assert upstream[0].method == "POST"


def test_the_prefix_itself_is_reachable(client: TestClient, upstream: list[httpx.Request]) -> None:
    """Reading the worklist is a GET of the prefix with nothing after it."""
    response = client.get("/api/fleet/music/actions/", params={"groups": 25})

    assert response.status_code == 200
    assert str(upstream[0].url) == "http://127.0.0.1:8500/api/sorting?groups=25"


def test_an_app_that_declares_no_actions_is_not_reachable(client: TestClient) -> None:
    """Chess has an agent and no actions prefix, which is every app but one."""
    with _wired(lambda request: httpx.Response(200)):
        response = client.post("/api/fleet/chess/actions/moves", json={})

    assert response.status_code == 404


def test_an_unknown_app_answers_the_same_as_one_without_actions(client: TestClient) -> None:
    """Telling the two apart only maps the fleet for whoever is asking."""
    with _wired(lambda request: httpx.Response(200)):
        unknown = client.post("/api/fleet/nope/actions/x", json={})
        no_actions = client.post("/api/fleet/chess/actions/x", json={})

    assert unknown.status_code == no_actions.status_code == 404
    assert unknown.json()["detail"].endswith("publishes actions")


def test_a_path_cannot_climb_out_of_the_prefix(client: TestClient) -> None:
    """The whole attack: /api/sorting/../agent/conversations is not /api/sorting.

    Percent-encoded, because that is the form that survives the trip — every
    HTTP client normalizes a literal `..` away before the request is sent, and
    the decoded segment is what arrives here as a path parameter.
    """
    forwarded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(request)
        return httpx.Response(200)

    with _wired(handler):
        response = client.post("/api/fleet/music/actions/%2e%2e/agent/conversations", json={})

    assert response.status_code == 400
    assert forwarded == [], "nothing was forwarded"


def test_the_app_s_own_refusal_reaches_the_page(client: TestClient) -> None:
    """A 422 saying "that artist is already sorted" is the app's answer to the
    person; rewriting it here would only make the page guess."""
    with _wired(lambda request: httpx.Response(422, json={"detail": "ask again as a correction"})):
        response = client.post("/api/fleet/music/actions/filings", json={"artist": "Zeds Dead"})

    assert response.status_code == 422
    assert response.json()["detail"] == "ask again as a correction"


def test_an_app_that_is_down_is_a_502(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _wired(handler):
        response = client.post("/api/fleet/music/actions/filings", json={})

    assert response.status_code == 502
    assert "music" in response.json()["detail"]


def test_the_browser_s_headers_are_not_forwarded(
    client: TestClient, upstream: list[httpx.Request]
) -> None:
    """A proxy that passes them on hands one origin's cookies to another service."""
    client.post(
        "/api/fleet/music/actions/filings",
        json={"artist": "Zeds Dead"},
        headers={"cookie": "session=secret", "authorization": "Bearer hunter2"},
    )

    forwarded = upstream[0].headers
    assert "cookie" not in forwarded
    assert "authorization" not in forwarded


def test_an_oversized_body_is_refused_before_it_is_forwarded(client: TestClient) -> None:
    forwarded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(request)
        return httpx.Response(200)

    with _wired(handler):
        response = client.post(
            "/api/fleet/music/actions/filings",
            content=b"x" * (routes_fleet._MAX_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert forwarded == []


def test_no_fleet_yet_is_a_503_not_a_crash(client: TestClient) -> None:
    fastapi_app.state.fleet = None

    assert client.post("/api/fleet/music/actions/x", json={}).status_code == 503
