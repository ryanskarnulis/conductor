"""Subagent-thread cleanup (#262), against httpx.MockTransport.

Each app answers the thread ``DELETE`` the way the scenario needs; what matters
is that every thread to go gets a ``DELETE`` with conductor's actor, that no
app fault escapes, that its rows are dropped either way, and that a thread a
question may still need is left alone.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api import routes_agent
from app.db.models import Conversation, DelegateThread
from app.fleet.delegate import CONDUCTOR_ACTOR, DelegateClient
from app.fleet.manifests import AgentSpec, Fleet, FleetApp
from app.fleet.thread_cleanup import SubagentThreadCleanup
from app.fleet.thread_store import DbThreadStore
from app.main import app as fastapi_app


def _agent_app(name: str, port: int) -> FleetApp:
    return FleetApp(
        name=name,
        title=name.title(),
        upstream=f"127.0.0.1:{port}",
        agent=AgentSpec(description=f"{name} agent.", api="/api/agent", examples=()),
    )


_FLEET = Fleet(
    apps=(
        _agent_app("tasks", 8100),
        _agent_app("chess", 8000),
        FleetApp(name="odysseus", title="Odysseus", upstream="127.0.0.1:7000", agent=None),
    )
)

Handler = Callable[[httpx.Request], httpx.Response]


def _cleanup(
    session_factory: sessionmaker[Session], handler: Handler, seen: list[httpx.Request]
) -> SubagentThreadCleanup:
    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(app: FleetApp) -> DelegateClient:
        http = httpx.Client(transport=httpx.MockTransport(recording))
        return DelegateClient(app.agent_base_url, client=http)

    return SubagentThreadCleanup(_FLEET, session_factory, client_factory=factory)


def _rows(session_factory: sessionmaker[Session], master: str) -> list[tuple[str, int]]:
    with session_factory() as session:
        return [
            (row.app_name, row.subagent_conversation_id)
            for row in session.execute(
                select(DelegateThread).where(DelegateThread.master_conversation_id == master)
            ).scalars()
        ]


def test_every_mapped_thread_is_deleted_and_rows_dropped(
    session_factory: sessionmaker[Session],
) -> None:
    store = DbThreadStore(session_factory)
    store.add("1", "tasks", 41)
    store.add("1", "chess", 7)
    store.add("2", "tasks", 99)  # another master's thread is left alone
    seen: list[httpx.Request] = []

    _cleanup(session_factory, lambda _: httpx.Response(204), seen)("1")

    assert sorted(str(request.url) for request in seen) == [
        "http://127.0.0.1:8000/api/agent/conversations/7",
        "http://127.0.0.1:8100/api/agent/conversations/41",
    ]
    assert all(request.method == "DELETE" for request in seen)
    assert all(request.headers["X-Agent-Actor"] == CONDUCTOR_ACTOR for request in seen)
    assert _rows(session_factory, "1") == []
    assert _rows(session_factory, "2") == [("tasks", 99)]


def test_app_faults_are_swallowed_and_rows_still_dropped(
    session_factory: sessionmaker[Session],
) -> None:
    """404 (already gone), 503, a transport error, and an app no longer in the
    fleet: none escapes, and the dead mapping goes regardless."""
    store = DbThreadStore(session_factory)
    store.add("1", "tasks", 41)
    store.add("1", "chess", 7)
    store.add("1", "odysseus", 3)  # inert now: no agent to tell
    store.add("1", "retired", 5)  # left the fleet entirely

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8100:
            return httpx.Response(404)
        raise httpx.ConnectError("connection refused", request=request)

    seen: list[httpx.Request] = []
    _cleanup(session_factory, handler, seen)("1")

    assert {request.url.port for request in seen} == {8000, 8100}
    assert _rows(session_factory, "1") == []


def test_a_master_with_no_threads_makes_no_calls(session_factory: sessionmaker[Session]) -> None:
    seen: list[httpx.Request] = []
    _cleanup(session_factory, lambda _: httpx.Response(204), seen)("1")
    assert seen == []


def test_deleting_a_conversation_cleans_up_its_threads(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The timeline is never deleted (#82); a side thread (#139) will be.
    with session_factory() as session:
        thread = Conversation(title="side", kind="thread")
        session.add(thread)
        session.commit()
        conversation_id = thread.id
    DbThreadStore(session_factory).add(str(conversation_id), "tasks", 41)
    seen: list[httpx.Request] = []
    fastapi_app.dependency_overrides[routes_agent.get_thread_cleanup] = lambda: _cleanup(
        session_factory, lambda _: httpx.Response(503), seen
    )

    response = client.delete(f"/api/agent/conversations/{conversation_id}")

    # An app fault never fails the person's delete.
    assert response.status_code == 204
    assert [str(request.url) for request in seen] == [
        "http://127.0.0.1:8100/api/agent/conversations/41"
    ]
    assert _rows(session_factory, str(conversation_id)) == []


def test_a_conversation_s_threads_go_from_every_window(
    session_factory: sessionmaker[Session],
) -> None:
    store = DbThreadStore(session_factory)
    store.add("1", "tasks", 41)
    store.add("1:30", "tasks", 42)  # a later window of the same conversation
    store.add("11", "tasks", 43)  # another conversation whose id starts the same
    seen: list[httpx.Request] = []

    _cleanup(session_factory, lambda _: httpx.Response(204), seen)("1")

    assert sorted(str(request.url).rsplit("/", 1)[1] for request in seen) == ["41", "42"]
    assert _rows(session_factory, "11") == [("tasks", 43)]


def test_settle_deletes_what_the_request_is_done_with_and_earlier_windows(
    session_factory: sessionmaker[Session],
) -> None:
    store = DbThreadStore(session_factory)
    store.add("1", "tasks", 40)  # an earlier window's: its cards lapsed with it
    store.add("1:30", "tasks", 41)  # this request's, nothing waits on it
    store.add("1:30", "tasks", 42)  # this window's, a card waits on it
    store.add("1:30", "chess", 7)  # another request's, still running
    store.add("2", "tasks", 99)  # another conversation's
    seen: list[httpx.Request] = []

    _cleanup(session_factory, lambda _: httpx.Response(204), seen).settle(
        1, "1:30", frozenset({("tasks", 41), ("tasks", 77)})
    )

    assert sorted(str(request.url).rsplit("/", 1)[1] for request in seen) == ["40", "41"]
    assert _rows(session_factory, "1:30") == [("tasks", 42), ("chess", 7)]
    assert _rows(session_factory, "2") == [("tasks", 99)]
