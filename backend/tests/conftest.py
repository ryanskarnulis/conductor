from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import foreground, think
from app.api import rate_limit, routes_agent, turn_activity
from app.config import get_settings
from app.db.models import Base
from app.db.session import get_db
from app.fleet.manifests import Fleet
from app.fleet.thread_cleanup import SubagentThreadCleanup
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.tools import registry


@pytest.fixture(autouse=True)
def registry_isolation() -> Generator[None, None, None]:
    """Keep the module-level tool registry clean across tests.

    The registry ships empty; tests register their own scratch tools against it.
    Snapshotting and restoring around every test stops a registration in one
    test from leaking into the next.
    """
    saved = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def fleet_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point fleet discovery at an empty directory.

    TestClient's lifespan discovers the fleet; against the real workspace that
    would register live ``ask_<app>`` tools in every API test. An empty dir
    keeps discovery deterministic and offline.
    """
    monkeypatch.setattr(get_settings(), "fleet_manifest_dir", tmp_path)


@pytest.fixture(autouse=True)
def household_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default household timezone (#130), whatever a dev's .env says: the
    tests' fixed timestamps are labelled in it. Tests that need another pin it."""
    monkeypatch.setattr(get_settings(), "conductor_timezone", "America/Denver")


@pytest.fixture(autouse=True)
def no_real_glance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan's `look` must never dial a real app (a dev PCC may be listening
    on its manifest port). Tests that look inject a reader; one that doesn't
    gets "couldn't look", the same as an app that is down."""
    from app.ai import interpreter
    from app.fleet.glance import GlanceError

    def refuse(app: object) -> dict[str, object]:
        raise GlanceError("tests don't read real apps")

    monkeypatch.setattr(interpreter, "read_glance", refuse)


@pytest.fixture(autouse=True)
def no_job_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job watcher (#76) never runs on its own in tests: it would open the
    configured database and dial real apps. Tests build one over the test
    database and drive it with ``sweep()``; a cancel to a job never leaves
    the process unless a test injects a canceller."""
    from app.ai import interpreter
    from app.fleet.jobs import JobError

    monkeypatch.setattr(get_settings(), "conductor_job_watcher", False)

    def refuse(app: object, ref: str) -> object:
        raise JobError("tests don't reach real apps")

    monkeypatch.setattr(interpreter, "cancel_job", refuse)


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recall by meaning (#147) never dials the embeddings service in tests (a
    dev box runs one on 8600): off, as CI has none. Tests that rank by meaning
    inject a fake (``tests/fake_embedder.py``)."""
    from app.services import fact_meaning, turn_vectors

    monkeypatch.setattr(get_settings(), "conductor_embeddings", False)
    turn_vectors._reset()
    fact_meaning._reset()


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    rate_limit._reset()


@pytest.fixture(autouse=True)
def _reset_turn_activity() -> None:
    turn_activity._reset()
    # Turns in flight and cancelled thinks (#78) are process-wide too.
    foreground._reset()
    think._reset()


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(test_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def client(
    db_session: Session, session_factory: sessionmaker[Session]
) -> Generator[TestClient, None, None]:
    """The app over the test database: routes share ``db_session``; the
    thread-store and thread-cleanup dependencies get their own sessions against
    the same engine (cleanup over an empty fleet, so nothing leaves the test)."""
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(session_factory)
    app.dependency_overrides[routes_agent.get_session_factory] = lambda: session_factory
    app.dependency_overrides[routes_agent.get_thread_cleanup] = lambda: SubagentThreadCleanup(
        Fleet(apps=()), session_factory
    )
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
