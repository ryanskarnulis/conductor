from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import rate_limit, routes_agent, turn_activity
from app.config import get_settings
from app.db.models import Base
from app.db.session import get_db
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
def _reset_rate_limiter() -> None:
    rate_limit._reset()


@pytest.fixture(autouse=True)
def _reset_turn_activity() -> None:
    turn_activity._reset()


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
    thread-store dependency gets its own sessions against the same engine."""
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(session_factory)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
