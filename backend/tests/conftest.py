from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import registry


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


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
