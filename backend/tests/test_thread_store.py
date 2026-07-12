"""DbThreadStore: the DB-backed ThreadStore behind the fleet/context.py seam."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from app.fleet.context import ThreadStore
from app.fleet.thread_store import DbThreadStore


def test_round_trip_update_and_forget(session_factory: sessionmaker[Session]) -> None:
    store = DbThreadStore(session_factory)

    assert store.get("1", "chess") is None

    store.set("1", "chess", 42)
    assert store.get("1", "chess") == 42

    # Same key overwrites (the 404-recreate path re-points the mapping).
    store.set("1", "chess", 43)
    assert store.get("1", "chess") == 43

    # Distinct apps and distinct master conversations don't collide.
    store.set("1", "tasks", 7)
    store.set("2", "chess", 9)
    assert store.get("1", "tasks") == 7
    assert store.get("2", "chess") == 9
    assert store.get("1", "chess") == 43

    store.forget("1", "chess")
    assert store.get("1", "chess") is None
    store.forget("1", "chess")  # idempotent


def test_persists_across_store_instances(session_factory: sessionmaker[Session]) -> None:
    """The point of the DB store: the mapping outlives the object (and process)."""
    DbThreadStore(session_factory).set("1", "chess", 42)
    fresh: ThreadStore = DbThreadStore(session_factory)
    assert fresh.get("1", "chess") == 42
