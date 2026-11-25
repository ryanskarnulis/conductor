"""DbThreadStore: the DB-backed ThreadStore behind the fleet/context.py seam."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import DelegateThread
from app.fleet.context import ThreadStore
from app.fleet.thread_store import DbThreadStore


def test_every_opened_thread_is_a_row_of_its_own(session_factory: sessionmaker[Session]) -> None:
    """#262: a window holds many threads per app; the rows outlive the store."""
    store: ThreadStore = DbThreadStore(session_factory)
    store.add("1", "tasks", 41)
    store.add("1", "tasks", 42)
    store.add("1:9", "chess", 7)
    with session_factory() as session:
        rows = [
            (row.master_conversation_id, row.app_name, row.subagent_conversation_id)
            for row in session.execute(select(DelegateThread).order_by(DelegateThread.id)).scalars()
        ]
    assert rows == [("1", "tasks", 41), ("1", "tasks", 42), ("1:9", "chess", 7)]
