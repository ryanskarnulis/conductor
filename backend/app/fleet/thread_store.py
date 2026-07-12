"""DB-backed :class:`ThreadStore`: the Slice 4 swap for ``InMemoryThreadStore``.

Implements the ``ThreadStore`` protocol (``app/fleet/context.py``) over the
``delegate_threads`` table, so subagent threads survive a restart: a follow-up
to the same app in the same master conversation keeps carrying context
app-side even after conductor comes back up. (A stale mapping is harmless — a
pruned subagent thread 404s and the delegate tool recreates it.)

Each operation opens its own short session from the given factory and commits
immediately: the master loop run holds no transaction open while a delegate
call is in flight, so a minutes-long subagent turn never pins a SQLite write
lock against the route's session.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DelegateThread

SessionFactory = Callable[[], Session]


class DbThreadStore:
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def get(self, master_conversation_id: str, app_name: str) -> int | None:
        with self._session_factory() as session:
            row = self._row(session, master_conversation_id, app_name)
            return row.subagent_conversation_id if row is not None else None

    def set(self, master_conversation_id: str, app_name: str, thread_id: int) -> None:
        with self._session_factory() as session:
            row = self._row(session, master_conversation_id, app_name)
            if row is None:
                session.add(
                    DelegateThread(
                        master_conversation_id=master_conversation_id,
                        app_name=app_name,
                        subagent_conversation_id=thread_id,
                    )
                )
            else:
                row.subagent_conversation_id = thread_id
            session.commit()

    def forget(self, master_conversation_id: str, app_name: str) -> None:
        with self._session_factory() as session:
            row = self._row(session, master_conversation_id, app_name)
            if row is not None:
                session.delete(row)
                session.commit()

    @staticmethod
    def _row(session: Session, master_conversation_id: str, app_name: str) -> DelegateThread | None:
        return session.execute(
            select(DelegateThread).where(
                DelegateThread.master_conversation_id == master_conversation_id,
                DelegateThread.app_name == app_name,
            )
        ).scalar_one_or_none()
