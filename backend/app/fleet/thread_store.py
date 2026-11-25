"""DB-backed :class:`ThreadStore`: the subagent threads conductor opened and
hasn't deleted yet (#262).

Implements the ``ThreadStore`` protocol (``app/fleet/context.py``) over the
``delegate_threads`` table: one row per app thread, under the window's key.
Nothing reuses a row; it is what ``app/fleet/thread_cleanup.py`` deletes
from, so a thread outlives its request only while a question waits on it.

Each operation opens its own short session from the given factory and commits
immediately: a turn holds no transaction open while a delegate
call is in flight, so a minutes-long subagent turn never pins a SQLite write
lock against the route's session.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.db.models import DelegateThread

SessionFactory = Callable[[], Session]


class DbThreadStore:
    def __init__(self, session_factory: SessionFactory) -> None:
        # Public: cleanup deletes from the database the run recorded into.
        self.session_factory = session_factory

    def add(self, master_conversation_id: str, app_name: str, thread_id: int) -> None:
        with self.session_factory() as session:
            session.add(
                DelegateThread(
                    master_conversation_id=master_conversation_id,
                    app_name=app_name,
                    subagent_conversation_id=thread_id,
                )
            )
            session.commit()
