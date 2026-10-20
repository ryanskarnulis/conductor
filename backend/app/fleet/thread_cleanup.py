"""Best-effort cleanup of a deleted master conversation's subagent threads.

When a master conversation is trashed, each app thread it opened would
otherwise sit orphaned in that app's own conversation list. This module walks
the conversation's ``delegate_threads`` rows, sends each app a
``DELETE /conversations/{id}`` (the app soft-deletes or resets the thread), and
drops the rows.

Best-effort on purpose: an orphaned thread is inert, not dangerous, so no app
fault ever fails the master delete. Every :class:`DelegateError` is swallowed
and logged (a 404 means the thread is already gone, which is the goal), and the
rows are dropped regardless — the master is gone, so the mapping is dead.

The route runs this as a background task after the delete commits, so the
person's click never waits on the fleet. Deleting a thread waits on no model,
so the client gets a short read timeout instead of the 300 s a delegate turn
needs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import structlog
from sqlalchemy import delete, select

from app.db.models import DelegateThread
from app.fleet.delegate import CONDUCTOR_ACTOR, DelegateClient, DelegateError, DelegateThreadGone
from app.fleet.manifests import Fleet, FleetApp
from app.fleet.thread_store import SessionFactory

logger = structlog.get_logger(__name__)

_CLEANUP_READ_TIMEOUT = 10.0

CleanupClientFactory = Callable[[FleetApp], DelegateClient]


def _cleanup_client(app: FleetApp) -> DelegateClient:
    return DelegateClient(
        app.agent_base_url, actor=CONDUCTOR_ACTOR, read_timeout=_CLEANUP_READ_TIMEOUT
    )


@dataclass(frozen=True)
class SubagentThreadCleanup:
    """Deletes one master conversation's subagent threads across the fleet.

    ``client_factory`` is injectable for tests (a :class:`DelegateClient` over
    ``httpx.MockTransport``).
    """

    fleet: Fleet
    session_factory: SessionFactory
    client_factory: CleanupClientFactory = field(default=_cleanup_client)

    def __call__(self, master_conversation_id: str) -> None:
        with self.session_factory() as session:
            rows = session.execute(
                select(DelegateThread.app_name, DelegateThread.subagent_conversation_id).where(
                    DelegateThread.master_conversation_id == master_conversation_id
                )
            ).all()
        if not rows:
            return

        deleted = 0
        for app_name, thread_id in rows:
            if self._delete_thread(master_conversation_id, app_name, thread_id):
                deleted += 1

        with self.session_factory() as session:
            session.execute(
                delete(DelegateThread).where(
                    DelegateThread.master_conversation_id == master_conversation_id
                )
            )
            session.commit()
        logger.info(
            "subagent_threads_cleaned",
            master_conversation_id=master_conversation_id,
            threads=len(rows),
            deleted=deleted,
        )

    def _delete_thread(self, master_conversation_id: str, app_name: str, thread_id: int) -> bool:
        """Delete one app thread; ``True`` if the app confirmed it (or it was gone)."""
        app = self.fleet.get(app_name)
        if app is None or not app.has_agent:
            # The app left the fleet (or dropped its agent block) since the
            # thread was opened; there is no one to tell.
            logger.info(
                "subagent_thread_cleanup_skipped",
                master_conversation_id=master_conversation_id,
                app=app_name,
                thread_id=thread_id,
                reason="not an agent app in the fleet",
            )
            return False
        try:
            with self.client_factory(app) as client:
                client.delete_conversation(thread_id)
        except DelegateThreadGone:
            return True
        except DelegateError as exc:
            logger.warning(
                "subagent_thread_cleanup_failed",
                master_conversation_id=master_conversation_id,
                app=app_name,
                thread_id=thread_id,
                error=str(exc),
            )
            return False
        return True
