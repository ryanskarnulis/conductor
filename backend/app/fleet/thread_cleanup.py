"""Best-effort deletion of the subagent threads conductor is done with (#262).

Every ask but an answer opens a fresh app thread (``app/fleet/context.py``),
recorded in ``delegate_threads`` under its window's key (``thread_key``:
``"<conversation id>"`` or ``"<conversation id>:<first message id>"``). A
thread is kept only while a question waits on it: an open confirm card (its
token is bound to the thread) or a waiting plan's app question (its answer
goes back there). Cards lapse with their window, and a closed side thread
takes no turns, so the threads to delete are:

- :meth:`SubagentThreadCleanup.settle`, after a request whose records are
  stored: the threads it sent to (or whose question it closed) that no
  question waits on, and every thread of the conversation's earlier windows.
  It only touches threads its own request used or a window that is over, so
  a turn still running elsewhere never loses its thread;
- :meth:`SubagentThreadCleanup.__call__`, when a conversation is deleted or a
  side thread closes: all of that conversation's threads.

Each one gets a ``DELETE /conversations/{id}`` (the app soft-deletes it) and
its row is dropped. Best-effort on purpose: an orphaned thread is inert, not
dangerous, so no app fault ever fails a request. Every :class:`DelegateError`
is swallowed and logged (a 404 means the thread is already gone, which is the
goal), and the rows are dropped regardless: one try per thread.

Routes run this as a background task after their commit, so the person never
waits on the fleet. The client comes from the same factory as the ask tools
(``app.state.client_factory``), so a harness that fakes the fleet fakes this
too; ``DelegateClient.delete_conversation`` waits on no model and has a short
timeout of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol

import structlog
from sqlalchemy import ColumnElement, delete, or_, select

from app.db.models import DelegateThread
from app.fleet.delegate import CONDUCTOR_ACTOR, DelegateClient, DelegateError, DelegateThreadGone
from app.fleet.manifests import Fleet, FleetApp
from app.fleet.thread_store import SessionFactory

logger = structlog.get_logger(__name__)


class ThreadDeleter(Protocol):
    """What cleanup needs of a delegate client: ``DELETE /conversations/{id}``."""

    def delete_conversation(self, conversation_id: int) -> None: ...


CleanupClientFactory = Callable[[FleetApp], AbstractContextManager[ThreadDeleter]]


def _cleanup_client(app: FleetApp) -> DelegateClient:
    return DelegateClient(app.agent_base_url, actor=CONDUCTOR_ACTOR)


def window_keys(conversation_id: str) -> ColumnElement[bool]:
    """Every window key of one conversation (``thread_key``'s two shapes)."""
    return or_(
        DelegateThread.master_conversation_id == conversation_id,
        DelegateThread.master_conversation_id.startswith(f"{conversation_id}:"),
    )


@dataclass(frozen=True)
class _Row:
    id: int
    key: str
    app: str
    thread: int


@dataclass(frozen=True)
class SubagentThreadCleanup:
    """Deletes subagent threads across the fleet.

    ``client_factory`` is injectable for tests (a :class:`DelegateClient` over
    ``httpx.MockTransport``); the routes pass the ask tools' own factory.
    """

    fleet: Fleet
    session_factory: SessionFactory
    client_factory: CleanupClientFactory = field(default=_cleanup_client)

    def __call__(self, conversation_id: str) -> None:
        """A conversation deleted or a side thread closed: all its threads."""
        self._delete(conversation_id, self._rows(conversation_id))

    def settle(
        self, conversation_id: int, window: str, done_with: Collection[tuple[str, int]]
    ) -> None:
        """After a request: the threads it is done with (``done_with``, by app
        and id, under this window's key), and every thread of this
        conversation's earlier windows."""
        done = set(done_with)
        rows = [
            row
            for row in self._rows(str(conversation_id))
            if row.key != window or (row.app, row.thread) in done
        ]
        self._delete(str(conversation_id), rows)

    def _rows(self, conversation_id: str) -> list[_Row]:
        with self.session_factory() as session:
            found = session.execute(
                select(
                    DelegateThread.id,
                    DelegateThread.master_conversation_id,
                    DelegateThread.app_name,
                    DelegateThread.subagent_conversation_id,
                ).where(window_keys(conversation_id))
            ).all()
        return [_Row(*row) for row in found]

    def _delete(self, conversation_id: str, rows: Sequence[_Row]) -> None:
        if not rows:
            return
        deleted = sum(1 for row in rows if self._delete_thread(conversation_id, row))
        with self.session_factory() as session:
            session.execute(
                delete(DelegateThread).where(DelegateThread.id.in_([r.id for r in rows]))
            )
            session.commit()
        logger.info(
            "subagent_threads_cleaned",
            master_conversation_id=conversation_id,
            threads=len(rows),
            deleted=deleted,
        )

    def _delete_thread(self, conversation_id: str, row: _Row) -> bool:
        """Delete one app thread; ``True`` if the app confirmed it (or it was gone)."""
        app = self.fleet.get(row.app)
        if app is None or not app.has_agent:
            # The app left the fleet (or dropped its agent block) since the
            # thread was opened; there is no one to tell.
            logger.info(
                "subagent_thread_cleanup_skipped",
                master_conversation_id=conversation_id,
                app=row.app,
                thread_id=row.thread,
                reason="not an agent app in the fleet",
            )
            return False
        try:
            with self.client_factory(app) as client:
                client.delete_conversation(row.thread)
        except DelegateThreadGone:
            return True
        except DelegateError as exc:
            logger.warning(
                "subagent_thread_cleanup_failed",
                master_conversation_id=conversation_id,
                app=row.app,
                thread_id=row.thread,
                error=str(exc),
            )
            return False
        return True
