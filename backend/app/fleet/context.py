"""Per-run delegation state: the thread map and the audit hook.

A :class:`DelegationContext` is bound (via a contextvar or, for the MCP server,
a process-global fallback) around a conductor turn. The ``ask_<app>``
tools read it for two things:

- **Which subagent thread should this app's call reuse?** The context holds a
  :class:`ThreadStore` keyed by ``(master_conversation_id, app_name)`` so a
  follow-up to the same app in the same master conversation carries context
  app-side. The store is an interface (the persistence seam): the HTTP route
  binds the DB-backed :class:`app.fleet.thread_store.DbThreadStore`, the MCP
  server a process-lifetime :class:`InMemoryThreadStore` — nothing else
  changes between them.
- **Audit.** Every delegate call — success or failure — emits one structlog
  ``delegate_call`` event (app, subagent thread id, latency, and either the
  stop reason or the error class), tagged with the driver identity.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from app.ai.confirmation import PendingConfirm

logger = structlog.get_logger(__name__)

# Driver identities: who is running this conductor turn. The delegate
# client always presents `agent:conductor` downstream regardless; this is
# conductor's own audit tag for the run.
LOOP_DRIVER = "agent:loop"
MCP_DRIVER = "agent:mcp"


class ThreadStore(Protocol):
    """Maps ``(master_conversation_id, app_name)`` to a subagent thread id.

    The persistence seam — the tools only ever touch this interface, so which
    store backs a run (the DB for the HTTP route, memory for the MCP server) is
    entirely the binder's choice.
    """

    def get(self, master_conversation_id: str, app_name: str) -> int | None: ...

    def set(self, master_conversation_id: str, app_name: str, thread_id: int) -> None: ...

    def forget(self, master_conversation_id: str, app_name: str) -> None: ...


class InMemoryThreadStore:
    """Process-lifetime :class:`ThreadStore` — the MCP server's session store.

    A pruned thread 404s and the tool recreates it, so a store that doesn't
    survive restart is contract-compliant; the HTTP route uses the DB-backed
    ``DbThreadStore`` (``app/fleet/thread_store.py``) behind the same
    interface instead.
    """

    def __init__(self) -> None:
        self._threads: dict[tuple[str, str], int] = {}

    def get(self, master_conversation_id: str, app_name: str) -> int | None:
        return self._threads.get((master_conversation_id, app_name))

    def set(self, master_conversation_id: str, app_name: str, thread_id: int) -> None:
        self._threads[(master_conversation_id, app_name)] = thread_id

    def forget(self, master_conversation_id: str, app_name: str) -> None:
        self._threads.pop((master_conversation_id, app_name), None)


@dataclass
class DelegationContext:
    """The delegation state for one master conversation / turn."""

    master_conversation_id: str
    thread_store: ThreadStore
    # Conductor's own audit tag for this run's driver.
    driver: str = LOOP_DRIVER
    # The question the last turn left waiting (#68), read from the stored turns
    # before this one, so a yes can only answer a question the person has
    # seen. A question asked during this run is never here.
    pending_confirm: PendingConfirm | None = None
    # A compiled plan's parallel reads (#72) run delegate calls from several
    # threads: the thread store is touched one at a time (its DB work is a few
    # short statements; only the network runs at once).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def thread_for(self, app_name: str) -> int | None:
        with self._lock:
            return self.thread_store.get(self.master_conversation_id, app_name)

    def remember_thread(self, app_name: str, thread_id: int) -> None:
        with self._lock:
            self.thread_store.set(self.master_conversation_id, app_name, thread_id)

    def forget_thread(self, app_name: str) -> None:
        with self._lock:
            self.thread_store.forget(self.master_conversation_id, app_name)

    def take_pending_confirm(self, app_name: str) -> PendingConfirm | None:
        """The waiting question if it is ``app_name``'s; used at most once."""
        pending = self.pending_confirm
        if pending is None or pending.app != app_name:
            return None
        self.pending_confirm = None
        return pending

    def audit_delegate_call(
        self,
        *,
        app: str,
        subagent_conversation_id: int | None,
        latency_ms: int,
        stop_reason: str | None = None,
        error: str | None = None,
        hop_id: str | None = None,
        app_request_id: str | None = None,
    ) -> None:
        """Emit the one audit event required for every delegate call.

        ``hop_id`` and ``app_request_id`` (#62) join the line to the turn's
        trace record and to the app's own logs for the request.
        """
        logger.info(
            "delegate_call",
            app=app,
            driver=self.driver,
            master_conversation_id=self.master_conversation_id,
            subagent_conversation_id=subagent_conversation_id,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            error=error,
            hop_id=hop_id,
            app_request_id=app_request_id,
        )


class NoDelegationContextError(RuntimeError):
    """A delegate tool ran with no delegation context bound — a wiring bug."""


# Request-scoped binding (the HTTP route wraps each turn in this); the MCP
# server has no per-request scope, so it registers a process-global fallback.
_current: ContextVar[DelegationContext | None] = ContextVar(
    "current_delegation_context", default=None
)
_process_default: DelegationContext | None = None


@contextmanager
def use_delegation_context(context: DelegationContext) -> Iterator[DelegationContext]:
    """Bind ``context`` for the duration of the block (request-scoped)."""
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def set_process_delegation_context(context: DelegationContext | None) -> None:
    """Register the process-wide fallback context (the MCP server uses this).

    A request-scoped context bound via :func:`use_delegation_context` always
    wins; this is only consulted when nothing is bound.
    """
    global _process_default
    _process_default = context


def current_delegation_context() -> DelegationContext:
    """The delegation context in force, or raise if none is bound."""
    context = _current.get()
    if context is not None:
        return context
    if _process_default is not None:
        return _process_default
    raise NoDelegationContextError("no delegation context is bound for this delegate call")
