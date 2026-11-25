"""Per-run delegation state: which app thread a call goes to, and the audit hook.

A :class:`DelegationContext` is bound (via a contextvar or, for the MCP server,
a process-global fallback) around a conductor turn. The ``ask_<app>``
tools read it for two things:

- **Which subagent thread does this call go to?** (#262) Code decides, never
  a model. A call goes to an app's existing thread only when it answers that
  app's own question: a confirm goes to the thread its token is bound to,
  and the step code resumes for an ``app_question`` goes to the thread that
  asked (:func:`answering`). Every other ask, reads included, opens a fresh
  thread, so no app answers from an earlier request's history. Each thread a
  run opens is recorded in its :class:`ThreadStore` (the HTTP route's
  ``DbThreadStore``) until it is deleted, and every thread it sends to is in
  ``used``, which the route checks once the request's records are stored
  (``app/fleet/thread_cleanup.py``). A run with no store (the MCP server:
  nothing there can answer an app later) deletes each thread after its call.
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
    """Where the subagent threads a run opens are recorded until they are
    deleted (#262), under the window's key (``thread_key``).

    The persistence seam: the HTTP route binds the DB-backed
    :class:`app.fleet.thread_store.DbThreadStore`; nothing ever looks a thread
    up here to reuse it.
    """

    def add(self, master_conversation_id: str, app_name: str, thread_id: int) -> None: ...


class InMemoryThreadStore:
    """A :class:`ThreadStore` in memory: what tests bind to see the threads a
    run opened."""

    def __init__(self) -> None:
        self.threads: list[tuple[str, str, int]] = []

    def add(self, master_conversation_id: str, app_name: str, thread_id: int) -> None:
        self.threads.append((master_conversation_id, app_name, thread_id))


@dataclass
class DelegationContext:
    """The delegation state for one master conversation / turn."""

    master_conversation_id: str
    # Where the threads this run opens are recorded (#262); ``None``: nothing
    # can answer one later, so each is deleted after its call (the MCP server).
    thread_store: ThreadStore | None
    # Conductor's own audit tag for this run's driver.
    driver: str = LOOP_DRIVER
    # The question the turn right before left waiting (#68), read from the
    # stored turns: what this turn's words may change or dismiss (#251). On a
    # card's tap, exactly the question tapped. One asked during this run is
    # never here.
    pending_confirm: PendingConfirm | None = None
    # Every question it left waiting, in the order asked (several gated calls
    # are several questions); ``pending_confirm`` is the first of them.
    pending_confirms: list[PendingConfirm] = field(default_factory=list)
    # The questions this run closed without sending their tokens, each with
    # why: the caller puts that on the turn that asked them.
    lapsed: list[tuple[PendingConfirm, str]] = field(default_factory=list)
    # Every app thread this run sent a message to, as ``(app, thread id)``:
    # once the request's records are stored, the route deletes the ones no
    # question waits on (#262).
    used: set[tuple[str, int]] = field(default_factory=set)
    # A compiled plan's parallel reads (#72) run delegate calls from several
    # threads: the thread store is touched one at a time (its DB work is a few
    # short statements; only the network runs at once).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def keeps_threads(self) -> bool:
        """Whether a thread outlives its call: only where a store records it."""
        return self.thread_store is not None

    def opened(self, app_name: str, thread_id: int) -> None:
        """A fresh thread this run opened: recorded until it is deleted."""
        with self._lock:
            if self.thread_store is not None:
                self.thread_store.add(self.master_conversation_id, app_name, thread_id)
            self.used.add((app_name, thread_id))

    def sent_to(self, app_name: str, thread_id: int) -> None:
        """An existing thread this run answered on (a confirm, an app's question)."""
        with self._lock:
            self.used.add((app_name, thread_id))

    def __post_init__(self) -> None:
        if self.pending_confirm is None and self.pending_confirms:
            self.pending_confirm = self.pending_confirms[0]

    def open_confirms(self) -> list[PendingConfirm]:
        """The waiting questions, in the order asked."""
        if self.pending_confirms:
            return list(self.pending_confirms)
        return [self.pending_confirm] if self.pending_confirm is not None else []

    def next_confirm(self, app_name: str) -> PendingConfirm | None:
        """The question a yes to ``app_name`` answers next: its first open one."""
        return next((one for one in self.open_confirms() if one.app == app_name), None)

    def take_pending_confirm(self, app_name: str) -> PendingConfirm | None:
        """``app_name``'s next waiting question, if it has one; each is used
        at most once."""
        pending = self.next_confirm(app_name)
        if pending is None:
            return None
        left = [one for one in self.open_confirms() if one is not pending]
        self.pending_confirms = left
        self.pending_confirm = left[0] if left else None
        return pending

    def lapse_confirms(self, why: str) -> None:
        """The waiting questions close: their tokens are never sent."""
        self.lapsed += [(one, why) for one in self.open_confirms()]
        self.pending_confirms = []
        self.pending_confirm = None

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


# The thread the step that answers an app's own question goes to (#262):
# ``(app, thread id)``, set by the interpreter around exactly that step's
# dispatch (:func:`answering`). Never set from anything a model wrote.
_answering: ContextVar[tuple[str, int] | None] = ContextVar("answering_thread", default=None)


@contextmanager
def answering(app_name: str, thread_id: int) -> Iterator[None]:
    """Within the block, an ask to ``app_name`` goes to ``thread_id``: the
    thread that asked the question it answers."""
    token = _answering.set((app_name, thread_id))
    try:
        yield
    finally:
        _answering.reset(token)


def answering_thread(app_name: str) -> int | None:
    """The thread an ask to ``app_name`` answers on, or ``None``: a fresh one."""
    pinned = _answering.get()
    return pinned[1] if pinned is not None and pinned[0] == app_name else None


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
