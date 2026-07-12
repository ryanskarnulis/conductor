"""Per-run delegation state: the thread map, the call budget, and the audit hook.

A :class:`DelegationContext` is bound (via a contextvar or, for the MCP server,
a process-global fallback) around a run of conductor's loop. The ``ask_<app>``
tools read it to answer three questions:

- **Which subagent thread should this app's call reuse?** The context holds a
  :class:`ThreadStore` keyed by ``(master_conversation_id, app_name)`` so a
  follow-up to the same app in the same master conversation carries context
  app-side. The store is an interface: an in-memory one for now, a DB-backed
  one in Slice 4 (this is the persistence seam) — nothing else changes.
- **Have we called this app too many times this turn?** A per-run, per-app
  counter enforces ``conductor_delegate_calls_per_turn_per_app``; exceeding it
  raises :class:`ToolError`, a domain error the model sees so it stops
  retrying one app and moves on.
- **Audit.** Every delegate call — success or failure — emits one structlog
  ``delegate_call`` event (app, subagent thread id, latency, and either the
  stop reason or the error class), tagged with the driver identity.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from app.tools.registry import ToolError

logger = structlog.get_logger(__name__)

# Driver identities: who is running conductor's loop this call. The delegate
# client always presents `agent:conductor` downstream regardless; this is
# conductor's own audit tag for the run.
LOOP_DRIVER = "agent:loop"
MCP_DRIVER = "agent:mcp"


class ThreadStore(Protocol):
    """Maps ``(master_conversation_id, app_name)`` to a subagent thread id.

    The seam Slice 4 swaps for a DB-backed store — the tools only ever touch
    this interface, so persistence slots in without touching them.
    """

    def get(self, master_conversation_id: str, app_name: str) -> int | None: ...

    def set(self, master_conversation_id: str, app_name: str, thread_id: int) -> None: ...

    def forget(self, master_conversation_id: str, app_name: str) -> None: ...


class InMemoryThreadStore:
    """Process-lifetime :class:`ThreadStore` (Slice 3 default).

    A pruned thread 404s and the tool recreates it, so a store that doesn't
    survive restart is contract-compliant; Slice 4 replaces it with a
    DB-backed one behind the same interface.
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
    """The delegation state for one master conversation / loop run."""

    master_conversation_id: str
    thread_store: ThreadStore
    # Per-turn per-app cap; <= 0 disables the budget (used by the MCP server,
    # whose driver is itself a trusted agent, not conductor's shallow loop).
    calls_per_turn_per_app: int
    # Conductor's own audit tag for this run's driver.
    driver: str = LOOP_DRIVER
    _calls_this_turn: dict[str, int] = field(default_factory=dict)

    def thread_for(self, app_name: str) -> int | None:
        return self.thread_store.get(self.master_conversation_id, app_name)

    def remember_thread(self, app_name: str, thread_id: int) -> None:
        self.thread_store.set(self.master_conversation_id, app_name, thread_id)

    def forget_thread(self, app_name: str) -> None:
        self.thread_store.forget(self.master_conversation_id, app_name)

    def charge_call(self, app_name: str) -> None:
        """Count one delegate call to ``app_name`` this turn; enforce the budget.

        Called before the network request. Raises :class:`ToolError` (a domain
        error the model reads) once the per-turn cap is reached, so the failed
        attempt never leaves conductor.
        """
        if self.calls_per_turn_per_app <= 0:
            return
        used = self._calls_this_turn.get(app_name, 0)
        if used >= self.calls_per_turn_per_app:
            raise ToolError(
                f"You've already asked {app_name} {used} time(s) this turn — that's the "
                f"per-turn limit. Stop retrying {app_name}; either try a different app or "
                f"tell the user what you found so far."
            )
        self._calls_this_turn[app_name] = used + 1

    def reset_turn(self) -> None:
        """Clear the per-turn call counters (a fresh master loop run)."""
        self._calls_this_turn.clear()

    def audit_delegate_call(
        self,
        *,
        app: str,
        subagent_conversation_id: int | None,
        latency_ms: int,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit the one audit event required for every delegate call."""
        logger.info(
            "delegate_call",
            app=app,
            driver=self.driver,
            master_conversation_id=self.master_conversation_id,
            subagent_conversation_id=subagent_conversation_id,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            error=error,
        )


class NoDelegationContextError(RuntimeError):
    """A delegate tool ran with no delegation context bound — a wiring bug."""


# Request-scoped binding (Slice 4's HTTP loop wraps each run in this); the MCP
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
