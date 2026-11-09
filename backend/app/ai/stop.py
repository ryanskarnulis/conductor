"""Stop (#66): one run's cancel, and the in-flight delegate call it reaches.

``POST …/cancel`` asks the run of the conversation's in-flight turn to stop
(``app/api/turn_activity.py`` holds one fresh :class:`Stop` per turn, so a
late Stop never cancels the next turn). The interpreter checks it before
every model call and step and ends ``cancelled``; a model call or a
delegate call already in flight finishes first.

A delegate call is the slow part of a conductor turn, so the Stop is also
passed on: the ``ask_<app>`` tool registers the app's own cancel
(``…/conversations/{id}/cancel``) while its message is in flight, and
:meth:`Stop.pass_on` calls it. The app stops at its own next boundary and
answers ``cancelled`` with the record of what it did. Passing on is
best-effort: an app without a cancel route runs its turn to the end, and the
turn still stops after it. A compiled plan's parallel reads (#72) put several
calls in flight at once; a Stop reaches each of them.

The tool reads the run's Stop from a contextvar (:func:`use_stop`) the
interpreter binds, so the MCP server, which has no Stop, needs no wiring.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

from app.ai.run_record import NOT_RUN_PREFIX

logger = structlog.get_logger(__name__)

# The answer to a call the Stop kept from starting: "not run", so the record
# sorts it as never having left conductor.
STOPPED_ERROR = f"{NOT_RUN_PREFIX}you stopped the run before this call"


class Stop:
    """One run's Stop: a flag, plus the cancels of the delegate calls in flight."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._in_flight: dict[int, Callable[[], object]] = {}

    def is_set(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        """Ask the run to stop at its next boundary. Idempotent, never blocks."""
        self._event.set()
        logger.info("agent_run_cancel_requested")

    def pass_on(self) -> None:
        """Send the app's cancel to each delegate call in flight, if any.

        Best-effort: a failure is logged, never raised (the turn stops after
        the calls regardless).
        """
        with self._lock:
            cancels = list(self._in_flight.values())
        for cancel in cancels:
            try:
                cancel()
            except Exception as exc:
                logger.warning(
                    "delegate_cancel_failed", error=str(exc), error_type=type(exc).__name__
                )

    @contextmanager
    def delegating(self, cancel: Callable[[], object]) -> Iterator[bool]:
        """Register ``cancel`` for the delegate call this block makes.

        Yields False, registering nothing, when the run is already stopped: the
        call must not start. Checked under the same lock :meth:`pass_on` reads,
        so a Stop either finds the call registered or the call finds the Stop.
        """
        key = id(cancel)
        with self._lock:
            if self._event.is_set():
                live = False
            else:
                self._in_flight[key] = cancel
                live = True
        try:
            yield live
        finally:
            if live:
                with self._lock:
                    self._in_flight.pop(key, None)


_current: ContextVar[Stop | None] = ContextVar("current_stop", default=None)


@contextmanager
def use_stop(stop: Stop | None) -> Iterator[None]:
    """Bind the run's Stop for the duration of the block."""
    token = _current.set(stop)
    try:
        yield
    finally:
        _current.reset(token)


def current_stop() -> Stop | None:
    """The Stop of the run in force, or ``None`` outside a stoppable run."""
    return _current.get()
