"""Typed step events for the live push channel (#38): the one seam.

The interpreter reports what a turn's steps do through these
calls; the route binds a sink for exactly the run (:func:`publishing`), and the
in-process registry (``app/api/turn_activity.py``) fans the events out to
``GET …/events``. With no sink bound (evals, direct callers) every call is a
no-op, and a sink that raises is logged and never costs the turn.

**Nothing streams ahead of its receipt.** Each payload is built from the record
the ledger journals: a step's outcome is :func:`run_record.classify` of its
:class:`~app.ai.runtime.ToolCallRecord`, and its receipts are the ones
:func:`app.services.ledger.live_step` says the ledger will write. A step is
announced as started before its call goes out and as landed only once the
call's record exists. ``seq`` is the step's place in the turn's record, which
is its ``seq`` in the ledger too.

The sink is a context variable, so the reads a plan runs at once (each in a
copy of the turn's context) publish to the same turn.

**Jobs outlive their turn** (#76). A job's progress and end come from the job
watcher, long after the turn that started it ended, so :func:`job` goes to a
process-wide sink (:func:`set_job_sink`, bound once at startup) keyed by the
conversation that started the job, never to a turn's sink.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from app.ai.runtime import ToolCallRecord

logger = structlog.get_logger(__name__)

EventKind = Literal["plan", "step_started", "receipt", "paused", "resumed"]
Sink = Callable[[EventKind, dict[str, Any]], None]
# A job event for a conversation (#76): ``(conversation_id, payload)``.
JobSink = Callable[[int, dict[str, Any]], None]

_SINK: ContextVar[Sink | None] = ContextVar("turn_event_sink", default=None)
_JOB_SINK: JobSink | None = None


@contextmanager
def publishing(sink: Sink) -> Iterator[None]:
    """Send this run's step events to ``sink`` for the duration."""
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def plan(body: dict[str, Any] | None, start: int) -> None:
    """The validated plan is about to run, from step ``start`` (0-based) of
    ``body``: one record per step from there, in order."""
    if body is None:
        return
    steps = body.get("steps")
    if isinstance(steps, list):
        _emit("plan", {"steps": steps, "start": start})


def step_started(seq: int, tool: str, arguments: dict[str, Any]) -> None:
    """The ``seq``-th call of the turn is going out now."""
    _emit("step_started", {"seq": seq, "tool": tool, "arguments": arguments})


def receipt(seq: int, record: ToolCallRecord) -> None:
    """The ``seq``-th call's record exists: what it did, as the ledger keeps it."""
    from app.services import ledger  # noqa: PLC0415 (the ledger imports the runtime)

    _emit("receipt", ledger.live_step(seq, record))


def paused(
    reason: Literal["ask_user", "gate", "app_confirm"],
    *,
    question: str | None,
    app: str | None = None,
    step: str | None = None,
) -> None:
    """The plan waits on the person: code's question, or an app's."""
    _emit("paused", {"reason": reason, "question": question, "app": app, "step": step})


def set_job_sink(sink: JobSink | None) -> None:
    """Where job events go for the life of the process (``None``: nowhere)."""
    global _JOB_SINK
    _JOB_SINK = sink


def job(conversation_id: int, payload: dict[str, Any]) -> None:
    """A job the conversation started moved (#76): its status, ``done`` and
    ``total`` as the watcher last read them, and once it ended, the note's
    message id. Built from the ledger row the watcher just committed."""
    sink = _JOB_SINK
    if sink is None:
        return
    try:
        sink(conversation_id, payload)
    except Exception as exc:
        logger.warning("turn_event_failed", kind="job", error=str(exc))


def resumed(
    reason: Literal["ask_user", "gate", "app_confirm", "proposal"],
    *,
    step: str | None = None,
    answer: str | None = None,
) -> None:
    """This turn picks up a plan that was waiting on the person (#89): why it
    waited, the id of the step it waited on (``None`` when an app's own
    question is answered), and, for an ``ask_user``, their answer, as they said
    it: what ``{s1}`` becomes in the steps that follow."""
    _emit("resumed", {"reason": reason, "step": step, "answer": answer})


def _emit(kind: EventKind, payload: dict[str, Any]) -> None:
    sink = _SINK.get()
    if sink is None:
        return
    try:
        sink(kind, payload)
    except Exception as exc:
        logger.warning("turn_event_failed", kind=kind, error=str(exc))
