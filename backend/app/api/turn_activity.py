"""In-memory registry of in-flight agent turns: the UI's live channel.

Conductor's turns are slow by construction — one master iteration can wrap a
subagent's full loop (warm delegate turns run ~1–2 s; thinking-on analysis
asks have measured ~12–22 s). While ``POST …/messages`` blocks, the page
follows the turn here, two ways:

- ``GET …/events`` (#38), a Server-Sent Events stream: every beat and step
  event of the conversation's turn, pushed as it happens (:func:`follow`);
- ``GET …/activity``, the poll fallback: the current beat only (:func:`get`).

The loop reports each provider call and tool dispatch through its
``on_activity`` seam (``app/ai/loop.py``) and each step's start, receipt and
pause through :mod:`app.ai.turn_events`; the route publishes them here, keyed
by conversation id, and closes the turn (``turn_ended``) only after the ledger
has committed it. Nothing is published ahead of the record it describes.

**Jobs** (#76): a job outlives the turn that started it, so the channel also
keeps the conversation's running jobs (at most :data:`MAX_JOBS`), fed by the
job watcher through :func:`publish_job`. Each change is a ``job`` event
(status, ``done``/``total``; ``ended`` with the note's message id when it
ends), a snapshot lists the running jobs (``jobs``, only when there are
any), and a stream stays open past ``turn_ended`` while one runs, ending
after the last one's ``job`` event says it ended. With no running job a
stream behaves exactly as before.

Each turn also carries its :class:`~app.ai.stop.Stop` (#66): ``POST …/cancel``
reaches the in-flight turn through here, and a fresh one per turn means a Stop
that lands after a turn ended does nothing, never cancelling the next.

**Bounded.** A conversation's channel holds one turn's reduced state (its
beat, plan, steps and pause; at most :data:`MAX_STEPS` steps) and at most
:data:`MAX_SUBSCRIBERS` followers, each with a queue of at most
:data:`QUEUE_LIMIT` events. A follower that falls behind is never allowed to
grow its queue: it is dropped to a fresh snapshot instead, so it loses no
state. A (re)connecting follower always starts from a snapshot of the turn,
so a reconnect needs no ``Last-Event-ID``. Event ids rise across the process.

Process-local on purpose, like the rate limiter: conductor runs SQLite on one
uvicorn worker (see ``backend/Dockerfile`` and CLAUDE.md), and this registry
relies on it.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import structlog

from app.ai.stop import Stop

logger = structlog.get_logger(__name__)

ActivityKind = Literal["model", "tool"]

# A follower's queue: past this it is resynced from a snapshot, never grown.
QUEUE_LIMIT = 64
# Followers per conversation; past this the oldest is closed.
MAX_SUBSCRIBERS = 8
# Running jobs kept per conversation (the watcher's backstop, never the design).
MAX_JOBS = 16
# Steps kept in one turn's state (the loop caps a run at 12 calls and a plan
# at 12 steps; this is the backstop, never the design).
MAX_STEPS = 64
# Seconds between heartbeats on a quiet stream (nginx and the browser keep it).
HEARTBEAT_SECONDS = 15.0
# How long a stream with no turn in flight waits for one before it ends.
IDLE_SECONDS = 30.0
# What the browser waits before reconnecting a dropped stream, in ms.
RETRY_MS = 2_000


@dataclass(frozen=True)
class TurnActivity:
    """The current beat of one in-flight turn."""

    kind: ActivityKind
    tool: str | None
    iteration: int
    # time.monotonic() when the whole turn began — elapsed time, not the beat's.
    started_at: float
    stop: Stop = field(default_factory=Stop, compare=False)
    # A compiled plan's honest ack (#70), once said; it stays for the turn.
    ack: str | None = None


@dataclass(frozen=True)
class TurnEvent:
    """One event on a conversation's stream: its SSE ``id``, ``event`` and ``data``."""

    id: int
    kind: str
    data: dict[str, Any]


@dataclass
class _Turn:
    """One in-flight turn's reduced state: what a snapshot sends."""

    activity: TurnActivity
    plan: dict[str, Any] | None = None
    steps: dict[int, dict[str, Any]] = field(default_factory=dict)
    paused: dict[str, Any] | None = None


class _Follower:
    """One open stream: a bounded queue on its event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[TurnEvent | str] = asyncio.Queue(maxsize=QUEUE_LIMIT)

    def offer(self, item: TurnEvent | str) -> None:
        """Queue ``item`` (on the follower's loop). A full queue is emptied and
        replaced by one resync marker: the follower re-reads a snapshot."""
        if item == _CLOSE:
            _drain(self.queue)
        elif self.queue.full():
            _drain(self.queue)
            item = _RESYNC
        self.queue.put_nowait(item)


@dataclass
class _Channel:
    turn: _Turn | None = None
    followers: list[_Follower] = field(default_factory=list)
    # Running jobs (#76), by ``app:ref``: the latest ``job`` event of each.
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def idle(self) -> bool:
        return not self.followers and self.turn is None and not self.jobs


_RESYNC = "resync"
_CLOSE = "close"

_LOCK = threading.Lock()
_CHANNELS: dict[int, _Channel] = {}
_IDS = itertools.count(1)


# --- the turn's side: begin, beats, step events, end ------------------------------------


def begin(conversation_id: int) -> Stop:
    """Mark a turn in flight (first beat: waiting on the model); its Stop."""
    activity = TurnActivity(kind="model", tool=None, iteration=1, started_at=time.monotonic())
    with _LOCK:
        channel = _CHANNELS.setdefault(conversation_id, _Channel())
        channel.turn = _Turn(activity)
        _publish(channel, "turn_started", _beat(activity))
    return activity.stop


def update(
    conversation_id: int,
    *,
    kind: ActivityKind,
    tool: str | None,
    iteration: int,
    ack: str | None = None,
) -> None:
    """Advance the in-flight turn's beat; a turn ``end()`` already cleared stays gone."""
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        if channel is None or channel.turn is None:
            return
        current = channel.turn.activity
        channel.turn.activity = replace(
            current, kind=kind, tool=tool, iteration=iteration, ack=ack or current.ack
        )
        _publish(channel, "activity", _beat(channel.turn.activity))


def publish(conversation_id: int, stop: Stop, kind: str, data: dict[str, Any]) -> None:
    """A step event of the turn ``stop`` belongs to (:mod:`app.ai.turn_events`).

    Dropped once that turn has ended or another has begun. A turn has one
    plan: the first announced stands.
    """
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        turn = channel.turn if channel is not None else None
        if channel is None or turn is None or turn.activity.stop is not stop:
            return
        if kind == "plan":
            if turn.plan is not None:
                return
            turn.plan = data
        elif kind in ("step_started", "receipt"):
            seq = int(data["seq"])
            if seq not in turn.steps and len(turn.steps) >= MAX_STEPS:
                return
            step = turn.steps.setdefault(seq, {"seq": seq, "receipt": None})
            if kind == "step_started":
                step.update(tool=data["tool"], arguments=data["arguments"])
            else:
                step.update(tool=data["tool"], arguments=data["arguments"], receipt=data)
        elif kind == "paused":
            turn.paused = data
        _publish(channel, kind, data)


def end(
    conversation_id: int,
    stop: Stop | None = None,
    *,
    stop_reason: str | None = None,
    message_id: int | None = None,
) -> None:
    """Clear the turn; with ``stop``, only if it is still that turn's.

    Followers get ``turn_ended`` (how it stopped; ``None`` when it ended with
    no record) and their streams end.
    """
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        if channel is None or channel.turn is None:
            return
        if stop is not None and channel.turn.activity.stop is not stop:
            return
        channel.turn = None
        _publish(channel, "turn_ended", {"stop_reason": stop_reason, "message_id": message_id})
        if channel.idle:
            del _CHANNELS[conversation_id]


def publish_job(conversation_id: int, payload: dict[str, Any]) -> None:
    """A job the conversation started moved (the watcher, through
    :func:`app.ai.turn_events.job`): kept while it runs, dropped once it ended,
    and sent to every follower as a ``job`` event."""
    key = _job_key(payload)
    with _LOCK:
        channel = _CHANNELS.setdefault(conversation_id, _Channel())
        if payload.get("ended"):
            channel.jobs.pop(key, None)
        elif key in channel.jobs or len(channel.jobs) < MAX_JOBS:
            channel.jobs[key] = payload
        _publish(channel, "job", payload)
        if channel.idle:
            del _CHANNELS[conversation_id]


def request_stop(conversation_id: int) -> Stop | None:
    """Stop the in-flight turn; its Stop, or ``None`` if nothing is running."""
    current = get(conversation_id)
    if current is None:
        return None
    current.stop.request()
    return current.stop


def get(conversation_id: int) -> TurnActivity | None:
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        return channel.turn.activity if channel is not None and channel.turn else None


# --- the page's side: following a conversation ------------------------------------------


@dataclass(frozen=True)
class Heartbeat:
    """Nothing happened for a while; the stream is still alive."""


async def follow(
    conversation_id: int,
    *,
    heartbeat: float | None = None,
    idle: float | None = None,
) -> AsyncGenerator[TurnEvent | Heartbeat, None]:
    """The conversation's stream: a snapshot, then each event as it happens.

    Ends after ``turn_ended``, when no turn begins within ``idle`` seconds of
    connecting, or when the stream is closed (the client went away, or too
    many others follow the same conversation). Quiet stretches yield a
    :class:`Heartbeat` every ``heartbeat`` seconds. Closing the iterator
    unsubscribes at once.
    """
    heartbeat = HEARTBEAT_SECONDS if heartbeat is None else heartbeat
    idle = IDLE_SECONDS if idle is None else idle
    follower = _Follower(asyncio.get_running_loop())
    with _LOCK:
        channel = _CHANNELS.setdefault(conversation_id, _Channel())
        channel.followers.append(follower)
        if len(channel.followers) > MAX_SUBSCRIBERS:
            oldest = channel.followers.pop(0)
            _deliver(oldest, _CLOSE)
        first = _snapshot(channel)
    try:
        yield first
        # What this stream has seen, so it ends in event order: a turn in
        # flight, and the jobs still running (#76).
        turn_on, running = _seen(first)
        # Events older than the last snapshot are already in it.
        floor = first.id
        waited = 0.0
        while True:
            try:
                item = await asyncio.wait_for(follower.queue.get(), timeout=heartbeat)
            except TimeoutError:
                waited += heartbeat
                if not (turn_on or running) and waited >= idle:
                    return
                yield Heartbeat()
                continue
            if item == _CLOSE:
                return
            if item == _RESYNC:
                with _LOCK:
                    event = _snapshot(_CHANNELS.setdefault(conversation_id, _Channel()))
                turn_on, running = _seen(event)
                floor = event.id
                yield event
                continue
            assert isinstance(item, TurnEvent)
            if item.id < floor:
                continue
            yield item
            if item.kind == "turn_started":
                turn_on = True
            elif item.kind == "turn_ended":
                turn_on = False
                if not running:
                    return
            elif item.kind == "job":
                key = _job_key(item.data)
                if not item.data.get("ended"):
                    running.add(key)
                elif key in running or not turn_on:
                    running.discard(key)
                    if not (turn_on or running):
                        return
    finally:
        _unsubscribe(conversation_id, follower)


def _seen(snapshot: TurnEvent) -> tuple[bool, set[str]]:
    """A snapshot's turn in flight and running jobs, as a stream tracks them."""
    jobs = snapshot.data.get("jobs") or []
    return bool(snapshot.data["active"]), {_job_key(job) for job in jobs}


def _job_key(job: dict[str, Any]) -> str:
    return f"{job.get('app')}:{job.get('ref')}"


def subscribers(conversation_id: int) -> int:
    """How many streams follow the conversation (for tests and logs)."""
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        return len(channel.followers) if channel is not None else 0


def _reset() -> None:
    """Clear every in-flight turn and stream. For tests only — keeps cases independent."""
    with _LOCK:
        for channel in _CHANNELS.values():
            for follower in channel.followers:
                _deliver(follower, _CLOSE)
        _CHANNELS.clear()


# --- internals (callers hold _LOCK) ---------------------------------------------------------


def _beat(activity: TurnActivity) -> dict[str, Any]:
    return {
        "kind": activity.kind,
        "tool": activity.tool,
        "iteration": activity.iteration,
        "ack": activity.ack,
        "elapsed_seconds": round(time.monotonic() - activity.started_at, 1),
    }


def _snapshot(channel: _Channel) -> TurnEvent:
    turn = channel.turn
    if turn is None:
        data: dict[str, Any] = {"active": False}
    else:
        data = {
            "active": True,
            "activity": _beat(turn.activity),
            "plan": turn.plan,
            "steps": [turn.steps[seq] for seq in sorted(turn.steps)],
            "paused": turn.paused,
        }
    if channel.jobs:
        data["jobs"] = list(channel.jobs.values())
    return TurnEvent(id=next(_IDS), kind="snapshot", data=data)


def _publish(channel: _Channel, kind: str, data: dict[str, Any]) -> None:
    event = TurnEvent(id=next(_IDS), kind=kind, data=data)
    for follower in channel.followers:
        _deliver(follower, event)


def _deliver(follower: _Follower, item: TurnEvent | str) -> None:
    """Hand ``item`` to the follower's loop; the publisher may be any thread."""
    try:
        follower.loop.call_soon_threadsafe(follower.offer, item)
    except RuntimeError:
        pass  # its loop is closed: the stream is gone, and unsubscribes itself


def _unsubscribe(conversation_id: int, follower: _Follower) -> None:
    with _LOCK:
        channel = _CHANNELS.get(conversation_id)
        if channel is None:
            return
        if follower in channel.followers:
            channel.followers.remove(follower)
        if channel.idle:
            del _CHANNELS[conversation_id]


def _drain(queue: asyncio.Queue[Any]) -> None:
    while not queue.empty():
        queue.get_nowait()
