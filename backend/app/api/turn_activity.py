"""In-memory registry of in-flight agent turns: the UI's poll target.

Conductor's turns are slow by construction — one master iteration can wrap a
subagent's full loop (warm delegate turns run ~1–2 s; thinking-on analysis
asks have measured ~12–22 s) — and v1 has no SSE. So while
``POST …/messages`` blocks, the frontend polls ``GET …/activity``, which reads
this registry: the loop reports each provider call and tool dispatch through
its ``on_activity`` seam (``app/ai/loop.py``), keyed here by conversation id.

Each turn also carries its :class:`~app.ai.stop.Stop` (#66): ``POST …/cancel``
reaches the in-flight turn through here, and a fresh one per turn means a Stop
that lands after a turn ended does nothing, never cancelling the next.

Process-local on purpose, like the rate limiter: the backend runs a single
uvicorn worker (see ``backend/Dockerfile``). If workers ever multiply, swap
the store behind this same module surface.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Literal

from app.ai.stop import Stop

ActivityKind = Literal["model", "tool"]


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


_LOCK = threading.Lock()
_ACTIVE: dict[int, TurnActivity] = {}


def begin(conversation_id: int) -> Stop:
    """Mark a turn in flight (first beat: waiting on the model); its Stop."""
    activity = TurnActivity(kind="model", tool=None, iteration=1, started_at=time.monotonic())
    with _LOCK:
        _ACTIVE[conversation_id] = activity
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
        current = _ACTIVE.get(conversation_id)
        if current is None:
            return
        _ACTIVE[conversation_id] = replace(
            current, kind=kind, tool=tool, iteration=iteration, ack=ack or current.ack
        )


def end(conversation_id: int, stop: Stop | None = None) -> None:
    """Clear the turn; with ``stop``, only if it is still that turn's."""
    with _LOCK:
        current = _ACTIVE.get(conversation_id)
        if current is not None and (stop is None or current.stop is stop):
            del _ACTIVE[conversation_id]


def request_stop(conversation_id: int) -> Stop | None:
    """Stop the in-flight turn; its Stop, or ``None`` if nothing is running."""
    with _LOCK:
        current = _ACTIVE.get(conversation_id)
    if current is None:
        return None
    current.stop.request()
    return current.stop


def get(conversation_id: int) -> TurnActivity | None:
    with _LOCK:
        return _ACTIVE.get(conversation_id)


def _reset() -> None:
    """Clear every in-flight turn. For tests only — keeps cases independent."""
    with _LOCK:
        _ACTIVE.clear()
