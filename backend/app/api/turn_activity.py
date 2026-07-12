"""In-memory registry of in-flight agent turns: the UI's poll target.

Conductor's turns are slow by construction — one master iteration can wrap a
subagent's full loop (warm delegate turns run ~1–2 s; thinking-on analysis
asks have measured ~12–22 s) — and v1 has no SSE. So while
``POST …/messages`` blocks, the frontend polls ``GET …/activity``, which reads
this registry: the loop reports each provider call and tool dispatch through
its ``on_activity`` seam (``app/ai/loop.py``), keyed here by conversation id.

Process-local on purpose, like the rate limiter: the backend runs a single
uvicorn worker (see ``backend/Dockerfile``). If workers ever multiply, swap
the store behind this same module surface.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Literal

ActivityKind = Literal["model", "tool"]


@dataclass(frozen=True)
class TurnActivity:
    """The current beat of one in-flight turn."""

    kind: ActivityKind
    tool: str | None
    iteration: int
    # time.monotonic() when the whole turn began — elapsed time, not the beat's.
    started_at: float


_LOCK = threading.Lock()
_ACTIVE: dict[int, TurnActivity] = {}


def begin(conversation_id: int) -> None:
    """Mark a turn in flight (first beat: waiting on the model)."""
    with _LOCK:
        _ACTIVE[conversation_id] = TurnActivity(
            kind="model", tool=None, iteration=1, started_at=time.monotonic()
        )


def update(conversation_id: int, *, kind: ActivityKind, tool: str | None, iteration: int) -> None:
    """Advance the in-flight turn's beat; a turn ``end()`` already cleared stays gone."""
    with _LOCK:
        current = _ACTIVE.get(conversation_id)
        if current is None:
            return
        _ACTIVE[conversation_id] = replace(current, kind=kind, tool=tool, iteration=iteration)


def end(conversation_id: int) -> None:
    with _LOCK:
        _ACTIVE.pop(conversation_id, None)


def get(conversation_id: int) -> TurnActivity | None:
    with _LOCK:
        return _ACTIVE.get(conversation_id)


def _reset() -> None:
    """Clear every in-flight turn. For tests only — keeps cases independent."""
    with _LOCK:
        _ACTIVE.clear()
