"""Whether a conversation turn is using the GPU now, or did lately (#78).

A ``think`` runs on the 26B in the background, and the GPU is exclusive
(``../llama-swap/config.yaml``): a turn's 12B call and a think's 26B call
can't both run. The person comes first, so a think yields: it stops when a
turn starts and only tries again after :data:`app.ai.think.QUIET` with no
turn. This is how it knows.

The route brackets each turn with :func:`entered`/:func:`left`.
Process-local, like the live channel: conductor is single-worker
(CLAUDE.md). After a restart nothing is running and nothing ran lately.
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_active = 0
_last: float | None = None


def entered() -> None:
    """A turn started."""
    global _active, _last
    with _lock:
        _active += 1
        _last = time.monotonic()


def left() -> None:
    """A turn ended."""
    global _active, _last
    with _lock:
        _active = max(0, _active - 1)
        _last = time.monotonic()


def busy() -> bool:
    """Whether a turn is running now."""
    with _lock:
        return _active > 0


def quiet_for(seconds: float) -> bool:
    """Whether no turn has run for ``seconds``."""
    with _lock:
        if _active > 0:
            return False
        return _last is None or time.monotonic() - _last >= seconds


def _reset() -> None:
    """Tests: forget every turn."""
    global _active, _last
    with _lock:
        _active = 0
        _last = None
