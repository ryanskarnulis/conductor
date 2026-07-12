"""In-process per-IP rate limiting, exposed as a FastAPI dependency factory.

A tiny sliding-window limiter attached to a route via ``Depends(rate_limit(...))``.
The one model-calling endpoint (``POST /agent/conversations/{id}/messages``)
uses it to cap runaway work — a conductor turn can fan out into subagent loops,
so one hammering client would tie up the whole fleet.

Deliberately dependency-free and process-local: this is a single-process,
single-user, local-first app, so a shared (Redis-backed) limiter would be
overkill. If the app ever scales to multiple workers, swap the in-memory
``_HITS`` store for a shared backend behind the same ``rate_limit`` factory.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

import structlog
from fastapi import HTTPException, Request, status

from app.api.request_ip import resolve_client_ip
from app.config import get_settings

logger = structlog.get_logger(__name__)

# Maps "{bucket}:{client_ip}" -> timestamps (monotonic seconds) of recent hits.
_HITS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()


def _reset() -> None:
    """Clear all recorded hits. For tests only — keeps cases independent."""
    with _LOCK:
        _HITS.clear()


def rate_limit(
    bucket: str, *, per_min_attr: str, window_seconds: float = 60.0
) -> Callable[[Request], None]:
    """Build a FastAPI dependency enforcing a per-IP requests/``window_seconds`` cap.

    ``per_min_attr`` names the ``Settings`` field holding the limit (read live on
    each request, so ``.env`` / test overrides take effect without re-import).
    Keyed per client IP within ``bucket`` so distinct endpoints throttle
    independently. Raises 429 with a ``Retry-After`` header on breach.
    """

    def dependency(request: Request) -> None:
        limit: int = getattr(get_settings(), per_min_attr)
        client_ip = resolve_client_ip(request) or "unknown"
        key = f"{bucket}:{client_ip}"
        now = time.monotonic()
        cutoff = now - window_seconds

        with _LOCK:
            hits = _HITS[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + window_seconds - now) + 1)
                logger.warning(
                    "rate_limited",
                    bucket=bucket,
                    client_ip=client_ip,
                    limit=limit,
                    window_seconds=window_seconds,
                    retry_after=retry_after,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="rate limit exceeded — slow down",
                    headers={"Retry-After": str(retry_after)},
                )

            if not hits:
                # The pruning loop emptied this key; drop it before re-adding so
                # the store doesn't accumulate stale deques for idle client IPs.
                del _HITS[key]
            _HITS[key].append(now)

    return dependency
