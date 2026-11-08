"""The glance reader (#73): one plain GET to an app's declared glance endpoint.

A glance (``../agent-standard/app-yaml-glance-block.md``) is an app's small,
read-only summary, answered with no agent loop and no model call. Like the
fleet action proxy (:mod:`app.api.routes_fleet`) this is the one other model-free
path to an app, and it keeps the same rules:

- **Only what the app declared.** The URL is the manifest's ``glance.path``
  on its upstream; nothing the person said or a model wrote reaches it.
- **No forwarded headers.** The request carries ``Accept`` and nothing else:
  no browser cookies or auth, no delegate actor, no trace ids.
- **Short timeouts.** A glance answers in ≤200 ms; nothing here waits on a
  model, so a slow one has failed. Redirects are not followed.
- **Small.** A body over :data:`MAX_BYTES` is not a glance.

Every failure is a :class:`GlanceError` whose message can be said to the
person ("it didn't answer in time"): conductor treats any of them as "couldn't
look" and may fall back to an ``ask``. Tests pass a fake :class:`GlanceReader`
to the interpreter; none dials the network.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx
import structlog

from app.fleet.manifests import FleetApp

logger = structlog.get_logger(__name__)

# Short, like the action proxy's: a person is waiting on a read that is meant
# to take a fifth of a second.
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 5.0
# The contract's glance is ~600 characters; this is generous, not a budget.
MAX_BYTES = 32 * 1024

_ACCEPT = {"Accept": "application/json"}


class GlanceError(Exception):
    """A glance that couldn't be read. The message is fit to tell the person."""


class GlanceReader(Protocol):
    """Read an app's glance: its JSON object, or :class:`GlanceError`."""

    def __call__(self, app: FleetApp, /) -> dict[str, Any]: ...


def read_glance(app: FleetApp) -> dict[str, Any]:
    """``GET`` the app's glance endpoint; the default :class:`GlanceReader`."""
    if app.glance is None:
        raise GlanceError(f"{app.name} has no glance")
    url = app.glance_url
    started = time.monotonic()
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.get(url, headers=_ACCEPT)
    except httpx.TimeoutException as exc:
        _failed(app, started, f"timed out: {exc}")
        raise GlanceError(f"{app.name} didn't answer in time") from exc
    except httpx.HTTPError as exc:
        _failed(app, started, f"unreachable: {exc}")
        raise GlanceError(f"{app.name} couldn't be reached") from exc
    if response.status_code != httpx.codes.OK:
        _failed(app, started, f"status {response.status_code}")
        raise GlanceError(f"{app.name} answered {response.status_code}")
    if len(response.content) > MAX_BYTES:
        _failed(app, started, f"{len(response.content)} bytes")
        raise GlanceError(f"{app.name}'s summary was too large to be a glance")
    try:
        body = response.json()
    except ValueError as exc:
        _failed(app, started, "not JSON")
        raise GlanceError(f"{app.name}'s summary wasn't readable") from exc
    if not isinstance(body, dict):
        _failed(app, started, "not an object")
        raise GlanceError(f"{app.name}'s summary wasn't readable")
    logger.info("glance_read", app=app.name, ms=_ms(started))
    return body


def _failed(app: FleetApp, started: float, why: str) -> None:
    logger.warning("glance_failed", app=app.name, ms=_ms(started), why=why)


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
