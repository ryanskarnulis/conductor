"""The jobs contract's client (#76): read and cancel an app's long-running job.

delegate-api "Jobs" (agent-standard#22): a step that starts long work returns
at once with a ``job`` on its record entry; the job is then read with
``GET {agent.api}/jobs/{ref}`` (no model call, no side effects, safe to poll)
and stopped with ``POST {agent.api}/jobs/{ref}/cancel`` (idempotent, never
blocks). Like the glance reader (:mod:`app.fleet.glance`):

- **Only the declared upstream.** The URL is the manifest's agent base plus
  the contract's path; the ref is the app's own, path-quoted.
- **No forwarded headers.** A poll carries ``Accept`` alone. A cancel is a
  write, so it also carries ``X-Agent-Actor`` for the app's audit trail, as
  the contract asks.
- **Short timeouts, no redirects, a small body.** Nothing here waits on a
  model.

Every failure is a :class:`JobError` whose message can be said to the person;
an unknown ref is :class:`JobGone`. Tests and the session evals pass fakes
(:class:`JobReader`, :class:`JobCanceller`); none dials the network.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx
import structlog
from pydantic import BaseModel, ValidationError, field_validator

from app.fleet.delegate import CONDUCTOR_ACTOR
from app.fleet.manifests import FleetApp

logger = structlog.get_logger(__name__)

CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 5.0
MAX_BYTES = 256 * 1024

# The A2A words (delegate-api "Jobs"). A job in one of FINISHED has ended.
JobStatus = Literal["submitted", "working", "input_required", "completed", "failed", "canceled"]
STATUSES: frozenset[str] = frozenset(
    {"submitted", "working", "input_required", "completed", "failed", "canceled"}
)
FINISHED: frozenset[str] = frozenset({"completed", "failed", "canceled"})

_ACCEPT = {"Accept": "application/json"}


class JobError(Exception):
    """A job that couldn't be read or cancelled. The message fits the person."""


class JobGone(JobError):
    """The app answered 404: it doesn't know the ref (or no longer keeps it)."""


class JobRead(BaseModel):
    """One job as the app reports it. Lenient: a status outside the A2A words
    reads as ``working`` (still going, never assumed done), and a record
    conductor can't read is ``None`` (what it did is unknown)."""

    ref: str
    status: str = "working"
    done: int = 0
    total: int | None = None
    title: str | None = None
    untrusted: bool = False
    question: str | None = None
    reason: str | None = None
    record: list[dict[str, Any]] | None = None
    # Only on a cancel's answer: whether it was still running when asked.
    cancel_requested: bool | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _a2a(cls, value: Any) -> str:
        return value if isinstance(value, str) and value in STATUSES else "working"

    @field_validator("done", mode="before")
    @classmethod
    def _count(cls, value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    @field_validator("total", mode="before")
    @classmethod
    def _total(cls, value: Any) -> int | None:
        return value if isinstance(value, int) and value >= 0 else None

    @field_validator("record", mode="before")
    @classmethod
    def _record(cls, value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, list) or not all(isinstance(e, dict) for e in value):
            return None
        return value

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class JobReader(Protocol):
    """Read one job: its report, or :class:`JobError`."""

    def __call__(self, app: FleetApp, ref: str, /) -> JobRead: ...


class JobCanceller(Protocol):
    """Ask one job to stop: its report with ``cancel_requested``, or :class:`JobError`."""

    def __call__(self, app: FleetApp, ref: str, /) -> JobRead: ...


def job_url(app: FleetApp, ref: str) -> str:
    return f"{app.agent_base_url.rstrip('/')}/jobs/{quote(ref, safe='')}"


def read_job(app: FleetApp, ref: str) -> JobRead:
    """``GET …/jobs/{ref}``; the default :class:`JobReader`."""
    return _request(app, ref, "GET", "", _ACCEPT)


def cancel_job(app: FleetApp, ref: str) -> JobRead:
    """``POST …/jobs/{ref}/cancel``; the default :class:`JobCanceller`."""
    headers = {**_ACCEPT, "X-Agent-Actor": CONDUCTOR_ACTOR}
    return _request(app, ref, "POST", "/cancel", headers)


def _request(app: FleetApp, ref: str, method: str, suffix: str, headers: dict[str, str]) -> JobRead:
    if app.agent is None:
        raise JobError(f"{app.name} has no agent to ask about its jobs")
    url = job_url(app, ref) + suffix
    started = time.monotonic()
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.request(method, url, headers=headers)
    except httpx.TimeoutException as exc:
        _failed(app, ref, method, started, f"timed out: {exc}")
        raise JobError(f"{app.name} didn't answer in time") from exc
    except httpx.HTTPError as exc:
        _failed(app, ref, method, started, f"unreachable: {exc}")
        raise JobError(f"{app.name} couldn't be reached") from exc
    if response.status_code == httpx.codes.NOT_FOUND:
        _failed(app, ref, method, started, "404")
        raise JobGone(f"{app.name} doesn't know that job")
    if response.status_code != httpx.codes.OK:
        _failed(app, ref, method, started, f"status {response.status_code}")
        raise JobError(f"{app.name} answered {response.status_code}")
    if len(response.content) > MAX_BYTES:
        _failed(app, ref, method, started, f"{len(response.content)} bytes")
        raise JobError(f"{app.name}'s job report was too large")
    try:
        job = JobRead.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        _failed(app, ref, method, started, "unreadable")
        raise JobError(f"{app.name}'s job report wasn't readable") from exc
    logger.info(
        "job_read" if method == "GET" else "job_cancel",
        app=app.name,
        ref=ref,
        status=job.status,
        done=job.done,
        total=job.total,
        ms=round((time.monotonic() - started) * 1000),
    )
    return job


def _failed(app: FleetApp, ref: str, method: str, started: float, why: str) -> None:
    logger.warning(
        "job_request_failed",
        app=app.name,
        ref=ref,
        method=method,
        why=why,
        ms=round((time.monotonic() - started) * 1000),
    )
