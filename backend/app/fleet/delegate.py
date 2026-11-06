"""Delegate REST client: conductor speaking the workspace agent contract.

Conductor is the *client* of every app agent's delegate API
(``../agent-standard/delegate-api.md``); the servers are PCC's
``routes_agent.py`` and chess's ``agent_api.py``. This module is plain
``httpx`` against ``http://{host}:{port}{agent.api}`` with the response bodies
Pydantic-validated at the boundary — the wire models mirror PCC/chess
field-for-field (``MessageExchange`` of two ``MessageRead`` turns, each with
``tool_calls`` of ``{tool, arguments, result, error}``).

Every failure mode the contract names becomes a typed error so the tool layer
can react precisely and no bare ``httpx`` exception ever escapes:

- :class:`DelegateThreadGone` (404) — thread unknown or soft-deleted; the tool
  recreates and retries exactly once.
- :class:`DelegateRateLimited` (429) — carries ``Retry-After``; surfaced, never
  auto-retried.
- :class:`DelegateUnavailable` — the app never acted: a connect-phase error
  (nothing was sent), a 503, or any 5xx/timeout on a request with no side
  effects. Retrying is safe.
- :class:`DelegateOutcomeUnknown` — a ``send_message`` that went out and then
  failed (a 500/502/504, or a read timeout/drop). The app's run may have done
  writes before dying (PCC and music persist the partial run and return 5xx),
  so the result is unknown and a retry could duplicate them (#60).
- :class:`DelegateRequestRejected` — any other 4xx (e.g. a 422 for a message
  the app's schema refuses): the request itself is invalid, so retrying the
  same input can never succeed.
- :class:`DelegateProtocolError` — a 2xx whose body doesn't match the contract.

Timeouts follow the fleet latency profile: a short connect timeout (~5s) but a
long read timeout (300s), because a delegate call can wrap a cold model load of
~100s before the first byte.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from app.ai import trace as run_trace

logger = structlog.get_logger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Transport faults raised before any request bytes reach the app: a retry is
# safe because the app cannot have acted. Every other transport fault (read
# timeout, dropped connection, write error) may land after the app started.
_NOT_SENT_ERRORS: tuple[type[httpx.HTTPError], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.UnsupportedProtocol,
)

# The actor conductor presents to every app agent. The app binds it as the
# audit actor for all mutations in the run (in place of its default
# `agent:loop`); apps honor only recognized delegate actors, so this is the one
# identity conductor may stamp. It is constant regardless of who drives
# conductor (its own loop, or the MCP host).
CONDUCTOR_ACTOR = "agent:conductor"

# Trace headers (#62). Conductor sends the turn and hop ids on every request of
# a delegate call; apps answer with their own request id (PCC's and music's
# RequestIDMiddleware), which the hop records.
TURN_HEADER = "X-Agent-Turn"
HOP_HEADER = "X-Agent-Hop"
APP_REQUEST_ID_HEADER = "X-Request-ID"

_DEFAULT_READ_TIMEOUT = 300.0
_DEFAULT_CONNECT_TIMEOUT = 5.0


class DelegateError(Exception):
    """Base for every typed delegate failure."""


class DelegateThreadGone(DelegateError):
    """The subagent thread is unknown or soft-deleted (HTTP 404)."""


class DelegateRateLimited(DelegateError):
    """The app is rate-limiting (HTTP 429). ``retry_after`` is seconds, if given."""

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DelegateUnavailable(DelegateError):
    """The app didn't act: nothing was sent, a 503, or a failed side-effect-free call."""


class DelegateOutcomeUnknown(DelegateError):
    """A message went out and the run failed: the app may have acted partway.

    Deliberately *not* a :class:`DelegateUnavailable` — nothing may treat it as
    retry-safe.
    """


class DelegateRequestRejected(DelegateError):
    """The app rejected the request as invalid (a 4xx other than 404/429)."""


class DelegateProtocolError(DelegateError):
    """A success response whose body doesn't match the delegate contract."""


# --- wire models (field names mirror PCC/chess schemas exactly) ---------------


class ToolCallRead(BaseModel):
    """One dispatched tool call as an app persisted it. Exactly one of
    ``result`` / ``error`` is set."""

    tool: str
    arguments: dict[str, Any] = {}
    result: str | None = None
    error: str | None = None


class MessageRead(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str | None = None
    # `null` (not `[]`) when no tools ran.
    tool_calls: list[ToolCallRead] | None = None
    stop_reason: str | None = None
    created_at: datetime


class MessageExchange(BaseModel):
    """What one ``POST …/messages`` returns: the stored user turn and the
    assistant turn the app's loop answered with."""

    user_message: MessageRead
    assistant_message: MessageRead


class ConversationCreated(BaseModel):
    """The created-thread response. Conductor needs only ``id``."""

    id: int


class ConversationDetail(BaseModel):
    """A thread with its history — for debugging/UI; conductor's routing
    doesn't need it, but the contract exposes it."""

    id: int
    title: str | None = None
    messages: list[MessageRead] = []


class DelegateClient:
    """Delegate-API client for one app, bound to its ``base_url``.

    ``base_url`` is ``http://{host}:{port}{agent.api}`` (see
    :attr:`app.fleet.manifests.FleetApp.agent_base_url`). Synchronous, matching
    the sync tool registry; FastAPI runs sync callers in worker threads.
    ``client`` is injectable for tests (``httpx.MockTransport``); otherwise the
    client owns one. Usable as a context manager to close its connections.
    """

    def __init__(
        self,
        base_url: str,
        *,
        actor: str = CONDUCTOR_ACTOR,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._actor = actor
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout)
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DelegateClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create_conversation(self, *, title: str | None = None) -> int:
        """``POST /conversations`` → the new thread id."""
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        response = self._send("POST", "/conversations", json=body)
        return self._parse(response, ConversationCreated).id

    def send_message(self, conversation_id: int, message: str) -> MessageExchange:
        """``POST /conversations/{id}/messages`` → the user + assistant turns.

        The delegate contract's request field is ``content``.
        """
        response = self._send(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json={"content": message},
            may_have_acted=True,
        )
        return self._parse(response, MessageExchange)

    def get_conversation(self, conversation_id: int) -> ConversationDetail:
        """``GET /conversations/{id}`` → the thread with its history."""
        response = self._send("GET", f"/conversations/{conversation_id}")
        return self._parse(response, ConversationDetail)

    def delete_conversation(self, conversation_id: int) -> None:
        """``DELETE /conversations/{id}`` — soft-delete/reset the thread (204)."""
        self._send("DELETE", f"/conversations/{conversation_id}")

    def _send(
        self, method: str, path: str, *, json: Any = None, may_have_acted: bool = False
    ) -> httpx.Response:
        """One request with the actor header, mapping every fault to a typed error.

        ``may_have_acted`` marks a request whose handler runs the app's agent
        (and so its writes): once it is sent, a failure is
        :class:`DelegateOutcomeUnknown` rather than :class:`DelegateUnavailable`.
        """
        url = f"{self._base_url}{path}"
        headers = {"X-Agent-Actor": self._actor}
        # Trace propagation (#62): the turn and this hop, so the app's records
        # of the request can join conductor's. The app's own request id comes
        # back on its response and is kept on the hop.
        hop = run_trace.current_hop()
        if hop is not None:
            headers[TURN_HEADER] = hop.run_id
            headers[HOP_HEADER] = hop.hop_id
        try:
            response = self._client.request(method, url, json=json, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("delegate_request_failed", url=url, error=str(exc))
            if may_have_acted and not isinstance(exc, _NOT_SENT_ERRORS):
                raise DelegateOutcomeUnknown(
                    f"request to {url} failed after it was sent: {exc}"
                ) from exc
            raise DelegateUnavailable(f"request to {url} failed: {exc}") from exc

        if hop is not None:
            hop.app_request_id = response.headers.get(APP_REQUEST_ID_HEADER) or hop.app_request_id
        status = response.status_code
        if status == httpx.codes.NOT_FOUND:
            raise DelegateThreadGone(f"{method} {path} → 404 (thread gone)")
        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise DelegateRateLimited(
                f"{method} {path} → 429 (rate limited)",
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        if status >= 500:
            detail = f"{method} {path} → {status}: {response.text[:300]}"
            # A 503 means the app turned the request away; any other 5xx on a
            # run may follow writes the app already made (and persisted).
            if may_have_acted and status != httpx.codes.SERVICE_UNAVAILABLE:
                raise DelegateOutcomeUnknown(detail)
            raise DelegateUnavailable(detail)
        if status >= 400:
            # A client error that isn't 404/429 (e.g. a 422 schema rejection).
            # The request is at fault, not the app — retrying it can't succeed.
            raise DelegateRequestRejected(f"{method} {path} → {status}: {response.text[:300]}")
        return response

    @staticmethod
    def _parse(response: httpx.Response, model: type[_ModelT]) -> _ModelT:
        try:
            return model.model_validate_json(response.text)
        except ValidationError as exc:
            raise DelegateProtocolError(
                f"delegate response failed {model.__name__} validation: {exc}"
            ) from exc


def _parse_retry_after(raw: str | None) -> int | None:
    """The ``Retry-After`` header as whole seconds, or ``None`` if absent/odd."""
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None
