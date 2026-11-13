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
  failed with no word on what it did (a v1 5xx, a read timeout, a 409 on a
  retried key). The app's run may have done writes before dying, so the
  result is unknown and a retry could duplicate them (#60).
  :class:`DelegateConnectionLost` is the one the tool layer may retry, under
  the same ``Idempotency-Key``: the connection dropped after the request went
  out.
- :class:`DelegateRunFailed` — a v1.1 failure body (#67): the run failed and
  the app said what it did. Any ``done`` entry means it partly happened; none
  means nothing changed.
- :class:`DelegateConfirmRefused` — a ``Confirm-Token`` the app can't honour
  (a 409 naming ``unknown`` / ``expired`` / ``used`` / ``stale``): nothing
  ran, and the question is asked again, never assumed (#68).
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
from typing import Annotated, Any, Literal, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError, ValidatorFunctionWrapHandler, WrapValidator

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
# conductor (its own turns, or the MCP host).
CONDUCTOR_ACTOR = "agent:conductor"

# Trace headers (#62). Conductor sends the turn and hop ids on every request of
# a delegate call; apps answer with their own request id (PCC's and music's
# RequestIDMiddleware), which the hop records.
TURN_HEADER = "X-Agent-Turn"
HOP_HEADER = "X-Agent-Hop"
APP_REQUEST_ID_HEADER = "X-Request-ID"

# delegate-api v1.1 (#67): the client's name for one exchange, so a retry of it
# never runs the app's loop twice. Minted per ``ask`` step, carried on the hop.
IDEMPOTENCY_HEADER = "Idempotency-Key"

# delegate-api "Confirmation" (#68): the app's single-use token for one armed
# call. A message carrying it is the person's yes; the app runs exactly that
# call, once, with no model turn.
CONFIRM_HEADER = "Confirm-Token"

_DEFAULT_READ_TIMEOUT = 300.0
_DEFAULT_CONNECT_TIMEOUT = 5.0
# A cancel never blocks app-side, so it never waits like a message does.
_CANCEL_TIMEOUT = 5.0


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


class DelegateConnectionLost(DelegateOutcomeUnknown):
    """The connection dropped after the message went out (not a timeout).

    The app may have run, so this is still an unknown outcome; but a retry
    under the same ``Idempotency-Key`` gets the app's stored answer instead of
    a second run, so the tool layer retries it once.
    """


class DelegateRunFailed(DelegateError):
    """A run that failed with a v1.1 failure body: the app said what it did.

    ``record`` is the failed run's partial record. Any ``done`` entry means the
    request partly happened, and resending it would repeat that; none means
    nothing changed.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int,
        stop_reason: str | None,
        record: list[RecordEntry],
    ) -> None:
        super().__init__(message)
        self.status = status
        self.stop_reason = stop_reason
        self.record = record

    @property
    def done(self) -> list[RecordEntry]:
        return [entry for entry in self.record if entry.outcome == "done"]


class DelegateConfirmRefused(DelegateError):
    """A ``Confirm-Token`` the app refused (409): nothing ran.

    ``kind`` is the app's word for why: ``unknown`` (not this thread's, or the
    app restarted), ``expired``, ``used`` or ``stale`` (what it previewed has
    changed).
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class DelegateRequestRejected(DelegateError):
    """The app rejected the request as invalid (a 4xx other than 404/429)."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


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


class EntityRef(BaseModel):
    """What a call changed or found: ``{app, kind, id, title}`` (v1.1).

    ``untrusted`` marks a ``title`` that is outside text (a web title, a file
    tag): shown or spoken quoted, never fed to anything that decides what to do.
    """

    app: str
    kind: str
    id: str
    title: str | None = None
    untrusted: bool = False


class Undo(BaseModel):
    """How a ``done`` call can be undone, or why it can't."""

    handle: str | None = None
    how: str | None = None
    none: str | None = None


class JobRef(BaseModel):
    ref: str
    status: str
    done: int = 0
    total: int | None = None


RecordOutcome = Literal["done", "refused", "looked_up", "not_run"]


class RecordEntry(BaseModel):
    """One receipt: what one of the app's tool calls did, built by the app's
    code from the call, never from its model's words (delegate-api v1.1)."""

    tool: str
    outcome: RecordOutcome
    touched: list[EntityRef] = []
    count: int | None = None
    reason: str | None = None
    undo: Undo | None = None
    job: JobRef | None = None
    untrusted: bool = False


def _lenient_record(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """A record conductor can't read is an absent one (outcome unknown), never a
    reason to drop the turn it rides on."""
    try:
        return handler(value)
    except ValidationError:
        logger.warning("delegate_record_unreadable")
        return None


# ``None`` is a v1 app (or an unreadable record); ``[]`` is a v1.1 app saying
# nothing ran.
Record = Annotated[list[RecordEntry] | None, WrapValidator(_lenient_record)]


class ConfirmPreview(BaseModel):
    """What a gated call would change: entity refs (at most 20) and the total."""

    count: int | None = None
    touched: list[EntityRef] = []


class NeedsInputRead(BaseModel):
    """An app's typed question to the person (delegate-api v1.2)."""

    question: str
    candidates: list[dict[str, Any]] = []


def _lenient_needs_input(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """A ``needs_input`` conductor can't read is no question: the plan runs on
    as for a v1.1 reply rather than failing the call."""
    try:
        return handler(value)
    except ValidationError:
        logger.warning("delegate_needs_input_unreadable")
        return None


NeedsInput = Annotated[NeedsInputRead | None, WrapValidator(_lenient_needs_input)]


class ConfirmRequired(BaseModel):
    """A gated call the app armed instead of running (delegate-api
    "Confirmation"): its token, and the question and preview its code built."""

    token: str
    tool: str
    question: str
    preview: ConfirmPreview | None = None
    expires_at: datetime | None = None


def _lenient_confirm(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """A confirmation conductor can't read is one it can't answer: the app's
    question still reaches the person, and a yes then goes back as words."""
    try:
        return handler(value)
    except ValidationError:
        logger.warning("delegate_confirm_unreadable")
        return None


Confirm = Annotated[ConfirmRequired | None, WrapValidator(_lenient_confirm)]


class MessageRead(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str | None = None
    # `null` (not `[]`) when no tools ran.
    tool_calls: list[ToolCallRead] | None = None
    stop_reason: str | None = None
    created_at: datetime
    # v1.1: one receipt per tool call.
    record: Record = None
    # Set on a ``confirm_required`` stop: the armed call's token and question.
    confirm_required: Confirm = None
    # v1.2 (#202): the app's own question when it did nothing,
    # ``{question, candidates}``; read only from an app that declares v1.2.
    needs_input: NeedsInput = None


class FailureBody(BaseModel):
    """A 5xx body from ``POST …/messages``. v1 apps send only ``detail``."""

    detail: Any = None
    stop_reason: str | None = None
    assistant_message_id: int | None = None
    record: Record = None


class MessageExchange(BaseModel):
    """What one ``POST …/messages`` returns: the stored user turn and the
    assistant turn the app's loop answered with."""

    user_message: MessageRead
    assistant_message: MessageRead


class RunCancelled(BaseModel):
    """What ``POST …/cancel`` returns: whether a run was in flight to stop."""

    cancel_requested: bool = False


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

    def send_message(
        self,
        conversation_id: int,
        message: str,
        *,
        idempotency_key: str | None = None,
        confirm_token: str | None = None,
        context: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> MessageExchange:
        """``POST /conversations/{id}/messages`` → the user + assistant turns.

        The delegate contract's request field is ``content``. The
        ``Idempotency-Key`` is ``idempotency_key``, else the current hop's
        (the tool layer mints one per ``ask`` step), else none (v1). A
        ``confirm_token`` answers the app's question yes (#68). ``context``
        and ``mode`` are delegate v1.2's (#202), sent only when given: a v1.1
        app may reject unknown fields.
        """
        hop = run_trace.current_hop()
        key = idempotency_key or (hop.idempotency_key if hop is not None else None)
        headers: dict[str, str] = {}
        if key:
            headers[IDEMPOTENCY_HEADER] = key
        if confirm_token:
            headers[CONFIRM_HEADER] = confirm_token
        response = self._send(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json={
                "content": message,
                **({"context": context} if context is not None else {}),
                **({"mode": mode} if mode is not None else {}),
            },
            may_have_acted=True,
            extra_headers=headers or None,
        )
        return self._parse(response, MessageExchange)

    def cancel(self, conversation_id: int) -> bool:
        """``POST /conversations/{id}/cancel`` → whether a run was in flight (#66).

        Asks the thread's in-flight run to stop at its next boundary; its own
        ``POST …/messages`` then answers ``cancelled`` with what it did. Not in
        delegate-api v1.1 (PCC exposes it); an app without the route answers
        404, raised as :class:`DelegateThreadGone`.
        """
        response = self._send(
            "POST", f"/conversations/{conversation_id}/cancel", timeout=_CANCEL_TIMEOUT
        )
        return self._parse(response, RunCancelled).cancel_requested

    def get_conversation(self, conversation_id: int) -> ConversationDetail:
        """``GET /conversations/{id}`` → the thread with its history."""
        response = self._send("GET", f"/conversations/{conversation_id}")
        return self._parse(response, ConversationDetail)

    def delete_conversation(self, conversation_id: int) -> None:
        """``DELETE /conversations/{id}`` — soft-delete/reset the thread (204)."""
        self._send("DELETE", f"/conversations/{conversation_id}")

    def _send(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        may_have_acted: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """One request with the actor header, mapping every fault to a typed error.

        ``may_have_acted`` marks a request whose handler runs the app's agent
        (and so its writes): once it is sent, a failure is
        :class:`DelegateOutcomeUnknown` rather than :class:`DelegateUnavailable`.
        """
        url = f"{self._base_url}{path}"
        headers = {"X-Agent-Actor": self._actor, **(extra_headers or {})}
        # Trace propagation (#62): the turn and this hop, so the app's records
        # of the request can join conductor's. The app's own request id comes
        # back on its response and is kept on the hop.
        hop = run_trace.current_hop()
        if hop is not None:
            headers[TURN_HEADER] = hop.run_id
            headers[HOP_HEADER] = hop.hop_id
        try:
            if timeout is None:
                response = self._client.request(method, url, json=json, headers=headers)
            else:
                response = self._client.request(
                    method, url, json=json, headers=headers, timeout=timeout
                )
        except httpx.HTTPError as exc:
            logger.warning("delegate_request_failed", url=url, error=str(exc))
            if may_have_acted and not isinstance(exc, _NOT_SENT_ERRORS):
                if not isinstance(exc, httpx.TimeoutException):
                    raise DelegateConnectionLost(
                        f"connection to {url} dropped after the request was sent: {exc}"
                    ) from exc
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
                failure = _failure_body(response)
                if failure is not None and failure.record is not None:
                    raise DelegateRunFailed(
                        detail,
                        status=status,
                        stop_reason=failure.stop_reason,
                        record=failure.record,
                    )
                raise DelegateOutcomeUnknown(detail)
            raise DelegateUnavailable(detail)
        if status == httpx.codes.CONFLICT:
            refusal = _confirm_refusal(response)
            if refusal is not None:
                raise DelegateConfirmRefused(
                    f"{method} {path} → 409: confirm token {refusal}", kind=refusal
                )
        if status >= 400:
            # A client error that isn't 404/429 (e.g. a 422 schema rejection).
            # The request is at fault, not the app — retrying it can't succeed.
            raise DelegateRequestRejected(
                f"{method} {path} → {status}: {response.text[:300]}", status=status
            )
        return response

    @staticmethod
    def _parse(response: httpx.Response, model: type[_ModelT]) -> _ModelT:
        try:
            return model.model_validate_json(response.text)
        except ValidationError as exc:
            raise DelegateProtocolError(
                f"delegate response failed {model.__name__} validation: {exc}"
            ) from exc


def _failure_body(response: httpx.Response) -> FailureBody | None:
    """A 5xx's v1.1 failure body, or ``None`` when it isn't one (a v1 app, a
    proxy's HTML page)."""
    try:
        return FailureBody.model_validate_json(response.text)
    except ValidationError:
        return None


def _confirm_refusal(response: httpx.Response) -> str | None:
    """A 409's ``confirm`` word, when it is a refused ``Confirm-Token``."""
    try:
        body = response.json()
    except ValueError:
        return None
    kind = body.get("confirm") if isinstance(body, dict) else None
    return kind if isinstance(kind, str) and kind else None


def _parse_retry_after(raw: str | None) -> int | None:
    """The ``Retry-After`` header as whole seconds, or ``None`` if absent/odd."""
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None
