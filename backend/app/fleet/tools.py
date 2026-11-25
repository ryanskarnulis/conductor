"""Fleet tools: ``ask_<app>`` per agent, ``open_<app>`` per openable app, plus
``list_agents``.

:func:`build_delegate_tools` turns each fleet app into tools on the shared
registry (``app/tools/registry.py``), from its manifest blocks:

- an **``agent:``** block yields ``ask_<app>`` (``ask_tasks``, …), whose
  docstring is the manifest's ``agent.description``. Calling it opens a fresh
  subagent thread (or, answering the app's own question, uses the thread that
  asked: #262), sends the message with ``X-Agent-Actor: agent:conductor``, and
  relays the assistant's reply plus a compact note of what the app actually
  did — never the raw transcript.
- an **``open:``** block yields ``open_<app>`` (``open_chess``, …) — a handoff,
  not a delegation. Some apps are places the user goes rather than services
  conductor calls: chess is a board you play *in*, and proxying moves through
  conductor's chat is a worse game than the app itself. The tool is purely
  local — it returns a handoff payload (target app, path, and the user's intent)
  that conductor's frontend reads off the persisted trajectory and navigates to.
  No network, no thread, no budget: nothing is asked of the app, the user is
  simply sent to it, and its agent picks up the intent on arrival.

The blocks are independent — an app may declare either, both, or neither.

Contract behaviors live here, not in the client:

- **A fresh thread per ask + 404 recreate-once** (#262). Code picks the
  thread, never a model: a confirm goes to the thread its token is bound to,
  an answer to an app's question to the thread that asked
  (:func:`app.fleet.context.answering`), and every other ask to a fresh one,
  so no app answers from an earlier request's history. The
  :class:`DelegationContext` records each; a :class:`DelegateThreadGone`
  triggers exactly one recreate-and-retry (a second 404 fails the call).
- **Typed delegate faults → domain errors.** Rate-limit, unavailable, outcome-unknown, protocol,
  and double-404 all become informative :class:`ToolError` s (the model reads
  them and adapts); a 429 surfaces its ``Retry-After`` and is never auto-retried.
- **Delegate v1.1 (#67).** Each ``ask`` step mints one ``Idempotency-Key``
  (on the hop; every message of the step carries it). A connection that drops
  after the message went out is retried once under that key, so the app
  answers from its store instead of running again. The app's typed record
  rides out beside the reply (``runtime.delegated_record``), and a failure
  body that lists ``done`` work is reported as partly happened, by name.
- **Structural confirmation (#68).** An app that answers ``confirm_required``
  armed a gated call instead of running it. The model gets the app's question
  to put to the person, word for word; the token stays on the turn's record
  (``runtime.delegated_confirm``) and the question is a card. A tap on it is
  ``confirm(app)``, run by code with no model (#251): it sends the token back
  on the same thread, so the app runs exactly that call. The context holds
  exactly the question tapped (``DelegationContext.pending_confirm``).

``list_agents`` is a local tool (no network) reporting the fleet from the
manifests. Routing hints live in the planner's prompt
(:func:`app.ai.compiler.planner_prompt`), built from the manifests.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import structlog

from app.ai import relay, run_record
from app.ai import trace as run_trace
from app.ai.confirmation import CONFIRM_TOOL, AppConfirm, PendingConfirm
from app.ai.facts import attach
from app.ai.stop import STOPPED_ERROR, current_stop
from app.fleet.context import DelegationContext, answering_thread, current_delegation_context
from app.fleet.delegate import (
    CONDUCTOR_ACTOR,
    DelegateClient,
    DelegateConfirmRefused,
    DelegateConnectionLost,
    DelegateError,
    DelegateOutcomeUnknown,
    DelegateProtocolError,
    DelegateRateLimited,
    DelegateRequestRejected,
    DelegateRunFailed,
    DelegateThreadGone,
    DelegateUnavailable,
    MessageExchange,
    MessageRead,
    RecordEntry,
)
from app.fleet.manifests import Fleet, FleetApp
from app.tools import registry, runtime
from app.tools.registry import ToolError

logger = structlog.get_logger(__name__)

# Both PCC and chess constrain a delegate message to 1–8000 chars after
# whitespace-stripping (their shared `AgentMessageText`). Guarding here keeps a
# doomed request from ever leaving conductor.
MAX_DELEGATE_MESSAGE_LENGTH = 8_000

# A handed-off intent rides in a query string and is one utterance, not an
# essay. Chess enforces the same cap on the receiving end.
MAX_INTENT_LENGTH = 500


class DelegateClientLike(Protocol):
    """The delegate-client surface the tools use (context manager + three calls).

    :class:`app.fleet.delegate.DelegateClient` satisfies it structurally; tests
    supply a fake at the same shape, so no live call happens in pytest.
    """

    def __enter__(self) -> DelegateClientLike: ...

    def __exit__(self, *exc: object) -> None: ...

    def create_conversation(self, *, title: str | None = None) -> int: ...

    def send_message(
        self,
        conversation_id: int,
        message: str,
        *,
        confirm_token: str | None = None,
        context: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> MessageExchange: ...

    def cancel(self, conversation_id: int) -> bool: ...

    def delete_conversation(self, conversation_id: int) -> None: ...


# Delegate v1.2's cap on refs per message (agent-standard delegate-api.md).
MAX_CONTEXT_REFS = 20

# A factory yields a fresh client bound to one app.
ClientFactory = Callable[[FleetApp], DelegateClientLike]


# The words a delegate call's error carries when the app may have written but
# its answer was lost (no record): the ledger files such a call as an
# ``unknown`` receipt, a possible write (#69).
OUTCOME_UNKNOWN = "stopped partway through"


def default_client_factory() -> ClientFactory:
    """A factory that opens a real :class:`DelegateClient` per call."""

    def factory(app: FleetApp) -> DelegateClient:
        return DelegateClient(app.agent_base_url, actor=CONDUCTOR_ACTOR)

    return factory


def ask_tool_name(app_name: str) -> str:
    """The tool name for an app: ``ask_<name>`` (slug hyphens → underscores)."""
    return f"ask_{app_name.replace('-', '_')}"


def open_tool_name(app_name: str) -> str:
    """The handoff tool name for an app: ``open_<name>`` (same slug rule)."""
    return f"open_{app_name.replace('-', '_')}"


def look_tool_name(app_name: str) -> str:
    """The name a glance read goes by on a plan's record: ``look_<name>``.

    Not a registry tool: the model is never offered it. The interpreter runs a
    ``look`` step itself (#73) and files it under this name.
    """
    return f"{run_record.LOOK_PREFIX}{app_name.replace('-', '_')}"


def build_delegate_tools(fleet: Fleet, client_factory: ClientFactory) -> list[str]:
    """Register the fleet's tools: ``ask_<app>``, ``open_<app>``, ``list_agents``.

    One ``ask_<app>`` per agent-bearing app and one ``open_<app>`` per openable
    app (an app can be both, or neither — the blocks are independent). Returns
    the registered tool names, in order. Mutates the shared registry — the one
    surface the turn runtime and the MCP server both consume. Inert fleet members get
    no tool (``list_agents`` still names them).
    """
    names: list[str] = []
    for app in fleet.agent_apps():
        fn = _make_ask_tool(app, client_factory)
        registry.tool(fn)
        names.append(fn.__name__)
    for app in fleet.open_apps():
        open_fn = _make_open_tool(app)
        registry.tool(open_fn)
        names.append(open_fn.__name__)
    if _confirming_apps(fleet):
        confirm_fn = _make_confirm_tool(fleet, client_factory)
        registry.tool(confirm_fn)
        names.append(confirm_fn.__name__)
    list_fn = _make_list_agents(fleet)
    registry.tool(list_fn)
    names.append(list_fn.__name__)
    logger.info("delegate_tools_built", tools=names)
    return names


def _make_ask_tool(app: FleetApp, client_factory: ClientFactory) -> Callable[[str], str]:
    """Build the ``ask_<app>`` callable for one agent-bearing app."""

    def ask_tool(
        message: str,
        facts: list[str] | None = None,
        refs: list[dict[str, Any]] | None = None,
        read: bool = False,
    ) -> str:
        assert app.agent is not None
        if app.agent.context:
            # Delegate v1.2 (#202): the words go as written, and the refs and
            # saved facts beside them as ``context``, data the app reads after
            # the words; a step that only reads goes ``mode: "read"``.
            given = {"refs": list(refs or [])[:MAX_CONTEXT_REFS], "facts": list(facts or [])}
            context = {k: v for k, v in given.items() if v} or None
            return _delegate(
                app, client_factory, message, app_context=context, mode="read" if read else None
            )
        # Saved facts code matched to the request (#134) follow it, quoted
        # and marked as context; a request too long to carry them goes alone.
        sent = attach(message.strip(), facts or [])
        return _delegate(
            app, client_factory, sent if len(sent) <= MAX_DELEGATE_MESSAGE_LENGTH else message
        )

    ask_tool.__name__ = ask_tool_name(app.name)
    ask_tool.__qualname__ = ask_tool.__name__
    # The description the model sees comes straight from the manifest.
    assert app.agent is not None  # agent_apps() guarantees this
    ask_tool.__doc__ = app.agent.description
    return ask_tool


def _make_open_tool(app: FleetApp) -> Callable[..., str]:
    """Build the ``open_<app>`` handoff callable for one openable app.

    Purely local: no network, no delegate thread, no call budget — nothing is
    *asked* of the app, the user is simply sent to it. The tool's return value
    is the handoff payload the frontend reads off the persisted trajectory to
    perform the navigation; the model sees it too, which is why it reads as a
    plain statement of what is about to happen.
    """
    assert app.open is not None  # open_apps() guarantees this

    def open_tool(intent: str = "") -> str:
        spec = app.open
        assert spec is not None
        payload: dict[str, Any] = {
            "handoff": app.name,
            "title": app.title,
            "path": spec.path,
            # The dev fallback: on a bare host (no dot) the frontend can't
            # derive `<name>.<domain>`, so it dials the upstream directly.
            "upstream": app.upstream,
        }
        cleaned = intent.strip()[:MAX_INTENT_LENGTH]
        if spec.intent_param and cleaned:
            payload["intent_param"] = spec.intent_param
            payload["intent"] = cleaned
        logger.info("app_handoff", app=app.name, intent=bool(payload.get("intent")))
        return json.dumps(payload)

    open_tool.__name__ = open_tool_name(app.name)
    open_tool.__qualname__ = open_tool.__name__
    # The description the model sees comes straight from the manifest; the
    # `intent` arg is documented here because the manifest can't type it.
    open_tool.__doc__ = (
        f"{app.open.description}\n\n"
        "Pass the user's request, in their own words, as `intent` — the app's "
        "agent picks it up on arrival and acts on it, so nothing they asked for "
        "is lost in the handoff."
    )
    return open_tool


def _confirming_apps(fleet: Fleet) -> list[FleetApp]:
    """The delegate apps that ask the person themselves (``agent.confirms``)."""
    return [app for app in fleet.agent_apps() if app.agent is not None and app.agent.confirms]


def _make_confirm_tool(fleet: Fleet, client_factory: ClientFactory) -> Callable[[str], str]:
    """Build ``confirm(app)``: the person's yes to the question an app asked."""
    apps = {app.name: app for app in _confirming_apps(fleet)}

    def confirm(app: str) -> str:
        """The person's tap on the card of a question an app asked: the app does
        exactly what it asked about. Code calls it, never a model; `app` is
        the app that asked (e.g. "tasks")."""
        target = apps.get(app.strip().lower())
        pending = (
            current_delegation_context().take_pending_confirm(target.name)
            if target is not None
            else None
        )
        if target is None or pending is None:
            raise ToolError(f"Nothing from {app} is waiting on a yes, so nothing was done.")
        return _delegate(target, client_factory, "yes", confirm=pending)

    confirm.__name__ = CONFIRM_TOOL
    confirm.__qualname__ = CONFIRM_TOOL
    return confirm


def _delegate(
    app: FleetApp,
    client_factory: ClientFactory,
    message: str,
    *,
    confirm: PendingConfirm | None = None,
    app_context: dict[str, Any] | None = None,
    mode: str | None = None,
) -> str:
    """Send ``message`` to ``app``'s agent, applying every contract behavior.

    ``app_context`` and ``mode`` are delegate v1.2's (#202), for an app that
    declares it; they ride every send of this step, its resend included.

    With ``confirm``, the message is the person's yes: it goes to the thread
    that asked, with that question's token, and is never recreated on a 404
    (a fresh thread never asked it). An answer to an app's question goes to
    the thread that asked it (:func:`answering_thread`); every other ask
    opens a fresh thread (#262).
    """
    message = message.strip()
    if not message:
        raise ToolError(
            f"The message to {app.name} was empty. Compose an actual request before asking."
        )
    if len(message) > MAX_DELEGATE_MESSAGE_LENGTH:
        raise ToolError(
            f"The message to {app.name} is {len(message)} characters; the limit is "
            f"{MAX_DELEGATE_MESSAGE_LENGTH}. Shorten it and ask again."
        )
    stop = current_stop()
    if stop is not None and stop.is_set():
        raise ToolError(STOPPED_ERROR)
    context = current_delegation_context()

    # One hop per delegate call (#62): its ids ride every request to the app,
    # and what came back lands on the turn's trace.
    with run_trace.hop(app.name) as hop:
        # One key for this step: a retry below, and a 404 → recreate (a fresh
        # thread has never seen it), resend it rather than run twice.
        hop.idempotency_key = uuid.uuid4().hex
        started = time.monotonic()
        subagent_id: int | None = None
        try:
            with (
                client_factory(app) as client,
                _released_unless_kept(client, app, context, lambda: subagent_id),
                _passing_stop_on(client, lambda: subagent_id),
            ):
                if confirm is not None:
                    subagent_id = confirm.thread_id
                    context.sent_to(app.name, subagent_id)
                    try:
                        exchange = _send_once_more_if_dropped(
                            client, subagent_id, message, confirm_token=confirm.token
                        )
                    except DelegateThreadGone as gone:
                        raise DelegateConfirmRefused(str(gone), kind="unknown") from gone
                    return _finish(app, context, hop, subagent_id, started, exchange)
                subagent_id = answering_thread(app.name)
                if subagent_id is not None:
                    context.sent_to(app.name, subagent_id)
                else:
                    subagent_id = client.create_conversation()
                    context.opened(app.name, subagent_id)
                try:
                    exchange = _send_once_more_if_dropped(
                        client, subagent_id, message, context=app_context, mode=mode
                    )
                except DelegateThreadGone:
                    # Contract rule: a pruned thread → recreate once and retry
                    # exactly once. A second 404 propagates and fails the call.
                    # An answer goes too: its request carries what was asked.
                    subagent_id = client.create_conversation()
                    context.opened(app.name, subagent_id)
                    exchange = _send_once_more_if_dropped(
                        client, subagent_id, message, context=app_context, mode=mode
                    )
        except DelegateError as exc:
            latency_ms = _elapsed_ms(started)
            context.audit_delegate_call(
                app=app.name,
                subagent_conversation_id=subagent_id,
                latency_ms=latency_ms,
                error=type(exc).__name__,
                hop_id=hop.hop_id,
                app_request_id=hop.app_request_id,
            )
            hop.thread_id, hop.ms, hop.error = subagent_id, latency_ms, type(exc).__name__
            runtime.delegated_thread.set(subagent_id)
            if isinstance(exc, DelegateRunFailed):
                # The app said what it did before failing: keep it on the turn.
                hop.app_stop_reason = exc.stop_reason
                _report(hop, exc.record, [entry.tool for entry in exc.record])
            raise _to_tool_error(app, exc) from exc
        return _finish(app, context, hop, subagent_id, started, exchange)


@contextmanager
def _released_unless_kept(
    client: DelegateClientLike,
    app: FleetApp,
    context: DelegationContext,
    thread: Callable[[], int | None],
) -> Iterator[None]:
    """A run with no thread store (the MCP server) deletes the thread once
    its call is over, however it ended: nothing there can answer it later
    (#262). Best-effort, like the route's cleanup."""
    try:
        yield
    finally:
        thread_id = thread()
        if not context.keeps_threads and thread_id is not None:
            try:
                client.delete_conversation(thread_id)
            except DelegateError as exc:
                logger.warning(
                    "subagent_thread_release_failed",
                    app=app.name,
                    thread_id=thread_id,
                    error=str(exc),
                )


@contextmanager
def _passing_stop_on(
    client: DelegateClientLike, thread: Callable[[], int | None]
) -> Iterator[None]:
    """While the block's message is in flight, a Stop sends the app's cancel
    for ``thread()`` (#66). A run already stopped never starts the call."""
    stop = current_stop()
    if stop is None:
        yield
        return

    def cancel() -> None:
        thread_id = thread()
        if thread_id is not None:
            client.cancel(thread_id)

    with stop.delegating(cancel) as live:
        if not live:
            raise ToolError(STOPPED_ERROR)
        yield


def _finish(
    app: FleetApp,
    context: DelegationContext,
    hop: run_trace.Hop,
    subagent_id: int,
    started: float,
    exchange: MessageExchange,
) -> str:
    """Audit and trace a delegate call that answered, and relay its reply."""
    assistant = exchange.assistant_message
    latency_ms = _elapsed_ms(started)
    context.audit_delegate_call(
        app=app.name,
        subagent_conversation_id=subagent_id,
        latency_ms=latency_ms,
        stop_reason=assistant.stop_reason,
        hop_id=hop.hop_id,
        app_request_id=hop.app_request_id,
    )
    hop.thread_id, hop.ms, hop.app_stop_reason = subagent_id, latency_ms, assistant.stop_reason
    runtime.delegated_thread.set(subagent_id)
    # Which of the app's own tools ran, recorded structurally as well as in the
    # `[app did: …]` note. The note is prose for the model; this is for the UI,
    # which has to know what a turn actually did rather than infer it from a
    # 12B's paraphrase of it.
    _report(hop, assistant.record, [call.tool for call in assistant.tool_calls or []])
    if app.agent is not None and app.agent.context:
        # A v1.2 app says when it asks (#202): its asking is read from here
        # alone, never from its words.
        typed = assistant.needs_input
        runtime.delegated_asks.set({"needs_input": None if typed is None else typed.model_dump()})
    if assistant.answer is not None and assistant.stop_reason == "completed":
        # A read's answer in the app's words (v1.3, #241): kept beside the
        # record for the person; whether it is passed on is ``relay``'s call.
        runtime.delegated_answer.set(relay.take(assistant.answer.model_dump()))
    armed = assistant.armed()
    if assistant.stop_reason == "confirm_required" and armed:
        kept = [
            AppConfirm(
                app=app.name,
                thread_id=subagent_id,
                token=asked.token,
                tool=asked.tool,
                question=asked.question,
                preview=asked.preview.model_dump(exclude_none=True) if asked.preview else None,
                expires_at=asked.expires_at.isoformat() if asked.expires_at else None,
            ).model_dump()
            for asked in armed
        ]
        runtime.delegated_confirm.set(kept[0])
        runtime.delegated_confirms.set(tuple(kept))
        return _confirm_reply(app, " ".join(asked.question for asked in armed))
    return _format_reply(app, assistant)


def _confirm_reply(app: FleetApp, question: str) -> str:
    """What the model hears when an app armed a call instead of running it."""
    return (
        f"{app.name} has done nothing yet: it needs the person's yes first, on the "
        f'card of its question: "{question}"'
    )


def _send_once_more_if_dropped(
    client: DelegateClientLike,
    conversation_id: int,
    message: str,
    *,
    confirm_token: str | None = None,
    context: dict[str, Any] | None = None,
    mode: str | None = None,
) -> MessageExchange:
    """Send, and resend once if the connection dropped after the message went out.

    The resend carries the hop's ``Idempotency-Key``, so the app answers with
    what the first send did (the stored exchange or failure) rather than
    running it again. A 409 there means the first run never finished and
    whatever it did has happened; a resend that can't reach the app leaves it
    unknown. Either way it is the first send's unknown outcome, never a retry.
    """

    def send() -> MessageExchange:
        if context is not None or mode is not None:
            return client.send_message(conversation_id, message, context=context, mode=mode)
        if confirm_token is None:
            return client.send_message(conversation_id, message)
        return client.send_message(conversation_id, message, confirm_token=confirm_token)

    try:
        return send()
    except DelegateConnectionLost as lost:
        hop = run_trace.current_hop()
        if hop is None or not hop.idempotency_key:
            raise  # without a key a resend could run it twice
        try:
            return send()
        except (DelegateRequestRejected, DelegateUnavailable, DelegateOutcomeUnknown) as again:
            if isinstance(again, DelegateRequestRejected) and again.status != 409:
                raise
            raise DelegateOutcomeUnknown(f"{lost}; the resend found: {again}") from again


def _report(hop: run_trace.Hop, record: list[RecordEntry] | None, tools: list[str]) -> None:
    """Hand what the app did to the dispatcher and the trace."""
    hop.app_tools = tools
    runtime.delegated_tools.set(tuple(tools))
    if record is not None:
        dumped = [entry.model_dump(exclude_none=True) for entry in record]
        hop.app_record = dumped
        runtime.delegated_record.set(tuple(dumped))


def _to_tool_error(app: FleetApp, exc: DelegateError) -> ToolError:
    """Map a typed delegate fault to an informative domain error for the model."""
    if isinstance(exc, DelegateThreadGone):
        return ToolError(
            f"{app.name}'s agent kept losing the conversation thread; the request didn't "
            f"go through. Treat it as failed."
        )
    if isinstance(exc, DelegateRateLimited):
        hint = f" Wait {exc.retry_after}s before trying again." if exc.retry_after else ""
        return ToolError(
            f"{app.name} is rate-limiting requests right now.{hint} Do not retry "
            f"automatically — tell the user to try again shortly."
        )
    if isinstance(exc, DelegateConfirmRefused):
        if exc.kind == "used":
            return ToolError(f"{app.name} already used that yes; nothing more was done.")
        if exc.kind == "stale" and exc.said:
            # The app re-checked what it would change (#251): its own words.
            return ToolError(f"{app.name} did nothing: {exc.said}.")
        why = {
            "expired": "it waited too long",
            "stale": "what it would change has changed since",
        }.get(exc.kind, "it no longer has that question open")
        return ToolError(
            f"{app.name} didn't take the yes ({why}), so nothing was done. Ask again "
            f"and it will check with you afresh."
        )
    if isinstance(exc, DelegateRequestRejected):
        return ToolError(
            f"{app.name} rejected the request as invalid. Do not retry the same message — "
            f"rephrase or shorten it, or report the failure."
        )
    if isinstance(exc, DelegateRunFailed):
        done = exc.done
        if done:
            return ToolError(
                f"{app.name}'s agent failed partway through this request, but part of it "
                f"happened. Done: {_describe_done(done)}. Anything else in the request was "
                f"not done. Do not resend the request: that would repeat what is done. Tell "
                f"the user what was done and what wasn't."
            )
        return ToolError(
            f"{app.name}'s agent failed before changing anything; nothing was changed. "
            f"Tell the user it failed; trying again shortly is safe."
        )
    if isinstance(exc, DelegateOutcomeUnknown):
        return ToolError(
            f"{app.name}'s agent {OUTCOME_UNKNOWN} this request, so what it did is "
            f"unknown: some changes may have been made. Do not retry or resend it. Tell the "
            f"user it stopped partway and to check {app.name} before asking again."
        )
    if isinstance(exc, DelegateUnavailable):
        return ToolError(
            f"{app.name}'s agent is unavailable right now (it may be down or its model is "
            f"still loading). Report it as unavailable; try again shortly."
        )
    if isinstance(exc, DelegateProtocolError):
        return ToolError(
            f"{app.name}'s agent returned a response I couldn't read. Treat the request as failed."
        )
    return ToolError(f"{app.name} delegate call failed: {exc}")


def _describe_done(entries: list[RecordEntry]) -> str:
    """The done receipts as one line: each tool and what it touched.

    An ``untrusted`` title (outside text) is never handed to the model, which
    decides what to do next from this; its kind and id name it instead.
    """
    parts: list[str] = []
    for entry in entries:
        names = [
            ref.title
            if ref.title and not (ref.untrusted or entry.untrusted)
            else f"{ref.kind} {ref.id}"
            for ref in entry.touched
        ]
        if entry.count is not None and entry.count > len(names):
            names.append(f"{entry.count - len(names)} more")
        parts.append(f"{entry.tool} ({', '.join(names)})" if names else entry.tool)
    return "; ".join(parts)


def _format_reply(app: FleetApp, assistant: MessageRead) -> str:
    """The assistant reply plus a short note of the app's tool activity.

    Keeps conductor's history clean: the reply text, and at most a one-line
    ``[app did: …]`` summary of which tools ran — never the raw transcript.
    """
    parts: list[str] = []
    if assistant.content and assistant.content.strip():
        parts.append(assistant.content.strip())
    else:
        parts.append(
            f"({app.name} finished without a text reply; stop reason: "
            f"{assistant.stop_reason or 'unknown'}.)"
        )
        # A cancelled run replies with no text (delegate-api): its record
        # still says what it did.
        done = [entry for entry in assistant.record or [] if entry.outcome == "done"]
        if done:
            parts.append(f"Done: {_describe_done(done)}.")
    note = _activity_note(app.name, assistant.tool_calls)
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def _activity_note(app_name: str, tool_calls: list[Any] | None) -> str:
    """A compact ``[app did: tool, tool(failed)]`` note, or empty if no tools ran."""
    if not tool_calls:
        return ""
    names = ", ".join(
        call.tool + ("(failed)" if call.error is not None else "") for call in tool_calls
    )
    return f"[{app_name} did: {names}]"


def _make_list_agents(fleet: Fleet) -> Callable[[], dict[str, Any]]:
    """Build the local ``list_agents`` tool over the discovered fleet."""

    def list_agents() -> dict[str, Any]:
        """List the fleet: which apps you can delegate to, which you can hand the
        user over to (and what each is for), plus any app you can do neither with.
        Local lookup — makes no delegate call."""
        return {
            "agents": [
                {
                    "app": app.name,
                    "tool": ask_tool_name(app.name),
                    "description": app.agent.description if app.agent else "",
                    "examples": list(app.agent.examples) if app.agent else [],
                }
                for app in fleet.agent_apps()
            ],
            "openable_apps": [
                {
                    "app": app.name,
                    "tool": open_tool_name(app.name),
                    "description": app.open.description if app.open else "",
                    "examples": list(app.open.examples) if app.open else [],
                }
                for app in fleet.open_apps()
            ],
            "inert_apps": [app.name for app in fleet.inert_apps()],
        }

    return list_agents


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
