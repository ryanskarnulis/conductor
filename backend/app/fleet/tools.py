"""Fleet tools: ``ask_<app>`` per agent, ``open_<app>`` per openable app, plus
``list_agents``.

:func:`build_delegate_tools` turns each fleet app into tools on the shared
registry (``app/tools/registry.py``), from its manifest blocks:

- an **``agent:``** block yields ``ask_<app>`` (``ask_tasks``, …), whose
  docstring is the manifest's ``agent.description``. Calling it resolves (or
  creates) the app's subagent thread for the current master conversation, sends
  the message with ``X-Agent-Actor: agent:conductor``, and relays the
  assistant's reply plus a compact note of what the app actually did — never the
  raw transcript.
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

- **Thread reuse + 404 recreate-once.** The subagent thread is looked up in the
  :class:`DelegationContext`; a :class:`DelegateThreadGone` triggers exactly one
  recreate-and-retry (a second 404 fails the call).
- **Typed delegate faults → domain errors.** Rate-limit, unavailable, protocol,
  and double-404 all become informative :class:`ToolError` s (the model reads
  them and adapts); a 429 surfaces its ``Retry-After`` and is never auto-retried.
- **Budget.** The per-turn per-app call is charged before the request, so an
  over-budget call never leaves conductor.

``list_agents`` is a local tool (no network) reporting the fleet from the
manifests. Routing *hints* live in the system prompt, not tool docstrings —
:func:`render_fleet_section` builds that layer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

import structlog

from app.fleet.context import current_delegation_context
from app.fleet.delegate import (
    CONDUCTOR_ACTOR,
    DelegateClient,
    DelegateError,
    DelegateProtocolError,
    DelegateRateLimited,
    DelegateRequestRejected,
    DelegateThreadGone,
    DelegateUnavailable,
    MessageExchange,
    MessageRead,
)
from app.fleet.manifests import Fleet, FleetApp
from app.tools import registry
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
    """The delegate-client surface the tools use (context manager + two calls).

    :class:`app.fleet.delegate.DelegateClient` satisfies it structurally; tests
    supply a fake at the same shape, so no live call happens in pytest.
    """

    def __enter__(self) -> DelegateClientLike: ...

    def __exit__(self, *exc: object) -> None: ...

    def create_conversation(self, *, title: str | None = None) -> int: ...

    def send_message(self, conversation_id: int, message: str) -> MessageExchange: ...


# A factory yields a fresh client bound to one app.
ClientFactory = Callable[[FleetApp], DelegateClientLike]


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


def build_delegate_tools(fleet: Fleet, client_factory: ClientFactory) -> list[str]:
    """Register the fleet's tools: ``ask_<app>``, ``open_<app>``, ``list_agents``.

    One ``ask_<app>`` per agent-bearing app and one ``open_<app>`` per openable
    app (an app can be both, or neither — the blocks are independent). Returns
    the registered tool names, in order. Mutates the shared registry — the one
    surface the loop and the MCP server both consume. Inert fleet members get
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
    list_fn = _make_list_agents(fleet)
    registry.tool(list_fn)
    names.append(list_fn.__name__)
    logger.info("delegate_tools_built", tools=names)
    return names


def _make_ask_tool(app: FleetApp, client_factory: ClientFactory) -> Callable[[str], str]:
    """Build the ``ask_<app>`` callable for one agent-bearing app."""

    def ask_tool(message: str) -> str:
        return _delegate(app, client_factory, message)

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


def _delegate(app: FleetApp, client_factory: ClientFactory, message: str) -> str:
    """Send ``message`` to ``app``'s agent, applying every contract behavior."""
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
    context = current_delegation_context()
    # Charge the budget before any network work; over-budget never leaves here.
    context.charge_call(app.name)

    started = time.monotonic()
    subagent_id: int | None = None
    try:
        with client_factory(app) as client:
            subagent_id = context.thread_for(app.name)
            if subagent_id is None:
                subagent_id = client.create_conversation()
                context.remember_thread(app.name, subagent_id)
            try:
                exchange = client.send_message(subagent_id, message)
            except DelegateThreadGone:
                # Contract rule: a pruned thread → recreate once and retry
                # exactly once. A second 404 propagates and fails the call.
                context.forget_thread(app.name)
                subagent_id = client.create_conversation()
                context.remember_thread(app.name, subagent_id)
                exchange = client.send_message(subagent_id, message)
    except DelegateError as exc:
        latency_ms = _elapsed_ms(started)
        context.audit_delegate_call(
            app=app.name,
            subagent_conversation_id=subagent_id,
            latency_ms=latency_ms,
            error=type(exc).__name__,
        )
        raise _to_tool_error(app, exc) from exc

    assistant = exchange.assistant_message
    context.audit_delegate_call(
        app=app.name,
        subagent_conversation_id=subagent_id,
        latency_ms=_elapsed_ms(started),
        stop_reason=assistant.stop_reason,
    )
    return _format_reply(app, assistant)


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
    if isinstance(exc, DelegateRequestRejected):
        return ToolError(
            f"{app.name} rejected the request as invalid. Do not retry the same message — "
            f"rephrase or shorten it, or report the failure."
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


def render_fleet_section(fleet: Fleet) -> str:
    """The system-prompt fleet layer: each app's tool, what it does, and its
    routing examples — the ask tools and the open tools alike. Empty when the
    fleet has no tool-bearing app.

    This is where routing hints belong (STANDARD.md) — not in tool docstrings.
    """
    agent_apps = fleet.agent_apps()
    open_apps = fleet.open_apps()
    if not agent_apps and not open_apps:
        return ""
    lines = [
        "Your fleet — the apps you can act through. Route each request to the single "
        "best-fit app by calling its tool; if two could fit, pick the primary one, and if "
        "none fit, say so plainly.",
    ]
    if agent_apps:
        lines.append("Apps you delegate to (their agent does the work and reports back):")
        for app in agent_apps:
            assert app.agent is not None
            lines.append(f"- {ask_tool_name(app.name)} — {app.agent.description}")
            if app.agent.examples:
                examples = "; ".join(f'"{example}"' for example in app.agent.examples)
                lines.append(f"  routes here: {examples}")
    if open_apps:
        lines.append(
            "Apps you hand the user OVER to — you don't do these things yourself, you send "
            "the user to the app and its agent takes over from there. Pass their request "
            "along as the `intent` so nothing is lost:"
        )
        for app in open_apps:
            assert app.open is not None
            lines.append(f"- {open_tool_name(app.name)} — {app.open.description}")
            if app.open.examples:
                examples = "; ".join(f'"{example}"' for example in app.open.examples)
                lines.append(f"  routes here: {examples}")
    inert = fleet.inert_apps()
    if inert:
        names = ", ".join(app.name for app in inert)
        lines.append(
            f"Also in the house but with no agent to delegate to (you can't act on these): {names}."
        )
    return "\n".join(lines)
