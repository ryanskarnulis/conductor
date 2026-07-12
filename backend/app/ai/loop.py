"""Conductor's agent loop: plan → call delegate tools → observe → respond.

Drives the local model (through the llama.cpp provider) against the shared tool
registry (``app/tools/registry.py``). The registry ships empty and is populated
at startup by ``app.fleet.tools.build_delegate_tools`` — one ``ask_<app>``
delegate tool per discovered fleet agent, plus ``list_agents`` (the HTTP app's
lifespan and the MCP server's ``main()`` both do this); loop tests exercise the
machinery with their own scratch tools against the same registry.

Termination is structural: at most ``max_iterations`` provider turns (kept
shallow — see :data:`app.config.Settings.conductor_max_iterations`), plus a
separate bounded budget of self-correction turns for schema-invalid tool calls
— arguments that aren't a JSON object (``ToolCallArgumentsError`` from the
provider), arguments that fail the tool's argument model, or a tool name the
registry doesn't know. Domain rejections from a tool ("no app handles that",
"conversation gone") are not corrections: they are fed back as ordinary tool
results for the model to react to within the iteration budget.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ValidationError

from app.ai.provider import (
    ChatProvider,
    ToolCall,
    ToolCallArgumentsError,
    tool_result_message,
)
from app.config import get_settings
from app.tools import registry
from app.tools.registry import ToolError, UnknownToolError

logger = structlog.get_logger(__name__)

# Stamped as the acting identity for every tool call the loop dispatches. NULL
# is the user; this is conductor's own in-app loop.
LOOP_ACTOR = "agent:loop"

# The system prompt is composed in layers (agent-standard/STANDARD.md §5):
#   1. app base prompt — conductor's behavioral contract (below);
#   2. global Glitch — the vendored house personality (verbatim canonical text,
#      see personality-global.md); conductor adds no app flavor on top;
#   3. dynamic layers — the fleet section (per-app routing hints rendered from
#      the discovered manifests by app.fleet.tools.render_fleet_section) and
#      today's date, injected per run.
# The vendored Glitch body is never edited here: fix drift by re-copying from
# agent-standard/ (../agent-standard/check-sync.sh).

# Layer 1 — app base prompt (app-owned behavioral contract). Seeded from
# ../agent-standard/conductor-prompt-material.md and kept tool-agnostic: it
# upholds the standard's invariants (tools-only action, no invented results,
# faithful relay, clarify-on-ambiguity) and adds the rule that is conductor's
# alone — it owns destructive-op confirmation, because app agents don't
# reliably confirm before irreversible actions.
_APP_BASE_PROMPT = """\
You are the household's local AI conductor, the master agent for the whole \
fleet, running entirely on our own hardware. You get things done ONLY by \
calling your tools: each tool hands a request to one app's agent and returns \
what that agent said. You have no abilities of your own beyond routing.

Non-negotiable behavior:
- Act only through your tools. Never claim an action you did not take, never \
invent a tool result, and never answer for an app from memory — if you did not \
ask the app's agent this turn, you do not know its answer.
- The app agents are the source of truth for their own domains. Relay what they \
actually said, faithfully — don't embellish, don't paper over gaps, don't \
substitute your own guess for their reply.
- If a request is ambiguous and a wrong guess would change state, ask one short \
clarifying question instead of guessing. If it's ambiguous which app should \
handle it, that's a clarifying question too.
- Destructive requests get confirmed FIRST, before any tool call. If the user \
asks to reset, restart, start over, resign, abandon, delete, undo, or \
overwrite something, do NOT call a tool this turn: reply with one short \
question asking them to confirm, and delegate only after they've said yes. \
App agents don't reliably confirm irreversible actions themselves, so you are \
the one safety stop. Example — user: "reset the chess game" → you, with no \
tool call: "That wipes the current game — sure?"
- If no app in the fleet can do something, say so plainly: no invented \
capabilities, no fake confidence.
- Prefer reversible actions and each app's safety rails; never work around them, \
even when asked.
- Everything stays local. Never suggest sending household data to outside services.

Working style:
- Route, don't micromanage: pass the user's intent to the app agent in one \
clear, self-contained message; don't split one ask into many calls.
- Delegate at most one level: you call app agents; they never call each other \
or you.
- Answer first, detail after. Attribute results to the app they came from when \
it matters."""

# Layer 2 — global Glitch, the vendored house personality (STANDARD.md §5).
_PERSONALITY_PATH = Path(__file__).with_name("personality-global.md")


def _load_global_personality() -> str:
    """The vendored Glitch text, minus its one leading ``<!-- vendored -->`` line.

    Read once at import from the copy shipped alongside this module. The body is
    canonical and must never be edited in place — re-vendor to change Glitch
    (fix drift by re-copying from ../agent-standard/, never by editing here).
    """
    lines = _PERSONALITY_PATH.read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("<!-- vendored")]
    return "\n".join(body).strip()


_GLOBAL_PERSONALITY = _load_global_personality()


def build_system_prompt(today: date, *, fleet_section: str | None = None) -> str:
    """Compose the layered system prompt: app base → global Glitch → fleet → date.

    The fleet section (each agent app's name, capability, and routing examples,
    from :func:`app.fleet.tools.render_fleet_section`) is a dynamic layer built
    per process from the discovered manifests; it sits after Glitch and before
    the date so routing hints live in the prompt, not in tool docstrings. It is
    omitted when empty (no agents discovered). Conductor ships no app-flavor
    layer.
    """
    layers = [_APP_BASE_PROMPT, _GLOBAL_PERSONALITY]
    if fleet_section:
        layers.append(fleet_section)
    layers.append(f"Today's date is {today.isoformat()}.")
    return "\n\n".join(layers)


_StopReason = Literal["completed", "max_iterations", "correction_limit"]


@dataclass(frozen=True)
class LoopActivity:
    """One progress beat of a run: what the loop is about to do.

    ``kind`` "model" — wait on a provider turn; "tool" — dispatch the named
    tool. Emitted to ``run()``'s ``on_activity`` callback so a caller can
    surface live "asking chess…" progress while the run blocks (conductor
    turns are slow by construction — a delegate call wraps a subagent's full
    loop). A callback failure is logged and never breaks the run.
    """

    kind: Literal["model", "tool"]
    iteration: int
    tool: str | None = None


OnActivity = Callable[[LoopActivity], None]


class ToolCallRecord(BaseModel):
    """One dispatched tool call: what ran and what came back.

    Exactly one of ``result``/``error`` is set. This is the loop's own record
    (for the caller, and for later persistence) — the model sees the same text
    via its ``role: tool`` message.
    """

    tool: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None


class AgentRunResult(BaseModel):
    """Outcome of one loop run. ``messages`` is the full transcript."""

    reply: str | None
    stop_reason: _StopReason
    tool_calls: list[ToolCallRecord]
    iterations: int
    messages: list[dict[str, Any]]


class AgentLoop:
    """Bounded tool-calling loop over one provider and the shared registry."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        max_iterations: int = 6,
        max_corrections: int = 3,
        fleet_section: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._provider = provider
        self._max_iterations = max_iterations
        self._max_corrections = max_corrections
        # The prompt's fleet layer (built once from the discovered manifests).
        # None keeps the prompt to base → Glitch → date (Slice 2 behavior).
        self._fleet_section = fleet_section

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[dict[str, Any]] | None = None,
        actor: str = LOOP_ACTOR,
        on_activity: OnActivity | None = None,
    ) -> AgentRunResult:
        """Run the loop for one user message. Always returns; never spins.

        Binds a request ID for the whole run unless the caller already bound one
        — every tool call and provider log line of the run then carries the same
        ID. ``actor`` is the identity threaded to every tool call (the default
        in-app loop identity, :data:`LOOP_ACTOR`). ``on_activity`` receives one
        :class:`LoopActivity` per provider turn and per tool dispatch — the
        progress seam the HTTP layer surfaces while a run blocks.
        """
        bindings: dict[str, str] = {}
        if "request_id" not in structlog.contextvars.get_contextvars():
            bindings["request_id"] = uuid.uuid4().hex[:8]
        with structlog.contextvars.bound_contextvars(**bindings):
            return self._run(user_message, history, actor, on_activity)

    def _run(
        self,
        user_message: str,
        history: Sequence[dict[str, Any]] | None,
        actor: str,
        on_activity: OnActivity | None,
    ) -> AgentRunResult:
        specs = registry.tool_specs()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(date.today(), fleet_section=self._fleet_section),
            },
            *(history or []),
            {"role": "user", "content": user_message},
        ]
        records: list[ToolCallRecord] = []
        corrections = 0
        iterations = 0
        logger.info("agent_run_started", tools=len(specs), history_messages=len(history or []))
        for iteration in range(1, self._max_iterations + 1):
            iterations = iteration
            self._emit(on_activity, LoopActivity(kind="model", iteration=iteration))
            try:
                result = self._provider.chat(messages, tools=specs)
            except ToolCallArgumentsError as exc:
                # The turn is unusable — its tool calls can't round-trip into
                # history — so the correction goes back as a user-role message
                # instead of a tool result.
                corrections += 1
                logger.warning(
                    "agent_unparseable_tool_call",
                    tool=exc.tool_name,
                    error=str(exc),
                    corrections=corrections,
                )
                if corrections > self._max_corrections:
                    return self._finish("correction_limit", None, records, iteration, messages)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your tool call failed before execution: {exc}. "
                            "Call the tool again with corrected JSON arguments."
                        ),
                    }
                )
                continue
            if not result.tool_calls:
                # A text turn terminates the loop, even when content is empty.
                messages.append(result.to_message())
                return self._finish("completed", result.content, records, iteration, messages)
            messages.append(result.to_message())
            schema_error_this_turn = False
            for call in result.tool_calls:
                self._emit(
                    on_activity, LoopActivity(kind="tool", iteration=iteration, tool=call.name)
                )
                record, feedback, schema_error = self._dispatch(call, actor)
                records.append(record)
                messages.append(tool_result_message(call.id, feedback))
                schema_error_this_turn = schema_error_this_turn or schema_error
            if schema_error_this_turn:
                corrections += 1
                if corrections > self._max_corrections:
                    return self._finish("correction_limit", None, records, iteration, messages)
        return self._finish("max_iterations", None, records, iterations, messages)

    def _dispatch(self, call: ToolCall, actor: str) -> tuple[ToolCallRecord, str, bool]:
        """Run one tool call as ``actor``.

        Returns the record, the feedback text for the model's ``role: tool``
        message, and whether the failure was schema-level (counts against the
        correction budget). Unexpected exceptions (bugs) propagate — the loop
        only self-corrects what the model can fix.
        """
        record = ToolCallRecord(tool=call.name, arguments=call.arguments)
        schema_error = False
        try:
            outcome = registry.call_tool(call.name, call.arguments, actor=actor)
        except UnknownToolError as exc:
            record.error = str(exc)
            schema_error = True
        except ValidationError as exc:
            record.error = f"Invalid arguments: {_validation_summary(exc)}"
            schema_error = True
        except ToolError as exc:
            # Domain rejection with a reason the model can act on.
            record.error = str(exc)
        else:
            record.result = _result_text(outcome)
            logger.info("agent_tool_call", tool=call.name, ok=True)
            return record, record.result, False
        logger.warning(
            "agent_tool_call",
            tool=call.name,
            ok=False,
            error=record.error,
            schema_error=schema_error,
        )
        return record, f"Error: {record.error}", schema_error

    @staticmethod
    def _emit(on_activity: OnActivity | None, activity: LoopActivity) -> None:
        """Report one progress beat; a broken callback must never sink the run."""
        if on_activity is None:
            return
        try:
            on_activity(activity)
        except Exception:
            logger.warning("agent_activity_callback_failed", kind=activity.kind, tool=activity.tool)

    @staticmethod
    def _finish(
        stop_reason: _StopReason,
        reply: str | None,
        records: list[ToolCallRecord],
        iterations: int,
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        logger.info(
            "agent_run_finished",
            stop_reason=stop_reason,
            iterations=iterations,
            tool_calls=len(records),
        )
        return AgentRunResult(
            reply=reply,
            stop_reason=stop_reason,
            tool_calls=records,
            iterations=iterations,
            messages=messages,
        )


def loop_from_settings(provider: ChatProvider, *, fleet_section: str | None = None) -> AgentLoop:
    """An :class:`AgentLoop` bounded by the configured ``conductor_max_iterations``.

    The provider factory's counterpart: the one place the loop's shallow
    iteration cap is wired from settings, so the ``CONDUCTOR_MAX_ITERATIONS``
    field is never an orphan. ``fleet_section`` is threaded through to the
    prompt's fleet layer (the HTTP app passes the one its lifespan rendered).
    """
    return AgentLoop(
        provider,
        max_iterations=get_settings().conductor_max_iterations,
        fleet_section=fleet_section,
    )


def _validation_summary(exc: ValidationError) -> str:
    """Pydantic's error list as one compact line the model can read."""
    return "; ".join(
        f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _result_text(value: Any) -> str:
    """A tool result as the text body of a ``role: tool`` message."""
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
