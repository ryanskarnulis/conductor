"""The turn runtime (#75): what every compiled turn runs on.

Conductor turns go planner → validator → interpreter → narrator only
(:mod:`app.ai.interpreter`). This module holds what the interpreter and its
control steps share, carried over from the retired tool-calling loop:

- the records: :class:`ToolCallRecord` (one step's call and what came back)
  and :class:`AgentRunResult` (the turn);
- **failures are stops, not exceptions** (STANDARD.md §3): a provider outage
  stops ``provider_error`` and a crash ``internal_error``; either raises
  :class:`AgentRunFailed` carrying the turn so far, and the route stores it
  before answering 502/500, so a write an app made first never vanishes from
  the thread;
- :class:`Runtime`: the metered model call (``max_tokens`` on every call, its
  status and tokens on the trace), one step's dispatch through the shared
  registry, the progress beat, and the reply: the narrator's for a completed
  turn (#65), the code-kept record for a stop short of ``completed``.

**Stop** (#66, :mod:`app.ai.stop`) ends a turn ``cancelled`` at the
interpreter's next boundary; a stopped turn is never narrated.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ValidationError

from app.ai import confirmation, incognito, narrator, run_record
from app.ai import trace as run_trace
from app.ai.plan import TurnPlan
from app.ai.provider import ChatProvider, ChatResult, ToolCall, ToolCallArgumentsError
from app.fleet.context import NoDelegationContextError, current_delegation_context
from app.tools import registry, runtime
from app.tools.registry import ToolError, UnknownToolError

logger = structlog.get_logger(__name__)

# Stamped as the acting identity for every call a turn dispatches, and sent to
# the apps as the delegate actor they recognize. The wire name predates #75.
ACTOR = "agent:loop"

# STANDARD.md §3's stop vocabulary, the same strings in every app. Conductor's
# turns end ``completed``, ``cancelled``, ``provider_error`` or
# ``internal_error``; the rest stay for the fleet's parity and for old rows.
StopReason = Literal[
    "completed",
    "max_iterations",
    "correction_limit",
    "truncated",
    "no_progress",
    "tool_call_limit",
    "cancelled",
    "provider_error",
    "timed_out",
    "internal_error",
]

# The HTTP status the route answers each failure stop with. Every other stop
# answers 200.
FAILURE_STATUS: dict[str, int] = {
    "provider_error": 502,
    "timed_out": 504,
    "internal_error": 500,
}

# Default per-call completion cap: model-profile's turn cap. llama-server's own
# default is unbounded.
DEFAULT_MAX_TOKENS = 2048

# ``finish_reason`` of a call the ``max_tokens`` cap cut off.
_FINISH_LENGTH = "length"

INTERNAL_ERROR = "internal error: the tool call failed unexpectedly"


@dataclass(frozen=True)
class Activity:
    """One progress beat of a turn: what it is about to do.

    ``kind`` "model": wait on a model call; "tool": run the named step.
    ``ack`` is a compiled plan's honest ack (#70): what's about to happen,
    said at once. A callback failure is logged and never breaks the turn.
    """

    kind: Literal["model", "tool"]
    tool: str | None = None
    ack: str | None = None


OnActivity = Callable[[Activity], None]


class ToolCallRecord(BaseModel):
    """One step's call: what ran and what came back.

    Exactly one of ``result``/``error`` is set, except for a step code never
    sent (``error`` starts "not run: ").
    """

    tool: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    # For a delegate call, the tools the *app* ran while answering. Persisted
    # and shown, never sent to a model.
    app_tools: list[str] | None = None
    # For a delegate call to a v1.1 app, its typed record (receipts) of what it
    # did, also when the call failed partway (#67). The narrator speaks from
    # it (#65); never sent to the planner.
    app_record: list[dict[str, Any]] | None = None
    # For a delegate call the app answered ``confirm_required`` (#68): the
    # token, thread and question kept for the person's yes
    # (``app.ai.confirmation.AppConfirm``). Persisted, never sent to a model.
    app_confirm: dict[str, Any] | None = None
    # For a call that carries an undo (#74, ``app.ai.control``): the ledger
    # receipts it undoes, marked undone once the app says it did.
    undoes: list[int] | None = None


class AgentRunResult(BaseModel):
    """Outcome of one turn. ``messages`` are the planner's exchange."""

    reply: str | None
    stop_reason: StopReason
    tool_calls: list[ToolCallRecord]
    messages: list[dict[str, Any]]
    # The turn id: what the turn's log lines, trace record and captured model
    # calls are keyed by (#62).
    run_id: str = ""
    # What the compiler made of the turn (#70), journaled by the ledger.
    plan: TurnPlan | None = None


class AgentRunFailed(Exception):
    """A turn that ended in a failure stop, carrying everything that ran first.

    The route stores ``result`` (a truthful partial turn, its reply built from
    the record) before answering ``http_status``.
    """

    def __init__(self, result: AgentRunResult, *, message: str) -> None:
        super().__init__(message)
        self.result = result
        self.http_status = FAILURE_STATUS[result.stop_reason]


class Runtime:
    """The machinery a turn runs on: one provider, the registry, a tracer."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tracer: run_trace.Tracer | None = None,
        narrate: bool = False,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.provider = provider
        self.max_tokens = max_tokens
        # Where finished turns are recorded (#62); ``None`` keeps no record.
        self.tracer = tracer
        # Whether a completed turn's reply is the narrator's (#65); off, it is
        # the code-built record reply.
        self.narrate = narrate

    # --- model calls --------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        trace: run_trace.RunTrace,
        *,
        role: run_trace.CallRole,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One metered model call: its status, time and tokens go on the trace."""
        call = run_trace.ModelCall(role=role)
        started = time.monotonic()
        try:
            if response_format is not None:
                result = self.provider.chat(
                    messages,
                    max_tokens=max_tokens or self.max_tokens,
                    response_format=response_format,
                )
            else:
                result = self.provider.chat(messages, max_tokens=max_tokens or self.max_tokens)
        except ToolCallArgumentsError as exc:
            call.status, call.failure = "bad_args", type(exc).__name__
            raise
        except Exception as exc:
            call.status, call.failure = "failed", type(exc).__name__
            raise
        else:
            call.status = "truncated" if result.finish_reason == _FINISH_LENGTH else "ok"
            call.finish_reason = result.finish_reason
            call.tool_calls = len(result.tool_calls)
            call.meter(result.usage)
            return result
        finally:
            call.ms = round((time.monotonic() - started) * 1000)
            trace.calls.append(call)

    # --- one step -----------------------------------------------------------------------

    def dispatch(self, call: ToolCall, actor: str) -> ToolCallRecord:
        """Run one step's call as ``actor`` through the registry.

        A refusal (an app said no, a call failed partway, arguments code got
        wrong) is the record's ``error``; what the app said it did rides along
        either way. Unexpected exceptions (bugs) propagate.
        """
        record = ToolCallRecord(tool=call.name, arguments=call.arguments)
        # A yes to an app's question about an undo carries what it undoes (#74).
        record.undoes = _answered_undo(call)
        # Cleared per dispatch and reset after, so what one delegate call
        # reported can never be read as the next call's.
        delegated = runtime.delegated_tools.set(None)
        delegated_record = runtime.delegated_record.set(None)
        delegated_confirm = runtime.delegated_confirm.set(None)
        try:
            outcome = registry.call_tool(call.name, call.arguments, actor=actor)
        except UnknownToolError as exc:
            record.error = str(exc)
        except ValidationError as exc:
            record.error = f"{run_record.INVALID_ARGUMENTS_PREFIX}{_validation_summary(exc)}"
        except ToolError as exc:
            record.error = str(exc)
            _take_app_report(record)
        else:
            record.result = _result_text(outcome)
            _take_app_report(record)
        finally:
            runtime.delegated_tools.reset(delegated)
            runtime.delegated_record.reset(delegated_record)
            runtime.delegated_confirm.reset(delegated_confirm)
        logger.info("agent_tool_call", tool=call.name, ok=record.error is None, error=record.error)
        return record

    # --- progress and endings -----------------------------------------------------------

    @staticmethod
    def emit(on_activity: OnActivity | None, activity: Activity) -> None:
        """Report one progress beat; a broken callback must never sink the turn."""
        if on_activity is None:
            return
        try:
            on_activity(activity)
        except Exception:
            logger.warning("agent_activity_callback_failed", kind=activity.kind, tool=activity.tool)

    @staticmethod
    def finish(
        stop_reason: StopReason,
        reply: str | None,
        records: list[ToolCallRecord],
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        logger.info("agent_run_finished", stop_reason=stop_reason, tool_calls=len(records))
        return AgentRunResult(
            reply=reply, stop_reason=stop_reason, tool_calls=records, messages=messages
        )

    def stop(
        self,
        stop_reason: StopReason,
        records: list[ToolCallRecord],
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        """End short of ``completed``, replying from the code-kept record."""
        return self.finish(
            stop_reason, run_record.stop_reply(stop_reason, records), records, messages
        )

    def failed(
        self,
        stop_reason: StopReason,
        message: str,
        records: list[ToolCallRecord],
        messages: list[dict[str, Any]],
    ) -> AgentRunFailed:
        """A failure stop as an exception carrying the partial turn."""
        return AgentRunFailed(self.stop(stop_reason, records, messages), message=message)

    def reply(
        self,
        words: str,
        records: Sequence[ToolCallRecord],
        trace: run_trace.RunTrace,
        on_activity: OnActivity | None,
    ) -> str | None:
        """A completed turn's reply: the narrator's (code-checked), or the
        code-built record reply when the narrator is off. Never raises: a
        narrator that fails falls back to the record inside
        :func:`narrator.narrate`."""
        if not self.narrate:
            trace.narration = {"path": "code:record", "appended": [], "model_calls": 0}
            return narrator.fallback_reply(records, trace.surface)
        started = time.monotonic()

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            self.emit(on_activity, Activity(kind="model"))
            return self.chat(messages, trace, max_tokens=max_tokens, role="narrator")

        try:
            told = narrator.narrate(words, None, records, surface=trace.surface, chat=chat)
        except Exception:
            # A bug in the narrator's code, not the model: the turn still stands.
            logger.exception("agent_narrator_crashed")
            told = narrator.Narration(narrator.fallback_reply(records, trace.surface), "fallback")
        trace.narration = {
            "path": told.path,
            "appended": told.appended,
            "model_calls": told.model_calls,
            "ms": round((time.monotonic() - started) * 1000),
        }
        logger.info("agent_narrated", path=told.path, appended=len(told.appended))
        return told.reply

    def write_trace(
        self, trace: run_trace.RunTrace, result: AgentRunResult, error: str, started: float
    ) -> None:
        """Record the finished turn; a write that fails never costs it. An
        incognito turn (#82) is never recorded: the trace holds its words."""
        if self.tracer is None or incognito.active():
            return
        try:
            total_ms = round((time.monotonic() - started) * 1000)
            self.tracer.write(trace.record(result, error=error, total_ms=total_ms))
        except Exception as exc:
            logger.warning("agent_trace_failed", error=str(exc), error_type=type(exc).__name__)


def _answered_undo(call: ToolCall) -> list[int] | None:
    """The ledger receipts a ``confirm`` call undoes (#74): those the app's
    waiting question was about, when it came from an undo. Read before the
    call takes the question."""
    if call.name != confirmation.CONFIRM_TOOL:
        return None
    try:
        pending = current_delegation_context().pending_confirm
    except NoDelegationContextError:
        return None
    app = str(call.arguments.get("app") or "").strip().lower()
    if pending is None or not pending.undoes or pending.app != app:
        return None
    return list(pending.undoes)


def _take_app_report(record: ToolCallRecord) -> None:
    """What the app said it did while answering this call, if anything."""
    record.app_tools = list(runtime.delegated_tools.get() or []) or None
    app_record = runtime.delegated_record.get()
    record.app_record = list(app_record) if app_record is not None else None
    record.app_confirm = runtime.delegated_confirm.get()


def _validation_summary(exc: ValidationError) -> str:
    """Pydantic's error list as one compact line."""
    return "; ".join(
        f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _result_text(value: Any) -> str:
    """A tool result as text."""
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
