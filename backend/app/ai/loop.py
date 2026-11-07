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

The safety nets are STANDARD.md §3's, after PCC's loop (#52):

- **Every model call is capped.** ``max_tokens`` goes out on every request. A
  text turn cut off at the cap (``finish_reason == "length"``) is dropped and
  the run stops ``truncated``; the tool calls of a cut-off turn arrived whole
  and still run.
- **Stall rule.** A turn whose every call repeats an exchange this run already
  saw (tool + canonicalised arguments + answer) stops ``no_progress``.
- **Per-run tool-call cap** (``max_tool_calls``). A call past it is answered
  "not run" and never dispatched; the run stops ``tool_call_limit``.
- **Failures are stops, not exceptions.** A provider failure stops
  ``provider_error`` and a tool or provider that raised unexpectedly stops
  ``internal_error``. Either raises :class:`AgentRunFailed` carrying the run so
  far, and the route stores it before answering 502/500, so a write an app made
  before the failure is never lost from the transcript.
- **A stop short of ``completed`` replies from what ran** (:mod:`app.ai.run_record`),
  never from the model's words. A stop where nothing ran has no reply.

A completed run's reply is the narrator's (#65, :mod:`app.ai.narrator`) when the
loop is built with ``narrate=True`` (:func:`loop_from_settings` does): a second,
tool-less call on the same provider speaks from the receipts of what ran, and
code checks it. The planner's own text turn ends the loop and stays in the
transcript; the narrator's words are the reply.

**Stop** (#66, :mod:`app.ai.stop`): ``run(stop=...)`` is checked before every
provider call (the narrator's too) and every tool dispatch; a set Stop ends
the run ``cancelled``, the rest of a batch answered "not run". A call already
in flight finishes first, and a delegate call in flight gets the app's own
cancel. A stopped run is never narrated: like every stop short of
``completed`` it replies from the code-kept record of what ran.

``timed_out`` is in the vocabulary for parity with the fleet; conductor has no
run deadline yet.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ValidationError

from app.ai import narrator, run_record
from app.ai.plan import PausedPlan, TurnPlan
from app.ai.stop import STOPPED_ERROR, Stop, use_stop
from app.ai import trace as run_trace
from app.ai.provider import (
    ChatProvider,
    ChatResult,
    ProviderError,
    ToolCall,
    ToolCallArgumentsError,
    ToolSpec,
    tool_result_message,
)
from app.config import get_settings
from app.fleet.manifests import Fleet
from app.tools import registry, runtime
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
# faithful relay, clarify-on-ambiguity). Destructive-op confirmation is per
# app (#68): an app with `agent.confirms` asks itself, and conductor's own
# question is the backstop for the rest; the fleet section renders both.
_APP_BASE_PROMPT = """\
You are the household's local AI conductor, the master agent for the whole \
fleet, running entirely on our own hardware. You get things done ONLY by \
calling your tools. Your tools come in two kinds: an `ask_*` tool hands a \
request to one app's agent and returns what that agent said, and an `open_*` \
tool hands the USER over to an app — it opens that app in their browser and \
its agent takes it from there. You have no abilities of your own beyond \
routing.

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
- An `open_*` app is somewhere the user GOES. Anything to do with that app is a \
handoff — including its destructive and its read-only asks alike. Don't confirm \
first, don't ask what they want to do there, don't try to answer for it: call \
the tool, pass what they said as the `intent`, and let the app's own agent take \
it from there. Then STOP — one `open_*` call is the whole turn. Don't call \
another tool after it; just say one short line telling them you're sending them \
over. Example — user: "let's play chess" → open_chess(intent="let's play \
chess") → you: "board's up — go."
- Destructive requests to an `ask_*` app follow the fleet section's \
"Destructive requests" rules: some apps ask the person themselves, and for the \
rest you are the one safety stop. None of it applies to `open_*` apps — you \
aren't doing anything to those, you're just opening the door, and their agent \
does its own confirming.
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


# STANDARD.md §3's stop vocabulary, the same strings in every app.
_StopReason = Literal[
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
# is the agent's doing, not a fault, and answers 200.
FAILURE_STATUS: dict[str, int] = {
    "provider_error": 502,
    "timed_out": 504,
    "internal_error": 500,
}

# Default per-call completion cap: model-profile's turn cap (PCC's
# ``COMPLETION_RESERVE_TOKENS``). llama-server's own default is unbounded.
DEFAULT_MAX_TOKENS = 2048

# Default per-run tool-call cap. Shallower than PCC's 24 because each conductor
# call can wrap an app's full loop: 6 iterations x 2 calls covers every real turn.
DEFAULT_MAX_TOOL_CALLS = 12

# ``finish_reason`` of a call the ``max_tokens`` cap cut off.
_FINISH_LENGTH = "length"

_TOOL_CALL_LIMIT_ERROR = (
    f"{run_record.NOT_RUN_PREFIX}the run reached its limit of tool calls before this call"
)
_INTERNAL_ERROR = "internal error: the tool call failed unexpectedly"
_CRASHED_BATCH_ERROR = (
    f"{run_record.NOT_RUN_PREFIX}an earlier tool call in this turn failed unexpectedly"
)


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
    # A compiled plan's honest ack (#70): what's about to happen, said at once.
    ack: str | None = None


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
    # For a delegate call, the tools the *app* ran while answering. Persisted
    # and shown, never sent to the model — its `role: tool` message carries
    # `result` alone, so this cannot spend context or be narrated back. The UI
    # reads it to know what a turn actually did, which a paraphrased reply
    # cannot be trusted to say.
    app_tools: list[str] | None = None
    # For a delegate call to a v1.1 app, its typed record (receipts) of what it
    # did, also when the call failed partway (#67). Same rule as `app_tools`:
    # persisted and shown, never sent to the model. The narrator speaks from
    # it (#65).
    app_record: list[dict[str, Any]] | None = None
    # For a delegate call the app answered ``confirm_required`` (#68): the
    # token, thread and question kept for the person's yes
    # (``app.ai.confirmation.AppConfirm``). Persisted, never sent to the model.
    app_confirm: dict[str, Any] | None = None


class AgentRunResult(BaseModel):
    """Outcome of one loop run. ``messages`` is the full transcript."""

    reply: str | None
    stop_reason: _StopReason
    tool_calls: list[ToolCallRecord]
    iterations: int
    messages: list[dict[str, Any]]
    # The turn id: what the turn's log lines, trace record and captured model
    # calls are keyed by (#62).
    run_id: str = ""
    # What the compiler made of the turn (#70), when the loop has one: the plan
    # that acted, or the one that fell back to the loop. Journaled by the ledger.
    plan: TurnPlan | None = None


class AgentRunFailed(Exception):
    """A run that ended in a failure stop, carrying everything that ran first.

    The route stores ``result`` (a truthful partial turn, its reply built from
    the record) before answering ``http_status``, so the next turn's history
    knows what the apps did.
    """

    def __init__(self, result: AgentRunResult, *, message: str) -> None:
        super().__init__(message)
        self.result = result
        self.http_status = FAILURE_STATUS[result.stop_reason]


class AgentLoop:
    """Bounded tool-calling loop over one provider and the shared registry."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        max_iterations: int = 6,
        max_corrections: int = 3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        fleet_section: str | None = None,
        tracer: run_trace.Tracer | None = None,
        narrate: bool = False,
        fleet: Fleet | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        self._provider = provider
        self._max_iterations = max_iterations
        self._max_corrections = max_corrections
        self._max_tokens = max_tokens
        self._max_tool_calls = max_tool_calls
        # The prompt's fleet layer (built once from the discovered manifests).
        # None keeps the prompt to base → Glitch → date (Slice 2 behavior).
        self._fleet_section = fleet_section
        # Where finished turns are recorded (#62); ``None`` keeps no record.
        self._tracer = tracer
        # Whether a completed run's reply is the narrator's (#65).
        self._narrate = narrate
        # Compiled plans (#70): with a fleet, each turn is compiled first and
        # the interpreter runs what it can; the rest falls back to the loop.
        from app.ai.interpreter import Interpreter  # noqa: PLC0415

        self._interpreter = Interpreter(self, fleet) if fleet is not None else None

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[dict[str, Any]] | None = None,
        actor: str = LOOP_ACTOR,
        on_activity: OnActivity | None = None,
        conversation_id: str | None = None,
        context: str | None = None,
        surface: str | None = None,
        stop: Stop | None = None,
        paused: PausedPlan | None = None,
        self_facts: Callable[[], str] | None = None,
    ) -> AgentRunResult:
        """Run the loop for one user message. Always returns; never spins.

        Binds a request ID for the whole run unless the caller already bound one
        — every tool call and provider log line of the run then carries the same
        ID. ``actor`` is the identity threaded to every tool call (the default
        in-app loop identity, :data:`LOOP_ACTOR`). ``on_activity`` receives one
        :class:`LoopActivity` per provider turn and per tool dispatch — the
        progress seam the HTTP layer surfaces while a run blocks.

        ``context`` (:mod:`app.ai.turn_context`: the state block and recall)
        opens the turn's user message, before the person's words; the route
        passes it instead of ``history``, which stays for direct callers.
        ``surface`` (``Now.surface``) shapes the narrator's reply: on voice and
        room_speaker, two sentences at most and no markdown. ``stop`` is the
        turn's Stop (#66): checked at every boundary, and bound for the tools
        so a delegate call in flight can pass it on to its app.

        A failure stop (``provider_error``, ``internal_error``) raises
        :class:`AgentRunFailed` with the partial run; every other stop returns.

        Every turn gets a ``run_id`` (reused if the caller bound one) on its
        log lines, its result, its trace record and its captured model calls.
        With a tracer configured the turn is recorded however it ends; a trace
        write that fails is logged and never costs the turn.

        Built with a fleet (#70), the turn is compiled first
        (:mod:`app.ai.interpreter`): ``paused`` is last turn's plan waiting on
        the person, ``self_facts`` what code knows about its own recent work
        (for "what did you just do?"). A plan the interpreter can't run yet
        falls back to the loop, and the result carries the plan either way.
        """
        bound = structlog.contextvars.get_contextvars()
        bindings: dict[str, str] = {}
        if "request_id" not in bound:
            bindings["request_id"] = uuid.uuid4().hex[:8]
        run_id = str(bound.get("run_id") or run_trace.new_run_id())
        bindings["run_id"] = run_id
        trace = run_trace.RunTrace(
            run_id=run_id,
            actor=actor,
            utterance=user_message,
            conversation_id=conversation_id,
            model=getattr(self._provider, "model", None),
            history=[dict(message) for message in history or []],
            context=context,
            surface=surface,
        )
        started = time.monotonic()
        with (
            structlog.contextvars.bound_contextvars(**bindings),
            run_trace.tracing(trace),
            use_stop(stop),
        ):
            try:
                result, turn_plan = None, None
                if self._interpreter is not None:
                    result, turn_plan = self._interpreter.run(
                        user_message,
                        context=context,
                        actor=actor,
                        trace=trace,
                        on_activity=on_activity,
                        stop=stop,
                        paused=paused,
                        self_facts=self_facts,
                    )
                    trace.plan = turn_plan.model_dump()
                if result is None:
                    opening = f"{context}\n\n{user_message}" if context else user_message
                    result = self._run(
                        opening, history, actor, on_activity, trace, words=user_message, stop=stop
                    )
                result.plan = turn_plan
            except AgentRunFailed as exc:
                exc.result.run_id = run_id
                self._write_trace(trace, exc.result, str(exc), started)
                raise
            result.run_id = run_id
            self._write_trace(trace, result, "", started)
            return result

    def _write_trace(
        self, trace: run_trace.RunTrace, result: AgentRunResult, error: str, started: float
    ) -> None:
        if self._tracer is None:
            return
        try:
            total_ms = round((time.monotonic() - started) * 1000)
            self._tracer.write(trace.record(result, error=error, total_ms=total_ms))
        except Exception as exc:
            logger.warning("agent_trace_failed", error=str(exc), error_type=type(exc).__name__)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        specs: Sequence[Any],
        trace: run_trace.RunTrace,
        *,
        max_tokens: int | None = None,
        role: run_trace.CallRole = "planner",
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One metered provider call: its status, time and tokens go on the trace."""
        call = run_trace.ModelCall(role=role)
        started = time.monotonic()
        try:
            if response_format is not None:
                result = self._provider.chat(
                    messages,
                    max_tokens=max_tokens or self._max_tokens,
                    response_format=response_format,
                )
            else:
                result = self._provider.chat(
                    messages, tools=specs or None, max_tokens=max_tokens or self._max_tokens
                )
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

    def _run(
        self,
        user_message: str,
        history: Sequence[dict[str, Any]] | None,
        actor: str,
        on_activity: OnActivity | None,
        trace: run_trace.RunTrace,
        *,
        words: str,
        stop: Stop | None = None,
    ) -> AgentRunResult:
        specs = self._tool_specs()
        system_prompt = self._system_prompt()
        trace.system_prompt_sha256 = run_trace.sha256(system_prompt)
        trace.tool_schemas_sha256 = run_trace.tool_schemas_sha256(specs)
        trace.tools_offered = len(specs)
        trace.limits = {
            "max_iterations": self._max_iterations,
            "max_corrections": self._max_corrections,
            "max_tokens": self._max_tokens,
            "max_tool_calls": self._max_tool_calls,
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": user_message},
        ]
        records: list[ToolCallRecord] = []
        corrections = 0
        iterations = 0
        dispatched = 0
        # Every exchange this run has seen: tool + canonical arguments + answer.
        seen: set[tuple[str, str, str]] = set()
        logger.info("agent_run_started", tools=len(specs), history_messages=len(history or []))
        for iteration in range(1, self._max_iterations + 1):
            iterations = iteration
            if _stopped(stop):
                return self._stop("cancelled", records, iteration, messages)
            self._emit(on_activity, LoopActivity(kind="model", iteration=iteration))
            try:
                result = self._chat(messages, specs, trace)
            except ToolCallArgumentsError as exc:
                # The turn is unusable — its tool calls can't round-trip into
                # history — so the correction goes back as a user-role message
                # instead of a tool result.
                corrections += 1
                trace.corrections = corrections
                logger.warning(
                    "agent_unparseable_tool_call",
                    tool=exc.tool_name,
                    error=str(exc),
                    corrections=corrections,
                )
                if corrections > self._max_corrections:
                    return self._stop("correction_limit", records, iteration, messages)
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
            except ProviderError as exc:
                # Conductor's own model died. Whatever the apps already did
                # stands; the route stores it before answering 502.
                logger.error("agent_provider_failed", error=str(exc))
                raise self._failed(
                    "provider_error", f"agent run failed: {exc}", records, iteration, messages
                ) from exc
            except Exception as exc:
                logger.exception("agent_provider_crashed", error_type=type(exc).__name__)
                raise self._failed(
                    "internal_error", f"agent run crashed: {exc}", records, iteration, messages
                ) from exc
            truncated = result.finish_reason == _FINISH_LENGTH
            if truncated:
                logger.warning(
                    "agent_turn_truncated",
                    max_tokens=self._max_tokens,
                    tool_calls=len(result.tool_calls),
                )
            if not result.tool_calls:
                if truncated:
                    # Cut off by the cap: the fragment is not a reply. Drop it
                    # (never returned, never in the transcript).
                    return self._stop("truncated", records, iteration, messages)
                # A text turn terminates the loop, even when content is empty.
                messages.append(result.to_message())
                if _stopped(stop):
                    # Stopped before the narrator's call: the record speaks.
                    return self._stop("cancelled", records, iteration, messages)
                reply = self._reply(words, result.content, records, trace, on_activity, iteration)
                return self._finish("completed", reply, records, iteration, messages)
            if truncated:
                # The calls arrived whole and run; only the cut-off text beside
                # them is dropped from history.
                result = ChatResult(
                    content=None,
                    tool_calls=result.tool_calls,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                )
            messages.append(result.to_message())
            schema_error_this_turn = False
            progressed = False
            over_cap = False
            cancelled = False
            for index, call in enumerate(result.tool_calls):
                if _stopped(stop):
                    # Stopped: this call and the rest of the turn never start.
                    self._skip(result.tool_calls[index:], records, messages, STOPPED_ERROR)
                    cancelled = True
                    break
                if dispatched >= self._max_tool_calls:
                    # Past the cap: this call and the rest of the turn are
                    # answered "not run" and never dispatched.
                    self._skip(result.tool_calls[index:], records, messages, _TOOL_CALL_LIMIT_ERROR)
                    over_cap = True
                    break
                dispatched += 1
                self._emit(
                    on_activity, LoopActivity(kind="tool", iteration=iteration, tool=call.name)
                )
                try:
                    record, feedback, schema_error = self._dispatch(call, actor)
                except Exception as exc:
                    # A bug, not something the model can correct. The call may
                    # or may not have reached an app, so its record claims
                    # neither; the rest of the batch never starts.
                    logger.exception("agent_tool_crashed", tool=call.name)
                    records.append(
                        ToolCallRecord(
                            tool=call.name, arguments=call.arguments, error=_INTERNAL_ERROR
                        )
                    )
                    messages.append(tool_result_message(call.id, f"Error: {_INTERNAL_ERROR}"))
                    self._skip(
                        result.tool_calls[index + 1 :], records, messages, _CRASHED_BATCH_ERROR
                    )
                    raise self._failed(
                        "internal_error",
                        f"agent run crashed in {call.name}: {exc}",
                        records,
                        iteration,
                        messages,
                    ) from exc
                records.append(record)
                messages.append(tool_result_message(call.id, feedback))
                schema_error_this_turn = schema_error_this_turn or schema_error
                # Judged on what came back, not the call alone: a repeat that
                # answers differently (a list after a write) is new work.
                exchange = _exchange_key(call.name, call.arguments, feedback)
                progressed = progressed or exchange not in seen
                seen.add(exchange)
            if cancelled or _stopped(stop):
                return self._stop("cancelled", records, iteration, messages)
            if over_cap:
                return self._stop("tool_call_limit", records, iteration, messages)
            if schema_error_this_turn:
                corrections += 1
                trace.corrections = corrections
                if corrections > self._max_corrections:
                    return self._stop("correction_limit", records, iteration, messages)
                # A malformed call is what the correction budget is for, so a
                # turn with one is never judged a stall.
            elif not progressed:
                logger.warning("agent_run_no_progress", tool_calls=len(result.tool_calls))
                return self._stop("no_progress", records, iteration, messages)
        return self._stop("max_iterations", records, iterations, messages)

    def _reply(
        self,
        words: str,
        planner_reply: str | None,
        records: list[ToolCallRecord],
        trace: run_trace.RunTrace,
        on_activity: OnActivity | None,
        iteration: int,
    ) -> str | None:
        """A completed run's reply: the narrator's, or the planner's words when
        the loop doesn't narrate. Never raises: a narrator that fails falls
        back to the code-built record reply inside :func:`narrator.narrate`."""
        if not self._narrate:
            return planner_reply
        started = time.monotonic()

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            self._emit(on_activity, LoopActivity(kind="model", iteration=iteration))
            return self._chat(messages, [], trace, max_tokens=max_tokens, role="narrator")

        try:
            told = narrator.narrate(words, planner_reply, records, surface=trace.surface, chat=chat)
        except Exception:
            # A bug in the narrator's code, not the model: the turn still stands.
            logger.exception("agent_narrator_crashed")
            told = narrator.Narration(
                narrator.fallback_reply(records, trace.surface) or planner_reply, "fallback"
            )
        trace.narration = {
            "path": told.path,
            "appended": told.appended,
            "model_calls": told.model_calls,
            "ms": round((time.monotonic() - started) * 1000),
        }
        logger.info("agent_narrated", path=told.path, appended=len(told.appended))
        return told.reply

    def _tool_specs(self) -> list[ToolSpec]:
        """The tools offered this turn. A seam for :mod:`app.ai.replay`."""
        return registry.tool_specs()

    def _system_prompt(self) -> str:
        """This turn's system prompt. A seam for :mod:`app.ai.replay`."""
        return build_system_prompt(date.today(), fleet_section=self._fleet_section)

    def _dispatch(self, call: ToolCall, actor: str) -> tuple[ToolCallRecord, str, bool]:
        """Run one tool call as ``actor``.

        Returns the record, the feedback text for the model's ``role: tool``
        message, and whether the failure was schema-level (counts against the
        correction budget). Unexpected exceptions (bugs) propagate — the loop
        only self-corrects what the model can fix.
        """
        record = ToolCallRecord(tool=call.name, arguments=call.arguments)
        schema_error = False
        # Cleared per dispatch and reset after, so what one delegate call
        # reported can never be read as the next call's.
        delegated = runtime.delegated_tools.set(None)
        delegated_record = runtime.delegated_record.set(None)
        delegated_confirm = runtime.delegated_confirm.set(None)
        try:
            outcome = registry.call_tool(call.name, call.arguments, actor=actor)
        except UnknownToolError as exc:
            record.error = str(exc)
            schema_error = True
        except ValidationError as exc:
            record.error = f"Invalid arguments: {_validation_summary(exc)}"
            schema_error = True
        except ToolError as exc:
            # Domain rejection with a reason the model can act on. A delegate
            # call that failed partway still reports what the app did.
            record.error = str(exc)
            self._take_app_report(record)
        else:
            record.result = _result_text(outcome)
            self._take_app_report(record)
            logger.info("agent_tool_call", tool=call.name, ok=True)
            return record, record.result, False
        finally:
            runtime.delegated_tools.reset(delegated)
            runtime.delegated_record.reset(delegated_record)
            runtime.delegated_confirm.reset(delegated_confirm)
        logger.warning(
            "agent_tool_call",
            tool=call.name,
            ok=False,
            error=record.error,
            schema_error=schema_error,
        )
        return record, f"Error: {record.error}", schema_error

    @staticmethod
    def _take_app_report(record: ToolCallRecord) -> None:
        """What the app said it did while answering this call, if anything."""
        record.app_tools = list(runtime.delegated_tools.get() or []) or None
        app_record = runtime.delegated_record.get()
        record.app_record = list(app_record) if app_record is not None else None
        record.app_confirm = runtime.delegated_confirm.get()

    @staticmethod
    def _skip(
        calls: Sequence[ToolCall],
        records: list[ToolCallRecord],
        messages: list[dict[str, Any]],
        error: str,
    ) -> None:
        """Record and answer calls that were never dispatched, so every call
        in the transcript has exactly one answer."""
        for call in calls:
            records.append(ToolCallRecord(tool=call.name, arguments=call.arguments, error=error))
            messages.append(tool_result_message(call.id, f"Error: {error}"))
        if calls:
            logger.warning("agent_tool_calls_skipped", count=len(calls), error=error)

    def _stop(
        self,
        stop_reason: _StopReason,
        records: list[ToolCallRecord],
        iterations: int,
        messages: list[dict[str, Any]],
    ) -> AgentRunResult:
        """End at a limit, replying from the code-kept record of what ran."""
        reply = run_record.stop_reply(stop_reason, records)
        return self._finish(stop_reason, reply, records, iterations, messages)

    def _failed(
        self,
        stop_reason: _StopReason,
        message: str,
        records: list[ToolCallRecord],
        iterations: int,
        messages: list[dict[str, Any]],
    ) -> AgentRunFailed:
        """A failure stop as an exception carrying the partial run."""
        return AgentRunFailed(
            self._stop(stop_reason, records, iterations, messages), message=message
        )

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


def loop_from_settings(
    provider: ChatProvider, *, fleet_section: str | None = None, fleet: Fleet | None = None
) -> AgentLoop:
    """An :class:`AgentLoop` bounded by the configured ``conductor_max_iterations``.

    The provider factory's counterpart: the one place the loop's shallow
    iteration cap is wired from settings, so the ``CONDUCTOR_MAX_ITERATIONS``
    field is never an orphan. ``fleet_section`` is threaded through to the
    prompt's fleet layer (the HTTP app passes the one its lifespan rendered).
    """
    settings = get_settings()
    return AgentLoop(
        provider,
        max_iterations=settings.conductor_max_iterations,
        max_tokens=settings.conductor_max_tokens,
        max_tool_calls=settings.conductor_max_tool_calls,
        fleet_section=fleet_section,
        tracer=run_trace.tracer_from_settings(),
        narrate=settings.conductor_narrator,
        # Compiled plans (#70) need the fleet the prompt and schema are built from.
        fleet=fleet if settings.conductor_compiler else None,
    )


def _stopped(stop: Stop | None) -> bool:
    return stop is not None and stop.is_set()


def _validation_summary(exc: ValidationError) -> str:
    """Pydantic's error list as one compact line the model can read."""
    return "; ".join(
        f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _exchange_key(name: str, arguments: dict[str, Any], answer: str) -> tuple[str, str, str]:
    """Identity of one exchange: the tool, its arguments (sorted keys) and its answer.

    The answer is part of the key: the same lookup after a write comes back
    different, and that is progress.
    """
    return name, json.dumps(arguments, sort_keys=True, default=str), answer


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
