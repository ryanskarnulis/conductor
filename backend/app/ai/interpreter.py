"""Interpreter v1 (#70): run a compiled plan for one turn.

The loop hands a turn here when it is built with a fleet
(``CONDUCTOR_COMPILER``). The compiler writes the plan (:mod:`app.ai.compiler`)
and code runs it, step by step, through the loop's own dispatch, so the
delegate budget, Idempotency-Keys, receipts, Stop and the narrator are the
same as the loop's. What v1 runs:

- ``ask``: the app's agent, through its ``ask_<app>`` tool. A ``destructive``
  ask to an app that doesn't confirm its own writes pauses first on code's
  question (the backstop gate); ``confirm`` yes runs it. An app's
  ``confirm_required`` pauses the plan the same way, on the app's question.
- ``open``: the handoff. Always last.
- ``ask_user``: the plan pauses on the question; the person's answer resumes
  the same plan, ``{s1}`` replaced by their words.
- ``confirm``: yes or no to whatever is waiting.
- The direct answers (:mod:`app.ai.speaker`), ``self`` from the ledger.

A step that fails ends the plan: the steps after it are on the record as not
run, and the narrator says so. A plan v1 can't run yet, or no valid plan after
one repair, returns ``None`` and the loop takes the turn; the compiled plan is
kept on the result either way (:class:`TurnPlan`), so the ledger records it.

For a slow plan (one that asks an app) code speaks an honest ack as soon as the
plan is valid: what's about to happen, never what's done.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from app.ai import compiler, narrator, run_record, speaker
from app.ai import plan as plans
from app.ai import trace as run_trace
from app.ai.provider import ChatResult, ProviderError, ToolCall
from app.ai.plan import PausedPlan, TurnPlan
from app.ai.stop import STOPPED_ERROR, Stop
from app.fleet.context import current_delegation_context
from app.fleet.manifests import Fleet
from app.fleet.tools import ask_tool_name, open_tool_name

if TYPE_CHECKING:
    from app.ai.loop import AgentLoop, AgentRunResult, LoopActivity, OnActivity, ToolCallRecord

logger = structlog.get_logger(__name__)

_AFTER_FAILURE = f"{run_record.NOT_RUN_PREFIX}an earlier step of the plan didn't finish"
_LEFT_IT = "Okay, I left it."
_CONFIRM_TOOL = "confirm"


@dataclass
class _Run:
    """One compiled turn's working state."""

    words: str
    actor: str
    trace: run_trace.RunTrace
    on_activity: OnActivity | None
    stop: Stop | None
    records: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class Interpreter:
    """Compiles and runs one turn on behalf of an :class:`AgentLoop`."""

    def __init__(self, loop: AgentLoop, fleet: Fleet) -> None:
        self._loop = loop
        self._fleet = fleet
        self._prompt = compiler.planner_prompt(fleet)

    @property
    def system_prompt(self) -> str:
        return self._prompt

    def run(
        self,
        words: str,
        *,
        context: str | None,
        actor: str,
        trace: run_trace.RunTrace,
        on_activity: OnActivity | None,
        stop: Stop | None,
        paused: PausedPlan | None,
        self_facts: Callable[[], str] | None,
    ) -> tuple[AgentRunResult | None, TurnPlan]:
        """The turn's result, or ``None`` when the loop should take it; and the plan."""
        loop = self._loop
        state = _Run(words, actor, trace, on_activity, stop)
        if _stopped(stop):
            return loop._stop("cancelled", [], 0, []), TurnPlan(status="ran", note="stopped")
        pending = current_delegation_context().pending_confirm
        gate = paused if paused is not None and paused.reason == "gate" else None
        checks = plans.Checks(
            fleet=self._fleet,
            calls_per_app=current_delegation_context().calls_per_turn_per_app,
            confirmable=pending is not None or gate is not None,
            resumable=paused is not None and paused.reason == "ask_user",
            context=context or "",
        )
        loop._emit(on_activity, _activity("model"))
        started = time.monotonic()
        try:
            compiled = compiler.compile_turn(
                self._planner_chat(trace),
                system_prompt=self._prompt,
                context=context,
                words=words,
                fleet=self._fleet,
                checks=checks,
            )
        except ProviderError as exc:
            logger.error("plan_compile_failed", error=str(exc))
            raise loop._failed("provider_error", f"agent run failed: {exc}", [], 1, []) from exc
        except Exception as exc:
            logger.exception("plan_compile_crashed", error_type=type(exc).__name__)
            raise loop._failed("internal_error", f"agent run crashed: {exc}", [], 1, []) from exc
        compile_ms = round((time.monotonic() - started) * 1000)
        state.messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": compiler.user_message(context, words)},
            *({"role": "assistant", "content": raw} for raw in compiled.raw),
        ]
        plan = compiled.plan
        if plan is None:
            logger.info("plan_fell_back", reason="invalid", problems=compiled.problems)
            return None, TurnPlan(
                status="invalid", note="; ".join(compiled.problems)[:500], acted=False
            )
        body = plans.dump(plan)
        confirm_app = (gate.app if gate else None) or (pending.app if pending else None)
        reason = plans.fallback_reason(plan, confirm_app=confirm_app)
        logger.info(
            "plan_compiled",
            plan=plans.describe(plan),
            repaired=compiled.repaired,
            ms=compile_ms,
            fallback=reason,
        )
        if reason is not None:
            return None, TurnPlan(
                body=body, status="fell_back", note=reason, acted=False, repaired=compiled.repaired
            )
        ack = plans.ack(plan)
        if ack is not None:
            loop._emit(on_activity, _activity("model", ack=ack))
        turn = TurnPlan(body=body, status="ran", ack=ack, repaired=compiled.repaired)
        if isinstance(plan, plans.Direct):
            turn.status = "answered"
            return self._answer(state, plan, self_facts), turn
        if isinstance(plan, plans.Resume):
            assert paused is not None  # the validator only allows a resume while paused
            resumed = plans.load(paused.body)
            assert isinstance(resumed, plans.StepPlan)
            turn.body = paused.body
            steps = resumed.steps[paused.resume_at :]
            return self._execute(state, turn, steps, paused.resume_at, {"s1": plan.resume}), turn
        assert isinstance(plan, plans.StepPlan)
        first = plan.steps[0]
        if isinstance(first, plans.Confirm):
            result = self._confirm(state, turn, plan, first, gate=gate, pending_app=confirm_app)
            return result, turn
        return self._execute(state, turn, plan.steps, 0, {}), turn

    # --- answering and confirming -------------------------------------------------------

    def _answer(
        self, state: _Run, plan: plans.Direct, self_facts: Callable[[], str] | None
    ) -> AgentRunResult:
        facts = None
        if plan.answer in ("self", "knowledge"):
            facts = self._facts(self_facts)
        reply, path = speaker.answer(
            plan.answer,
            state.words,
            surface=state.trace.surface,
            chat=self._spoken_chat(state),
            facts=facts,
            structured=self._structured_chat(state),
        )
        state.trace.narration = {"path": f"speaker:{path}", "appended": [], "model_calls": 0}
        return self._loop._finish("completed", reply, [], 1, state.messages)

    def _facts(self, self_facts: Callable[[], str] | None) -> str:
        lines = []
        if self_facts is not None:
            try:
                lines.append(self_facts())
            except Exception as exc:
                logger.warning("self_facts_failed", error=str(exc))
        apps = [
            f"- {app.name}: {(app.agent or app.open).description}"  # type: ignore[union-attr]
            for app in self._fleet.apps
            if app.agent is not None or app.open is not None
        ]
        if apps:
            lines.append("What I can do, through the apps:\n" + "\n".join(apps))
        return "\n\n".join(line for line in lines if line)

    def _confirm(
        self,
        state: _Run,
        turn: TurnPlan,
        plan: plans.StepPlan,
        step: plans.Confirm,
        *,
        gate: PausedPlan | None,
        pending_app: str | None,
    ) -> AgentRunResult:
        rest = plan.steps[1:]
        if gate is not None:
            # Conductor's own question: a yes runs the paused plan from the gated step.
            turn.body = gate.body
            if step.answer == "no":
                turn.note = "declined at the gate"
                return self._loop._finish("completed", _LEFT_IT, [], 1, state.messages)
            paused = plans.load(gate.body)
            assert isinstance(paused, plans.StepPlan)
            steps = paused.steps[gate.resume_at :]
            return self._execute(
                state, turn, steps, gate.resume_at, {}, gate_passed=steps[0].id if steps else None
            )
        if step.answer == "no":
            # The app's question lapses on its own; nothing is sent.
            turn.note = "declined the app's question"
            return self._loop._finish("completed", _LEFT_IT, [], 1, state.messages)
        record = self._dispatch(state, _CONFIRM_TOOL, {"app": pending_app or ""})
        if record is None or not self._went_well(record):
            return self._finish(state, turn, rest, 1)
        return self._execute(state, turn, rest, 1, {})

    # --- running steps ---------------------------------------------------------------------

    def _execute(
        self,
        state: _Run,
        turn: TurnPlan,
        steps: Sequence[plans.Step],
        offset: int,
        answers: dict[str, str],
        *,
        gate_passed: str | None = None,
    ) -> AgentRunResult:
        """Run ``steps`` in order; ``offset`` is the first one's index in the plan."""
        loop = self._loop
        for index, step in enumerate(steps):
            position = offset + index
            if _stopped(state.stop):
                self._skip(steps[index:], state, STOPPED_ERROR)
                turn.note = f"stopped before {step.id}"
                return loop._stop("cancelled", state.records, 1, state.messages)
            if isinstance(step, plans.AskUser):
                turn.status, turn.pause, turn.resume_at = "paused", "ask_user", position + 1
                return self._ask_user(state, step)
            if isinstance(step, plans.Ask):
                app = self._fleet.get(step.app)
                confirms = app is not None and app.agent is not None and app.agent.confirms
                if step.destructive and not confirms and step.id != gate_passed:
                    turn.status, turn.pause = "paused", "gate"
                    turn.resume_at, turn.pause_app = position, step.app
                    return self._gate(state, step, answers)
                request = plans.substitute(step.request, answers)
                record = self._dispatch(state, ask_tool_name(step.app), {"message": request})
                if record is not None and record.app_confirm and record.app_confirm.get("question"):
                    if index + 1 < len(steps):
                        turn.status, turn.pause = "paused", "app_confirm"
                        turn.resume_at, turn.pause_app = position + 1, step.app
                    return self._finish(state, turn, [], 1)
            elif isinstance(step, plans.Open):
                intent = plans.substitute(step.intent, answers)
                record = self._dispatch(state, open_tool_name(step.app), {"intent": intent})
            else:  # the validator and fallback_reason keep other kinds out
                raise AssertionError(f"interpreter v1 can't run {step.do}")
            if record is None or not self._went_well(record):
                turn.note = f"stopped after {step.id}"
                return self._finish(state, turn, steps[index + 1 :], 1)
        return self._finish(state, turn, [], 1)

    def _finish(
        self, state: _Run, turn: TurnPlan, unrun: Sequence[plans.Step], iteration: int
    ) -> AgentRunResult:
        loop = self._loop
        self._skip(unrun, state, _AFTER_FAILURE)
        if _stopped(state.stop):
            return loop._stop("cancelled", state.records, iteration, state.messages)
        reply = loop._reply(
            state.words, None, state.records, state.trace, state.on_activity, iteration
        )
        return loop._finish("completed", reply, state.records, iteration, state.messages)

    def _dispatch(self, state: _Run, tool: str, arguments: dict[str, Any]) -> ToolCallRecord | None:
        """One step through the loop's dispatch; a crash fails the turn like the loop's."""
        loop = self._loop
        loop._emit(state.on_activity, _activity("tool", tool=tool))
        call = ToolCall(id=f"plan-{len(state.records) + 1}", name=tool, arguments=arguments)
        try:
            record, _feedback, _schema = loop._dispatch(call, state.actor)
        except Exception as exc:
            logger.exception("plan_step_crashed", tool=tool)
            from app.ai.loop import ToolCallRecord  # noqa: PLC0415

            state.records.append(
                ToolCallRecord(tool=tool, arguments=arguments, error=_INTERNAL_ERROR)
            )
            raise loop._failed(
                "internal_error", f"agent run crashed in {tool}: {exc}", state.records, 1, []
            ) from exc
        state.records.append(record)
        return record

    @staticmethod
    def _went_well(record: ToolCallRecord) -> bool:
        return run_record.classify(record) in ("done", "looked_up")

    def _skip(self, steps: Sequence[plans.Step], state: _Run, error: str) -> None:
        from app.ai.loop import ToolCallRecord  # noqa: PLC0415

        for step in steps:
            tool, arguments = _call_for(step)
            state.records.append(ToolCallRecord(tool=tool, arguments=arguments, error=error))

    # --- questions -------------------------------------------------------------------------

    def _ask_user(self, state: _Run, step: plans.AskUser) -> AgentRunResult:
        question = step.question.strip()
        options = [option.strip() for option in step.options if option.strip()]
        if options:
            question = f"{question} ({' / '.join(options)})"
        reply = self._with_record_line(state, question)
        return self._loop._finish("completed", reply, state.records, 1, state.messages)

    def _gate(self, state: _Run, step: plans.Ask, answers: dict[str, str]) -> AgentRunResult:
        request = plans.substitute(step.request, answers)
        question = f'Just to check: should I ask {step.app} to "{request}"?'
        reply = self._with_record_line(state, question)
        return self._loop._finish("completed", reply, state.records, 1, state.messages)

    def _with_record_line(self, state: _Run, question: str) -> str:
        line = narrator.record_line(state.records, state.trace.surface)
        return f"{line} {question}" if line else question

    # --- model calls -----------------------------------------------------------------------

    def _planner_chat(self, trace: run_trace.RunTrace) -> compiler.PlannerChat:
        loop = self._loop

        def chat(
            messages: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any]
        ) -> ChatResult:
            return loop._chat(
                messages,
                [],
                trace,
                max_tokens=max_tokens,
                role="compiler",
                response_format=response_format,
            )

        return chat

    def _structured_chat(self, state: _Run) -> speaker.StructuredChat:
        loop = self._loop

        def chat(
            messages: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any]
        ) -> ChatResult:
            loop._emit(state.on_activity, _activity("model"))
            return loop._chat(
                messages,
                [],
                state.trace,
                max_tokens=max_tokens,
                role="speaker",
                response_format=response_format,
            )

        return chat

    def _spoken_chat(self, state: _Run) -> narrator.Chat:
        loop = self._loop

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            loop._emit(state.on_activity, _activity("model"))
            return loop._chat(messages, [], state.trace, max_tokens=max_tokens, role="speaker")

        return chat


def _call_for(step: plans.Step) -> tuple[str, dict[str, Any]]:
    """The tool call a step would have been, for the record of steps not run."""
    if isinstance(step, plans.Ask):
        return ask_tool_name(step.app), {"message": step.request}
    if isinstance(step, plans.Open):
        return open_tool_name(step.app), {"intent": step.intent}
    return f"plan:{step.do}", {}


_INTERNAL_ERROR = "internal error: the tool call failed unexpectedly"


def _activity(
    kind: Literal["model", "tool"], *, tool: str | None = None, ack: str | None = None
) -> LoopActivity:
    from app.ai.loop import LoopActivity  # noqa: PLC0415

    return LoopActivity(kind=kind, iteration=1, tool=tool, ack=ack)


def _stopped(stop: Stop | None) -> bool:
    return stop is not None and stop.is_set()
