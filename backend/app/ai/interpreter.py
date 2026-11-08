"""The interpreter (#70, #72): run a compiled plan for one turn.

The loop hands a turn here when it is built with a fleet
(``CONDUCTOR_COMPILER``). The compiler writes the plan (:mod:`app.ai.compiler`)
and code runs it, step by step, through the loop's own dispatch, so the
delegate budget, Idempotency-Keys, receipts, Stop and the narrator are the
same as the loop's. What it runs, across apps (#72):

- ``ask``: the app's agent, through its ``ask_<app>`` tool. A ``destructive``
  ask to an app that doesn't confirm its own writes pauses first on code's
  question (the backstop gate); ``confirm`` yes runs it. An app's
  ``confirm_required`` pauses the plan the same way, on the app's question.
- ``look`` (#73): a plain GET to the app's glance endpoint (declared in its
  manifest), through :mod:`app.fleet.glance`: no agent loop, no model call, no
  budget charge. The record is a ``looked_up`` receipt built by code
  (:mod:`app.ai.glance`). If the glance can't be read (down, 5xx, timeout) the
  same app is asked the person's words instead, when that can't repeat a
  write the plan already made; else the step fails and the plan ends.
- ``open``: the handoff. Always last.
- ``ask_user``: the plan pauses on the question; the person's answer resumes
  the same plan, ``{s1}`` replaced by their words.
- ``{sN}`` in a step's text: the main title step N's receipt touched, quoted,
  filled in by code when the step runs (:mod:`app.ai.dataflow`, which holds
  the rules for no title, several titles and outside text).
- ``confirm``: yes or no to whatever is waiting.
- ``cancel``/``amend``/``undo``: on what's waiting or what just ran, over the
  ledger (:mod:`app.ai.control`, #74), each as a plan of its own; ``cancel``
  and ``amend`` on a running job go to the jobs runtime
  (:mod:`app.ai.job_control`, #76).
- A step whose receipt carries a job ref detaches: the app accepted the work
  and the reply says so now ("queued 40"); the ledger's job row and the
  watcher (:mod:`app.services.job_watcher`) follow it from there.
- The direct answers (:mod:`app.ai.speaker`), ``self`` from the ledger.

Independent reads (:func:`plans.reads_only`: asks to different apps, phrased
as questions, no refs; and looks) that sit next to each other run at once, each in its
own thread with the turn's context; everything else runs in order, one call
at a time, so there is never more than one writer in flight. The per-app
budget is charged under a lock and a Stop reaches every call in flight.

A step that fails ends the plan: the steps after it are on the record as not
run, and the narrator says so. In a parallel group the reads beside it have
already run; the plan ends after the group.

A plan the interpreter can't run yet, or no valid plan after one repair,
returns ``None`` and the loop takes the turn; the compiled plan is
kept on the result either way (:class:`TurnPlan`), so the ledger records it.

For a slow plan (one that asks an app) code speaks an honest ack as soon as the
plan is valid: what's about to happen, never what's done.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from app.ai import compiler, control, dataflow, narrator, run_record, speaker, turn_events
from app.ai import glance as glances
from app.ai import plan as plans
from app.ai import trace as run_trace
from app.ai.provider import ChatResult, ProviderError, ToolCall
from app.ai.plan import Ledgered, PausedPlan, TurnPlan
from app.ai.stop import STOPPED_ERROR, Stop, current_stop
from app.fleet.context import current_delegation_context
from app.fleet.glance import GlanceError, GlanceReader, read_glance
from app.fleet.jobs import JobCanceller, cancel_job
from app.fleet.manifests import Fleet
from app.fleet.tools import ask_tool_name, look_tool_name, open_tool_name

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
    # Whether ``words`` is what the person asked for. A resumed or confirmed
    # plan was told "the first one" or "yes": that restates nothing.
    restate: bool = True
    # What the ledger says control steps act on (#74), read only if needed.
    view: control.View = field(default_factory=lambda: control.View(None))


class Interpreter:
    """Compiles and runs one turn on behalf of an :class:`AgentLoop`."""

    def __init__(
        self,
        loop: AgentLoop,
        fleet: Fleet,
        glance: GlanceReader | None = None,
        jobs: JobCanceller | None = None,
    ) -> None:
        self._loop = loop
        self._fleet = fleet
        # How a look reads a glance; tests pass a fake, none dials the network.
        self._glance: GlanceReader = glance or read_glance
        # How a cancel reaches a running job (#76); the same rule.
        self._cancel_job: JobCanceller = jobs or cancel_job
        self._looks = {look_tool_name(app.name): app for app in fleet.apps}
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
        control_view: Callable[[], Ledgered] | None = None,
    ) -> tuple[AgentRunResult | None, TurnPlan]:
        """The turn's result, or ``None`` when the loop should take it; and the plan."""
        loop = self._loop
        state = _Run(words, actor, trace, on_activity, stop, view=control.View(control_view))
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
        reason = plans.fallback_reason(
            plan,
            confirm_app=confirm_app,
            pending=paused is not None or pending is not None,
        )
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
        if isinstance(plan, plans.Resume) or (
            isinstance(plan, plans.StepPlan) and isinstance(plan.steps[0], plans.Confirm)
        ):
            state.restate = False
        if isinstance(plan, plans.Direct):
            turn.status = "answered"
            return self._answer(state, plan, self_facts), turn
        if isinstance(plan, plans.Resume):
            assert paused is not None  # the validator only allows a resume while paused
            resumed = plans.load(paused.body)
            assert isinstance(resumed, plans.StepPlan)
            turn.body = paused.body
            steps = resumed.steps[paused.resume_at :]
            values = {"s1": dataflow.answer(plan.resume)}
            return self._execute(state, turn, steps, paused.resume_at, values), turn
        assert isinstance(plan, plans.StepPlan)
        first = plan.steps[0]
        if isinstance(first, plans.Confirm):
            result = self._confirm(
                state,
                turn,
                plan,
                first,
                gate=gate,
                pending_app=confirm_app,
                undoing=control.undo_waiting(paused, pending),
            )
            return result, turn
        if first.do in plans.CONTROL_KINDS:
            result = control.run(
                self, state, turn, first, state.view, paused=paused, pending=pending
            )
            return result, turn
        return self._execute(state, turn, plan.steps, 0, {}), turn

    # --- answering and confirming -------------------------------------------------------

    def _answer(
        self, state: _Run, plan: plans.Direct, self_facts: Callable[[], str] | None
    ) -> AgentRunResult:
        facts = None
        if plan.answer in ("self", "knowledge", "chat"):
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
        undoing: tuple[plans.Undo, int, tuple[int, ...]] | None = None,
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
            if steps and isinstance(steps[0], plans.Undo):
                undo, at = steps[0], gate.resume_at
                return control.undo(
                    self, state, turn, undo, state.view, position=at, gate_app=gate.app
                )
            return self._execute(
                state, turn, steps, gate.resume_at, {}, gate_passed=steps[0].id if steps else None
            )
        if step.answer == "no":
            # The app's question lapses on its own; nothing is sent.
            turn.note = "declined the app's question"
            return self._loop._finish("completed", _LEFT_IT, [], 1, state.messages)
        turn_events.plan(turn.body, 0)
        record = self._dispatch(state, _CONFIRM_TOOL, {"app": pending_app or ""})
        if undoing is not None and record is not None and self._went_well(record):
            # The yes was to an undo's question: the rest of that undo goes on.
            undo_step, position, handled = undoing
            turn.body = {"steps": [undo_step.model_dump()]}
            return control.undo(
                self, state, turn, undo_step, state.view, position=position, handled=handled
            )
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
        values: dict[str, dataflow.Value],
        *,
        gate_passed: str | None = None,
    ) -> AgentRunResult:
        """Run ``steps``; ``offset`` is the first one's index in the plan."""
        loop = self._loop
        turn_events.plan(turn.body, offset)
        index = 0
        while index < len(steps):
            step = steps[index]
            position = offset + index
            if _stopped(state.stop):
                self._skip(steps[index:], state, STOPPED_ERROR)
                turn.note = f"stopped before {step.id}"
                return loop._stop("cancelled", state.records, 1, state.messages)
            if isinstance(step, plans.AskUser):
                turn.status, turn.pause, turn.resume_at = "paused", "ask_user", position + 1
                return self._ask_user(state, step)
            group = _read_group(steps, index)
            if len(group) > 1:
                results = self._dispatch_all(state, [_call_for(s) for s in group])
                for n, read in enumerate(group):
                    if isinstance(read, plans.Look) and not self._went_well(results[n]):
                        results[n] = self._look_fell_back(state, read, results[n], steps)
                for read, got in zip(group, results, strict=True):
                    self._keep(values, read, got)
                after = index + len(group)
                asked = next(
                    (
                        (read, got)
                        for read, got in zip(group, results, strict=True)
                        if got.app_confirm and got.app_confirm.get("question")
                    ),
                    None,
                )
                if asked is not None:
                    return self._app_asked(state, turn, steps, offset + after, asked[0], values)
                failed = [
                    read
                    for read, got in zip(group, results, strict=True)
                    if not self._went_well(got)
                ]
                if failed:
                    turn.note = f"stopped after {failed[0].id}"
                    return self._finish(state, turn, steps[after:], 1)
                index = after
                continue
            if isinstance(step, plans.Ask):
                filled = dataflow.fill(step.request, values)
                app = self._fleet.get(step.app)
                confirms = app is not None and app.agent is not None and app.agent.confirms
                refusal = _refusal(step, filled, confirms=confirms)
                if refusal is not None:
                    return self._refuse(state, turn, steps, index, filled, refusal)
                if step.destructive and not confirms and step.id != gate_passed:
                    turn.status, turn.pause = "paused", "gate"
                    turn.resume_at, turn.pause_app = position, step.app
                    self._carry(turn, position, values)
                    return self._gate(state, step, filled.text)
                record = self._dispatch(state, ask_tool_name(step.app), {"message": filled.text})
                if record is not None and record.app_confirm and record.app_confirm.get("question"):
                    return self._app_asked(state, turn, steps, position + 1, step, values)
            elif isinstance(step, plans.Look):
                record = self._dispatch(state, look_tool_name(step.app), {})
                if record is not None and not self._went_well(record):
                    record = self._look_fell_back(state, step, record, steps)
                    if record.app_confirm and record.app_confirm.get("question"):
                        return self._app_asked(state, turn, steps, position + 1, step, values)
            elif isinstance(step, plans.Open):
                filled = dataflow.fill(step.intent, values)
                refusal = _refusal(step, filled, confirms=False)
                if refusal is not None:
                    return self._refuse(state, turn, steps, index, filled, refusal)
                record = self._dispatch(state, open_tool_name(step.app), {"intent": filled.text})
            else:  # the validator and fallback_reason keep other kinds out
                raise AssertionError(f"the interpreter can't run {step.do}")
            if record is None or not self._went_well(record):
                turn.note = f"stopped after {step.id}"
                return self._finish(state, turn, steps[index + 1 :], 1)
            self._keep(values, step, record)
            index += 1
        return self._finish(state, turn, [], 1)

    @staticmethod
    def _keep(values: dict[str, dataflow.Value], step: plans.Step, record: ToolCallRecord) -> None:
        """What a step gives later steps as ``{sN}``, from its receipt."""
        value = dataflow.from_record(record.app_record)
        if value is not None:
            values[step.id] = value

    def _app_asked(
        self,
        state: _Run,
        turn: TurnPlan,
        steps: Sequence[plans.Step],
        resume_at: int,
        step: plans.Ask | plans.Look,
        values: dict[str, dataflow.Value],
    ) -> AgentRunResult:
        """An app asked the person to confirm: the plan waits on their answer
        if anything is left to run after it."""
        plan_length = len((turn.body or {}).get("steps") or [])
        asked = next((r.app_confirm for r in reversed(state.records) if r.app_confirm), None)
        question = asked.get("question") if asked else None
        turn_events.paused("app_confirm", question=question, app=step.app, step=step.id)
        if resume_at < plan_length:
            turn.status, turn.pause = "paused", "app_confirm"
            turn.resume_at, turn.pause_app = resume_at, step.app
            self._carry(turn, resume_at, values)
        return self._finish(state, turn, [], 1)

    @staticmethod
    def _carry(turn: TurnPlan, start: int, values: dict[str, dataflow.Value]) -> None:
        """A pausing plan keeps what earlier steps found for the turn that resumes it."""
        if turn.body is not None:
            turn.body = dataflow.carry(turn.body, start, values)

    def _refuse(
        self,
        state: _Run,
        turn: TurnPlan,
        steps: Sequence[plans.Step],
        index: int,
        filled: dataflow.Filled,
        refusal: str,
    ) -> AgentRunResult:
        """A step code won't run (#72): on the record as not run, with why, and
        the plan ends there."""
        from app.ai.loop import ToolCallRecord  # noqa: PLC0415

        step = steps[index]
        tool, arguments = _call_for(step)
        arguments = {name: filled.text for name in arguments}
        state.records.append(
            ToolCallRecord(
                tool=tool, arguments=arguments, error=run_record.NOT_RUN_PREFIX + refusal
            )
        )
        turn_events.receipt(len(state.records), state.records[-1])
        turn.note = f"{step.id} not run: {refusal}"[:500]
        logger.info("plan_step_refused", step=step.id, reason=refusal)
        return self._finish(state, turn, steps[index + 1 :], 1)

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
        turn_events.step_started(len(state.records) + 1, tool, arguments)
        call = ToolCall(id=f"plan-{len(state.records) + 1}", name=tool, arguments=arguments)
        try:
            record, _feedback, _schema = self._call(call, state.actor)
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
        turn_events.receipt(len(state.records), record)
        return record

    def _dispatch_all(
        self, state: _Run, calls: Sequence[tuple[str, dict[str, Any]]]
    ) -> list[ToolCallRecord]:
        """Independent reads at once (#72): one thread each, each running in a
        copy of the turn's context (its delegation context, Stop and trace),
        records kept in plan order. A crash fails the turn like the loop's,
        once every call has come back."""
        from app.ai.loop import ToolCallRecord  # noqa: PLC0415

        loop = self._loop
        first = len(state.records)
        tool_calls = [
            ToolCall(id=f"plan-{first + n + 1}", name=tool, arguments=arguments)
            for n, (tool, arguments) in enumerate(calls)
        ]
        for n, call in enumerate(tool_calls):
            loop._emit(state.on_activity, _activity("tool", tool=call.name))
            turn_events.step_started(first + n + 1, call.name, call.arguments)
        logger.info("plan_reads_parallel", tools=[call.name for call in tool_calls])
        with ThreadPoolExecutor(max_workers=len(tool_calls), thread_name_prefix="plan") as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, self._call, call, state.actor)
                for call in tool_calls
            ]
        records: list[ToolCallRecord] = []
        crash: tuple[str, Exception] | None = None
        for n, (call, future) in enumerate(zip(tool_calls, futures, strict=True)):
            try:
                record, _feedback, _schema = future.result()
            except Exception as error:
                logger.exception("plan_step_crashed", tool=call.name)
                record = ToolCallRecord(
                    tool=call.name, arguments=call.arguments, error=_INTERNAL_ERROR
                )
                crash = crash or (call.name, error)
            if any(entry.get("outcome") == "done" for entry in record.app_record or []):
                # Taken for a read, it wrote: harmless once, worth knowing.
                logger.warning("plan_read_wrote", tool=call.name)
            records.append(record)
            turn_events.receipt(first + n + 1, record)
        state.records.extend(records)
        if crash is not None:
            tool, cause = crash
            raise loop._failed(
                "internal_error", f"agent run crashed in {tool}: {cause}", state.records, 1, []
            ) from cause
        return records

    def _call(self, call: ToolCall, actor: str) -> tuple[ToolCallRecord, str, bool]:
        """One call: a look is read here, never through the registry the model
        is offered; everything else is the loop's own dispatch."""
        if run_record.is_look(call.name):
            return self._look(call)
        return self._loop._dispatch(call, actor)

    def _look(self, call: ToolCall) -> tuple[ToolCallRecord, str, bool]:
        """Read one app's glance: a ``looked_up`` record built by code, or the
        reason it couldn't be read. No model call, no delegate budget."""
        from app.ai.loop import ToolCallRecord  # noqa: PLC0415

        record = ToolCallRecord(tool=call.name, arguments={})
        app = self._looks.get(call.name)
        stop = current_stop()
        try:
            if stop is not None and stop.is_set():
                record.error = STOPPED_ERROR
            elif app is None or app.glance is None:
                record.error = "couldn't look: that app has no glance"
            else:
                payload = self._glance(app)
                record.result, entry = glances.receipt(payload)
                record.app_record = [entry]
        except GlanceError as exc:
            record.error = f"couldn't look at {app.name if app else 'the app'}: {exc}"
        logger.info("plan_look", tool=call.name, ok=record.error is None)
        return record, record.result or f"Error: {record.error}", False

    def _look_fell_back(
        self,
        state: _Run,
        step: plans.Look,
        failed: ToolCallRecord,
        steps: Sequence[plans.Step],
    ) -> ToolCallRecord:
        """A glance that couldn't be read: ask the same app the person's words
        instead (#73). Returns the ask's record, or ``failed`` when asking
        wouldn't be right: Stop, an app with no agent, words that aren't the
        request (a resumed plan was told "yes"), words that aren't a plain
        question (a look replaces a read; its stand-in never writes), or an
        ask already in the plan for this app, which the words would repeat."""
        app = self._fleet.get(step.app)
        if (
            run_record.classify(failed) == "not_run"
            or app is None
            or app.agent is None
            or not state.restate
            or not plans.reads_only(
                plans.Ask(id="s0", do="ask", app=app.name, request=state.words, destructive=False)
            )
        ):
            return failed
        tool = ask_tool_name(app.name)
        if any(isinstance(other, plans.Ask) and other.app == app.name for other in steps) or any(
            record.tool == tool for record in state.records
        ):
            return failed
        failed.error = run_record.LOOK_FELL_BACK + (failed.error or "")
        logger.info("plan_look_fell_back", app=app.name, reason=failed.error)
        record = self._dispatch(state, tool, {"message": state.words})
        assert record is not None
        return record

    @staticmethod
    def _went_well(record: ToolCallRecord) -> bool:
        return run_record.classify(record) in ("done", "looked_up")

    def _skip(self, steps: Sequence[plans.Step], state: _Run, error: str) -> None:
        from app.ai.loop import ToolCallRecord  # noqa: PLC0415

        for step in steps:
            tool, arguments = _call_for(step)
            state.records.append(ToolCallRecord(tool=tool, arguments=arguments, error=error))
            turn_events.receipt(len(state.records), state.records[-1])

    # --- questions -------------------------------------------------------------------------

    def _ask_user(self, state: _Run, step: plans.AskUser) -> AgentRunResult:
        question = step.question.strip()
        options = [option.strip() for option in step.options if option.strip()]
        if options:
            question = f"{question} ({' / '.join(options)})"
        turn_events.paused("ask_user", question=question, step=step.id)
        reply = self._with_record_line(state, question)
        return self._loop._finish("completed", reply, state.records, 1, state.messages)

    def _gate(self, state: _Run, step: plans.Ask, request: str) -> AgentRunResult:
        question = f'Just to check: should I ask {step.app} to "{request}"?'
        turn_events.paused("gate", question=question, app=step.app, step=step.id)
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
    if isinstance(step, plans.Look):
        return look_tool_name(step.app), {}
    return f"plan:{step.do}", {}


_INTERNAL_ERROR = "internal error: the tool call failed unexpectedly"


def _read_group(steps: Sequence[plans.Step], start: int) -> list[plans.Ask | plans.Look]:
    """The reads from ``start`` that can run at once: next to each other, each
    to a different app (one call per app thread at a time). A look is always a
    read (#73)."""
    group: list[plans.Ask | plans.Look] = []
    for step in steps[start:]:
        if not isinstance(step, plans.Look) and not (
            isinstance(step, plans.Ask) and plans.reads_only(step)
        ):
            break
        if any(read.app == step.app for read in group):
            break
        group.append(step)
    return group


def _refusal(
    step: plans.Ask | plans.Open, filled: dataflow.Filled, *, confirms: bool
) -> str | None:
    """Why code won't run this step with what it was given, or ``None`` (#72)."""
    if filled.missing:
        return f"{filled.missing[0]} gave no name to pass on"
    if not filled.untrusted:
        return None
    source = filled.untrusted[0]
    if isinstance(step, plans.Open):
        return f"outside text from {source} can't go into a handoff"
    if step.destructive:
        return f"outside text from {source} can't choose what to delete"
    if not confirms:
        return f"outside text from {source} can't go to {step.app}, which doesn't confirm changes"
    return None


def _activity(
    kind: Literal["model", "tool"], *, tool: str | None = None, ack: str | None = None
) -> LoopActivity:
    from app.ai.loop import LoopActivity  # noqa: PLC0415

    return LoopActivity(kind=kind, iteration=1, tool=tool, ack=ack)


def _stopped(stop: Stop | None) -> bool:
    return stop is not None and stop.is_set()
