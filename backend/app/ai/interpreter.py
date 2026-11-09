"""The interpreter (#70, #72, #75): every conductor turn, compiled and run.

Conductor turns go planner → validator → interpreter → narrator only (#75).
The compiler writes the plan (:mod:`app.ai.compiler`) and code runs it, step
by step, on the turn runtime (:mod:`app.ai.runtime`: the registry's dispatch,
the delegate budget, Idempotency-Keys, receipts, Stop and the narrator). What
it runs, across apps (#72):

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
- ``recall`` (#80): memory over the ledger (:mod:`app.services.memory`): the
  person's words and the apps' receipts from every conversation and day,
  cited by day and conversation. No app, no model. ``{sN}`` on it is the one
  entity it resolved, if one clearly answers.
- ``ask(app="memory")`` (#81): conductor's own app, run by code
  (:mod:`app.services.facts`): remember, forget or list what the person
  saved, in their own words. No agent, no model, no budget, no gate.
- ``ask_user``: the plan pauses on the question; the person's answer resumes
  the same plan. Each option is resolved in code to the entity State named
  for it (#120, :mod:`app.ai.options`), so ``{s1}`` is the picked entity's
  title with its trust; an answer that picks no option goes in as they said it.
- ``{sN}`` in a step's text: the main title step N's receipt touched, quoted,
  filled in by code when the step runs (:mod:`app.ai.dataflow`, which holds
  the rules for no title, several titles and outside text).
- ``confirm``: yes or no to whatever is waiting.
- ``cancel``/``amend``/``undo``: on what's waiting or what just ran, over the
  ledger (:mod:`app.ai.control`, #74), each as a plan of its own; ``cancel``
  and ``amend`` on a running job go to the jobs runtime
  (:mod:`app.ai.job_control`, #76). One inside a longer plan is split off and
  runs first (#75); then the rest runs, unless it waits on a question the
  control step left or depended on what it acted on (then it is not run,
  and said).
- A step whose receipt carries a job ref detaches: the app accepted the work
  and the reply says so now ("queued 40"); the ledger's job row and the
  watcher (:mod:`app.services.job_watcher`) follow it from there.
- ``think`` (#78): handed to the background as a job conductor owns
  (:mod:`app.ai.think`, run by :mod:`app.services.thinker`) with what its
  ``inputs`` found; code replies at once. The plan ends at the think (#75):
  steps after it are not run, and said. Its proposal posts later as the
  job's note and waits like a gate. With the lane off (``CONDUCTOR_THINK``)
  the think is not run and code says so.
- The direct answers (:mod:`app.ai.speaker`), ``self`` from the ledger.

Independent reads (:func:`plans.reads_only`: asks to different apps, phrased
as questions, no refs; and looks) that sit next to each other run at once, each in its
own thread with the turn's context; everything else runs in order, one call
at a time, so there is never more than one writer in flight. The per-app
budget is charged under a lock and a Stop reaches every call in flight.

A step that fails ends the plan: the steps after it are on the record as not
run, and the narrator says so. In a parallel group the reads beside it have
already run; the plan ends after the group.

**No plan, no guess** (#75). A plan the validator still refuses after the
compiler's repair, or one with a ``{sN}`` no step can name, gets one more
round (:func:`compiler.repair_turn`). Still nothing runnable: code asks which
app was meant when the tries disagreed on it, else says plainly it couldn't
put the request together. A model outage is a failed turn (502), never a
question.

For a slow plan (one that asks an app) code speaks an honest ack as soon as the
plan is valid: what's about to happen, never what's done.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import structlog

from app import clock
from app.ai import (
    compiler,
    control,
    dataflow,
    facts,
    memory,
    narrator,
    options,
    run_record,
    speaker,
    think,
    turn_events,
)
from app.ai import glance as glances
from app.ai import plan as plans
from app.ai import trace as run_trace
from app.ai.confirmation import PendingConfirm
from app.ai.plan import Ledgered, PausedPlan, TurnPlan
from app.ai.provider import (
    ChatProvider,
    ChatResult,
    ProviderError,
    ProviderResponseError,
    ToolCall,
)
from app.ai.runtime import (
    ACTOR,
    DEFAULT_MAX_TOKENS,
    INTERNAL_ERROR,
    Activity,
    AgentRunFailed,
    AgentRunResult,
    OnActivity,
    Runtime,
    ToolCallRecord,
)
from app.ai.stop import STOPPED_ERROR, Stop, current_stop, use_stop
from app.config import get_settings
from app.fleet.context import current_delegation_context
from app.fleet.glance import GlanceError, GlanceReader, read_glance
from app.fleet.jobs import JobCanceller, cancel_job
from app.fleet.manifests import Fleet
from app.fleet.tools import ask_tool_name, look_tool_name, open_tool_name

logger = structlog.get_logger(__name__)

# How a recall step reads the ledger (#80): its query in, what memory found out.
MemoryReader = Callable[[str], memory.Recalled]
# The memory app (#81): the step's request and the person's words in; its
# result line and receipts out.
MemoryApp = Callable[[str, str], tuple[str, list[dict[str, Any]]]]

_AFTER_FAILURE = f"{run_record.NOT_RUN_PREFIX}an earlier step of the plan didn't finish"
# A think's proposal comes back later (#78), so nothing runs after it (#75).
_AFTER_THINK = (
    f"{run_record.NOT_RUN_PREFIX}nothing runs after handing a problem to the thinking model"
)
_THINK_OFF = f"{run_record.NOT_RUN_PREFIX}the thinking model is switched off here"
# The rest of a plan whose control step ran first (#75), when it can't go on.
_WAITS_ON_QUESTION = f"{run_record.NOT_RUN_PREFIX}the question before it needs your answer first"
_LEFT_IT = "Okay, I left it."
_SET_ASIDE = "Okay, I set that plan aside. Nothing changed."
_CONFIRM_TOOL = "confirm"
# Code's answers when no plan can run (#75): never a guessed action.
UNPLANNED = "I couldn't put that together. Try it in two parts?"
ONE_CONTROL = "I can only undo, cancel or change one thing at a time. Say them one by one?"
THINK_OFF = "I can't think that through here: the thinking model is switched off."
# What a control step did, as the reason the rest of its plan didn't run.
_ACTED_ON = {"cancel": "cancelled", "amend": "changed", "undo": "undone"}


@dataclass
class _Run:
    """One compiled turn's working state."""

    words: str
    actor: str
    trace: run_trace.RunTrace
    on_activity: OnActivity | None
    stop: Stop | None
    context: str | None = None
    records: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Whether ``words`` is what the person asked for. A resumed or confirmed
    # plan was told "the first one" or "yes": that restates nothing.
    restate: bool = True
    # What the ledger says control steps act on (#74), read only if needed.
    view: control.View = field(default_factory=lambda: control.View(None))
    # Each step's record by id, for a think's inputs (#78).
    found: dict[str, ToolCallRecord] = field(default_factory=dict)
    # Memory over the ledger (#80): a recall step's query in, what it found out.
    memory: MemoryReader | None = None
    # The memory app (#81): what an ask to ``memory`` reaches.
    memory_app: MemoryApp | None = None
    # The turn's plan, once compiled: what the ledger journals.
    plan: TurnPlan | None = None
    # Where this plan's records start: a control step split off and run first
    # (#75) has its own reply, so the rest's reply narrates only its own.
    since: int = 0
    # Steps after a think in the plan running now (#75): never run, said.
    dropped: list[plans.Step] = field(default_factory=list)


class Interpreter:
    """Compiles and runs every conductor turn over one provider and the fleet."""

    def __init__(
        self,
        provider: ChatProvider,
        fleet: Fleet,
        *,
        glance: GlanceReader | None = None,
        jobs: JobCanceller | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tracer: run_trace.Tracer | None = None,
        narrate: bool = False,
    ) -> None:
        self._rt = Runtime(provider, max_tokens=max_tokens, tracer=tracer, narrate=narrate)
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
        actor: str = ACTOR,
        on_activity: OnActivity | None = None,
        conversation_id: str | None = None,
        context: str | None = None,
        surface: str | None = None,
        stop: Stop | None = None,
        paused: PausedPlan | None = None,
        self_facts: Callable[[], str] | None = None,
        control_view: Callable[[], Ledgered] | None = None,
        recall: MemoryReader | None = None,
        memory_app: MemoryApp | None = None,
    ) -> AgentRunResult:
        """Run one turn. Always returns, except on a failure stop.

        Binds a request id (unless the caller bound one) and the turn's
        ``run_id`` for every log line, its result, its trace record and its
        captured model calls; with a tracer the turn is recorded however it
        ends, and a trace write that fails never costs the turn. ``context``
        (:mod:`app.ai.turn_context`: the state block and recall) opens the
        planner's message before the person's words. ``surface`` shapes the
        reply (on voice, two sentences and no markdown). ``stop`` is the
        turn's Stop (#66), checked at every boundary and bound for the tools
        so a delegate call in flight can pass it on. ``paused`` is last
        turn's plan waiting on the person, ``self_facts`` what code knows
        about its own recent work, ``control_view`` what the ledger says a
        control step acts on (#74), ``recall`` how a recall step reads
        memory (#80), ``memory_app`` what an ask to the built-in ``memory``
        app reaches (#81).

        A failure stop (``provider_error``, ``internal_error``) raises
        :class:`AgentRunFailed` with the partial turn and its plan.
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
            utterance=words,
            conversation_id=conversation_id,
            model=getattr(self._rt.provider, "model", None),
            context=context,
            surface=surface,
            system_prompt_sha256=run_trace.sha256(self._prompt),
        )
        state = _Run(
            words,
            actor,
            trace,
            on_activity,
            stop,
            context=context,
            view=control.View(control_view),
            memory=recall,
            memory_app=memory_app,
        )
        started = time.monotonic()
        with (
            structlog.contextvars.bound_contextvars(**bindings),
            run_trace.tracing(trace),
            use_stop(stop),
        ):
            try:
                result = self._turn(state, paused, self_facts)
            except AgentRunFailed as exc:
                exc.result.run_id, exc.result.plan = run_id, state.plan
                trace.plan = state.plan.model_dump() if state.plan else None
                self._rt.write_trace(trace, exc.result, str(exc), started)
                raise
            result.run_id, result.plan = run_id, state.plan
            trace.plan = state.plan.model_dump() if state.plan else None
            self._rt.write_trace(trace, result, "", started)
            return result

    def _turn(
        self, state: _Run, paused: PausedPlan | None, self_facts: Callable[[], str] | None
    ) -> AgentRunResult:
        """Compile the turn, then answer, resume, confirm, control or run it."""
        rt = self._rt
        if _stopped(state.stop):
            state.plan = TurnPlan(status="ran", note="stopped")
            return rt.stop("cancelled", [], [])
        pending = current_delegation_context().pending_confirm
        # Code's question and a think's proposal (#78) both take a yes that
        # runs a paused plan.
        gate = paused if paused is not None and paused.reason in ("gate", "proposal") else None
        checks = plans.Checks(
            fleet=self._fleet,
            calls_per_app=current_delegation_context().calls_per_turn_per_app,
            confirmable=pending is not None or gate is not None,
            resumable=paused is not None and paused.reason == "ask_user",
            context=state.context or "",
        )
        rt.emit(state.on_activity, Activity(kind="model"))
        started = time.monotonic()
        compiled = self._compile(state, checks)
        compile_ms = round((time.monotonic() - started) * 1000)
        state.messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": compiler.user_message(state.context, state.words)},
            *({"role": "assistant", "content": raw} for raw in compiled.raw),
        ]
        plan = compiled.plan
        if plan is None:
            state.plan = TurnPlan(
                status="invalid",
                note="; ".join(compiled.problems)[:500] or None,
                repaired=compiled.repaired,
            )
            return self._unplanned(state, compiled)
        body = plans.dump(plan)
        confirm_app = (gate.app if gate else None) or (pending.app if pending else None)
        logger.info(
            "plan_compiled", plan=plans.describe(plan), repaired=compiled.repaired, ms=compile_ms
        )
        ack = plans.ack(plan)
        if ack is not None:
            rt.emit(state.on_activity, Activity(kind="model", ack=ack))
        turn = TurnPlan(body=body, status="ran", ack=ack, repaired=compiled.repaired)
        state.plan = turn
        if isinstance(plan, plans.Resume) or (
            isinstance(plan, plans.StepPlan) and isinstance(plan.steps[0], plans.Confirm)
        ):
            state.restate = False
        if isinstance(plan, plans.Direct):
            turn.status = "answered"
            return self._answer(state, plan, self_facts)
        if isinstance(plan, plans.Resume):
            assert paused is not None  # the validator only allows a resume while paused
            resumed = plans.load(paused.body)
            assert isinstance(resumed, plans.StepPlan)
            turn.body = paused.body
            steps = resumed.steps[paused.resume_at :]
            asked = resumed.steps[paused.resume_at - 1 : paused.resume_at]
            offered = asked[0].options if asked and isinstance(asked[0], plans.AskUser) else []
            # The person's pick, as the entity State named for it (#120).
            value, picked = options.chosen(plan.resume, options.stored(paused.body, offered))
            values = {"s1": value}
            turn_events.resumed(
                "ask_user",
                step=asked[0].id if asked else None,
                answer=value.text,
                untrusted=value.untrusted,
                entity=picked.entity if picked is not None else None,
            )
            return self._execute(state, turn, steps, paused.resume_at, values)
        assert isinstance(plan, plans.StepPlan)
        controls = [step for step in plan.steps if step.do in plans.CONTROL_KINDS]
        if controls and len(plan.steps) > 1:
            return self._split(state, turn, plan, controls, paused=paused, pending=pending)
        first = plan.steps[0]
        if isinstance(first, plans.Confirm):
            return self._confirm(
                state,
                turn,
                plan,
                first,
                gate=gate,
                pending_app=confirm_app,
                undoing=control.undo_waiting(paused, pending),
            )
        if first.do in plans.CONTROL_KINDS:
            return control.run(self, state, turn, first, state.view, paused=paused, pending=pending)
        return self._execute(state, turn, plan.steps, 0, {})

    # --- compiling ---------------------------------------------------------------------------

    def _compile(self, state: _Run, checks: plans.Checks) -> compiler.Compiled:
        """The plan, after the compiler's repair and, when it still can't
        run, one more round (#75). A server answer that wasn't a completion
        counts as a reply that isn't a plan; an outage fails the turn."""
        chat = self._planner_chat(state.trace)
        try:
            try:
                compiled = compiler.compile_turn(
                    chat,
                    system_prompt=self._prompt,
                    context=state.context,
                    words=state.words,
                    fleet=self._fleet,
                    checks=checks,
                )
            except ProviderResponseError as exc:
                logger.warning("plan_compile_unusable", error=str(exc))
                compiled = compiler.Compiled(plan=None, problems=[_not_a_plan(exc)])
            found = compiled.problems if compiled.plan is None else plans.unrunnable(compiled.plan)
            if found:
                logger.info("plan_repair_again", problems=found)
                try:
                    compiled = compiler.repair_turn(
                        chat,
                        system_prompt=self._prompt,
                        context=state.context,
                        words=state.words,
                        fleet=self._fleet,
                        checks=checks,
                        previous=compiled,
                        found=found,
                    )
                except ProviderResponseError as exc:
                    logger.warning("plan_compile_unusable", error=str(exc))
                    compiled.plan, compiled.problems = None, [_not_a_plan(exc)]
        except ProviderError as exc:
            logger.error("plan_compile_failed", error=str(exc))
            raise self._rt.failed("provider_error", f"agent run failed: {exc}", [], []) from exc
        except Exception as exc:
            logger.exception("plan_compile_crashed", error_type=type(exc).__name__)
            raise self._rt.failed("internal_error", f"agent run crashed: {exc}", [], []) from exc
        return compiled

    def _unplanned(self, state: _Run, compiled: compiler.Compiled) -> AgentRunResult:
        """No plan code can run, after every repair (#75): code asks which app
        was meant when the tries disagreed on it, else says it couldn't put
        the request together. Nothing runs and nothing is guessed."""
        question = clarifying(compiled.raw)
        path = "code:clarify" if question else "code:unplanned"
        logger.info("plan_unplanned", problems=compiled.problems, clarify=question is not None)
        state.trace.narration = {"path": path, "appended": [], "model_calls": 0}
        reply = narrator.fit(question or UNPLANNED, state.trace.surface)
        return self._rt.finish("completed", reply, [], state.messages)

    # --- a control step inside a longer plan ------------------------------------------------

    def _split(
        self,
        state: _Run,
        turn: TurnPlan,
        plan: plans.StepPlan,
        controls: Sequence[plans.Step],
        *,
        paused: PausedPlan | None,
        pending: PendingConfirm | None,
    ) -> AgentRunResult:
        """A control step inside a longer plan (#75) runs first, as a plan of
        its own; then the rest runs. The rest is not run, and said, when the
        control step left a question waiting or the rest depended on what it
        acted on (a ``{sN}`` on it, or a confirm of what it cancelled)."""
        if len(controls) > 1:
            turn.note = "more than one control step"
            state.trace.narration = {"path": "code:control", "appended": [], "model_calls": 0}
            return self.say(state, ONE_CONTROL)
        step = controls[0]
        rest = [other for other in plan.steps if other is not step]
        first = TurnPlan(body={"steps": [step.model_dump()]}, status="ran", repaired=turn.repaired)
        turn.first = first
        turn.body = {"steps": [other.model_dump() for other in rest]}
        logger.info("plan_control_split", control=step.do, rest=len(rest))
        done = control.run(self, state, first, step, state.view, paused=paused, pending=pending)
        turn.first_steps = len(state.records)
        state.since = len(state.records)
        if done.stop_reason != "completed":
            self._skip(rest, state, STOPPED_ERROR)
            return self._rt.finish(done.stop_reason, done.reply, state.records, state.messages)
        why: str | None = None
        if first.status == "paused" or narrator.live_confirm(state.records) is not None:
            why = _WAITS_ON_QUESTION
        elif isinstance(rest[0], plans.Confirm) or any(
            step.id in plans.REF.findall(plans.step_text(other)) for other in rest
        ):
            why = f"{run_record.NOT_RUN_PREFIX}it depended on what was {_ACTED_ON[step.do]}"
        if why is not None:
            self._skip(rest, state, why)
            turn.note = why[:500]
            line = narrator.record_line(state.records[state.since :], state.trace.surface)
            reply = _joined(done.reply, line, state.trace.surface)
            return self._rt.finish("completed", reply, state.records, state.messages)
        state.restate = False  # the person's words were about the control step too
        after = self._execute(state, turn, rest, 0, {})
        reply = _joined(done.reply, after.reply, state.trace.surface)
        return self._rt.finish(after.stop_reason, reply, state.records, state.messages)

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
        return self._rt.finish("completed", reply, [], state.messages)

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
        if gate is not None and gate.reason == "proposal":
            return self._proposal(state, turn, gate, step.answer)
        if gate is not None:
            # Conductor's own question: a yes runs the paused plan from the gated step.
            turn.body = gate.body
            if step.answer == "no":
                turn.note = "declined at the gate"
                return self._rt.finish("completed", _LEFT_IT, [], state.messages)
            paused = plans.load(gate.body)
            assert isinstance(paused, plans.StepPlan)
            steps = paused.steps[gate.resume_at :]
            turn_events.resumed("gate", step=steps[0].id if steps else None)
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
            return self._rt.finish("completed", _LEFT_IT, [], state.messages)
        turn_events.resumed("app_confirm")
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
            return self._finish(state, turn, rest)
        return self._execute(state, turn, rest, 1, {})

    def _proposal(
        self, state: _Run, turn: TurnPlan, proposal: PausedPlan, answer: str
    ) -> AgentRunResult:
        """The person's answer to a think's proposal (#78). Yes runs it as a
        normal plan: a destructive ask to an app that doesn't confirm still
        stops at code's gate, and an app still asks its own question. No sets
        it aside; nothing runs."""
        turn.body = proposal.body
        turn.ends = proposal.plan_id
        if answer == "no":
            turn.ends_as = "set_aside"
            turn.note = "set the proposal aside"
            return self._rt.finish("completed", _SET_ASIDE, [], state.messages)
        turn.ends_as = "accepted"
        accepted = plans.load(proposal.body)
        assert isinstance(accepted, plans.StepPlan)
        ack = plans.ack(accepted)
        if ack is not None:
            turn.ack = ack
            self._rt.emit(state.on_activity, Activity(kind="model", ack=ack))
        turn_events.resumed("proposal")
        logger.info("proposal_accepted", plan=proposal.plan_id, steps=len(accepted.steps))
        return self._execute(state, turn, accepted.steps, 0, {})

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
        """Run ``steps``; ``offset`` is the first one's index in the plan.

        The plan ends at a think (#75): the steps after it are kept aside and
        go on the record as not run when the plan ends."""
        turn_events.plan(turn.body, offset)
        cut = next((n for n, step in enumerate(steps) if isinstance(step, plans.Think)), None)
        if cut is not None:
            state.dropped = list(steps[cut + 1 :])
            steps = steps[: cut + 1]
            if state.dropped:
                logger.info("plan_ends_at_think", dropped=[step.id for step in state.dropped])
        index = 0
        while index < len(steps):
            step = steps[index]
            position = offset + index
            if _stopped(state.stop):
                self._skip([*steps[index:], *state.dropped], state, STOPPED_ERROR)
                turn.note = f"stopped before {step.id}"
                return self._rt.stop("cancelled", state.records, state.messages)
            if isinstance(step, plans.AskUser):
                turn.status, turn.pause, turn.resume_at = "paused", "ask_user", position + 1
                if turn.body is not None:
                    turn.body = options.annotate(turn.body, step.options, state.context or "")
                return self._ask_user(state, step)
            group = _read_group(steps, index)
            if len(group) > 1:
                results = self._dispatch_all(state, [_call_for(s) for s in group])
                for n, read in enumerate(group):
                    if isinstance(read, plans.Look) and not self._went_well(results[n]):
                        results[n] = self._look_fell_back(state, read, results[n], steps)
                for read, got in zip(group, results, strict=True):
                    self._keep(state, values, read, got)
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
                    return self._finish(state, turn, steps[after:])
                index = after
                continue
            if isinstance(step, plans.Ask):
                filled = dataflow.fill(step.request, values)
                app = self._fleet.get(step.app)
                confirms = app is not None and app.agent is not None and app.agent.confirms
                refusal = _refusal(step, filled, confirms=confirms)
                if refusal is not None:
                    return self._refuse(state, turn, steps, index, filled, refusal)
                if step.app == facts.APP:
                    record: ToolCallRecord | None = self._memory_app(state, filled.text)
                elif step.destructive and not confirms and step.id != gate_passed:
                    turn.status, turn.pause = "paused", "gate"
                    turn.resume_at, turn.pause_app = position, step.app
                    self._carry(turn, position, values)
                    return self._gate(state, step, filled.text)
                else:
                    record = self._dispatch(
                        state, ask_tool_name(step.app), {"message": filled.text}
                    )
                if record is not None and record.app_confirm and record.app_confirm.get("question"):
                    return self._app_asked(state, turn, steps, position + 1, step, values)
            elif isinstance(step, plans.Look):
                record = self._dispatch(state, look_tool_name(step.app), {})
                if record is not None and not self._went_well(record):
                    record = self._look_fell_back(state, step, record, steps)
                    if record.app_confirm and record.app_confirm.get("question"):
                        return self._app_asked(state, turn, steps, position + 1, step, values)
            elif isinstance(step, plans.Think):
                return self._think(state, turn, step, values)
            elif isinstance(step, plans.Recall):
                record = self._recall(state, step)
            elif isinstance(step, plans.Open):
                filled = dataflow.fill(step.intent, values)
                refusal = _refusal(step, filled, confirms=False)
                if refusal is not None:
                    return self._refuse(state, turn, steps, index, filled, refusal)
                record = self._dispatch(state, open_tool_name(step.app), {"intent": filled.text})
            else:  # confirm and the control steps are handled before a plan runs
                raise AssertionError(f"the interpreter can't run {step.do} here")
            if record is None or not self._went_well(record):
                turn.note = f"stopped after {step.id}"
                return self._finish(state, turn, steps[index + 1 :])
            self._keep(state, values, step, record)
            index += 1
        return self._finish(state, turn, [])

    @staticmethod
    def _keep(
        state: _Run, values: dict[str, dataflow.Value], step: plans.Step, record: ToolCallRecord
    ) -> None:
        """What a step gives later steps as ``{sN}``, from its receipt; and
        its record, for a think's inputs (#78)."""
        state.found[step.id] = record
        value = (
            dataflow.from_recall(record.app_record)
            if memory.is_recall(record.tool)
            else dataflow.from_record(record.app_record)
        )
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
            state.dropped = []  # still in the paused plan: said when it gets there
        return self._finish(state, turn, [])

    def _think(
        self,
        state: _Run,
        turn: TurnPlan,
        step: plans.Think,
        values: dict[str, dataflow.Value],
    ) -> AgentRunResult:
        """Hand a think to the background (#78): its record carries the job
        and what its inputs found; the reply is code's, at once. The thinker
        runs it on the 26B later; nothing here calls a model. The plan ends
        here: steps after it are not run, and said (#75). With the lane off
        the think is not run either, and code says so."""
        record = think.record(step, state.found, values)
        off = not get_settings().conductor_think
        if off:
            record = ToolCallRecord(
                tool=record.tool,
                arguments={"problem": record.arguments["problem"], "inputs": list(step.inputs)},
                error=_THINK_OFF,
            )
        self._rt.emit(state.on_activity, Activity(kind="tool", tool=think.TOOL))
        seq = len(state.records) + 1
        turn_events.step_started(seq, record.tool, record.arguments)
        earlier = list(state.records[state.since :])
        state.records.append(record)
        turn_events.receipt(seq, record)
        after = len(state.records)
        self._skip(state.dropped, state, _AFTER_THINK)
        state.dropped = []
        line = narrator.record_line([*earlier, *state.records[after:]], state.trace.surface)
        if off:
            turn.note = "think not run: the lane is off"
            logger.info("plan_think_off")
            state.trace.narration = {"path": "code:think_off", "appended": [], "model_calls": 0}
            reply = narrator.fit(f"{line} {THINK_OFF}".strip(), state.trace.surface)
            return self._rt.finish("completed", reply, state.records, state.messages)
        ref = think.ref_of(record)
        turn.note = f"thinking in the background ({ref})"
        logger.info("plan_think_detached", ref=ref, inputs=list(step.inputs))
        reply = think.thinking_reply(line, state.trace.surface)
        state.trace.narration = {"path": "code:think", "appended": [], "model_calls": 0}
        return self._rt.finish("completed", reply, state.records, state.messages)

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
        return self._finish(state, turn, steps[index + 1 :])

    def _finish(self, state: _Run, turn: TurnPlan, unrun: Sequence[plans.Step]) -> AgentRunResult:
        """End the plan: what didn't run goes on the record with why, then the
        narrator speaks from this plan's records (a stopped turn: the record)."""
        self._skip(unrun, state, _AFTER_FAILURE)
        self._skip(state.dropped, state, _AFTER_THINK)
        state.dropped = []
        if _stopped(state.stop):
            return self._rt.stop("cancelled", state.records, state.messages)
        reply = self._rt.reply(
            state.words, state.records[state.since :], state.trace, state.on_activity
        )
        return self._rt.finish("completed", reply, state.records, state.messages)

    def _dispatch(self, state: _Run, tool: str, arguments: dict[str, Any]) -> ToolCallRecord | None:
        """One step through the runtime's dispatch; a crash fails the turn."""
        self._rt.emit(state.on_activity, Activity(kind="tool", tool=tool))
        turn_events.step_started(len(state.records) + 1, tool, arguments)
        call = ToolCall(id=f"plan-{len(state.records) + 1}", name=tool, arguments=arguments)
        try:
            record = self._call(call, state.actor)
        except Exception as exc:
            logger.exception("plan_step_crashed", tool=tool)
            state.records.append(
                ToolCallRecord(tool=tool, arguments=arguments, error=INTERNAL_ERROR)
            )
            raise self._rt.failed(
                "internal_error", f"agent run crashed in {tool}: {exc}", state.records, []
            ) from exc
        state.records.append(record)
        turn_events.receipt(len(state.records), record)
        return record

    def _dispatch_all(
        self, state: _Run, calls: Sequence[tuple[str, dict[str, Any]]]
    ) -> list[ToolCallRecord]:
        """Independent reads at once (#72): one thread each, each running in a
        copy of the turn's context (its delegation context, Stop and trace),
        records kept in plan order. A crash fails the turn, once every call
        has come back."""
        first = len(state.records)
        tool_calls = [
            ToolCall(id=f"plan-{first + n + 1}", name=tool, arguments=arguments)
            for n, (tool, arguments) in enumerate(calls)
        ]
        for n, call in enumerate(tool_calls):
            self._rt.emit(state.on_activity, Activity(kind="tool", tool=call.name))
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
                record = future.result()
            except Exception as error:
                logger.exception("plan_step_crashed", tool=call.name)
                record = ToolCallRecord(
                    tool=call.name, arguments=call.arguments, error=INTERNAL_ERROR
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
            raise self._rt.failed(
                "internal_error", f"agent run crashed in {tool}: {cause}", state.records, []
            ) from cause
        return records

    def _call(self, call: ToolCall, actor: str) -> ToolCallRecord:
        """One call: a look is read here, never through the registry;
        everything else is the runtime's dispatch."""
        if run_record.is_look(call.name):
            return self._look(call)
        return self._rt.dispatch(call, actor)

    def _look(self, call: ToolCall) -> ToolCallRecord:
        """Read one app's glance: a ``looked_up`` record built by code, or the
        reason it couldn't be read. No model call, no delegate budget."""
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
        return record

    def _recall(self, state: _Run, step: plans.Recall) -> ToolCallRecord:
        """Read memory over the ledger (#80): no app, no model, no budget. The
        record is code's: one ``looked_up`` entry per cited turn, or "nothing
        on record" (still a read that went well; a later ``{sN}`` on it has
        nothing to pass on)."""
        arguments = {"query": step.query}
        self._rt.emit(state.on_activity, Activity(kind="tool", tool=memory.RECALL_TOOL))
        turn_events.step_started(len(state.records) + 1, memory.RECALL_TOOL, arguments)
        record = ToolCallRecord(tool=memory.RECALL_TOOL, arguments=arguments)
        if state.memory is None:
            record.error = "couldn't look back: memory isn't available here"
        else:
            try:
                found = state.memory(step.query)
                record.result, record.app_record = memory.record(found, clock.local_today())
            except Exception as exc:
                logger.warning("plan_recall_failed", error=str(exc), error_type=type(exc).__name__)
                record.error = "couldn't look back: memory couldn't be read"
        logger.info("plan_recall", hits=len(record.app_record or []), ok=record.error is None)
        state.records.append(record)
        turn_events.receipt(len(state.records), record)
        return record

    def _memory_app(self, state: _Run, request: str) -> ToolCallRecord:
        """An ask to ``memory`` (#81): run by code over the saved facts, filed
        as ``ask_memory`` with v1.1 receipts like any app's. The fact comes
        from the person's own words, so a turn that only said "yes" gives
        none."""
        arguments = {"message": request}
        self._rt.emit(state.on_activity, Activity(kind="tool", tool=facts.TOOL))
        turn_events.step_started(len(state.records) + 1, facts.TOOL, arguments)
        record = ToolCallRecord(tool=facts.TOOL, arguments=arguments)
        if state.memory_app is None:
            record.error = "couldn't reach memory: it isn't available here"
        else:
            try:
                said = state.words if state.restate else ""
                record.result, record.app_record = state.memory_app(request, said)
            except Exception as exc:
                logger.warning("plan_memory_failed", error=str(exc), error_type=type(exc).__name__)
                record.error = "couldn't reach memory: it couldn't be read or written"
        logger.info("plan_memory", ok=record.error is None)
        state.records.append(record)
        turn_events.receipt(len(state.records), record)
        return record

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
        for step in steps:
            tool, arguments = _call_for(step)
            state.records.append(ToolCallRecord(tool=tool, arguments=arguments, error=error))
            turn_events.receipt(len(state.records), state.records[-1])

    # --- questions -------------------------------------------------------------------------

    def _ask_user(self, state: _Run, step: plans.AskUser) -> AgentRunResult:
        question = step.question.strip()
        offered = [option.strip() for option in step.options if option.strip()]
        if offered:
            question = f"{question} ({' / '.join(offered)})"
        turn_events.paused("ask_user", question=question, step=step.id)
        reply = self._with_record_line(state, question)
        return self._rt.finish("completed", reply, state.records, state.messages)

    def _gate(self, state: _Run, step: plans.Ask, request: str) -> AgentRunResult:
        question = f'Just to check: should I ask {step.app} to "{request}"?'
        turn_events.paused("gate", question=question, app=step.app, step=step.id)
        reply = self._with_record_line(state, question)
        return self._rt.finish("completed", reply, state.records, state.messages)

    def _with_record_line(self, state: _Run, question: str) -> str:
        line = narrator.record_line(state.records[state.since :], state.trace.surface)
        return f"{line} {question}" if line else question

    def say(self, state: _Run, reply: str) -> AgentRunResult:
        """Code's own reply for a control step: plain facts, no model call."""
        return self._rt.finish(
            "completed", narrator.fit(reply, state.trace.surface), state.records, state.messages
        )

    # --- model calls -----------------------------------------------------------------------

    def _planner_chat(self, trace: run_trace.RunTrace) -> compiler.PlannerChat:
        rt = self._rt

        def chat(
            messages: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any]
        ) -> ChatResult:
            return rt.chat(
                messages,
                trace,
                max_tokens=max_tokens,
                role="compiler",
                response_format=response_format,
            )

        return chat

    def _structured_chat(self, state: _Run) -> speaker.StructuredChat:
        rt = self._rt

        def chat(
            messages: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any]
        ) -> ChatResult:
            rt.emit(state.on_activity, Activity(kind="model"))
            return rt.chat(
                messages,
                state.trace,
                max_tokens=max_tokens,
                role="speaker",
                response_format=response_format,
            )

        return chat

    def _spoken_chat(self, state: _Run) -> narrator.Chat:
        rt = self._rt

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            rt.emit(state.on_activity, Activity(kind="model"))
            return rt.chat(messages, state.trace, max_tokens=max_tokens, role="speaker")

        return chat


def conductor_from_settings(provider: ChatProvider, fleet: Fleet) -> Interpreter:
    """The turn runtime as configured: the completion cap, the tracer and the
    narrator switch. The one place settings reach a turn."""
    settings = get_settings()
    return Interpreter(
        provider,
        fleet,
        max_tokens=settings.conductor_max_tokens,
        tracer=run_trace.tracer_from_settings(),
        narrate=settings.conductor_narrator,
    )


def _call_for(step: plans.Step) -> tuple[str, dict[str, Any]]:
    """The tool call a step would have been, for the record of steps not run."""
    if isinstance(step, plans.Ask):
        return ask_tool_name(step.app), {"message": step.request}
    if isinstance(step, plans.Open):
        return open_tool_name(step.app), {"intent": step.intent}
    if isinstance(step, plans.Look):
        return look_tool_name(step.app), {}
    if isinstance(step, plans.Recall):
        return memory.RECALL_TOOL, {"query": step.query}
    return f"plan:{step.do}", {}


def _read_group(steps: Sequence[plans.Step], start: int) -> list[plans.Ask | plans.Look]:
    """The reads from ``start`` that can run at once: next to each other, each
    to a different app (one call per app thread at a time). A look is always a
    read (#73)."""
    group: list[plans.Ask | plans.Look] = []
    for step in steps[start:]:
        if not isinstance(step, plans.Look) and not (
            isinstance(step, plans.Ask) and plans.reads_only(step) and step.app != facts.APP
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


def clarifying(raws: Sequence[str]) -> str | None:
    """Code's question when the planner's tries disagreed on which app the
    request is for (#75): each try a plan to one app, two different apps.
    ``None`` when there's no such ambiguity to name."""
    picks: list[tuple[str, str]] = []
    for raw in raws:
        try:
            plan = plans.parse(raw)
        except ValueError:
            continue
        if not isinstance(plan, plans.StepPlan):
            continue
        touching = [step for step in plan.steps if isinstance(step, plans.Ask | plans.Open)]
        if not touching or len({step.app for step in touching}) != 1:
            continue
        step = touching[0]
        if any(app == step.app for app, _ in picks):
            continue
        said = step.request if isinstance(step, plans.Ask) else step.intent
        picks.append((step.app, " ".join(plans.REF.sub("that", said).split())))
    if len(picks) < 2:
        return None
    (first, said_first), (second, said_second) = picks[:2]
    if 0 < len(said_first) <= _QUOTED and 0 < len(said_second) <= _QUOTED:
        return f'Do you mean "{said_first}" in {first}, or "{said_second}" in {second}?'
    return f"Do you mean that for {first} or for {second}?"


# How long a planner's request can be to quote in code's question.
_QUOTED = 60


def _not_a_plan(exc: Exception) -> str:
    return f"not a plan: {str(exc).splitlines()[0][:200]}"


def _joined(first: str | None, second: str | None, surface: str | None) -> str | None:
    """Two replies of one turn (a control step's, then the rest's); on voice
    the first keeps its first sentence."""
    if first and surface in narrator.SPOKEN:
        sentences = narrator.fit(first, surface).split(". ")
        first = sentences[0] if len(sentences) == 1 else sentences[0] + "."
    parts = [part.strip() for part in (first, second) if part and part.strip()]
    return " ".join(parts) or None


def _stopped(stop: Stop | None) -> bool:
    return stop is not None and stop.is_set()
