"""The interpreter (#70, #72, #75): every conductor turn, compiled and run.

Conductor turns go planner → validator → interpreter → narrator only (#75).
The compiler writes the plan (:mod:`app.ai.compiler`) and code runs it, step
by step, on the turn runtime (:mod:`app.ai.runtime`: the registry's dispatch,
the delegate budget, Idempotency-Keys, receipts, Stop and the narrator). What
it runs, across apps (#72):

- ``ask``: the app's agent, through its ``ask_<app>`` tool. A ``destructive``
  ask to an app that doesn't confirm its own writes pauses first on code's
  question (the backstop gate); a tap on its card runs it. An app's
  ``confirm_required`` is a card per question, the same way.
  An ask whose words clearly point at a saved fact (#134,
  :func:`app.ai.facts.related`) carries it in its ``facts`` argument; the
  tool sends it after the request, quoted and marked as context.
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
- ``ask_user``: the plan pauses on the question; the person's pick resumes
  the same plan. Each option is resolved in code to the entity State named
  for it (#120, :mod:`app.ai.options`), so ``{s1}`` is the picked entity's
  title with its trust. One with no options waits on their words instead.
- ``{sN}`` in a step's text: the main title step N's receipt touched, quoted,
  filled in by code when the step runs (:mod:`app.ai.dataflow`, which holds
  the rules for no title, several titles and outside text).
- ``cancel``/``amend``/``undo``: on what's waiting or what just ran, over the
  ledger (:mod:`app.ai.control`, #74), each as a plan of its own; ``cancel``
  and ``amend`` on a running job go to the jobs runtime
  (:mod:`app.ai.job_control`, #76). One inside a longer plan is split off and
  runs first (#75); then the rest runs, unless it waits on a question the
  control step left (then it is not run, and said).
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
at a time, so there is never more than one writer in flight. A Stop reaches
every call in flight.

A step that fails ends the plan: the steps after it are on the record as not
run, and the narrator says so. In a parallel group the reads beside it have
already run; the plan ends after the group.

**No plan, no guess** (#75). A plan the validator still refuses after the
compiler's repair, or one with a ``{sN}`` no step can name, gets one more
round (:func:`compiler.repair_turn`). Still nothing runnable: code asks which
app was meant when the tries disagreed on it; else, for a request the planner
never shaped into steps (#146), files a ``think`` for the person's own words,
as a planner-planned one is filed, and says it is thinking it over; else (the
lane off, an answer to a waiting question, a yes with nothing waiting) says
plainly it couldn't put the request together. A model outage is a failed turn
(502), never a question and never a think.

For a slow plan (one that asks an app) code speaks an honest ack as soon as the
plan is valid: what's about to happen, never what's done.

**Questions never block the chat** (#251). A question is a card, and only a
tap approves it: there is no ``confirm`` step, and a bare "yes" with a card
waiting gets code's :data:`TAP_THE_CARD`. Code, never the planner, reads the
person's words against the question asked in the turn right before: an
option's name picks it, "never mind" dismisses it, and a correction ("no,
make it Saturday", ``plans.changes_what_waits``) amends it, so the question
is asked again on a new card.
Anything else is a new request and the card stays. Only the steps that need
the answer wait with it (:func:`plans.waiting_on`); the rest of the plan runs
at once. A tap resumes the waiting steps with no chat turn
(:meth:`Interpreter.resume`).
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import structlog

from app import clock
from app.ai import (
    compiler,
    confirmation,
    control,
    dataflow,
    facts,
    memory,
    narrator,
    options,
    relay,
    run_record,
    speaker,
    think,
    turn_events,
)
from app.ai import glance as glances
from app.ai.state_block import Entity
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
from app.fleet import context as delegation
from app.fleet.context import current_delegation_context
from app.fleet.glance import GlanceError, GlanceReader, read_glance
from app.fleet.jobs import JobCanceller, cancel_job
from app.fleet.manifests import Fleet
from app.fleet.tools import ask_tool_name, look_tool_name, open_tool_name

logger = structlog.get_logger(__name__)

# Delegate v1.2 (#202): how many of an app's own recent entities ride its ask
# as ``context.refs``, and the contract's cap on refs per message.
STATE_REFS = 5
MAX_CONTEXT_REFS = 20
# Where a paused body keeps a v1.2 app's ``needs_input`` candidates (#196).
APP_CANDIDATES = "app_candidates"
# ... and the app thread that asked an ``app_question`` (#262): the answer
# goes back there; every other ask opens a fresh one.
APP_THREAD = "app_thread"

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
# How the person's answer to an app's question rides after the step that asked
# it, when the paused plan resumes (#168).
APP_ANSWER_LEAD = "the person's answer to your question"
_THINK_OFF = f"{run_record.NOT_RUN_PREFIX}the thinking model is switched off here"
# The rest of a plan whose control step ran first (#75), when it can't go on.
_WAITS_ON_QUESTION = f"{run_record.NOT_RUN_PREFIX}the question before it needs your answer first"
_SET_ASIDE = "Okay, I set that plan aside. Nothing changed."
_CONFIRM_TOOL = "confirm"
# No words approve anything (#251).
TAP_THE_CARD = "I can't take a yes in words. Tap the card to approve it."
# Where a paused body keeps the ids of the steps that wait with its question
# (#251): the rest of the plan already ran.
WAITING = "waiting"
# ... and the question it asked, word for word, for its card.
QUESTION = "question"
# Code's answers when no plan can run (#75): never a guessed action.
UNPLANNED = "I couldn't put that together. Try it in two parts?"
ONE_CONTROL = "I can only undo, cancel or change one thing at a time. Say them one by one?"
THINK_OFF = "I can't think that through here: the thinking model is switched off."
# Code's lead when it files the think itself (#146); the think's own line follows.
UNPLANNED_THINKS = "I couldn't put that together as steps."


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
    # The saved facts (#134), newest first: read once, when an app ask needs them.
    saved_facts: Callable[[], Sequence[str]] | None = None
    saved: list[str] | None = None
    # The one saved fact a request means, when it shares no words with any
    # (#204, :func:`app.services.fact_meaning.nearest`): None, words alone.
    fact_meaning: Callable[[str, Sequence[str]], str | None] | None = None
    # What this window's steps touched, newest first (#202): a v1.2 app's ask
    # carries its own as ``context.refs``. Read once, when an ask needs them.
    entities: Callable[[], Sequence[Entity] | None] | None = None
    touched: list[Entity] | None = None
    # The candidate the person's answer picked from a v1.2 app's question
    # (#196): the resumed ask carries it, marked ``picked``, as its only ref.
    picked: dict[str, Any] | None = None
    # The step that answers an app's own question, its app and the thread
    # that asked (#262): that step's call goes there, and only that one.
    answering: tuple[str, str, int] | None = None
    # The turn's plan, once compiled: what the ledger journals.
    plan: TurnPlan | None = None
    # Where this plan's records start: a control step split off and run first
    # (#75) has its own reply, so the rest's reply narrates only its own.
    since: int = 0
    # Steps after a think in the plan running now (#75): never run, said.
    dropped: list[plans.Step] = field(default_factory=list)
    # A card waits somewhere in this window, also an older one (#251).
    cards_waiting: bool = False
    # Running the steps a question doesn't hold up (#251): nothing is
    # narrated until the plan's own reply.
    holding: int = 0


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
        saved_facts: Callable[[], Sequence[str]] | None = None,
        entities: Callable[[], Sequence[Entity] | None] | None = None,
        fact_meaning: Callable[[str, Sequence[str]], str | None] | None = None,
        cards_waiting: bool = False,
        resume: Resume | None = None,
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
        app reaches (#81), ``saved_facts`` the person's saved facts, newest
        first, which an app ask whose words point at one carries (#134), and
        ``fact_meaning`` the one a non-read ask means when no words match (#204).
        ``cards_waiting``: a card waits in this window, the turn right before
        or older, so a bare yes gets "tap the card" (#251). ``resume`` is a
        card's tap instead of words (:meth:`resume`): no plan is compiled.

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
            saved_facts=saved_facts,
            entities=entities,
            fact_meaning=fact_meaning,
            cards_waiting=cards_waiting,
        )
        started = time.monotonic()
        with (
            structlog.contextvars.bound_contextvars(**bindings),
            run_trace.tracing(trace),
            use_stop(stop),
        ):
            try:
                if resume is not None:
                    result = self._resume(state, resume)
                else:
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
        """Read the words against what the turn before asked (code's: a yes, a
        pick, a dismissal, #251); else compile the turn, then answer, resume,
        control or run it."""
        rt = self._rt
        if _stopped(state.stop):
            state.plan = TurnPlan(status="ran", note="stopped")
            return rt.stop("cancelled", [], [])
        pending = current_delegation_context().pending_confirm
        # What the turn right before asked (#251): a card (an app's question,
        # code's gate, a proposal, a pick-one), or an open question only words
        # can answer. Their words may pick, change or dismiss it; nothing they
        # say approves it.
        card = pending is not None or (paused is not None and _on_a_card(paused))
        opened = paused is not None and not _on_a_card(paused)
        if plans.bare_yes(state.words, just_asked=card):
            if card or state.cards_waiting:
                return self._tap_the_card(state)
            if not opened:
                # Nothing waits at all: a yes answers nothing. Small talk, by
                # code: the planner has no step for a yes (#251, #176).
                return self._yes_to_nothing(state, self_facts)
        if paused is not None and paused.reason in ("ask_user", "app_question"):
            picked = self._picked(state, paused)
            if picked is not None:
                return picked
        if (card or opened) and plans.dismisses(state.words):
            return self._dismiss(state, paused, pending)
        if card and plans.changes_what_waits(state.words):
            return self._correct(state, paused, pending)
        checks = plans.Checks(
            fleet=self._fleet,
            asked=card,
            resumable=opened,
            context=state.context or "",
            # A correction with nothing waiting (#198): an amend, when the
            # ledger has a write the last plan finished.
            changeable=paused is None
            and pending is None
            and plans.is_correction(state.words)
            and control.changeable(state.view),
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
            return self._unplanned(state, compiled, waiting=checks.asked or checks.resumable)
        body = plans.dump(plan)
        logger.info(
            "plan_compiled", plan=plans.describe(plan), repaired=compiled.repaired, ms=compile_ms
        )
        ack = plans.ack(plan)
        if ack is not None:
            rt.emit(state.on_activity, Activity(kind="model", ack=ack))
        turn = TurnPlan(body=body, status="ran", ack=ack, repaired=compiled.repaired)
        state.plan = turn
        if isinstance(plan, plans.Resume):
            state.restate = False
        if isinstance(plan, plans.Direct):
            turn.status = "answered"
            return self._answer(state, plan, self_facts)
        if isinstance(plan, plans.Resume):
            assert paused is not None  # the validator only allows a resume while paused
            turn.ends, turn.ends_as = paused.plan_id, "resumed"
            if paused.reason == "app_question":
                return self._resume_app_question(state, turn, paused)
            value, _ = options.chosen(plan.resume, [])
            return self._resume_ask_user(state, turn, paused, value, None)
        assert isinstance(plan, plans.StepPlan)
        # The planner's cancel never closes a card (#251): it reaches what an
        # open question left waiting, or a running job.
        reach = paused if opened else None
        controls = [step for step in plan.steps if step.do in plans.CONTROL_KINDS]
        if controls and len(plan.steps) > 1:
            return self._split(
                state, turn, plan, controls, paused=paused, pending=pending, reach=reach
            )
        first = plan.steps[0]
        if isinstance(first, plans.Cancel):
            return control.run(
                self, state, turn, first, state.view, paused=reach, pending=None, cards=card
            )
        if first.do in plans.CONTROL_KINDS:
            return control.run(self, state, turn, first, state.view, paused=paused, pending=pending)
        return self._execute(state, turn, plan.steps, 0, {})

    # --- code's reading of an answer (#251) --------------------------------------------------

    def _tap_the_card(self, state: _Run) -> AgentRunResult:
        """A bare yes with a card waiting: nothing is planned and nothing is
        approved. Code says how to approve it."""
        state.plan = TurnPlan(status="answered", note="a yes approves nothing: tap the card")
        state.trace.narration = {"path": "code:tap_the_card", "appended": [], "model_calls": 0}
        logger.info("yes_approves_nothing")
        return self.say(state, TAP_THE_CARD)

    def _yes_to_nothing(self, state: _Run, self_facts: Callable[[], str] | None) -> AgentRunResult:
        """A bare yes with nothing waiting anywhere: the speaker answers it
        as small talk, with no planner call. Nothing runs."""
        plan = plans.Direct(answer="chat")
        state.plan = TurnPlan(
            body=plans.dump(plan), status="answered", note="a yes with nothing waiting"
        )
        logger.info("yes_with_nothing_waiting")
        return self._answer(state, plan, self_facts)

    def _picked(self, state: _Run, paused: PausedPlan) -> AgentRunResult | None:
        """The person's words name one of the options the turn right before
        offered: the same plan resumes with it, matched in code. ``None``
        when they pick none (then they are a correction or a new request)."""
        if paused.reason == "app_question":
            candidates = [c for c in paused.body.get(APP_CANDIDATES) or [] if isinstance(c, dict)]
            titles = [str(c.get("title") or "") for c in candidates]
            if options.answers(state.words, titles) is None:
                return None
            turn = self._resumed_turn(state, paused)
            return self._resume_app_question(state, turn, paused)
        offered = _offered(paused)
        if not offered:
            return None
        stored = options.stored(paused.body, offered)
        if options.answers(state.words, [option.option for option in stored]) is None:
            return None
        value, picked = options.chosen(state.words, stored)
        turn = self._resumed_turn(state, paused)
        return self._resume_ask_user(state, turn, paused, value, picked)

    def _resumed_turn(self, state: _Run, paused: PausedPlan) -> TurnPlan:
        """The plan row of a turn that picks up a waiting plan with no
        planner call: the waiting row ends as ``resumed``."""
        turn = TurnPlan(body=paused.body, status="ran", ends=paused.plan_id, ends_as="resumed")
        state.plan, state.restate = turn, False
        return turn

    def _dismiss(
        self, state: _Run, paused: PausedPlan | None, pending: PendingConfirm | None
    ) -> AgentRunResult:
        """ "Never mind" to what the turn right before asked: that question
        and what waited on it are dropped, nothing older. A cancel that code
        compiled from the words, so a running job they name still takes it."""
        step = plans.Cancel(id="s1", do="cancel", target="pending")
        turn = TurnPlan(body={"steps": [step.model_dump()]}, status="ran")
        state.plan, state.restate = turn, False
        state.trace.narration = {"path": "code:dismiss", "appended": [], "model_calls": 0}
        logger.info("question_dismissed", plan=paused.plan_id if paused else None)
        if paused is not None and paused.reason == "proposal":
            # A think's proposal (#78) is set aside; nothing of it ran.
            turn.ends, turn.ends_as = paused.plan_id, "set_aside"
            turn.note = "set the proposal aside"
            return self.say(state, _SET_ASIDE)
        return control.run(self, state, turn, step, state.view, paused=paused, pending=pending)

    def _correct(
        self, state: _Run, paused: PausedPlan | None, pending: PendingConfirm | None
    ) -> AgentRunResult:
        """A spoken correction of what the turn right before asked (#77: "no,
        Saturday", "no, just the paint cabinets task"): the amend, compiled
        by code from their own words. The waiting step is asked again with
        the change, on a new card. (The planner's own amend still reaches
        here for a change these words don't show.)"""
        step = plans.Amend(id="s1", do="amend", target="pending", change=state.words)
        turn = TurnPlan(body={"steps": [step.model_dump()]}, status="ran")
        state.plan, state.restate = turn, False
        logger.info("question_corrected", plan=paused.plan_id if paused else None)
        return control.run(self, state, turn, step, state.view, paused=paused, pending=pending)

    def _resume_ask_user(
        self,
        state: _Run,
        turn: TurnPlan,
        paused: PausedPlan,
        value: dataflow.Value,
        picked: options.Option | None,
    ) -> AgentRunResult:
        """The plan an ``ask_user`` paused goes on with the answer as
        ``{s1}``: the picked option's entity (#120), or their words."""
        resumed = plans.load(paused.body)
        assert isinstance(resumed, plans.StepPlan)
        turn.body = paused.body
        steps = _waiting_steps(resumed, paused)
        asked = resumed.steps[paused.resume_at - 1 : paused.resume_at]
        if picked is not None and steps and plans.changes_too(state.words, value.text):
            # "No, the bill one on Saturday" (#198): the pick fills {s1}
            # and the rest of their words change the step it resumes.
            target = steps[0]
            if isinstance(target, plans.Ask | plans.Open):
                changed = control.changed(target, state.words)
                steps = [changed, *steps[1:]]
                kept = [changed if s.id == changed.id else s for s in resumed.steps]
                turn.body = {**paused.body, "steps": [s.model_dump() for s in kept]}
        turn_events.resumed(
            "ask_user",
            step=asked[0].id if asked else None,
            answer=value.text,
            untrusted=value.untrusted,
            entity=picked.entity if picked is not None else None,
        )
        return self._execute(state, turn, steps, paused.resume_at, {"s1": value})

    # --- a card's tap (#251) -----------------------------------------------------------------

    def resume(self, tapped: Resume, **turn: Any) -> AgentRunResult:
        """Pick a waiting plan up from its card, with no chat turn: the steps
        that waited on the question run now. ``turn`` is :meth:`run`'s
        keywords; the words are the request that plan came from, for the
        narrator, and restate nothing."""
        return self.run(tapped.words, resume=tapped, **turn)

    def _resume(self, state: _Run, tapped: Resume) -> AgentRunResult:
        paused = tapped.paused
        state.restate = False
        if _stopped(state.stop):
            state.plan = TurnPlan(status="ran", note="stopped")
            return self._rt.stop("cancelled", [], [])
        turn = self._resumed_turn(state, paused)
        turn.ends_as = "accepted"
        if paused.reason == "proposal":
            return self._proposal(state, turn, paused)
        body = plans.load(paused.body)
        assert isinstance(body, plans.StepPlan)
        steps = _waiting_steps(body, paused)
        if paused.reason == "gate":
            turn_events.resumed("gate", step=steps[0].id if steps else None)
            if steps and isinstance(steps[0], plans.Undo):
                return control.undo(
                    self,
                    state,
                    turn,
                    steps[0],
                    state.view,
                    position=paused.resume_at,
                    gate_app=paused.app,
                    only=_receipts(paused),
                )
            return self._execute(
                state, turn, steps, paused.resume_at, {}, gate_passed=steps[0].id if steps else None
            )
        if paused.reason == "ask_user":
            stored = options.stored(paused.body, _offered(paused))
            if tapped.option is None or not 0 <= tapped.option < len(stored):
                raise ValueError("that card has no such option")
            option = stored[tapped.option]
            # Their pick, by its place: the tap said which, word for word.
            state.words = option.option
            value, picked = options.chosen(option.option, stored)
            return self._resume_ask_user(state, turn, paused, value, picked or option)
        if paused.reason == "app_question":
            candidates = [c for c in paused.body.get(APP_CANDIDATES) or [] if isinstance(c, dict)]
            if tapped.option is None or not 0 <= tapped.option < len(candidates):
                raise ValueError("that card has no such option")
            state.words = str(candidates[tapped.option].get("title") or "")
            state.picked = {**candidates[tapped.option], "picked": True}
            return self._resume_app_question(state, turn, paused, picked=True)
        # An app's questions, every one answered from its card: what waited
        # on them runs with what the approved calls touched.
        turn_events.resumed("app_confirm")
        undoing = control.undo_step(paused)
        if undoing is not None:
            undo_step, position = undoing
            turn.body = {"steps": [undo_step.model_dump()]}
            return control.undo(
                self,
                state,
                turn,
                undo_step,
                state.view,
                position=position,
                handled=tapped.undone,
                only=_receipts(paused) or [],
            )
        values: dict[str, dataflow.Value] = {}
        asking = body.steps[paused.resume_at - 1 : paused.resume_at]
        touched = dataflow.from_record(tapped.app_record)
        if asking and touched is not None:
            values[asking[0].id] = touched
        return self._execute(state, turn, steps, paused.resume_at, values)

    # --- a card's answer ---------------------------------------------------------------------

    def undo_goes_on(
        self, paused: PausedPlan, view: Callable[[], Ledgered], handled: Sequence[int]
    ) -> bool:
        """Whether a waiting undo has more to ask once its cards are answered."""
        return control.undo_goes_on(self, control.View(view), paused, handled)

    def answer_confirm(self, app: str, *, actor: str = ACTOR) -> ToolCallRecord:
        """The person's yes from one question's card: that question's token
        goes to its app (the bound delegation context holds exactly that
        question) and the app runs that one call. No plan, no model call."""
        call = ToolCall(id="card-1", name=_CONFIRM_TOOL, arguments={"app": app})
        return self._rt.dispatch(call, actor)

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

    def _unplanned(
        self, state: _Run, compiled: compiler.Compiled, *, waiting: bool
    ) -> AgentRunResult:
        """No plan code can run, after every repair (#75): code asks which app
        was meant when the tries disagreed on it; else, when nothing waits on
        the person and the words are a request the planner never shaped into
        steps, hands them to a think (#146); else says it couldn't put the
        request together. Nothing runs and nothing is guessed."""
        question = clarifying(compiled.raw)
        if question is None and not waiting and not answering(compiled.raw):
            if get_settings().conductor_think and not _stopped(state.stop):
                return self._think_unplanned(state, compiled)
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
        reach: PausedPlan | None = None,
    ) -> AgentRunResult:
        """A control step inside a longer plan (#75) runs first, as a plan of
        its own; then the rest runs. The rest is not run, and said, when the
        control step left a question waiting (a ``{sN}`` on a control step
        never compiles: ``plans.unrunnable``). ``reach`` is what a cancel may
        end (an open question's plan, never a card's, #251)."""
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
        if isinstance(step, plans.Cancel):
            done = control.run(
                self,
                state,
                first,
                step,
                state.view,
                paused=reach,
                pending=None,
                cards=pending is not None or (paused is not None and reach is None),
            )
        else:
            done = control.run(self, state, first, step, state.view, paused=paused, pending=pending)
        turn.first_steps = len(state.records)
        state.since = len(state.records)
        if done.stop_reason != "completed":
            self._skip(rest, state, STOPPED_ERROR)
            return self._rt.finish(done.stop_reason, done.reply, state.records, state.messages)
        why: str | None = None
        if first.status == "paused" or narrator.live_confirm(state.records) is not None:
            why = _WAITS_ON_QUESTION
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

    def _proposal(self, state: _Run, turn: TurnPlan, proposal: PausedPlan) -> AgentRunResult:
        """A tap on a think's proposal (#78) runs it as a normal plan: a
        destructive ask to an app that doesn't confirm still stops at code's
        gate, and an app still asks its own question."""
        turn.body = proposal.body
        turn.ends, turn.ends_as = proposal.plan_id, "accepted"
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
        go on the record as not run when the plan ends. A step that stops for
        a question holds up only the steps that need its answer
        (:meth:`_hold`, #251)."""
        if not state.holding:
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
            position = _place(turn.body, step, offset + index)
            if _stopped(state.stop):
                self._skip([*steps[index:], *state.dropped], state, STOPPED_ERROR)
                turn.note = f"stopped before {step.id}"
                return self._rt.stop("cancelled", state.records, state.messages)
            if isinstance(step, plans.AskUser):
                if turn.body is not None:
                    turn.body = options.annotate(turn.body, step.options, state.context or "")
                stopped = self._hold(state, turn, step, steps[index + 1 :], values)
                if stopped is not None:
                    return stopped
                turn.status, turn.pause, turn.resume_at = "paused", "ask_user", position + 1
                self._carry(turn, position + 1, values)
                return self._ask_user(state, step, turn)
            group = _read_group(steps, index, self._reads)
            if len(group) > 1:
                calls = [
                    (ask_tool_name(s.app), self._ask_arguments(state, s, s.request, values))
                    if isinstance(s, plans.Ask)
                    else _call_for(s)
                    for s in group
                ]
                with self._on_asking_thread(state, group):
                    results = self._dispatch_all(state, calls)
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
                    return self._app_asked(state, turn, steps[after:], asked[0], values)
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
                app = self._fleet.get(step.app)
                typed = app is not None and app.agent is not None and app.agent.context
                # A v1.2 app whose evals pass on refs alone gets "that album"
                # in the words and the title as a context ref (#204).
                filled = dataflow.fill(
                    step.request, values, pointers=typed and step.app in dataflow.TITLES_OUT
                )
                confirms = app is not None and app.agent is not None and app.agent.confirms
                refusal = _refusal(step, filled, confirms=confirms)
                if refusal is not None:
                    return self._refuse(state, turn, steps, index, filled, refusal)
                if step.app == facts.APP:
                    record: ToolCallRecord | None = self._memory_app(state, filled.text)
                elif step.destructive and not confirms and step.id != gate_passed:
                    stopped = self._hold(state, turn, step, steps[index + 1 :], values, itself=True)
                    if stopped is not None:
                        return stopped
                    turn.status, turn.pause = "paused", "gate"
                    turn.resume_at, turn.pause_app = position, step.app
                    self._carry(turn, position, values)
                    return self._gate(state, step, filled.text, turn)
                else:
                    arguments = self._ask_arguments(state, step, filled.text, values)
                    with self._on_asking_thread(state, [step]):
                        record = self._dispatch(state, ask_tool_name(step.app), arguments)
                if record is not None and record.app_confirm and record.app_confirm.get("question"):
                    return self._app_asked(state, turn, steps[index + 1 :], step, values)
            elif isinstance(step, plans.Look):
                record = self._dispatch(state, look_tool_name(step.app), {})
                if record is not None and not self._went_well(record):
                    record = self._look_fell_back(state, step, record, steps)
                    if record.app_confirm and record.app_confirm.get("question"):
                        return self._app_asked(state, turn, steps[index + 1 :], step, values)
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
            else:  # the control steps are handled before a plan runs
                raise AssertionError(f"the interpreter can't run {step.do} here")
            if record is None or not self._went_well(record):
                turn.note = f"stopped after {step.id}"
                return self._finish(state, turn, steps[index + 1 :])
            self._keep(state, values, step, record)
            if isinstance(step, plans.Ask) and narrator.app_asks(record) is not None:
                # The app asked the person (#168): the plan waits on their
                # answer, which goes back to the same step.
                return self._app_questioned(state, turn, position, step, values, steps[index + 1 :])
            if narrator.app_declined(record):
                # It turned its part down: what follows would run on a guess.
                owner = confirmation.record_app(record.tool, record.arguments) or "the app"
                why = f"{owner} turned down its part"
                turn.note = f"stopped after {step.id}: {why}"
                return self._finish(
                    state, turn, steps[index + 1 :], run_record.NOT_RUN_PREFIX + why
                )
            index += 1
        return self._finish(state, turn, [])

    @staticmethod
    def _on_asking_thread(state: _Run, steps: Sequence[plans.Step]) -> AbstractContextManager[None]:
        """Around the dispatch of ``steps``: the one that answers an app's own
        question goes to the thread that asked it (#262). Code's choice, made
        when the answer resumed the plan; every other ask opens a fresh one."""
        pinned = state.answering
        if pinned is None or not any(step.id == pinned[0] for step in steps):
            return nullcontext()
        state.answering = None
        return delegation.answering(pinned[1], pinned[2])

    def _ask_arguments(
        self,
        state: _Run,
        step: plans.Ask,
        message: str,
        values: Mapping[str, dataflow.Value] | None = None,
    ) -> dict[str, Any]:
        """An app ask's arguments: the request as the plan wrote it (refs
        filled), and the saved facts its words clearly point at (#134), which
        the ask tool sends after it as quoted context. Matched on the plan's
        own words, never on a title a ref put in, and on the person's words
        when they are the request (#179: the plan may write what the fact
        says instead of the words that pointed at it)."""
        arguments: dict[str, Any] = {"message": message}
        request = plans.REF.sub(" ", step.request)
        attached = facts.related(request, self._saved(state), state.words if state.restate else "")
        if not attached and not self._reads(step):
            attached = self._meant(state, request)
        if attached:
            arguments[facts.ARGUMENT] = attached
            logger.info("plan_facts_attached", step=step.id, app=step.app, count=len(attached))
        app = self._fleet.get(step.app)
        if app is not None and app.agent is not None and app.agent.context:
            # Delegate v1.2 (#202): what the words point at goes as refs, with
            # ids, beside the titles spliced in; a step code runs as a read
            # goes ``mode: "read"``, so a wrong guess costs a refused write.
            refs = self._context_refs(state, step, values or {})
            if refs:
                arguments[run_record.CONTEXT_REFS] = refs
            if self._reads(step):
                arguments[run_record.READ_MODE] = True
        return arguments

    def _reads(self, step: plans.Ask) -> bool:
        """Whether an ask runs as a read: the planner's mark for a v1.2 app,
        the words' reading for a v1.1 one (``plans.marked_read``, #204)."""
        app = self._fleet.get(step.app)
        typed = app is not None and app.agent is not None and app.agent.context
        return plans.marked_read(step, typed=typed)

    def _context_refs(
        self, state: _Run, step: plans.Ask, values: Mapping[str, dataflow.Value]
    ) -> list[dict[str, Any]]:
        """The refs a v1.2 app's ask carries: each ``{sN}``'s, then the app's
        own entities this window touched, newest first (:data:`STATE_REFS`), so
        "their other album" reaches music with the album's id. Deduplicated,
        at most :data:`MAX_CONTEXT_REFS`."""
        if state.picked is not None and state.picked.get("app") == step.app:
            picked, state.picked = state.picked, None
            return [picked]
        refs: list[dict[str, Any]] = []
        for match in plans.REF.finditer(step.request):
            value = values.get(match.group(1))
            if value is not None:
                refs += [dict(ref) for ref in value.refs]
        own = [e for e in self._touched(state) if e.app == step.app][:STATE_REFS]
        refs += [
            {"app": e.app, "kind": e.kind, "id": e.id, "title": e.title, "untrusted": e.untrusted}
            for e in own
        ]
        seen: set[tuple[str, str, str]] = set()
        kept: list[dict[str, Any]] = []
        for ref in refs:
            key = (str(ref.get("app")), str(ref.get("kind")), str(ref.get("id")))
            if key in seen or not all(key):
                continue
            seen.add(key)
            kept.append({k: v for k, v in ref.items() if v is not None})
        return kept[:MAX_CONTEXT_REFS]

    @staticmethod
    def _touched(state: _Run) -> list[Entity]:
        """This window's entities, read once a turn; none when the ledger
        can't be read (the ask still goes, with its ``{sN}`` refs)."""
        if state.touched is None:
            state.touched = []
            if state.entities is not None:
                try:
                    state.touched = list(state.entities() or [])
                except Exception as exc:
                    logger.warning(
                        "turn_entities_failed", error=str(exc), error_type=type(exc).__name__
                    )
        return state.touched

    def _meant(self, state: _Run, request: str) -> list[str]:
        """The one saved fact a write's request means though it shares no
        words with it (#204), or none; a failure is words alone, never the ask's."""
        saved = self._saved(state)
        if state.fact_meaning is None or not saved:
            return []
        try:
            meant = state.fact_meaning(request, saved)
        except Exception as exc:
            logger.warning("fact_meaning_failed", error=str(exc), error_type=type(exc).__name__)
            return []
        return [] if meant is None else [meant]

    @staticmethod
    def _saved(state: _Run) -> list[str]:
        """The person's saved facts, newest first, read once a turn; none
        when memory can't be read (the ask still goes, as the plan wrote it)."""
        if state.saved is None:
            state.saved = []
            if state.saved_facts is not None:
                try:
                    state.saved = list(state.saved_facts())
                except Exception as exc:
                    logger.warning(
                        "saved_facts_failed", error=str(exc), error_type=type(exc).__name__
                    )
        return state.saved

    def _keep(
        self,
        state: _Run,
        values: dict[str, dataflow.Value],
        step: plans.Step,
        record: ToolCallRecord,
    ) -> None:
        """What a step gives later steps as ``{sN}``, from its receipt; and
        its record, for a think's inputs (#78). An app's answer (#241) is
        stamped with whether the step was a read, and marked when it names
        outside text; neither is ever a value for ``{sN}``."""
        state.found[step.id] = record
        if record.app_answer is not None:
            # A look that fell back to asking (#73) only ever asks a plain question.
            record.app_answer["read"] = (
                self._reads(step) if isinstance(step, plans.Ask) else isinstance(step, plans.Look)
            )
            relay.mark_outside(
                record, (e.title for e in self._touched(state) if e.untrusted and e.title)
            )
        value = (
            dataflow.from_recall(record.app_record)
            if memory.is_recall(record.tool)
            else dataflow.from_record(record.app_record)
        )
        if value is not None:
            values[step.id] = value

    def _hold(
        self,
        state: _Run,
        turn: TurnPlan,
        asking: plans.Step,
        rest: Sequence[plans.Step],
        values: dict[str, dataflow.Value],
        *,
        itself: bool = False,
    ) -> AgentRunResult | None:
        """A question never holds up what doesn't need its answer (#251).

        Of the steps after the one that asked, those that use what it will
        give (a ``{sN}`` on it, or on a step that waits) wait with it, and so
        does a step that would stop for a question of its own; the rest run
        now, in order. The plan's body keeps the ids of the waiting steps
        (``itself``: the asking step runs again when it resumes, as a gated
        one does). Returns the stopped turn when a Stop cut that short, else
        ``None`` and the caller pauses the plan."""
        waiting, free = plans.waiting_on(asking.id, rest, stops=self._would_stop)
        ids = [asking.id] if itself else []
        ids += [step.id for step in waiting]
        if free and not _stopped(state.stop):
            logger.info(
                "plan_runs_past_question",
                asking=asking.id,
                runs=[step.id for step in free],
                waits=[step.id for step in waiting],
            )
            beside = TurnPlan(body={}, status="ran")
            state.holding += 1
            try:
                self._execute(state, beside, free, 0, values)
            finally:
                state.holding -= 1
            ids += [str(one) for one in (beside.body or {}).get(WAITING) or []]
            turn.note = beside.note or turn.note
        if _stopped(state.stop):
            self._skip(waiting, state, STOPPED_ERROR)
            turn.note = f"stopped at {asking.id}'s question"
            return self._rt.stop("cancelled", state.records, state.messages)
        if turn.body is not None:
            turn.body = {**turn.body, WAITING: sorted(set(ids), key=_step_number)}
        return None

    def _would_stop(self, step: plans.Step) -> bool:
        """Whether a step stops for a question or ends the plan when it runs:
        a destructive ask to an app that doesn't confirm (code's gate), an
        ``ask_user`` or a think. Beside an open question it waits instead."""
        if isinstance(step, plans.AskUser | plans.Think):
            return True
        if not isinstance(step, plans.Ask) or not step.destructive or step.app == facts.APP:
            return False
        app = self._fleet.get(step.app)
        return not (app is not None and app.agent is not None and app.agent.confirms)

    def _app_asked(
        self,
        state: _Run,
        turn: TurnPlan,
        rest: Sequence[plans.Step],
        step: plans.Ask | plans.Look,
        values: dict[str, dataflow.Value],
    ) -> AgentRunResult:
        """An app asked the person to confirm: each question is a card. The
        steps that need what the confirmed call will touch wait with it; the
        rest of the plan runs now (#251)."""
        asking = next((r for r in reversed(state.records) if r.app_confirm), None)
        question = "\n".join(narrator.asked_questions(asking)) if asking else None
        turn_events.paused("app_confirm", question=question or None, app=step.app, step=step.id)
        position = _place(turn.body, step, 0)
        stopped = self._hold(state, turn, step, rest, values)
        if stopped is not None:
            return stopped
        if (turn.body or {}).get(WAITING):
            turn.status, turn.pause = "paused", "app_confirm"
            turn.resume_at, turn.pause_app = position + 1, step.app
            self._carry(turn, position + 1, values)
            state.dropped = []  # still in the paused plan: said when it gets there
        return self._finish(state, turn, [])

    def _app_questioned(
        self,
        state: _Run,
        turn: TurnPlan,
        position: int,
        step: plans.Ask,
        values: dict[str, dataflow.Value],
        rest: Sequence[plans.Step] = (),
    ) -> AgentRunResult:
        """An app that did nothing asked the person something (#168: music
        naming the albums to pick from). The plan pauses at that step; what
        needs its answer waits unrun and unsaid, the rest runs now (#251). The
        person's pick, or their words for an open question, goes back to the
        step and the plan goes on (:meth:`_resume_app_question`). The reply
        is the app's question in its own words (``narrator``)."""
        question = narrator.app_asks(state.records[-1])
        typed = (state.records[-1].app_asks or {}).get("needs_input") or {}
        candidates = typed.get("candidates") if isinstance(typed, dict) else None
        if candidates and turn.body is not None:
            # What the person picks from goes with the plan (#196): their
            # answer is matched to one by code when it resumes.
            turn.body = {**turn.body, APP_CANDIDATES: list(candidates)}
        asking = state.records[-1].app_thread
        if asking is not None and turn.body is not None:
            # The answer goes back to the thread that asked (#262).
            turn.body = {**turn.body, APP_THREAD: asking}
        _asks(turn, question)
        turn_events.paused("app_question", question=question, app=step.app, step=step.id)
        stopped = self._hold(state, turn, step, rest, values, itself=True)
        if stopped is not None:
            return stopped
        turn.status, turn.pause = "paused", "app_question"
        turn.resume_at, turn.pause_app = position, step.app
        turn.note = f"{step.id} waits on the person: {step.app} asked"
        self._carry(turn, position, values)
        state.dropped = []  # still in the paused plan: said when it gets there
        return self._finish(state, turn, [])

    def _resume_app_question(
        self, state: _Run, turn: TurnPlan, paused: PausedPlan, *, picked: bool = False
    ) -> AgentRunResult:
        """The person answered an app's question (#168): the step that asked
        goes to the app again with their answer, in their own words (trusted),
        after its request, on the thread that asked it (#262); the rest of the
        plan follows."""
        resumed = plans.load(paused.body)
        assert isinstance(resumed, plans.StepPlan)
        steps = list(resumed.steps)
        asked = steps[paused.resume_at]
        assert isinstance(asked, plans.Ask)
        thread = paused.body.get(APP_THREAD)
        if isinstance(thread, int) and not isinstance(thread, bool):
            state.answering = (asked.id, asked.app, thread)
        answer = dataflow.answer(state.words).text
        candidates = [c for c in paused.body.get(APP_CANDIDATES) or [] if isinstance(c, dict)]
        chosen = (
            None
            if picked
            else options.pick(state.words, [str(c.get("title") or "") for c in candidates])
        )
        if chosen is not None:
            # The answer names one of the app's candidates (#196): the app gets
            # that ref, with its id and marked as the pick, not a title to reread.
            state.picked = {**candidates[chosen], "picked": True}
        steps[paused.resume_at] = asked.model_copy(
            update={"request": f"{asked.request} ({APP_ANSWER_LEAD}: {answer})"}
        )
        turn.body = {**paused.body, "steps": [step.model_dump() for step in steps]}
        turn_events.resumed("app_question", step=asked.id, answer=answer)
        resumed = resumed.model_copy(update={"steps": steps})
        return self._execute(state, turn, _waiting_steps(resumed, paused), paused.resume_at, {})

    def _think_unplanned(self, state: _Run, compiled: compiler.Compiled) -> AgentRunResult:
        """A request no round could shape into a plan (#146): the person's own
        words go to the background as a think, through the same path a planned
        one takes. The ledger plan stays ``invalid`` (the planner wrote none);
        the reply says it is thinking it over, never that anything ran."""
        step = plans.Think(id="s1", do="think", problem=state.words, inputs=[])
        turn_events.plan({"steps": [step.model_dump()]}, 0)
        why = "; ".join(compiled.problems)[:350] or "no plan"
        logger.info("plan_unplanned_to_think", problems=compiled.problems)
        return self._think(
            state,
            state.plan or TurnPlan(status="invalid"),
            step,
            {},
            lead=UNPLANNED_THINKS,
            why=f"no plan ({why})",
        )

    def _think(
        self,
        state: _Run,
        turn: TurnPlan,
        step: plans.Think,
        values: dict[str, dataflow.Value],
        *,
        lead: str = "",
        why: str | None = None,
    ) -> AgentRunResult:
        """Hand a think to the background (#78): its record carries the job
        and what its inputs found; the reply is code's, at once. The thinker
        runs it on the 26B later; nothing here calls a model. The plan ends
        here: steps after it are not run, and said (#75). With the lane off
        the think is not run either, and code says so. ``lead`` opens the reply
        and ``why`` the plan's note when code filed the think itself (#146)."""
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
        turn.note = (
            f"{why}; thinking in the background ({ref})"
            if why
            else (f"thinking in the background ({ref})")
        )
        line = f"{lead} {line}".strip()
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

    def _finish(
        self,
        state: _Run,
        turn: TurnPlan,
        unrun: Sequence[plans.Step],
        why: str = _AFTER_FAILURE,
    ) -> AgentRunResult:
        """End the plan: what didn't run goes on the record with why, then the
        narrator speaks from this plan's records (a stopped turn: the record)."""
        self._skip(unrun, state, why)
        self._skip(state.dropped, state, _AFTER_THINK)
        state.dropped = []
        if _stopped(state.stop):
            return self._rt.stop("cancelled", state.records, state.messages)
        if state.holding:
            # Steps run beside a question: the plan's own reply says them.
            return self._rt.finish("completed", None, state.records, state.messages)
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

    def _ask_user(self, state: _Run, step: plans.AskUser, turn: TurnPlan) -> AgentRunResult:
        question = step.question.strip()
        _asks(turn, question)
        offered = [option.strip() for option in step.options if option.strip()]
        if offered:
            question = f"{question} ({' / '.join(offered)})"
        turn_events.paused("ask_user", question=question, step=step.id)
        reply = self._with_record_line(state, question)
        return self._rt.finish("completed", reply, state.records, state.messages)

    def _gate(self, state: _Run, step: plans.Ask, request: str, turn: TurnPlan) -> AgentRunResult:
        question = f'Just to check: should I ask {step.app} to "{request}"?'
        _asks(turn, question)
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


def _read_group(
    steps: Sequence[plans.Step],
    start: int,
    reads: Callable[[plans.Ask], bool] = plans.reads_only,
) -> list[plans.Ask | plans.Look]:
    """The reads from ``start`` that can run at once: next to each other, each
    to a different app (one call per app at a time). A look is always a
    read (#73); an ask is one when ``reads`` says so (#204)."""
    group: list[plans.Ask | plans.Look] = []
    for step in steps[start:]:
        if not isinstance(step, plans.Look) and not (
            isinstance(step, plans.Ask) and reads(step) and step.app != facts.APP
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
        return f"{filled.missing[0]} {run_record.NO_NAME_REASON}"
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


@dataclass(frozen=True)
class Resume:
    """A card's tap (#251): the waiting plan it picks up, with no words.

    ``words`` is the request that plan came from (the narrator's context);
    ``option`` the place of the option tapped on a pick-one. For an app's
    questions answered from their cards, ``app_record`` is what the approved
    calls did and ``undone`` the receipts they undid."""

    paused: PausedPlan
    words: str = ""
    option: int | None = None
    app_record: tuple[dict[str, Any], ...] = ()
    undone: tuple[int, ...] = ()


def _offered(paused: PausedPlan) -> list[str]:
    """The options the ``ask_user`` a plan is paused on offered."""
    try:
        body = plans.load(paused.body)
    except ValueError:
        return []
    if not isinstance(body, plans.StepPlan) or paused.reason != "ask_user":
        return []
    asked = body.steps[paused.resume_at - 1 : paused.resume_at]
    if not asked or not isinstance(asked[0], plans.AskUser):
        return []
    return [option for option in asked[0].options if option.strip()]


def _on_a_card(paused: PausedPlan) -> bool:
    """Whether a waiting plan's question is a card a tap answers (#251):
    code's gate, a proposal, an app's confirm, or a pick-one. An open
    question (an ``ask_user`` with no options, an app's with no candidates)
    waits on the person's words instead, for one turn."""
    if paused.reason == "ask_user":
        return bool(_offered(paused))
    if paused.reason == "app_question":
        return bool(paused.body.get(APP_CANDIDATES))
    return True


def on_a_card(paused: PausedPlan) -> bool:
    """:func:`_on_a_card`, for the page's cards and the state block."""
    return _on_a_card(paused)


def waiting_steps(paused: PausedPlan) -> list[plans.Step]:
    """The steps a waiting plan still has to run."""
    body = plans.load(paused.body)
    return _waiting_steps(body, paused) if isinstance(body, plans.StepPlan) else []


def _waiting_steps(plan: plans.StepPlan, paused: PausedPlan) -> list[plans.Step]:
    """The steps a waiting plan runs when it resumes: those its body lists as
    waiting, or everything from where it paused (a body from before #251)."""
    rest = list(plan.steps[paused.resume_at :])
    ids = paused.body.get(WAITING)
    if not isinstance(ids, list):
        return rest
    return [step for step in rest if step.id in ids]


def _asks(turn: TurnPlan, question: str | None) -> None:
    """Keep a pausing plan's question with it, for its card (#251)."""
    if turn.body is not None and question:
        turn.body = {**turn.body, QUESTION: question}


def _receipts(paused: PausedPlan) -> list[int] | None:
    """The receipts a waiting undo chose when it asked, if its body says."""
    kept = paused.body.get(control.UNDO_RECEIPTS)
    return [i for i in kept if isinstance(i, int)] if isinstance(kept, list) else None


def _place(body: dict[str, Any] | None, step: plans.Step, fallback: int) -> int:
    """A step's index in the plan's body: what runs may be a part of it."""
    for index, stored in enumerate((body or {}).get("steps") or []):
        if isinstance(stored, dict) and stored.get("id") == step.id:
            return index
    return fallback


def _step_number(step_id: str) -> int:
    return int(step_id[1:]) if step_id[1:].isdigit() else 0


def answering(raws: Sequence[str]) -> bool:
    """Whether any try read the words as an answer or a control (#146): a
    resume, a confirm (refused: no step approves, #251), or a cancel, amend
    or undo. Those fail on what's
    waiting, not on how hard the request is, so there is nothing to think
    over; code says it couldn't put that together."""
    for raw in raws:
        try:
            plan = plans.parse(raw)
        except ValueError:
            continue
        if isinstance(plan, plans.Resume):
            return True
        if isinstance(plan, plans.StepPlan) and any(
            isinstance(step, plans.Confirm) or step.do in plans.CONTROL_KINDS for step in plan.steps
        ):
            return True
    return False


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
