"""Control steps (#74): ``cancel``, ``amend`` and ``undo``, over the ledger.

"Actually, stop", "make it the first twenty" and "undo that" act on what's
waiting or on what just ran, not on an app. Each runs as a plan of its own
(:func:`app.ai.plan.fallback_reason`); the interpreter hands it here.

- **cancel** ends the plan waiting on the person: paused on ``ask_user``, on
  code's own question (the gate), or on an app's ``confirm_required``. The
  app's token lapses (it is never sent), the waiting plan's ledger row says
  ``cancelled``, and code says plainly what won't happen. Nothing waiting: it
  says so.
- **amend** rewrites the waiting step's words with the change and resumes the
  same plan from it. A gated step is asked about again in its new words; an
  app's question lapses and the app gets the changed request (it may ask
  again); a plan waiting on ``ask_user`` asks its question again. Nothing
  waiting: it says there's nothing to change.
- **undo** reads the last plan's write receipts from the ledger (or older
  ones only when the target names them) and, per app, asks the owning app to
  redeem its undo handles: through the app's own confirmation when it has
  ``confirms``, else after code's ask-first pause, as for any destructive
  ask. A receipt with no undo (``undo_none``), an ``unknown`` receipt, or one
  from an app conductor can't reach is said plainly, never tried. A receipt
  is undone once: the call that redeems it carries its ids
  (``ToolCallRecord.undoes``) and the ledger marks it when the app says it
  did. "Undo that" twice finds the same plan, already undone.

What couldn't be undone rides in the turn's record as a step not run, with
its reason, so the narrator says it and code appends it when the narrator
doesn't (#113's mechanism): also on the turn that answers the app's question,
after which the rest of the undo carries on.

Cancelling or amending a **running job** is the jobs runtime's (#76,
:mod:`app.ai.job_control`): a cancel or amend goes there when a job runs and
nothing is waiting, or when it names the job; otherwise it acts on what's
waiting, as above.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import structlog

from app.ai import dataflow, job_control, narrator, run_record, turn_events
from app.ai import plan as plans
from app.ai.confirmation import PendingConfirm
from app.ai.plan import Ledgered, PausedPlan, TurnPlan, UndoReceipt
from app.fleet.context import current_delegation_context
from app.fleet.tools import ask_tool_name

if TYPE_CHECKING:
    from app.ai.interpreter import Interpreter, _Run
    from app.ai.loop import AgentRunResult, ToolCallRecord

logger = structlog.get_logger(__name__)

NOTHING_TO_CANCEL = "There's nothing waiting to cancel."
NOTHING_TO_CHANGE = "There's nothing waiting to change."
NOTHING_TO_UNDO = "There's nothing I've done here to undo."
CHANGED_NOTHING = "The last thing I did didn't change anything, so there's nothing to undo."
NO_RECORD = "I can't read my record of what I did right now, so I haven't undone anything."
CANT_AMEND_UNDO = "I can't change an undo partway, so I've dropped it; say undo again to redo it."
CANT_UNDO = "None of that can be undone."
CANT_AMEND_PROPOSAL = (
    "I can't edit a plan I proposed, so I've set it aside and nothing changed. "
    "Ask again with the change and I'll think it over."
)
_UNDO_STOPPED = f"{run_record.NOT_RUN_PREFIX}an earlier part of the undo didn't finish"

# How many things one undo request or note names before it counts the rest.
MAX_NAMED = 3
# Words that never name a thing on their own ("undo the last one").
_VAGUE = frozenset({"last", "that", "this", "undo", "the last", "it", "them", "all"})
_BRACED_REF = re.compile(r"\{\s*s\d+(?:\.[a-z_]+)?\s*\}")


class View:
    """The ledger's view for this turn, read at most once and only when a
    control step needs it. ``None``: there is no ledger, or it can't be read."""

    def __init__(self, read: Callable[[], Ledgered] | None) -> None:
        self._read = read
        self._got: tuple[Ledgered | None] | None = None

    def __call__(self) -> Ledgered | None:
        if self._got is None:
            try:
                self._got = (self._read() if self._read is not None else None,)
            except Exception as exc:  # a failed read is no view, never no turn
                logger.warning("control_view_failed", error=str(exc))
                self._got = (None,)
        return self._got[0]

    def running(self) -> tuple[plans.RunningJob, ...]:
        """This conversation's running jobs (#76); none when there's no view."""
        view = self()
        return view.running if view is not None else ()


def run(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    step: plans.Step,
    view: View,
    *,
    paused: PausedPlan | None,
    pending: PendingConfirm | None,
) -> AgentRunResult:
    """Run a control plan's one step."""
    if isinstance(step, plans.Cancel | plans.Amend):
        running = view.running()
        waiting = paused is not None or pending is not None
        if job_control.aims_at_job(step, running, waiting=waiting, words=state.words):
            if isinstance(step, plans.Cancel):
                return job_control.cancel(interp, state, turn, step, running)
            return job_control.amend(interp, state, turn, step, running)
    if isinstance(step, plans.Cancel):
        return cancel(interp, state, turn, paused=paused, pending=pending)
    if isinstance(step, plans.Amend):
        return amend(interp, state, turn, step, paused=paused, pending=pending)
    assert isinstance(step, plans.Undo)
    return undo(interp, state, turn, step, view, position=0, words=state.words)


# --- cancel ----------------------------------------------------------------------------------


def cancel(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    *,
    paused: PausedPlan | None,
    pending: PendingConfirm | None,
) -> AgentRunResult:
    if paused is None and pending is None:
        turn.note = "cancel: nothing was waiting"
        return _say(interp, state, NOTHING_TO_CANCEL)
    _lapse()
    wont: list[str] = []
    if pending is not None:
        wont.append(f'{pending.app} won\'t get a yes to "{_flat(pending.question)}"')
    if paused is not None:
        wont += _waiting(paused)
        turn.ends, turn.ends_as = paused.plan_id, "cancelled"
    turn.note = "cancelled what was waiting"
    logger.info("plan_cancelled", plan=paused.plan_id if paused else None)
    if not wont:
        return _say(interp, state, "Okay, cancelled.")
    return _say(interp, state, f"Okay, cancelled: {_join(wont)}.")


def _waiting(paused: PausedPlan) -> list[str]:
    """What the waiting plan would still have done, in plain words."""
    try:
        body = plans.load(paused.body)
    except ValueError:
        return []
    if not isinstance(body, plans.StepPlan):
        return []
    lines: list[str] = []
    for step in body.steps[paused.resume_at :]:
        if isinstance(step, plans.Ask):
            lines.append(f'I won\'t ask {step.app} to "{_unref(step.request)}"')
        elif isinstance(step, plans.Open):
            lines.append(f"I won't open {step.app}")
        elif isinstance(step, plans.Undo):
            lines.append("I won't undo anything")
    if len(lines) > MAX_NAMED:
        lines = [*lines[:MAX_NAMED], "nor the rest of the plan"]
    return lines


# --- amend -----------------------------------------------------------------------------------


def amend(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    step: plans.Amend,
    *,
    paused: PausedPlan | None,
    pending: PendingConfirm | None,
) -> AgentRunResult:
    change = " ".join(step.change.split())
    if (paused is None and pending is None) or not change:
        turn.note = "amend: nothing was waiting"
        return _say(interp, state, NOTHING_TO_CHANGE)
    if paused is None:
        assert pending is not None
        return _amend_question(interp, state, turn, pending, change)
    if paused.reason == "proposal":
        # A think's proposal (#78) runs only as the person saw it: a change
        # sets it aside, never edits it into something they didn't see.
        _lapse()
        turn.ends, turn.ends_as = paused.plan_id, "set_aside"
        turn.note = "amend: a proposal can't be edited; set it aside"
        return _say(interp, state, CANT_AMEND_PROPOSAL)
    body = plans.load(paused.body)
    assert isinstance(body, plans.StepPlan)
    steps = list(body.steps)
    at = _amended_at(paused, steps)
    if at is None:
        turn.note = "amend: nothing in the waiting plan to change"
        return _say(interp, state, NOTHING_TO_CHANGE)
    target = steps[at]
    if isinstance(target, plans.Undo):
        _lapse()
        turn.ends, turn.ends_as = paused.plan_id, "cancelled"
        turn.note = "amend: an undo can't be changed; dropped it"
        return _say(interp, state, CANT_AMEND_UNDO)
    assert isinstance(target, plans.Ask | plans.Open)
    steps[at] = _changed(target, change)
    _lapse()
    turn.body = {"steps": [s.model_dump() for s in steps]}
    turn.ends, turn.ends_as = paused.plan_id, "amended"
    turn.note = f"amended {target.id}"
    logger.info("plan_amended", plan=paused.plan_id, step=target.id)
    if paused.reason == "ask_user":
        question = steps[paused.resume_at - 1]
        assert isinstance(question, plans.AskUser)
        turn.status, turn.pause, turn.resume_at = "paused", "ask_user", paused.resume_at
        return interp._ask_user(state, question)
    return interp._execute(state, turn, steps[at:], at, {})


def _amended_at(paused: PausedPlan, steps: Sequence[plans.Step]) -> int | None:
    """The waiting step a change applies to: the gated step, the step that
    asked the app's question, or the step after an ``ask_user``."""
    if paused.reason == "app_confirm":
        for index in range(min(paused.resume_at, len(steps)) - 1, -1, -1):
            step = steps[index]
            if isinstance(step, plans.Ask | plans.Undo) and _app_of(step, paused) == paused.app:
                return index
        return None
    at = paused.resume_at
    if at >= len(steps) or not isinstance(steps[at], plans.Ask | plans.Open | plans.Undo):
        return None
    return at


def _app_of(step: plans.Ask | plans.Undo, paused: PausedPlan) -> str | None:
    return step.app if isinstance(step, plans.Ask) else paused.app


def _amend_question(
    interp: Interpreter, state: _Run, turn: TurnPlan, pending: PendingConfirm, change: str
) -> AgentRunResult:
    """An app's question from a turn the loop ran: the app gets what it was
    asked, changed (no compiled plan was waiting)."""
    if pending.request is None or pending.undoes:
        turn.note = "amend: the app's question can't be changed"
        return _say(interp, state, NOTHING_TO_CHANGE)
    _lapse()
    request = _with_change(pending.request, change)
    ask = plans.Ask(id="s1", do="ask", app=pending.app, request=request, destructive=False)
    turn.body = {"steps": [ask.model_dump()]}
    turn.note = "amended the app's question"
    return interp._execute(state, turn, [ask], 0, {})


def _changed(step: plans.Ask | plans.Open, change: str) -> plans.Ask | plans.Open:
    """The step with the person's change after its words."""
    if isinstance(step, plans.Ask):
        return step.model_copy(update={"request": _with_change(step.request, change)})
    return step.model_copy(update={"intent": _with_change(step.intent, change)})


def _with_change(words: str, change: str) -> str:
    return f"{words} (change: {change})"


# --- undo ------------------------------------------------------------------------------------


def undo(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    step: plans.Undo,
    view: View,
    *,
    position: int,
    gate_app: str | None = None,
    handled: Sequence[int] = (),
    words: str = "",
) -> AgentRunResult:
    """Undo what ``step.target`` names (the last plan, by default).

    ``position`` is the step's index in the plan (where a pause resumes),
    ``gate_app`` the app the person just said yes to at code's gate, and
    ``handled`` the receipts the yes this turn just redeemed: the undo goes on
    with what's left. ``words`` are the person's own, when this turn is the
    undo they asked for.
    """
    ledgered = view()
    if ledgered is None:
        return _say(interp, state, NO_RECORD)
    chosen = pick(ledgered, step.target, words)
    if not chosen and not handled:
        turn.note = "undo: nothing to undo"
        return _say(
            interp, state, NOTHING_TO_UNDO if ledgered.last_plan is None else CHANGED_NOTHING
        )
    groups: dict[str, list[UndoReceipt]] = {}
    already: list[UndoReceipt] = []
    notes: list[ToolCallRecord] = []
    for receipt in chosen:
        app = interp._fleet.get(receipt.app)
        if receipt.id in handled:
            continue
        if receipt.undone:
            already.append(receipt)
        elif receipt.outcome == "unknown":
            notes.append(
                _note(receipt, f"there's no record of what {receipt.app} did; check it there")
            )
        elif not (receipt.handle or receipt.how):
            notes.append(_note(receipt, receipt.none or f"{receipt.app} gave no way to undo it"))
        elif app is None or app.agent is None:
            notes.append(_note(receipt, f"conductor can't reach {receipt.app}"))
        else:
            groups.setdefault(receipt.app, []).append(receipt)
    turn.note = f"undo of plan {chosen[0].plan_id}" if chosen else turn.note
    asks = [
        plans.Ask(
            id=step.id, do="ask", app=app_name, request=_undo_request(receipts), destructive=True
        )
        for app_name, receipts in groups.items()
    ]
    # What this undo will do, one record per step, as the live channel reads
    # a plan (#38): what can't be undone first, then one ask per app.
    turn_events.plan(
        {
            "steps": [
                *(_note_step(step.id, note) for note in notes),
                *(a.model_dump() for a in asks),
            ]
        },
        0,
    )
    for note in notes:
        _add(state, note)
    if not asks:
        if handled:
            return interp._finish(state, turn, [], 1)
        lead = f"Already undone: {_what(already)}." if already else CANT_UNDO
        line = narrator.record_line(state.records, state.trace.surface)
        return _say(interp, state, f"{lead} {line}".strip())
    for index, ask in enumerate(asks):
        app = interp._fleet.get(ask.app)
        assert app is not None and app.agent is not None
        if not app.agent.confirms and ask.app != gate_app:
            # Code's own ask-first pause, as for any destructive ask; a yes
            # resumes this undo here (what's done by then is marked).
            turn.status, turn.pause = "paused", "gate"
            turn.resume_at, turn.pause_app = position, ask.app
            return interp._gate(state, ask, ask.request)
        record = interp._dispatch(state, ask_tool_name(ask.app), {"message": ask.request})
        if record is None:
            break
        record.undoes = [receipt.id for receipt in groups[ask.app]]
        question = (record.app_confirm or {}).get("question")
        if question:
            # The app's own question; the yes carries what it undoes, and
            # the rest of the undo goes on after it.
            turn_events.paused("app_confirm", question=question, app=ask.app, step=step.id)
            turn.status, turn.pause = "paused", "app_confirm"
            turn.resume_at, turn.pause_app = position + 1, ask.app
            return interp._finish(state, turn, [], 1)
        if not interp._went_well(record):
            turn.note = f"undo stopped at {ask.app}"
            for later in asks[index + 1 :]:
                _add(state, _unrun(later.app, later.request, _UNDO_STOPPED))
            break
    return interp._finish(state, turn, [], 1)


def undo_waiting(
    paused: PausedPlan | None, pending: PendingConfirm | None
) -> tuple[plans.Undo, int, tuple[int, ...]] | None:
    """The undo an app's waiting question belongs to, if it is one: the undo
    step, its index, and the receipts the yes redeems."""
    if paused is None or pending is None or paused.reason != "app_confirm":
        return None
    if not pending.undoes or pending.app != paused.app:
        return None
    try:
        body = plans.load(paused.body)
    except ValueError:
        return None
    index = paused.resume_at - 1
    if not isinstance(body, plans.StepPlan) or not 0 <= index < len(body.steps):
        return None
    step = body.steps[index]
    return (step, index, pending.undoes) if isinstance(step, plans.Undo) else None


def pick(ledgered: Ledgered, target: str, words: str = "") -> list[UndoReceipt]:
    """The receipts an undo means.

    The last plan's, all of them, unless the person's own ``words`` name some
    of them ("undo the milk task"): the planner's target pointing at one
    thing in State doesn't narrow "undo that". An older plan's only when the
    target names something in it and nothing in the last plan: nothing older
    is ever picked unnamed.
    """
    last = [receipt for receipt in ledgered.receipts if receipt.plan_id == ledgered.last_plan]
    said = [receipt for receipt in last if words and names(words, receipt)]
    if said:
        return said
    if any(names(target, receipt) for receipt in last):
        return last
    older = [
        receipt
        for receipt in ledgered.receipts
        if receipt.plan_id != ledgered.last_plan and names(target, receipt)
    ]
    if older:
        newest = older[0].plan_id
        return [receipt for receipt in older if receipt.plan_id == newest]
    return last


def names(target: str, receipt: UndoReceipt) -> bool:
    """Whether a target clearly names something this receipt touched: its
    title, or its kind and id (``task:301``, ``task 301``), or its id."""
    words = f" {_norm(target)} "
    raw = target.lower()
    for item in receipt.touched:
        title = _norm(item.title or "")
        if len(title) >= 4 and title not in _VAGUE and f" {title} " in words:
            return True
        ref = re.escape(item.ref.lower())
        if re.search(rf"\b{re.escape(item.kind.lower())}[\s:#]*{ref}(?![\w])", raw):
            return True
        if len(item.ref) >= 2 and re.search(rf"(?<![\w]){ref}(?![\w])", raw):
            return True
    return False


def _undo_request(receipts: Sequence[UndoReceipt]) -> str:
    """One self-contained request to the owning app: each thing, how to undo
    it and its handle. Outside titles go by kind and id (the request is a
    write)."""
    parts = []
    for receipt in receipts:
        handle = f" (undo handle: {receipt.handle})" if receipt.handle else ""
        parts.append(f"{_what([receipt], to_app=True)}: {receipt.how or 'undo it'}{handle}")
    return "Undo what you did earlier — " + "; ".join(parts) + "."


def _what(receipts: Sequence[UndoReceipt], *, to_app: bool = False) -> str:
    """The things some receipts touched, by name: quoted titles (outside ones
    cleaned, or by kind and id when it goes to an app), else a count or the
    action."""
    names_: list[str] = []
    for receipt in receipts:
        items = [item for item in receipt.touched if not item.kind.endswith("job")]
        titled = [item for item in items if item.title]
        if titled:
            for item in titled:
                if to_app and item.untrusted:
                    names_.append(f"{item.kind} {item.ref}")
                elif to_app:
                    names_.append(f"{item.kind} “{dataflow.clean(item.title or '')}”")
                else:
                    names_.append(f"“{dataflow.clean(item.title or '')}”")
        elif items:
            names_ += [f"{item.kind} {item.ref}" for item in items]
        elif receipt.count:
            names_.append(f"{receipt.count} things")
        else:
            action = (receipt.tool or "").replace("_", " ").strip()
            names_.append(f"the {action}" if action else "its last change")
    unique = list(dict.fromkeys(names_))
    if len(unique) > MAX_NAMED:
        return f"{', '.join(unique[:MAX_NAMED])} and {len(unique) - MAX_NAMED} more"
    return (
        " and ".join(unique) if len(unique) <= 2 else f"{', '.join(unique[:-1])} and {unique[-1]}"
    )


def _note(receipt: UndoReceipt, reason: str) -> ToolCallRecord:
    """What couldn't be undone, as a step not run: the narrator says it, and
    code appends it when the narrator doesn't."""
    return _unrun(receipt.app, f"undo {_what([receipt])}", run_record.NOT_RUN_PREFIX + reason)


def _note_step(step_id: str, note: ToolCallRecord) -> dict[str, Any]:
    """A note as a plan step, for the live channel: an ask that won't run."""
    return {
        "id": step_id,
        "do": "ask",
        "app": note.tool.removeprefix("ask_"),
        "request": note.arguments.get("message", ""),
        "destructive": False,
    }


def _add(state: _Run, record: ToolCallRecord) -> None:
    """A record code made itself, on the turn's record and the live channel."""
    state.records.append(record)
    turn_events.receipt(len(state.records), record)


def _unrun(app: str, message: str, error: str) -> ToolCallRecord:
    from app.ai.loop import ToolCallRecord  # noqa: PLC0415

    return ToolCallRecord(tool=ask_tool_name(app), arguments={"message": message}, error=error)


# --- shared ----------------------------------------------------------------------------------


def _lapse() -> None:
    """The app's waiting question lapses: its token is never sent."""
    current_delegation_context().pending_confirm = None


def _say(interp: Interpreter, state: _Run, reply: str) -> AgentRunResult:
    """Code's own reply for a control step: plain facts, no model call."""
    return interp._loop._finish(
        "completed", narrator.fit(reply, state.trace.surface), state.records, 1, state.messages
    )


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _unref(text: str) -> str:
    return " ".join(_BRACED_REF.sub("that", text).split())


def _flat(text: str) -> str:
    return " ".join(text.split())


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'", "")))
