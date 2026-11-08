"""The ledger (#69): one exact record of what happened.

The only write path for the ``ledger_*`` tables. A turn is opened before the
loop runs (the person's words, ``running``) and closed after it with the plan
it ran: for today's loop, one ``loop`` plan whose steps are its tool calls,
each with the receipts from the app's typed record (delegate v1.1). Receipts
fill the entity registry from their ``touched`` refs and the job store from
their ``job`` reports.

**Every app write has a receipt row.** A step to an app that ran but left no
record conductor could read (a v1 app, or a call that stopped partway) gets
conductor's own ``unknown`` receipt: a possible write, never "nothing changed".

Journaling never costs the person their turn: each write runs in a savepoint,
and a failure is logged (``ledger_*_failed``) and rolled back alone. The store
is unbounded; each turn's context stays bounded (:func:`recent_entities` reads
a fixed window).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import confirmation, run_record
from app.ai.loop import AgentRunResult, ToolCallRecord
from app.ai.plan import Ledgered, PausedPlan, Touched, TurnPlan, UndoReceipt
from app.ai.state_block import Entity
from app.db.models import (
    Conversation,
    ConversationMessage,
    LedgerEntity,
    LedgerJob,
    LedgerPlan,
    LedgerReceipt,
    LedgerStep,
    LedgerTouch,
    LedgerTurn,
    utcnow,
)
from app.fleet.tools import OUTCOME_UNKNOWN

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Receipts whose touched refs count as recent entities: what a call changed or found.
ENTITY_OUTCOMES = ("done", "looked_up")
# How far back :func:`recent_entities` looks, in touches: bounded so the state
# block's cost never grows with the ledger.
ENTITY_TOUCHES_SCANNED = 200

UNKNOWN_REASON = "the app ran but conductor has no record of what it did"


def open_turn(
    db: Session,
    conversation: Conversation,
    message: ConversationMessage,
    *,
    surface: str | None = None,
    panel: str | None = None,
) -> LedgerTurn | None:
    """Put the person's words on the record as a ``running`` turn."""
    return _safely(
        db,
        "ledger_open_turn_failed",
        lambda: _add_turn(
            db,
            LedgerTurn(
                conversation_id=conversation.id,
                user_message_id=message.id,
                kind="message",
                words=message.content or "",
                surface=surface,
                panel=panel,
            ),
        ),
    )


def note_turn(
    db: Session, conversation: Conversation, message: ConversationMessage
) -> LedgerTurn | None:
    """Put something the person did in the UI on the record: words, no plan."""
    return _safely(
        db,
        "ledger_note_failed",
        lambda: _add_turn(
            db,
            LedgerTurn(
                conversation_id=conversation.id,
                user_message_id=message.id,
                kind="note",
                words=message.content or "",
                status="ended",
                ended_at=utcnow(),
            ),
        ),
    )


def close_turn(
    db: Session,
    turn: LedgerTurn | None,
    run: AgentRunResult,
    assistant_message: ConversationMessage | None = None,
) -> None:
    """End ``turn`` with the plan the loop ran: its steps and their receipts."""
    if turn is None:
        return

    def write() -> None:
        turn.status = "ended"
        turn.stop_reason = run.stop_reason
        turn.run_id = run.run_id or None
        turn.ended_at = utcnow()
        if assistant_message is not None:
            turn.assistant_message_id = assistant_message.id
        compiled = None if run.plan is None else _compiled_plan(turn, run.plan)
        if compiled is not None:
            db.add(compiled)
            _end_waiting_plan(db, turn, run.plan)
        if compiled is not None and compiled.acted:
            plan = compiled  # the compiled plan acted: its steps are the turn's calls
        else:
            plan = LedgerPlan(turn_id=turn.id, source="loop")
            db.add(plan)
        db.flush()
        for seq, record in enumerate(run.tool_calls, start=1):
            _journal_step(db, plan, seq, record)
        db.flush()

    _safely(db, "ledger_close_turn_failed", write)
    logger.info(
        "ledger_turn_closed",
        turn_id=turn.id,
        stop_reason=run.stop_reason,
        steps=len(run.tool_calls),
        run_id=run.run_id,
    )


def paused_plan(db: Session, conversation_id: int) -> PausedPlan | None:
    """Last turn's compiled plan, if it is waiting on the person (#70).

    Only the newest turn's plan can resume: anything said or done since (a
    note included) lets the question lapse, as an app's question does.
    """
    last = db.execute(
        select(LedgerTurn)
        .where(LedgerTurn.conversation_id == conversation_id, LedgerTurn.status == "ended")
        .order_by(LedgerTurn.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if last is None or last.kind != "message":
        return None
    for plan in last.plans:
        if (
            plan.source == "compiled"
            and plan.acted
            and plan.status == "paused"
            and plan.body is not None
            and plan.pause in ("ask_user", "gate", "app_confirm")
            and plan.resume_at is not None
        ):
            return PausedPlan(
                body=plan.body,
                reason=plan.pause,  # type: ignore[arg-type]
                resume_at=plan.resume_at,
                app=plan.pause_app,
                plan_id=plan.id,
            )
    return None


# How many of this conversation's turns the self-report reads.
SELF_REPORT_TURNS = 3

# Plain words for the self-report, so the speaker never reads a tool name or a
# bare "done" as a claim that something was completed (S16).
_STEP_WORDS = {
    "ask": "asked",
    "look": "looked at",
    "open": "opened",
    "confirm": "said yes to",
    "local": "looked up",
}
_OUTCOME_WORDS = {
    "done": "it answered",
    "looked_up": "it answered",
    "failed": "it failed",
    "not_run": "it didn't run",
}
_RECEIPT_WORDS = {
    "done": "the app did",
    "looked_up": "the app found",
    "refused": "the app refused",
    "not_run": "the app didn't run",
    "unknown": "unknown whether the app did",
}


def self_report(db: Session, conversation_id: int) -> str:
    """What conductor did lately and what's still running, from the ledger
    alone, for "what are you doing / what did you just do?" (S16)."""
    turns = (
        db.execute(
            select(LedgerTurn)
            .where(
                LedgerTurn.conversation_id == conversation_id,
                LedgerTurn.status == "ended",
                LedgerTurn.kind == "message",
            )
            .order_by(LedgerTurn.id.desc())
            .limit(SELF_REPORT_TURNS)
        )
        .scalars()
        .all()
    )
    lines: list[str] = []
    for turn in turns:
        lines.append(f'You were asked: "{_clip(turn.words)}"')
        acted = [plan for plan in turn.plans if plan.acted]
        steps = [step for plan in acted for step in plan.steps]
        if not steps:
            lines.append("- nothing ran; you answered directly")
        for step in steps:
            verb = _STEP_WORDS.get(step.kind, step.kind)
            outcome = _OUTCOME_WORDS.get(step.outcome, step.outcome)
            lines.append(f"- {verb} {step.app or step.tool}: {outcome}")
            for receipt in step.receipts:
                titles = [f'"{_clip(touch.title)}"' for touch in receipt.touched if touch.title][:3]
                what = f" {', '.join(titles)}" if titles else ""
                action = (receipt.tool or "something").replace("_", " ")
                did = _RECEIPT_WORDS.get(receipt.outcome, receipt.outcome)
                since = " (since undone)" if receipt.undone_by_step_id is not None else ""
                lines.append(f"  - {did}: {action}{what}{since}")
        if turn.stop_reason and turn.stop_reason != "completed":
            lines.append(f"- the turn ended {turn.stop_reason}")
    jobs = (
        db.execute(
            select(LedgerJob)
            .where(LedgerJob.status.not_in(("completed", "failed", "canceled")))
            .order_by(LedgerJob.updated_at.desc())
            .limit(5)
        )
        .scalars()
        .all()
    )
    for job in jobs:
        total = f" of {job.total}" if job.total is not None else ""
        lines.append(f"Running: {job.app} job {job.ref}, {job.status}, {job.done}{total} done")
    if not lines:
        return "Nothing on record yet in this conversation."
    return "\n".join(lines)


# How far back an undo looks (#74): the last plan, plus a few before it for a
# target that names something older. Bounded, like every turn's context.
CONTROL_TURNS = 20
CONTROL_PLANS = 5
_CONTROL_STEPS = frozenset({"cancel", "amend", "undo"})
_WRITE_OUTCOMES = ("done", "unknown")


def control_view(db: Session, conversation_id: int) -> Ledgered:
    """What a control step acts on (#74): the last plan that ran steps, the
    write receipts of the last few, and the jobs still running.

    A control plan (a compiled ``cancel``/``amend``/``undo``) and a plan with
    a step that undid something are passed over, so "undo that" twice finds
    the same plan, already undone, never the undo itself.
    """
    turns = (
        db.execute(
            select(LedgerTurn)
            .where(
                LedgerTurn.conversation_id == conversation_id,
                LedgerTurn.status == "ended",
                LedgerTurn.kind == "message",
            )
            .order_by(LedgerTurn.id.desc())
            .limit(CONTROL_TURNS)
        )
        .scalars()
        .all()
    )
    plans: list[LedgerPlan] = []
    for turn in turns:
        for plan in reversed(turn.plans):
            if plan.acted and plan.steps and not _is_control(db, plan):
                plans.append(plan)
                break
        if len(plans) == CONTROL_PLANS:
            break
    receipts = [
        _undo_receipt(plan, receipt)
        for plan in plans
        for step in plan.steps
        for receipt in step.receipts
        if receipt.outcome in _WRITE_OUTCOMES
    ]
    return Ledgered(
        last_plan=plans[0].id if plans else None,
        receipts=tuple(receipts),
        jobs=tuple(_running_jobs(db, conversation_id)),
    )


def _is_control(db: Session, plan: LedgerPlan) -> bool:
    body_steps = (plan.body or {}).get("steps") or []
    if any(isinstance(s, dict) and s.get("do") in _CONTROL_STEPS for s in body_steps):
        return True
    step_ids = [step.id for step in plan.steps]
    undid = db.execute(
        select(LedgerReceipt.id).where(LedgerReceipt.undone_by_step_id.in_(step_ids)).limit(1)
    ).first()
    return undid is not None


def _undo_receipt(plan: LedgerPlan, receipt: LedgerReceipt) -> UndoReceipt:
    return UndoReceipt(
        id=receipt.id,
        plan_id=plan.id,
        app=receipt.app,
        tool=receipt.tool,
        outcome=receipt.outcome,
        touched=tuple(
            Touched(
                kind=touch.entity.kind,
                ref=touch.entity.ref,
                title=touch.title,
                untrusted=touch.untrusted or receipt.untrusted,
            )
            for touch in receipt.touched
        ),
        count=receipt.count,
        handle=receipt.undo_handle,
        how=receipt.undo_how,
        none=receipt.undo_none,
        undone=receipt.undone_by_step_id is not None,
    )


def _running_jobs(db: Session, conversation_id: int) -> list[str]:
    """This conversation's jobs that haven't finished, as ``app:ref``."""
    rows = db.execute(
        select(LedgerJob.app, LedgerJob.ref)
        .join(LedgerStep, LedgerStep.id == LedgerJob.started_by_step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
        .where(
            LedgerTurn.conversation_id == conversation_id,
            LedgerJob.status.not_in(("completed", "failed", "canceled")),
        )
        .order_by(LedgerJob.updated_at.desc())
        .limit(10)
    ).all()
    return [f"{app}:{ref}" for app, ref in rows]


def _clip(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _compiled_plan(turn: LedgerTurn, compiled: TurnPlan) -> LedgerPlan:
    body = compiled.body
    answer = body.get("answer") if isinstance(body, dict) else None
    return LedgerPlan(
        turn_id=turn.id,
        source="compiled",
        body=body,
        answer=str(answer) if answer else None,
        acted=compiled.acted,
        status=compiled.status,
        note=compiled.note,
        pause=compiled.pause,
        resume_at=compiled.resume_at,
        pause_app=compiled.pause_app,
    )


def recent_entities(db: Session, conversation_id: int, *, limit: int = 10) -> list[Entity] | None:
    """What this conversation's steps changed or found, newest first, each
    with its title as of the receipt that touched it.

    ``None`` for a conversation the ledger has never closed a turn of (one from
    before #69): the state block then reads its stored records instead.
    """
    journaled = db.execute(
        select(LedgerTurn.id)
        .where(LedgerTurn.conversation_id == conversation_id, LedgerTurn.status == "ended")
        .limit(1)
    ).first()
    if journaled is None:
        return None
    rows = db.execute(
        select(LedgerTouch, LedgerEntity)
        .join(LedgerEntity, LedgerEntity.id == LedgerTouch.entity_id)
        .join(LedgerReceipt, LedgerReceipt.id == LedgerTouch.receipt_id)
        .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
        .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
        .join(LedgerTurn, LedgerTurn.id == LedgerPlan.turn_id)
        .where(
            LedgerTurn.conversation_id == conversation_id,
            LedgerPlan.acted.is_(True),
            LedgerReceipt.outcome.in_(ENTITY_OUTCOMES),
        )
        # Newest receipt first; within one receipt, the order the app listed.
        .order_by(LedgerReceipt.id.desc(), LedgerTouch.id)
        .limit(ENTITY_TOUCHES_SCANNED)
    ).all()
    seen: set[int] = set()
    recent: list[Entity] = []
    for touch, entity in rows:
        if entity.id in seen:
            continue
        seen.add(entity.id)
        recent.append(
            Entity(
                app=entity.app,
                kind=entity.kind,
                id=entity.ref,
                title=touch.title,
                untrusted=touch.untrusted,
            )
        )
        if len(recent) == limit:
            break
    return recent


def step_kind(tool: str, arguments: dict[str, Any]) -> tuple[str, str | None]:
    """A loop tool call as a step kind and the app it spoke to."""
    if tool.startswith("ask_"):
        return "ask", tool.removeprefix("ask_")
    if tool.startswith("open_"):
        return "open", tool.removeprefix("open_")
    if run_record.is_look(tool):
        # A glance (#73): the app's name as the plan wrote it (hyphens flatten).
        return "look", tool.removeprefix(run_record.LOOK_PREFIX)
    if tool == confirmation.CONFIRM_TOOL:
        return "confirm", confirmation.record_app(tool, arguments)
    if tool in run_record.READ_TOOLS:
        return "local", None
    return "unknown", None


def live_step(seq: int, record: ToolCallRecord) -> dict[str, Any]:
    """One step as the ledger will journal it, for the live channel (#38).

    The same kind, app and outcome as its ``LedgerStep`` row, and the same
    receipts: the app's typed record entries as the app wrote them, or
    conductor's ``unknown`` receipt for an app call with no readable record.
    An app's confirm question goes out without its token.
    """
    kind, app = step_kind(record.tool, record.arguments)
    outcome = run_record.classify(record)
    receipts: list[dict[str, Any]] = []
    if app is not None and kind in ("ask", "confirm"):
        if record.app_record is not None:
            receipts = [{**entry, "source": "app"} for entry in record.app_record]
        elif _may_have_written(record, outcome):
            receipts = [{"source": "conductor", "outcome": "unknown", "reason": UNKNOWN_REASON}]
    asked = record.app_confirm or {}
    return {
        "seq": seq,
        "kind": kind,
        "app": app,
        "tool": record.tool,
        "arguments": record.arguments,
        "outcome": outcome,
        "error": record.error,
        "receipts": receipts,
        "confirm": (
            {"app": asked.get("app"), "tool": asked.get("tool"), "question": asked["question"]}
            if asked.get("question")
            else None
        ),
    }


# --- internals ----------------------------------------------------------------------------


def _add_turn(db: Session, turn: LedgerTurn) -> LedgerTurn:
    db.add(turn)
    db.flush()
    return turn


def _safely(db: Session, event: str, write: Callable[[], T]) -> T | None:
    """Run one ledger write in a savepoint; a failure is logged and undone alone."""
    try:
        with db.begin_nested():
            return write()
    except Exception as exc:
        logger.error(event, error=str(exc), error_type=type(exc).__name__)
        return None


def _journal_step(db: Session, plan: LedgerPlan, seq: int, record: ToolCallRecord) -> None:
    kind, app = step_kind(record.tool, record.arguments)
    outcome = run_record.classify(record)
    step = LedgerStep(
        plan_id=plan.id,
        seq=seq,
        kind=kind,
        app=app,
        tool=record.tool,
        arguments=record.arguments,
        outcome=outcome,
        result=record.result,
        error=record.error,
    )
    db.add(step)
    db.flush()
    _mark_undone(db, step, record, outcome)
    if app is None or kind not in ("ask", "confirm", "look"):
        return
    if record.app_record is not None:
        for entry_seq, entry in enumerate(record.app_record, start=1):
            _journal_receipt(db, step, app, entry_seq, entry)
    elif _may_have_written(record, outcome):
        db.add(
            LedgerReceipt(
                step_id=step.id,
                seq=1,
                source="conductor",
                app=app,
                outcome="unknown",
                reason=UNKNOWN_REASON,
            )
        )


def _end_waiting_plan(db: Session, turn: LedgerTurn, compiled: TurnPlan | None) -> None:
    """A control step ended the plan that was waiting (#74): say so on its row."""
    if compiled is None or compiled.ends is None or compiled.ends_as is None:
        return
    waiting = db.get(LedgerPlan, compiled.ends)
    if waiting is None or waiting.status != "paused":
        return
    waiting.status = compiled.ends_as
    waiting.note = f"{compiled.ends_as} by the next turn ({turn.id})"


def _mark_undone(db: Session, step: LedgerStep, record: ToolCallRecord, outcome: str) -> None:
    """The receipts an undo call redeemed (#74), once the app says it did it
    (or may have: an unknown is never redeemed again). A question back, a
    refusal or a call that never ran leaves them undoable."""
    if not record.undoes or record.app_confirm:
        return
    if record.app_record is not None:
        landed = any(entry.get("outcome") == "done" for entry in record.app_record)
    else:
        landed = _may_have_written(record, outcome)
    if not landed:
        return
    receipts = db.execute(
        select(LedgerReceipt).where(
            LedgerReceipt.id.in_(record.undoes), LedgerReceipt.undone_by_step_id.is_(None)
        )
    ).scalars()
    for receipt in receipts:
        receipt.undone_by_step_id = step.id


def _may_have_written(record: ToolCallRecord, outcome: str) -> bool:
    """An app call with no readable record that reached the app: an answer
    from a v1 app, or one that stopped partway."""
    if outcome == "done":
        return True
    return outcome == "failed" and OUTCOME_UNKNOWN in (record.error or "")


def _journal_receipt(
    db: Session, step: LedgerStep, app: str, seq: int, entry: dict[str, Any]
) -> None:
    undo = entry.get("undo") or {}
    receipt = LedgerReceipt(
        step_id=step.id,
        seq=seq,
        source="app",
        app=app,
        tool=_text(entry.get("tool")),
        outcome=str(entry.get("outcome") or "unknown"),
        count=entry.get("count") if isinstance(entry.get("count"), int) else None,
        reason=_text(entry.get("reason")),
        untrusted=bool(entry.get("untrusted")),
        undo_handle=_text(undo.get("handle")),
        undo_how=_text(undo.get("how")),
        undo_none=_text(undo.get("none")),
    )
    job = entry.get("job")
    if isinstance(job, dict) and job.get("ref") is not None:
        receipt.job_id = _report_job(db, step, app, job).id
    db.add(receipt)
    db.flush()
    for ref in entry.get("touched") or []:
        if not isinstance(ref, dict) or ref.get("id") is None:
            continue
        untrusted = bool(entry.get("untrusted") or ref.get("untrusted"))
        title = _text(ref.get("title"))
        entity = _see_entity(
            db,
            app=str(ref.get("app") or app),
            kind=str(ref.get("kind") or "thing"),
            ref=str(ref["id"]),
            title=title,
            untrusted=untrusted,
        )
        db.add(
            LedgerTouch(
                receipt_id=receipt.id, entity_id=entity.id, title=title, untrusted=untrusted
            )
        )


def _see_entity(
    db: Session, *, app: str, kind: str, ref: str, title: str | None, untrusted: bool
) -> LedgerEntity:
    entity = db.execute(
        select(LedgerEntity).where(
            LedgerEntity.app == app, LedgerEntity.kind == kind, LedgerEntity.ref == ref
        )
    ).scalar_one_or_none()
    if entity is None:
        entity = LedgerEntity(app=app, kind=kind, ref=ref)
        db.add(entity)
    if title is not None:
        entity.title = title
        entity.untrusted = untrusted
    entity.last_seen_at = utcnow()
    db.flush()
    return entity


def _report_job(db: Session, step: LedgerStep, app: str, job: dict[str, Any]) -> LedgerJob:
    """The job's latest report wins; the step that first reported it started it."""
    ref = str(job["ref"])
    row = db.execute(
        select(LedgerJob).where(LedgerJob.app == app, LedgerJob.ref == ref)
    ).scalar_one_or_none()
    if row is None:
        row = LedgerJob(app=app, ref=ref, status="submitted", started_by_step_id=step.id)
        db.add(row)
    row.status = str(job.get("status") or row.status)
    done = job.get("done")
    row.done = done if isinstance(done, int) else row.done
    total = job.get("total")
    row.total = total if isinstance(total, int) else None
    db.flush()
    return row


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def receipts_for(db: Session, turn_id: int) -> Sequence[LedgerReceipt]:
    """A turn's receipts, in step then record order (for the trajectory walks)."""
    return (
        db.execute(
            select(LedgerReceipt)
            .join(LedgerStep, LedgerStep.id == LedgerReceipt.step_id)
            .join(LedgerPlan, LedgerPlan.id == LedgerStep.plan_id)
            .where(LedgerPlan.turn_id == turn_id)
            .order_by(LedgerStep.seq, LedgerReceipt.seq)
        )
        .scalars()
        .all()
    )
