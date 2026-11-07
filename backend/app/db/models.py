from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Timezone-aware UTC now — the app's single timestamp representation.

    ``func.now()`` writes naive strings on SQLite; mixing naive and aware
    shapes makes serialized JSON ambiguous (JS parses the naive form as local
    time). All Python-side writes go through this. ``server_default=func.now()``
    stays on the columns purely as DDL; the ORM default always wins on insert.
    """
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=utcnow, onupdate=utcnow, server_default=func.now()
    )


class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)


class ConversationRole(enum.StrEnum):
    user = "user"
    assistant = "assistant"


class Conversation(Base, TimestampMixin, SoftDeleteMixin):
    """One chat thread with conductor's master agent.

    Soft delete is conversation-level only: messages are immutable children
    that ride along with their conversation (no per-message delete).
    ``updated_at`` is touched on every appended message so the conversation
    list can order by recency.
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Auto-derived from the first user message when not provided explicitly.
    title: Mapped[str | None] = mapped_column(default=None)

    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="conversation", order_by="ConversationMessage.id"
    )


class ConversationMessage(Base, TimestampMixin):
    """One user or assistant turn in a master conversation.

    The assistant turn persists the loop's outcome denormalized: ``content``
    is the reply text (null when the run stopped without one), ``tool_calls``
    is the list of dispatched delegate calls with arguments and result/error
    (shape: ``app/ai/loop.py::ToolCallRecord``), ``stop_reason`` is the loop's
    termination cause. This is the display/chat-trajectory store; what each
    turn did, step by step with the apps' receipts, is the ledger's
    (:class:`LedgerTurn`, #69). No ``deleted_at``: messages are immutable once written
    and share their conversation's soft-delete fate.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[ConversationRole]
    content: Mapped[str | None] = mapped_column(default=None)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=None)
    stop_reason: Mapped[str | None] = mapped_column(default=None)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class DelegateThread(Base, TimestampMixin):
    """One master-conversation ↔ app subagent-thread mapping (the ThreadStore rows).

    Keyed by ``(master_conversation_id, app_name)`` — the same key the
    ``ThreadStore`` protocol (``app/fleet/context.py``) uses. The master id is
    a string on purpose: HTTP runs use the master :class:`Conversation` id
    stringified, but the protocol allows any driver-scoped key, so the column
    carries the key verbatim and takes no FK. Deleting a master conversation
    asks each app to delete its thread and then drops the rows, best-effort
    (``app/fleet/thread_cleanup.py``); a row that survives is inert, not
    dangerous — a pruned subagent thread 404s and the delegate tool recreates
    it. No ``deleted_at`` is needed: rows are hard-deleted cache entries, not
    user content.
    """

    __tablename__ = "delegate_threads"
    __table_args__ = (UniqueConstraint("master_conversation_id", "app_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    master_conversation_id: Mapped[str] = mapped_column(index=True)
    app_name: Mapped[str]
    subagent_conversation_id: Mapped[int]


# --- the ledger (#69) --------------------------------------------------------------------
#
# One exact record of what happened: the person's words, the plan each turn ran,
# its steps, the apps' receipts for each step, the jobs those started and the
# entities they touched. Episodic memory, the source for undo and "do the same",
# and the job store (#56). Append-only: rows are never edited after their turn
# ends, except a job's latest report and an entity's latest title. No soft
# delete: trashing a conversation leaves what the apps did on the record
# (forgetting is #81's, on purpose).


class LedgerTurn(Base, TimestampMixin):
    """One thing the person said or did, and how conductor's turn on it ended.

    ``kind`` is ``message`` (a turn the loop ran) or ``note`` (something done in
    the UI, ``POST …/notes``: words, no plan). Opened before the loop runs, so a
    turn in flight is on the record as ``running``; ``stop_reason`` and
    ``ended_at`` are set when it ends.
    """

    __tablename__ = "ledger_turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    user_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_messages.id"), default=None
    )
    assistant_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_messages.id"), default=None
    )
    kind: Mapped[str] = mapped_column(default="message")
    words: Mapped[str]
    surface: Mapped[str | None] = mapped_column(default=None)
    panel: Mapped[str | None] = mapped_column(default=None)
    # The loop's run id (#62): joins the turn to its logs, trace and captures.
    run_id: Mapped[str | None] = mapped_column(default=None, index=True)
    status: Mapped[str] = mapped_column(default="running")  # running | ended
    stop_reason: Mapped[str | None] = mapped_column(default=None)
    ended_at: Mapped[datetime | None] = mapped_column(default=None)

    plans: Mapped[list[LedgerPlan]] = relationship(back_populates="turn", order_by="LedgerPlan.id")


class LedgerPlan(Base, TimestampMixin):
    """What a turn set out to do.

    ``source`` is ``loop`` when the plan is whatever the loop's tool calls
    were, recorded after the fact (``body`` null), or ``compiled`` (#70) with
    its typed ``body``. A compiled plan the interpreter couldn't run yet is
    kept with ``acted`` false beside the loop plan that took the turn.
    ``answer`` is a direct answer's kind (``knowledge``/``decline``/``self``/
    ``chat``) when a compiled plan has no steps.
    """

    __tablename__ = "ledger_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("ledger_turns.id"), index=True)
    source: Mapped[str] = mapped_column(default="loop")
    body: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    answer: Mapped[str | None] = mapped_column(default=None)
    acted: Mapped[bool] = mapped_column(default=True)  # false: the loop took the turn
    # A compiled plan's fate (#70): ran, answered, paused, fell_back or invalid,
    # with why (the fallback reason, or where it stopped). A paused plan keeps
    # why it waits, the 0-based step it resumes at, and the app it's about.
    status: Mapped[str | None] = mapped_column(default=None)
    note: Mapped[str | None] = mapped_column(default=None)
    pause: Mapped[str | None] = mapped_column(default=None)
    resume_at: Mapped[int | None] = mapped_column(default=None)
    pause_app: Mapped[str | None] = mapped_column(default=None)

    turn: Mapped[LedgerTurn] = relationship(back_populates="plans")
    steps: Mapped[list[LedgerStep]] = relationship(back_populates="plan", order_by="LedgerStep.seq")


class LedgerStep(Base, TimestampMixin):
    """One step of a plan: for the loop, one tool call.

    ``kind`` is a step kind (``ask``, ``open``, ``confirm``, …; ``local`` for
    conductor's own read-only tools), ``app`` the fleet app it spoke to.
    ``outcome`` is conductor's sorting of the call (``app.ai.run_record``:
    ``done``/``failed``/``looked_up``/``not_run``); what the app itself did is
    its receipts.
    """

    __tablename__ = "ledger_steps"
    __table_args__ = (UniqueConstraint("plan_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("ledger_plans.id"), index=True)
    seq: Mapped[int]
    kind: Mapped[str]
    app: Mapped[str | None] = mapped_column(default=None, index=True)
    tool: Mapped[str]
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)
    outcome: Mapped[str]
    result: Mapped[str | None] = mapped_column(default=None)
    error: Mapped[str | None] = mapped_column(default=None)

    plan: Mapped[LedgerPlan] = relationship(back_populates="steps")
    receipts: Mapped[list[LedgerReceipt]] = relationship(
        back_populates="step", order_by="LedgerReceipt.seq"
    )


class LedgerReceipt(Base, TimestampMixin):
    """What one of the app's tool calls did, from its typed record (delegate
    v1.1), or conductor's own ``unknown`` when an app ran and said nothing it
    could read (a v1 app, or a call that stopped partway).

    ``source`` is ``app`` or ``conductor``. ``outcome`` is the app's
    (``done``/``refused``/``looked_up``/``not_run``) or ``unknown``, which
    counts as a possible write: the person can check and undo it, never
    "nothing changed". The undo handle is opaque and only its app reads it.
    """

    __tablename__ = "ledger_receipts"
    __table_args__ = (UniqueConstraint("step_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    step_id: Mapped[int] = mapped_column(ForeignKey("ledger_steps.id"), index=True)
    seq: Mapped[int]
    source: Mapped[str] = mapped_column(default="app")
    app: Mapped[str] = mapped_column(index=True)
    tool: Mapped[str | None] = mapped_column(default=None)
    outcome: Mapped[str]
    count: Mapped[int | None] = mapped_column(default=None)
    reason: Mapped[str | None] = mapped_column(default=None)
    untrusted: Mapped[bool] = mapped_column(default=False)
    undo_handle: Mapped[str | None] = mapped_column(default=None)
    undo_how: Mapped[str | None] = mapped_column(default=None)
    undo_none: Mapped[str | None] = mapped_column(default=None)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("ledger_jobs.id"), default=None)

    step: Mapped[LedgerStep] = relationship(back_populates="receipts")
    touched: Mapped[list[LedgerTouch]] = relationship(
        back_populates="receipt", order_by="LedgerTouch.id"
    )


class LedgerEntity(Base, TimestampMixin):
    """One thing in an app, as conductor has seen it: ``(app, kind, ref)``.

    The registry fills from receipts' ``touched`` refs. ``title`` is the latest
    seen (each receipt keeps the title as of then on its :class:`LedgerTouch`);
    ``untrusted`` marks outside text, shown quoted and never fed to anything that
    decides what to do.
    """

    __tablename__ = "ledger_entities"
    __table_args__ = (UniqueConstraint("app", "kind", "ref"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app: Mapped[str]
    kind: Mapped[str]
    ref: Mapped[str]
    title: Mapped[str | None] = mapped_column(default=None)
    untrusted: Mapped[bool] = mapped_column(default=False)
    last_seen_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class LedgerTouch(Base):
    """A receipt touched an entity: the title and trust as of that receipt."""

    __tablename__ = "ledger_touches"

    id: Mapped[int] = mapped_column(primary_key=True)
    receipt_id: Mapped[int] = mapped_column(ForeignKey("ledger_receipts.id"), index=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("ledger_entities.id"), index=True)
    title: Mapped[str | None] = mapped_column(default=None)
    untrusted: Mapped[bool] = mapped_column(default=False)

    receipt: Mapped[LedgerReceipt] = relationship(back_populates="touched")
    entity: Mapped[LedgerEntity] = relationship()


class LedgerJob(Base, TimestampMixin):
    """Long work an app started, by ``(app, ref)``, with its latest report.

    ``status`` uses the A2A words (``submitted``, ``working``,
    ``input_required``, ``completed``, ``failed``, ``canceled``). Today a job
    is only seen through later receipts that report it; the watcher that polls
    it is #76.
    """

    __tablename__ = "ledger_jobs"
    __table_args__ = (UniqueConstraint("app", "ref"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app: Mapped[str]
    ref: Mapped[str]
    status: Mapped[str]
    done: Mapped[int] = mapped_column(default=0)
    total: Mapped[int | None] = mapped_column(default=None)
    started_by_step_id: Mapped[int | None] = mapped_column(
        ForeignKey("ledger_steps.id"), default=None
    )
