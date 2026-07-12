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
    termination cause. This is the display/chat-trajectory store; the audit
    source of truth for delegation stays the structured ``delegate_call`` log
    event (``app/fleet/context.py``) — conductor has no local mutations to
    audit beyond that. No ``deleted_at``: messages are immutable once written
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
    carries the key verbatim and takes no FK. Rows whose master conversation
    is gone are inert, not dangerous — a pruned subagent thread 404s and the
    delegate tool recreates it — so nothing cascades and no ``deleted_at`` is
    needed (``forget`` hard-deletes the one row; it is a cache entry, not user
    content).
    """

    __tablename__ = "delegate_threads"
    __table_args__ = (UniqueConstraint("master_conversation_id", "app_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    master_conversation_id: Mapped[str] = mapped_column(index=True)
    app_name: Mapped[str]
    subagent_conversation_id: Mapped[int]
