"""Master-conversation persistence: threads and their user/assistant turns.

The only write path for ``conversations`` / ``conversation_messages`` (routes
are the callers today). Messages are immutable once appended; soft delete is
conversation-level only. The turn itself lives in ``app/ai/interpreter.py`` — this
module persists what it consumed and produced.

Delta from PCC's module: no ``activity_events`` rows — conductor's audit trail
for what a run *did* is the structured ``delegate_call`` log event
(``app/fleet/context.py``), so conversation lifecycle changes log a line rather
than landing in a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import confirmation, narrator
from app.ai.runtime import AgentRunResult, ToolCallRecord
from app.ai.recall import EarlierCall
from app.db.models import Conversation, ConversationMessage, ConversationRole, utcnow
from app.services.common import active, soft_delete

logger = structlog.get_logger(__name__)

# Auto-titles derived from the first user message are cut here, on a word
# boundary where one exists.
MAX_DERIVED_TITLE_LENGTH = 60


def list_conversations(db: Session, *, limit: int = 50) -> Sequence[Conversation]:
    """Active conversations, most recently touched first."""
    return (
        db.execute(
            active(Conversation)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def get_conversation(db: Session, conversation_id: int) -> Conversation | None:
    return db.execute(
        active(Conversation).where(Conversation.id == conversation_id)
    ).scalar_one_or_none()


def create_conversation(db: Session, *, title: str | None = None) -> Conversation:
    conversation = Conversation(title=title)
    db.add(conversation)
    db.flush()
    db.refresh(conversation)
    return conversation


def soft_delete_conversation(db: Session, conversation: Conversation) -> None:
    """Trash a conversation (messages ride along; nothing is hard-deleted)."""
    soft_delete(conversation)
    db.flush()


def list_messages(db: Session, conversation_id: int) -> Sequence[ConversationMessage]:
    """A conversation's messages, oldest first."""
    return (
        db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.id)
        )
        .scalars()
        .all()
    )


def stored_turns(db: Session, conversation_id: int) -> list[dict[str, Any]]:
    """A conversation's turns as :mod:`app.ai.recall` and the state block read
    them: role, content and the stored tool calls, oldest first.

    Never replayed as chat turns (#64): the model gets recall built from them,
    as data, so a false reply of its own is never read back as fact. Jobs'
    notes are left out (#76).
    """
    return [
        {
            "role": message.role.value,
            "content": message.content,
            "tool_calls": message.tool_calls,
        }
        for message in list_messages(db, conversation_id)
        # A job's note (#76) is not an exchange: it never voids the question
        # the last turn left, and the ledger keeps what the job did.
        if message.stop_reason != narrator.JOB_NOTE
    ]


# How far back recall looks across conversations: the newest assistant turns
# that called an app. Bounded so recall's cost never grows with the database.
EARLIER_TURNS_SCANNED = 200


def earlier_calls(db: Session, conversation_id: int) -> list[EarlierCall]:
    """Delegate calls from other live conversations, newest first, each with
    what the person had asked — recall's candidates for "that song" days later."""
    assistant_turns = (
        db.execute(
            select(ConversationMessage)
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(
                ConversationMessage.conversation_id != conversation_id,
                ConversationMessage.role == ConversationRole.assistant,
                ConversationMessage.tool_calls.is_not(None),
                Conversation.deleted_at.is_(None),
            )
            .order_by(ConversationMessage.id.desc())
            .limit(EARLIER_TURNS_SCANNED)
        )
        .scalars()
        .all()
    )
    calls: list[EarlierCall] = []
    for turn in assistant_turns:
        asked = db.execute(
            select(ConversationMessage.content)
            .where(
                ConversationMessage.conversation_id == turn.conversation_id,
                ConversationMessage.role == ConversationRole.user,
                ConversationMessage.id < turn.id,
            )
            .order_by(ConversationMessage.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        for stored in turn.tool_calls or []:
            if (
                confirmation.record_app(str(stored.get("tool", "")), stored.get("arguments") or {})
                is None
            ):
                continue
            calls.append(
                EarlierCall(
                    when=turn.created_at,
                    asked=asked or "",
                    record=ToolCallRecord.model_validate(stored),
                )
            )
    return calls


def append_user_message(
    db: Session, conversation: Conversation, content: str
) -> ConversationMessage:
    """Store one user turn; an untitled conversation is titled from it."""
    if conversation.title is None:
        conversation.title = _derive_title(content)
    message = ConversationMessage(
        conversation_id=conversation.id,
        role=ConversationRole.user,
        content=content,
    )
    conversation.updated_at = utcnow()
    db.add(message)
    db.flush()
    db.refresh(message)
    logger.info(
        "conversation_user_message",
        conversation_id=conversation.id,
        message_id=message.id,
    )
    return message


def append_assistant_message(
    db: Session, conversation: Conversation, run: AgentRunResult
) -> ConversationMessage:
    """Store the turn's outcome as one assistant turn."""
    message = ConversationMessage(
        conversation_id=conversation.id,
        role=ConversationRole.assistant,
        content=run.reply,
        tool_calls=([record.model_dump() for record in run.tool_calls] if run.tool_calls else None),
        stop_reason=run.stop_reason,
    )
    conversation.updated_at = utcnow()
    db.add(message)
    db.flush()
    db.refresh(message)
    logger.info(
        "conversation_assistant_message",
        conversation_id=conversation.id,
        message_id=message.id,
        stop_reason=run.stop_reason,
        tool_calls=len(run.tool_calls),
        run_id=run.run_id,
    )
    return message


def _derive_title(content: str) -> str:
    """First user message, cut to a title on a word boundary."""
    first_line = content.strip().splitlines()[0]
    if len(first_line) <= MAX_DERIVED_TITLE_LENGTH:
        return first_line
    cut = first_line[: MAX_DERIVED_TITLE_LENGTH + 1]
    head, _, _ = cut.rpartition(" ")
    return (head or cut[:MAX_DERIVED_TITLE_LENGTH]).rstrip() + "…"
