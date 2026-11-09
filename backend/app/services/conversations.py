"""Master-conversation persistence: the timeline and its user/assistant turns.

The only write path for ``conversations`` / ``conversation_messages`` (routes
are the callers today). Messages are immutable once appended; soft delete is
conversation-level only. The turn itself lives in ``app/ai/interpreter.py`` — this
module persists what it consumed and produced.

**One timeline** (#82): every surface writes to one endless conversation,
created on first use (:func:`timeline`). Conversations from before it are
``legacy``: gone from the listing and the API, reachable only through recall
(#58 (c)). The planner's "this conversation" is the timeline since its last
quiet gap (:data:`QUIET_GAP`, :func:`window_since`); anything older reaches it
only as cross-day recall, labelled with its day.

**Incognito** (#82, :mod:`app.ai.incognito`): an incognito turn's words and
reply are stored as the placeholder, its calls without their words; what
the apps did stays on the record.

Delta from PCC's module: no ``activity_events`` rows — conductor's audit trail
for what a run *did* is the structured ``delegate_call`` log event
(``app/fleet/context.py``), so conversation lifecycle changes log a line rather
than landing in a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.ai import confirmation, facts, incognito, narrator
from app.ai.runtime import AgentRunResult, ToolCallRecord
from app.ai.recall import EarlierCall
from app.db.models import (
    Conversation,
    ConversationKind,
    ConversationMessage,
    ConversationRole,
    LedgerTurn,
    utcnow,
)
from app.services.common import active, soft_delete

logger = structlog.get_logger(__name__)

# Auto-titles derived from the first user message are cut here, on a word
# boundary where one exists.
MAX_DERIVED_TITLE_LENGTH = 60


# The timeline's name, where a conversation's title is shown (recall's citations).
TIMELINE_TITLE = "Timeline"

# The quiet gap (#82): the planner's "this conversation" is the timeline's
# turns since the last stretch this long with nothing said. The UI's
# incognito switch turns itself off after the same gap.
QUIET_GAP = timedelta(hours=3)
# How many of the newest messages :func:`window_since` reads: a busy day past
# this is cut here, and recall's own character budgets cap what's shown.
WINDOW_SCANNED = 400
# How many of the timeline's newest messages the page is sent.
TIMELINE_MESSAGES = 500

_SHOWN_KINDS = (ConversationKind.timeline.value, ConversationKind.thread.value)


def list_conversations(db: Session, *, limit: int = 50) -> Sequence[Conversation]:
    """Active conversations (the timeline; never a legacy one), most recently
    touched first."""
    return (
        db.execute(
            active(Conversation)
            .where(Conversation.kind.in_(_SHOWN_KINDS))
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def get_conversation(db: Session, conversation_id: int) -> Conversation | None:
    """A live conversation the API may show: a legacy one is reachable only
    through recall (#58 (c)), so it reads as not found."""
    return db.execute(
        active(Conversation).where(
            Conversation.id == conversation_id, Conversation.kind.in_(_SHOWN_KINDS)
        )
    ).scalar_one_or_none()


def timeline(db: Session) -> Conversation:
    """The one timeline, created (clean, #58 (c)) on first use."""
    found = db.execute(
        active(Conversation)
        .where(Conversation.kind == ConversationKind.timeline.value)
        .order_by(Conversation.id)
        .limit(1)
    ).scalar_one_or_none()
    if found is not None:
        return found
    conversation = Conversation(title=TIMELINE_TITLE, kind=ConversationKind.timeline.value)
    db.add(conversation)
    db.flush()
    db.refresh(conversation)
    logger.info("timeline_created", conversation_id=conversation.id)
    return conversation


def create_conversation(db: Session, *, title: str | None = None) -> Conversation:
    """There is no new conversation (#82): the timeline, as it is. ``title``
    is accepted for old clients and ignored."""
    return timeline(db)


def is_timeline(conversation: Conversation) -> bool:
    return conversation.kind == ConversationKind.timeline.value


def window_since(
    db: Session, conversation_id: int, *, now: datetime | None = None
) -> datetime | None:
    """Where the planner's "this conversation" starts (#82): the first message
    after the last quiet gap (:data:`QUIET_GAP`), as a UTC timestamp. Messages
    at or after it are in the window. ``now`` itself counts: when the newest
    message is a gap old, the window is empty (it starts now). ``None``: every
    scanned message is in it (no gap found). Jobs' notes never break a gap."""
    now = now or utcnow()
    stamps = (
        db.execute(
            select(ConversationMessage.created_at)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                or_(
                    ConversationMessage.stop_reason.is_(None),
                    ConversationMessage.stop_reason != narrator.JOB_NOTE,
                ),
            )
            .order_by(ConversationMessage.id.desc())
            .limit(WINDOW_SCANNED)
        )
        .scalars()
        .all()
    )
    later = _aware(now)
    for stamp in stamps:
        at = _aware(stamp)
        if later - at >= QUIET_GAP:
            return later
        later = at
    if len(stamps) == WINDOW_SCANNED:
        return later
    return None


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; conductor writes UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def soft_delete_conversation(db: Session, conversation: Conversation) -> None:
    """Trash a conversation (messages ride along; nothing is hard-deleted)."""
    soft_delete(conversation)
    db.flush()


def list_messages(
    db: Session,
    conversation_id: int,
    *,
    since: datetime | None = None,
    newest: int | None = None,
) -> Sequence[ConversationMessage]:
    """A conversation's messages, oldest first: those at or after ``since``,
    and only the ``newest`` few when given."""
    query = select(ConversationMessage).where(
        ConversationMessage.conversation_id == conversation_id
    )
    if since is not None:
        query = query.where(ConversationMessage.created_at >= since)
    if newest is not None:
        kept = (
            db.execute(query.order_by(ConversationMessage.id.desc()).limit(newest)).scalars().all()
        )
        return list(reversed(kept))
    return db.execute(query.order_by(ConversationMessage.id)).scalars().all()


def stored_turns(
    db: Session, conversation_id: int, *, since: datetime | None = None
) -> list[dict[str, Any]]:
    """A conversation's turns as :mod:`app.ai.recall` and the state block read
    them: role, content and the stored tool calls, oldest first; on the
    timeline, those since the quiet gap (``since``, :func:`window_since`).

    Never replayed as chat turns (#64): the model gets recall built from them,
    as data, so a false reply of its own is never read back as fact. Jobs'
    notes are left out (#76). A turn the person had forgotten (#81) stays a
    turn, with nothing of it: no words, no reply, no record. An incognito
    turn (#82) is the same, except an app's question still waiting on the
    person, so their yes can answer it. The memory app's records never ride
    along (the state block lists what is saved now).
    """
    forgotten = forgotten_messages(db, conversation_id)
    hidden = incognito_messages(db, conversation_id)
    turns = []
    for message in list_messages(db, conversation_id, since=since, newest=WINDOW_SCANNED):
        # A job's note (#76) is not an exchange: it never voids the question
        # the last turn left, and the ledger keeps what the job did.
        if message.stop_reason == narrator.JOB_NOTE:
            continue
        content: str | None
        calls: list[dict[str, Any]] | None
        if message.id in forgotten:
            content, calls = FORGOTTEN, None
        elif message.id in hidden:
            content, calls = incognito.PLACEHOLDER, _waiting_only(message.tool_calls)
        else:
            content, calls = message.content, _without_memory(message.tool_calls)
        turns.append({"role": message.role.value, "content": content, "tool_calls": calls})
    return turns


def incognito_messages(db: Session, conversation_id: int | None = None) -> set[int]:
    """The messages of incognito turns (#82)."""
    query = select(LedgerTurn.user_message_id, LedgerTurn.assistant_message_id).where(
        LedgerTurn.incognito.is_(True)
    )
    if conversation_id is not None:
        query = query.where(LedgerTurn.conversation_id == conversation_id)
    return {message for pair in db.execute(query).all() for message in pair if message is not None}


def _waiting_only(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """An incognito turn's calls that left an app's question waiting: what a
    yes needs, and nothing the app answered."""
    kept = [
        {**incognito.tool_call(call), "result": None, "app_record": None}
        for call in calls or []
        if call.get("app_confirm")
    ]
    return kept or None


# What a forgotten turn's words and reply become in what the model is shown (#81).
FORGOTTEN = "(forgotten at the person's request)"


def forgotten_messages(db: Session, conversation_id: int | None = None) -> set[int]:
    """The messages of turns the person had forgotten (#81)."""
    query = select(LedgerTurn.user_message_id, LedgerTurn.assistant_message_id).where(
        LedgerTurn.forgotten_at.is_not(None)
    )
    if conversation_id is not None:
        query = query.where(LedgerTurn.conversation_id == conversation_id)
    return {message for pair in db.execute(query).all() for message in pair if message is not None}


def _without_memory(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not calls:
        return calls
    kept = [call for call in calls if call.get("tool") != facts.TOOL]
    return kept or None


# How far back recall looks across conversations: the newest assistant turns
# that called an app. Bounded so recall's cost never grows with the database.
EARLIER_TURNS_SCANNED = 200


def earlier_calls(
    db: Session, conversation_id: int, *, since: datetime | None = None
) -> list[EarlierCall]:
    """Delegate calls from before this conversation's window, newest first,
    each with what the person had asked — recall's candidates for "that song"
    days later. That is every other live conversation (legacy ones included),
    and on the timeline its turns before the quiet gap (``since``); each hit
    is labelled with its day. Incognito and forgotten turns are never one."""
    outside = ConversationMessage.conversation_id != conversation_id
    before = outside if since is None else or_(outside, ConversationMessage.created_at < since)
    assistant_turns = (
        db.execute(
            select(ConversationMessage)
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(
                before,
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
    skipped = forgotten_messages(db) | incognito_messages(db) if assistant_turns else set()
    calls: list[EarlierCall] = []
    for turn in assistant_turns:
        if turn.id in skipped:
            continue
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
            if stored.get("tool") == facts.TOOL or (
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
    db: Session, conversation: Conversation, content: str, *, hidden: bool = False
) -> ConversationMessage:
    """Store one user turn; an untitled conversation is titled from it. An
    incognito turn (``hidden``, #82) stores the placeholder, never the words."""
    if hidden:
        content = incognito.PLACEHOLDER
    if conversation.title is None and not hidden:
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
    db: Session, conversation: Conversation, run: AgentRunResult, *, hidden: bool = False
) -> ConversationMessage:
    """Store the turn's outcome as one assistant turn. An incognito turn
    (``hidden``, #82) stores the placeholder for the reply and its calls
    without their words: the apps' receipts and a waiting question stay."""
    calls = [record.model_dump() for record in run.tool_calls] if run.tool_calls else None
    if hidden and calls:
        calls = [incognito.tool_call(call) for call in calls]
    message = ConversationMessage(
        conversation_id=conversation.id,
        role=ConversationRole.assistant,
        content=incognito.PLACEHOLDER if hidden and run.reply is not None else run.reply,
        tool_calls=calls,
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
