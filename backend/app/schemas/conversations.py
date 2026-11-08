from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.db.models import ConversationRole
from app.schemas.common import NonBlankStr, UTCDateTime

# Cap on one chat turn. Generous for typed input while bounding what a run
# feeds the local model's context window — the same cap the fleet's delegate
# APIs enforce (see app/fleet/tools.py::MAX_DELEGATE_MESSAGE_LENGTH).
MAX_AGENT_MESSAGE_LENGTH = 8_000

AgentMessageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_AGENT_MESSAGE_LENGTH),
]


class ConversationCreate(BaseModel):
    # Optional: an untitled conversation is titled from its first user message.
    title: NonBlankStr | None = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: UTCDateTime
    # Touched on every appended message — the conversation list's recency order.
    updated_at: UTCDateTime


class MessageCreate(BaseModel):
    content: AgentMessageText
    # Where the person is (#64), for the state block: the surface they are on
    # and the app panel they have open. Optional; absent says nothing.
    surface: Literal["desktop", "phone", "voice", "room_speaker"] | None = None
    panel: Annotated[str, StringConstraints(strip_whitespace=True, max_length=60)] | None = None


class ToolCallRead(BaseModel):
    """One dispatched tool call as persisted on an assistant message.

    Mirrors ``app/ai/loop.py::ToolCallRecord``; exactly one of ``result`` /
    ``error`` is set.
    """

    tool: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    # For a delegate call, the tools the app ran while answering — persisted and
    # sent to the UI, never to the model.
    app_tools: list[str] | None = None
    # For a delegate call to a v1.1 app, its typed record of what it did (#67).
    app_record: list[dict[str, Any]] | None = None
    # The app's armed question and its token, kept for the person's yes (#68).
    app_confirm: dict[str, Any] | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: ConversationRole
    content: str | None
    tool_calls: list[ToolCallRead] | None = None
    stop_reason: str | None = None
    created_at: UTCDateTime


class ConversationDetail(ConversationRead):
    """A conversation with its full message history, oldest first."""

    messages: list[MessageRead]


class MessageExchange(BaseModel):
    """What one ``POST …/messages`` produced: the stored user turn and the
    assistant turn the loop answered it with."""

    user_message: MessageRead
    assistant_message: MessageRead


class RunCancelRead(BaseModel):
    """Outcome of a Stop (#66): ``cancel_requested`` is True when a turn was in
    flight and has been asked to stop; False means nothing was running."""

    cancel_requested: bool


class JobCancelCreate(BaseModel):
    """A tap on a job card's Cancel (#90): the job by its app and ref."""

    app: NonBlankStr
    ref: NonBlankStr


class JobCancelRead(BaseModel):
    """What the app answered to a card's Cancel, and the note it left.

    ``done``/``total`` are the job's count at the moment of the cancel.
    ``cancel_requested`` is False when the job had already ended; ``status``
    is the app's own word, so a job still ``working`` has been asked to stop,
    not stopped. ``message`` is the person's note turn recording the tap.
    """

    app: str
    ref: str
    status: str
    done: int
    total: int | None
    cancel_requested: bool
    message: MessageRead


class TurnActivityRead(BaseModel):
    """What conductor is doing right now for one conversation's in-flight turn.

    The poll fallback while ``POST …/messages`` blocks (``GET …/events`` is
    the stream, #38): ``kind``
    "model" means waiting on a provider turn, "tool" means the named tool —
    for an ``ask_<app>`` tool, that is a live delegate call the UI can render
    as "asking chess…". All fields but ``active`` are null when idle.
    """

    active: bool
    kind: Literal["model", "tool"] | None = None
    tool: str | None = None
    iteration: int | None = None
    elapsed_seconds: float | None = None
    # A compiled plan's ack (#70): what conductor is about to do, said as soon
    # as the plan is valid ("On it, asking music."). Never a claim of done.
    ack: str | None = None
