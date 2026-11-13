from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

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


# Bounds on the voice client's heard report (#77): an hour of audio, and a
# reply far longer than any spoken one.
MAX_HEARD_MS = 3_600_000
MAX_HEARD_CHARS = 100_000


class HeardReport(BaseModel):
    """How much of the last spoken reply the person heard before they cut in
    (#77): the voice client's ``Heard`` (``frontend/src/voice/tts.ts``), its
    fields in snake case. Characters are the browser's (UTF-16 code units)."""

    model_config = ConfigDict(extra="forbid")

    played_ms: Annotated[int, Field(ge=0, le=MAX_HEARD_MS)]
    total_ms: Annotated[int, Field(ge=0, le=MAX_HEARD_MS)] | None = None
    heard_chars: Annotated[int, Field(ge=0, le=MAX_HEARD_CHARS)] | None = None
    total_chars: Annotated[int, Field(ge=0, le=MAX_HEARD_CHARS)]

    @model_validator(mode="after")
    def _within_the_reply(self) -> HeardReport:
        # The client rounds an estimate; a report is never worth a lost turn.
        if self.heard_chars is not None and self.heard_chars > self.total_chars:
            self.heard_chars = self.total_chars
        return self


class MessageCreate(BaseModel):
    content: AgentMessageText
    # Where the person is (#64), for the state block: the surface they are on
    # and the app panel they have open. Optional; absent says nothing.
    surface: Literal["desktop", "phone", "voice", "room_speaker"] | None = None
    panel: Annotated[str, StringConstraints(strip_whitespace=True, max_length=60)] | None = None
    # Incognito (#82): this turn's words and reply are never written. The page
    # carries the switch per request and turns it off itself.
    incognito: bool = False
    # Barge-in (#77): the person spoke over the last reply; how much of it they
    # heard. Kept on that reply's ledger turn, never for an incognito turn.
    heard: HeardReport | None = None


class ToolCallRead(BaseModel):
    """One dispatched tool call as persisted on an assistant message.

    Mirrors ``app/ai/runtime.py::ToolCallRecord``; exactly one of ``result`` /
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


class NoteThreadRead(BaseModel):
    """The side thread a timeline note came from (#139)."""

    id: int
    title: str | None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: ConversationRole
    content: str | None
    tool_calls: list[ToolCallRead] | None = None
    stop_reason: str | None = None
    created_at: UTCDateTime
    # The turn's surface (#82: each timeline entry is tagged with it), and
    # whether it was incognito (its words and reply were never written).
    surface: str | None = None
    incognito: bool = False
    # A side thread's result note (#139): the thread it came from.
    thread: NoteThreadRead | None = None


class ConversationDetail(ConversationRead):
    """A conversation with its message history, oldest first (on the
    timeline, its newest messages)."""

    messages: list[MessageRead]


class TimelineDetail(ConversationDetail):
    """The one timeline (#82), with what the page needs to run it: the quiet
    gap after which incognito turns itself off, and the line it shows."""

    quiet_gap_seconds: int
    incognito_notice: str


class TimelineHit(BaseModel):
    """One timeline turn that matches a search (#82): what was said, the
    reply, and the receipt titles that matched. Incognito and forgotten turns
    are never hits."""

    turn_id: int
    message_id: int | None
    at: UTCDateTime
    surface: str | None
    kind: str
    said: str | None
    reply: str | None
    titles: list[str]
    matched: list[Literal["words", "reply", "receipt"]]


class MessageExchange(BaseModel):
    """What one ``POST …/messages`` produced: the stored user turn and the
    assistant turn conductor answered it with."""

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


# --- side threads (#139) ------------------------------------------------------------------


class ThreadCreate(BaseModel):
    """Start a side thread from the page: its name, if the person gave one.
    Incognito keeps no name."""

    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=60)] | None = None
    incognito: bool = False


class CarryRead(BaseModel):
    """One thing a side thread carries in, as data: an entity by its title
    (``untrusted``: outside text, shown quoted), or a saved fact's words."""

    type: Literal["entity", "fact"]
    app: str | None = None
    kind: str | None = None
    title: str | None = None
    untrusted: bool = False
    text: str | None = None


class ThreadRead(BaseModel):
    """A side thread: its name, what it carries (as it stands now), and, once
    closed, when and the result it posted to the timeline."""

    id: int
    title: str | None
    created_at: UTCDateTime
    updated_at: UTCDateTime
    closed_at: UTCDateTime | None = None
    result: str | None = None
    carries: list[CarryRead]


class ThreadClose(BaseModel):
    """Close a side thread: the surface the person closed it from."""

    surface: Literal["desktop", "phone", "voice", "room_speaker"] | None = None


class ThreadClosed(BaseModel):
    """A closed side thread and the result note it posted to the timeline."""

    thread: ThreadRead
    message: MessageRead
