"""The memory list's shapes (#81): saved facts and where each came from."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, StringConstraints

from app.ai.facts import MAX_FACT_CHARS
from app.schemas.common import UTCDateTime
from app.schemas.conversations import MessageRead

FactText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_FACT_CHARS)
]


class FactThread(BaseModel):
    id: int
    title: str | None


class FactSource(BaseModel):
    """The turn a fact came from: what the person said (``None`` once that
    turn was forgotten), when, on what surface, and in which conversation
    (``None`` once it was trashed). ``kind`` is ``message`` (they said it) or
    ``note`` (they changed it in the memory list)."""

    turn_id: int
    kind: str
    words: str | None
    said_at: UTCDateTime
    surface: str | None
    thread: FactThread | None


class FactRead(BaseModel):
    id: int
    text: str
    created_at: UTCDateTime
    updated_at: UTCDateTime
    # ``None``: changed in the memory list outside any conversation.
    source: FactSource | None


class FactUpdate(BaseModel):
    """The person's new words for a fact, and the conversation open on screen
    (where the change is recorded as their note turn), if any."""

    text: FactText
    conversation_id: int | None = None


class FactChange(BaseModel):
    """What an edit or a forget did: the fact as it is now (``None`` once
    forgotten) and the note turn that recorded it, if a conversation was open."""

    fact: FactRead | None
    message: MessageRead | None
