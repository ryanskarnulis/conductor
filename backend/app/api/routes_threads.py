"""Side threads API (#139): start one from the page, list the open ones, close one.

A side thread is a conversation (``kind = thread``): its turns, notes, live
channel and Stop go through the usual ``/agent/conversations/{id}/…``
routes, which read it whole and with what it carries
(:mod:`app.services.threads`). These routes are the page's controls around
it. Starting one is the person's act in the UI, never a step a plan takes.

``POST /agent/threads/{id}/close`` narrates the thread's result from its
receipts (:mod:`app.ai.thread_note`) and posts it into the timeline as a
note; **409** for a thread already closed or with a turn still running.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai import foreground, narrator, thread_note
from app.ai.provider import ChatResult
from app.ai.providers.llamacpp import provider_from_settings
from app.api import turn_activity
from app.api.rate_limit import rate_limit
from app.db.models import Conversation, utcnow
from app.db.session import get_db
from app.schemas.conversations import (
    CarryRead,
    MessageRead,
    NoteThreadRead,
    ThreadClose,
    ThreadClosed,
    ThreadCreate,
    ThreadRead,
)
from app.services import threads as threads_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/agent/threads", tags=["agent"])


def get_thread_chat() -> Generator[narrator.Chat | None, None, None]:
    """The narrator's model for a thread's result; tests override this
    dependency (``None``: code's own note)."""
    provider = provider_from_settings()

    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        return provider.chat(messages, max_tokens=max_tokens, role="narrator")

    try:
        yield chat
    finally:
        provider.close()


def thread_read(db: Session, thread: Conversation) -> ThreadRead:
    """A thread as the page reads it, with what it carries as it stands now."""
    now = threads_service.carried(db, thread)
    carries = [
        CarryRead(
            type="entity",
            app=entity.app,
            kind=entity.kind,
            title=entity.title or f"{entity.kind} {entity.id}",
            untrusted=entity.untrusted,
        )
        for entity in now.entities
    ] + [CarryRead(type="fact", text=fact.text) for fact in now.facts]
    return ThreadRead(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        closed_at=thread.closed_at,
        result=thread.result,
        carries=carries,
    )


def _get_or_404(db: Session, thread_id: int) -> Conversation:
    thread = threads_service.get(db, thread_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Side thread not found")
    return thread


@router.get("", response_model=list[ThreadRead])
def list_threads(db: Session = Depends(get_db)) -> Sequence[ThreadRead]:
    """The side threads still open, oldest first."""
    return [thread_read(db, thread) for thread in threads_service.open_threads(db)]


@router.post(
    "",
    response_model=ThreadRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit("agent_notes", per_min_attr="fleet_actions_per_min"))],
)
def start_thread(data: ThreadCreate, db: Session = Depends(get_db)) -> ThreadRead:
    """Start a side thread carrying the entities and saved facts of the
    timeline's current window. **No model runs.**"""
    thread = threads_service.start(db, title=data.title, hidden=data.incognito)
    db.commit()
    db.refresh(thread)
    return thread_read(db, thread)


@router.get("/{thread_id}", response_model=ThreadRead)
def get_thread(thread_id: int, db: Session = Depends(get_db)) -> ThreadRead:
    return thread_read(db, _get_or_404(db, thread_id))


@router.post(
    "/{thread_id}/close",
    response_model=ThreadClosed,
    dependencies=[Depends(rate_limit("agent_messages", per_min_attr="agent_messages_per_min"))],
)
def close_thread(
    thread_id: int,
    data: ThreadClose,
    db: Session = Depends(get_db),
    chat: narrator.Chat | None = Depends(get_thread_chat),
) -> ThreadClosed:
    """Close a side thread: its result, narrated from its receipts with code's
    checks, is posted into the timeline as a note (it lapses no waiting
    question there). Posted once: a second close is **409**, as is a close
    while a turn still runs in the thread."""
    thread = _get_or_404(db, thread_id)
    if threads_service.is_closed(thread):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already closed")
    if turn_activity.get(thread.id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a turn is still running in this side thread; stop it or let it finish",
        )
    ending = threads_service.ending(db, thread)
    # Narrate outside any transaction; the GPU is the person's meanwhile (#78).
    db.commit()
    foreground.entered()
    try:
        note = thread_note.thread_note(ending.end, surface=data.surface, chat=chat)
    finally:
        foreground.left()
    text = note.reply or thread_note.fallback(ending.end, data.surface)
    message = threads_service.close(db, thread, text, ending.receipts, now=utcnow())
    if message is None:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already closed")
    db.commit()
    db.refresh(message)
    db.refresh(thread)
    logger.info("agent_thread_closed", thread_id=thread.id, path=note.path)
    read = MessageRead.model_validate(message)
    read.thread = NoteThreadRead(id=thread.id, title=thread.title)
    return ThreadClosed(thread=thread_read(db, thread), message=read)
