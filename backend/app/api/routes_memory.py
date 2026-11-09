"""The memory list (#81): what conductor remembers about the person, with where
each fact came from; edit and forget. **No model runs.**

Edits are the person's own words. Each edit or forget made with a
conversation open is recorded there as the person's note turn (the same as
``POST …/notes``), so the thread shows what they did; an edit's note turn is
the fact's new source. Forgetting takes the fact's turns out of recall and the
state block (:func:`app.services.facts.forget`); the forget note names the
fact, so it is forgotten too.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.rate_limit import rate_limit
from app.db.models import ConversationMessage, MemoryFact
from app.db.session import get_db
from app.schemas.conversations import MessageRead
from app.schemas.memory import FactChange, FactRead, FactSource, FactThread, FactUpdate
from app.services import conversations as conversations_service
from app.services import facts as facts_service
from app.services import ledger

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


def _read(db: Session, row: MemoryFact) -> FactRead:
    found = facts_service.source(db, row)
    return FactRead(
        id=row.id,
        text=row.text,
        created_at=row.created_at,
        updated_at=row.updated_at,
        source=(
            FactSource(
                turn_id=found.turn_id,
                kind=found.kind,
                words=found.words,
                said_at=found.said_at,
                surface=found.surface,
                thread=(
                    FactThread(id=found.thread_id, title=found.thread_title)
                    if found.thread_id is not None
                    else None
                ),
            )
            if found is not None
            else None
        ),
    )


def _fact_or_404(db: Session, fact_id: int) -> MemoryFact:
    row = facts_service.get(db, fact_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="fact not found")
    return row


def _note(
    db: Session, conversation_id: int | None, words: str
) -> tuple[ConversationMessage | None, int | None]:
    """The person's note turn in the open conversation: the message and its
    ledger turn's id. No conversation open, no note."""
    if conversation_id is None:
        return None, None
    conversation = conversations_service.get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")
    message = conversations_service.append_user_message(db, conversation, words)
    turn = ledger.note_turn(db, conversation, message)
    return message, turn.id if turn is not None else None


@router.get("/facts", response_model=list[FactRead])
def list_facts(db: Session = Depends(get_db)) -> list[FactRead]:
    """Every saved fact, newest first, each with where it came from."""
    return [_read(db, row) for row in facts_service.live(db)]


@router.patch(
    "/facts/{fact_id}",
    response_model=FactChange,
    dependencies=[Depends(rate_limit("memory_edits", per_min_attr="fleet_actions_per_min"))],
)
def edit_fact(fact_id: int, data: FactUpdate, db: Session = Depends(get_db)) -> FactChange:
    """The person's new words for a fact, recorded as their note turn."""
    row = _fact_or_404(db, fact_id)
    message, turn_id = _note(db, data.conversation_id, f"Changed a memory to “{data.text}”")
    facts_service.edit(db, row, data.text, note_turn_id=turn_id)
    db.commit()
    db.refresh(row)
    if message is not None:
        db.refresh(message)
    logger.info("memory_fact_edit", fact_id=fact_id, noted=message is not None)
    return FactChange(
        fact=_read(db, row),
        message=MessageRead.model_validate(message) if message is not None else None,
    )


@router.delete(
    "/facts/{fact_id}",
    response_model=FactChange,
    dependencies=[Depends(rate_limit("memory_edits", per_min_attr="fleet_actions_per_min"))],
)
def forget_fact(
    fact_id: int, conversation_id: int | None = None, db: Session = Depends(get_db)
) -> FactChange:
    """Forget a fact: gone from the list, recall and the state block. The
    note turn that records it names the fact, so it is forgotten as well."""
    row = _fact_or_404(db, fact_id)
    message, turn_id = _note(db, conversation_id, f"Forgot “{row.text}” in the memory list")
    facts_service.forget(db, row, by_turn_id=turn_id)
    db.commit()
    if message is not None:
        db.refresh(message)
    logger.info("memory_fact_forget", fact_id=fact_id, noted=message is not None)
    return FactChange(
        fact=None, message=MessageRead.model_validate(message) if message is not None else None
    )
