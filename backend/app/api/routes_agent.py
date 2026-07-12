"""Agent chat API: conversations, history, and the message → loop round trip.

``POST /agent/conversations/{id}/messages`` is the one model-calling endpoint:
it stores the user turn, binds the delegation context (DB-backed thread store,
per-turn budget, audit driver), runs the master loop synchronously — v1 is
non-streaming — then stores and returns the assistant turn. While it blocks,
``GET /agent/conversations/{id}/activity`` reports what the loop is doing
(``app/api/turn_activity.py``) so the UI can show "asking chess…" progress.

Deltas from PCC's routes_agent: no ``X-Agent-Actor`` handling — conductor is
the delegation root (STANDARD.md depth-1), so no other agent may drive this
API and every run carries the loop's own identity; plus the delegation-context
binding and the activity endpoint, which PCC's local-tool agent doesn't need.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Sequence

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.ai.loop import AgentLoop, LoopActivity, loop_from_settings
from app.ai.provider import ProviderError
from app.ai.providers.llamacpp import provider_from_settings
from app.api import turn_activity
from app.api.rate_limit import rate_limit
from app.config import get_settings
from app.db.models import Conversation
from app.db.session import SessionLocal, get_db
from app.fleet.context import DelegationContext, ThreadStore, use_delegation_context
from app.fleet.thread_store import DbThreadStore
from app.schemas.conversations import (
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    MessageCreate,
    MessageExchange,
    MessageRead,
    TurnActivityRead,
)
from app.services import conversations as conversations_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_loop(request: Request) -> Generator[AgentLoop, None, None]:
    """The loop over the configured provider, carrying the fleet prompt layer
    the lifespan rendered; tests override this dependency."""
    provider = provider_from_settings()
    try:
        yield loop_from_settings(
            provider, fleet_section=getattr(request.app.state, "fleet_section", None)
        )
    finally:
        provider.close()


def get_thread_store() -> ThreadStore:
    """The DB-backed subagent-thread map; tests override this dependency."""
    return DbThreadStore(SessionLocal)


def _get_or_404(db: Session, conversation_id: int) -> Conversation:
    conversation = conversations_service.get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.get("/conversations", response_model=list[ConversationRead])
def list_conversations(db: Session = Depends(get_db)) -> Sequence[Conversation]:
    return conversations_service.list_conversations(db)


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)) -> Conversation:
    conversation = conversations_service.create_conversation(db, title=data.title)
    db.commit()
    db.refresh(conversation)
    logger.info("conversation_created", conversation_id=conversation.id)
    return conversation


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)) -> ConversationDetail:
    conversation = _get_or_404(db, conversation_id)
    return ConversationDetail(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=[
            MessageRead.model_validate(message)
            for message in conversations_service.list_messages(db, conversation.id)
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)) -> None:
    conversation = _get_or_404(db, conversation_id)
    conversations_service.soft_delete_conversation(db, conversation)
    db.commit()
    logger.info("conversation_deleted", conversation_id=conversation_id)


@router.get("/conversations/{conversation_id}/activity", response_model=TurnActivityRead)
def get_turn_activity(conversation_id: int, db: Session = Depends(get_db)) -> TurnActivityRead:
    """What the conversation's in-flight turn is doing right now, if any.

    The poll target while ``POST …/messages`` blocks — v1's no-SSE progress
    channel. 404s for an unknown conversation so a stale tab learns the thread
    is gone rather than polling forever.
    """
    _get_or_404(db, conversation_id)
    activity = turn_activity.get(conversation_id)
    if activity is None:
        return TurnActivityRead(active=False)
    return TurnActivityRead(
        active=True,
        kind=activity.kind,
        tool=activity.tool,
        iteration=activity.iteration,
        elapsed_seconds=round(time.monotonic() - activity.started_at, 1),
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageExchange,
    dependencies=[Depends(rate_limit("agent_messages", per_min_attr="agent_messages_per_min"))],
)
def post_message(
    conversation_id: int,
    data: MessageCreate,
    db: Session = Depends(get_db),
    loop: AgentLoop = Depends(get_agent_loop),
    thread_store: ThreadStore = Depends(get_thread_store),
) -> MessageExchange:
    """Store the user turn, run the master loop, store and return the assistant turn.

    The user message is committed *before* the loop runs: a run can take
    minutes (each iteration may wrap a subagent's full loop) and a provider
    failure — surfaced as 502 — must not swallow what the user said. The
    delegation context is bound for exactly the run: the DB-backed thread
    store keyed by this conversation, the per-turn per-app call budget, and
    the ``agent:loop`` audit driver.
    """
    conversation = _get_or_404(db, conversation_id)
    history = conversations_service.history_for_loop(db, conversation.id)
    user_message = conversations_service.append_user_message(db, conversation, data.content)
    db.commit()

    context = DelegationContext(
        master_conversation_id=str(conversation_id),
        thread_store=thread_store,
        calls_per_turn_per_app=get_settings().conductor_delegate_calls_per_turn_per_app,
    )

    def report(activity: LoopActivity) -> None:
        turn_activity.update(
            conversation_id, kind=activity.kind, tool=activity.tool, iteration=activity.iteration
        )

    turn_activity.begin(conversation_id)
    try:
        with use_delegation_context(context):
            run = loop.run(data.content, history=history, on_activity=report)
    except ProviderError as exc:
        logger.error("agent_run_failed", conversation_id=conversation_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"agent run failed: {exc}",
        ) from exc
    finally:
        turn_activity.end(conversation_id)

    assistant_message = conversations_service.append_assistant_message(db, conversation, run)
    db.commit()
    return MessageExchange(
        user_message=MessageRead.model_validate(user_message),
        assistant_message=MessageRead.model_validate(assistant_message),
    )
