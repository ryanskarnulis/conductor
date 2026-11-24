"""The cards waiting on the person (#251): every open question, one read.

A question never blocks the chat. It becomes a card, and only a tap approves
it. This is the one place that says which questions are open:

- an app's ``confirm_required`` questions, from the asking turn's record
  (:mod:`app.ai.confirmation`): open until its card is answered, its token's
  window closes, or a newer question about the same thing replaces it;
- a waiting plan's question, from the ledger (:func:`ledger.waiting_plans`):
  code's gate, an undo's, a think's proposal, a pick-one.

All of them lapse when the conversation's window ends (the quiet gap). An
open question with nothing to tap (an ``ask_user`` with no options) is not a
card: it waits on the person's words for one turn, as before.

One request's cards are a group (``group``: the message that asked them).
``latest`` marks the cards the turn right before asked: the only ones the
person's words can pick from, change or dismiss.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.ai import confirmation, interpreter
from app.ai.confirmation import PendingConfirm
from app.ai.plan import PausedPlan
from app.db.models import Conversation, ConversationRole
from app.services import conversations as conversations_service
from app.services import ledger

CardKind = Literal["app_confirm", "gate", "proposal", "pick"]

PROPOSAL_QUESTION = "Run the plan I proposed?"
MAX_QUESTION_CHARS = 600
MAX_OPTION_CHARS = 120


@dataclass(frozen=True)
class Card:
    """One open question, as the page shows it and a tap answers it."""

    id: str
    kind: CardKind
    question: str
    asked_at: datetime
    # The message that asked it: one request's cards share it.
    group: int | None
    app: str | None = None
    # A pick-one's options, in the order offered.
    options: tuple[str, ...] = ()
    expires_at: str | None = None
    latest: bool = False
    incognito: bool = False
    # What answers it: an app's token, or the waiting plan.
    pending: PendingConfirm | None = None
    plan: ledger.WaitingPlan | None = None


def confirm_id(token: str) -> str:
    return f"confirm:{token}"


def plan_id(plan: int | None) -> str:
    return f"plan:{plan}"


def open_cards(
    db: Session, conversation: Conversation, *, since: datetime | None, now: datetime
) -> list[Card]:
    """Every card open in this window, oldest first."""
    last = conversations_service.last_exchange(db, conversation.id, since=since)
    newest = last.id if last is not None else None
    forgotten = conversations_service.forgotten_messages(db, conversation.id)
    hidden = conversations_service.incognito_messages(db, conversation.id)
    by_subject: dict[tuple[Any, ...], Card] = {}
    for message in conversations_service.list_messages(
        db, conversation.id, since=since, newest=conversations_service.WINDOW_SCANNED
    ):
        if message.role != ConversationRole.assistant or message.id in forgotten:
            continue
        calls = message.tool_calls or []
        # A newer question about the same thing replaces the old card, also
        # once the newer one is answered: the old one never comes back.
        for asked in confirmation.open_in([call for call in calls if not call.get("answers")]):
            by_subject.pop(asked.subject, None)
        for pending in confirmation.open_in(calls):
            if confirmation.expired(pending.expires_at, now):
                continue
            by_subject[pending.subject] = Card(
                id=confirm_id(pending.token),
                kind="app_confirm",
                question=_clip(pending.question, MAX_QUESTION_CHARS),
                asked_at=_aware(message.created_at),
                group=message.id,
                app=pending.app,
                expires_at=pending.expires_at,
                latest=message.id == newest,
                incognito=message.id in hidden,
                pending=pending,
            )
    cards = list(by_subject.values())
    just_asked = ledger.paused_plan(db, conversation.id, since=since)
    for waiting in ledger.waiting_plans(db, conversation.id, since=since):
        paused = waiting.paused
        if paused.reason == "app_confirm" or not interpreter.on_a_card(paused):
            continue
        kind: CardKind = (
            "gate"
            if paused.reason == "gate"
            else "proposal"
            if paused.reason == "proposal"
            else "pick"
        )
        cards.append(
            Card(
                id=plan_id(paused.plan_id),
                kind=kind,
                question=_clip(_question(paused), MAX_QUESTION_CHARS),
                asked_at=waiting.asked_at,
                group=waiting.message_id,
                app=paused.app,
                options=tuple(_clip(option, MAX_OPTION_CHARS) for option in options_of(paused)),
                latest=just_asked is not None and just_asked.plan_id == paused.plan_id,
                incognito=waiting.incognito,
                plan=waiting,
            )
        )
    return sorted(cards, key=lambda card: (card.asked_at, card.id))


def options_of(paused: PausedPlan) -> list[str]:
    """What a pick-one's card offers: an ``ask_user``'s options, or the
    titles of the candidates an app named."""
    if paused.reason == "ask_user":
        return interpreter._offered(paused)
    if paused.reason == "app_question":
        candidates = paused.body.get(interpreter.APP_CANDIDATES) or []
        return [str(c.get("title") or "") for c in candidates if isinstance(c, dict)]
    return []


def _question(paused: PausedPlan) -> str:
    kept = paused.body.get(interpreter.QUESTION)
    if isinstance(kept, str) and kept.strip():
        return kept.strip()
    if paused.reason == "proposal":
        return PROPOSAL_QUESTION
    # A plan paused before its question was kept with it.
    return f"Go ahead with {paused.app}?" if paused.app else "Go ahead?"


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; conductor writes UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def tap_note(cards: list[Card], option: str | None = None) -> str:
    """The person's tap, as their own turn on the record (a note, as any act
    in the UI is): which card, and what they picked."""
    asked = " ".join(card.question for card in cards[:3])
    more = f" (and {len(cards) - 3} more)" if len(cards) > 3 else ""
    if option is not None:
        return f"Tapped “{option}”: {asked}"
    return f"Tapped Approve: {asked}{more}"
