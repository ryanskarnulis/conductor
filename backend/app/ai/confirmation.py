"""The questions an app is waiting on the person to answer (#68, #251).

delegate-api "Confirmation": an app that gates a write answers
``confirm_required`` with a single-use token and its own question. Conductor
keeps the token on the turn's record (``ToolCallRecord.app_confirm``:
persisted and shown, never sent to the model) and shows the question as a
card. Only a tap on the card sends the token back (the ``confirm`` tool, run
by code with no model); no words approve one, and other turns leave it open.

Several gated calls in one turn are several questions
(``ToolCallRecord.app_confirms``, in call order; ``app_confirm`` is the first),
each with its own token and answered on its own: a ``confirm`` record says
which one it answered (``answers``), and that closes only that one. A card's
answer is appended to the asking turn's record, so the others stay open. A
question code closes without sending anything (a dismissal, an amend, a newer
question about the same thing) gets a record too: ``not run``, with why.
Pure: no session, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

# The local tool that answers yes; its record names the app in ``arguments``.
CONFIRM_TOOL = "confirm"

# A card's "Not now" on the record: nothing was sent to the app.
DECLINED = "not run: declined"
# The person turned the question down in words, or changed what was asked.
DISMISSED = "not run: dismissed"
AMENDED = "not run: asked again with a change"


class AppConfirm(BaseModel):
    """What conductor keeps of one armed confirmation of an app's."""

    app: str
    thread_id: int
    token: str
    tool: str
    question: str
    # What the call would change: ``{count, touched}`` as the app sent it.
    preview: dict[str, Any] | None = None
    expires_at: str | None = None


@dataclass(frozen=True)
class PendingConfirm:
    """An app's question that is still open."""

    app: str
    thread_id: int
    token: str
    tool: str
    question: str
    # What conductor asked the app, word for word (an ``amend`` resends it
    # changed, #74), and the ledger receipts the asked-for call undoes.
    request: str | None = None
    undoes: tuple[int, ...] = ()
    expires_at: str | None = None
    # The call that armed it armed others too: which of an undo's receipts
    # this one redeems isn't known.
    several: bool = False
    # What it would change, as the app previewed it.
    preview: dict[str, Any] | None = None

    @property
    def subject(self) -> tuple[Any, ...]:
        """The entity and action it is about: a later question with the same
        subject replaces it (#251). The app, the tool and what the preview
        touches; one with no preview is told apart by its words."""
        touched = tuple(
            sorted(
                (str(ref.get("kind")), str(ref.get("id")))
                for ref in (self.preview or {}).get("touched") or []
                if isinstance(ref, dict)
            )
        )
        return (self.app, self.tool, touched or " ".join(self.question.split()).lower())


def record_app(tool: str, arguments: dict[str, Any]) -> str | None:
    """The fleet app a stored call spoke to: ``ask_<app>``, or ``confirm(app=…)``."""
    if tool.startswith("ask_"):
        return tool.removeprefix("ask_")
    if tool == CONFIRM_TOOL:
        app = arguments.get("app")
        return str(app) if app else None
    return None


def asked(stored: dict[str, Any]) -> list[AppConfirm]:
    """Every confirmation a stored call armed, in call order: its
    ``app_confirms``, else its one ``app_confirm`` (rows from before the list)."""
    one = stored.get("app_confirm")
    kept = []
    for raw in stored.get("app_confirms") or ([one] if one else []):
        try:
            kept.append(AppConfirm.model_validate(raw))
        except ValidationError:
            continue
    return kept


def open_in(calls: Sequence[dict[str, Any]]) -> list[PendingConfirm]:
    """The questions one turn's stored calls leave open, in the order asked.

    A record that says which one it answered (``answers``: a card's yes or
    no, or code closing it) closes that one alone. Nothing else does: other
    messages to the app leave its questions open (#251)."""
    live: list[PendingConfirm] = []
    for stored in calls:
        answered = stored.get("answers")
        if answered:
            live = [one for one in live if one.token != answered]
        armed = asked(stored)
        for kept in armed:
            live.append(
                PendingConfirm(
                    app=kept.app,
                    thread_id=kept.thread_id,
                    token=kept.token,
                    tool=kept.tool,
                    question=kept.question,
                    request=_text((stored.get("arguments") or {}).get("message")),
                    undoes=tuple(i for i in stored.get("undoes") or [] if isinstance(i, int)),
                    expires_at=kept.expires_at,
                    several=len(armed) > 1,
                    preview=kept.preview,
                )
            )
    return live


def open_confirms(history: Sequence[dict[str, Any]]) -> list[PendingConfirm]:
    """Every question the turn right before left waiting, from stored turns
    (oldest first): the ones the person's words may still change or dismiss."""
    if not history or history[-1].get("role") != "assistant":
        return []
    return open_in(history[-1].get("tool_calls") or [])


def pending(history: Sequence[dict[str, Any]]) -> PendingConfirm | None:
    """The first question the turn right before left waiting, if any."""
    waiting = open_confirms(history)
    return waiting[0] if waiting else None


def closing(pending: PendingConfirm, why: str) -> dict[str, Any]:
    """The record that closes a question nothing was sent for."""
    return {
        "tool": CONFIRM_TOOL,
        "arguments": {"app": pending.app},
        "error": why,
        "answers": pending.token,
    }


def expired(expires_at: str | None, now: datetime) -> bool:
    """Whether a question's window has closed by conductor's clock. One with
    no readable expiry is the app's to judge."""
    if not expires_at:
        return False
    try:
        at = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return at <= now


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
