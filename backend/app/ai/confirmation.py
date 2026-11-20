"""The question an app is waiting on the person to answer (#68).

delegate-api "Confirmation": an app that gates a write answers
``confirm_required`` with a single-use token and its own question. Conductor
keeps the token on the turn's record (``ToolCallRecord.app_confirm``:
persisted and shown, never sent to the model) and relays the question. The
person's next message settles it: a yes is the ``confirm`` tool, which sends
the token back so the app runs exactly that call; anything else lets it lapse.

So a confirmation is live only while it sits on the conversation's last turn,
and only if nothing later in that turn spoke to the same app (the app voids
its question on the thread's next message).

Several gated calls in one turn are several questions
(``ToolCallRecord.app_confirms``, in call order; ``app_confirm`` is the first),
each with its own token and answered on its own: a ``confirm`` record says
which one it answered (``answers``), and that closes only that one. A card's
answer is appended to the asking turn's record, so the others stay open.
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
    """A live question, if the person's newest message may answer it."""

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

    A later message to the same app without a token voids all of that app's;
    a ``confirm`` that says which one it answered (``answers``) closes that
    one alone."""
    live: list[PendingConfirm] = []
    for stored in calls:
        app = record_app(str(stored.get("tool", "")), stored.get("arguments") or {})
        answered = stored.get("answers")
        if answered:
            live = [one for one in live if one.token != answered]
        elif app is not None:
            # The app's thread heard another message: its questions are void.
            live = [one for one in live if one.app != app]
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
                )
            )
    return live


def open_confirms(history: Sequence[dict[str, Any]]) -> list[PendingConfirm]:
    """Every question the last turn left waiting, from stored turns (oldest first)."""
    if not history or history[-1].get("role") != "assistant":
        return []
    return open_in(history[-1].get("tool_calls") or [])


def pending(history: Sequence[dict[str, Any]]) -> PendingConfirm | None:
    """The first question the last turn left waiting, if any."""
    waiting = open_confirms(history)
    return waiting[0] if waiting else None


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
