"""The question an app is waiting on the person to answer (#68).

delegate-api "Confirmation": an app that gates a write answers
``confirm_required`` with a single-use token and its own question. Conductor
keeps the token on the turn's record (``ToolCallRecord.app_confirm``:
persisted and shown, never sent to the model) and relays the question. The
person's next message settles it: a yes is the ``confirm`` tool, which sends
the token back so the app runs exactly that call; anything else lets it lapse.

So a confirmation is live only while it sits on the conversation's last turn,
and only if nothing later in that turn spoke to the same app (the app voids
its question on the thread's next message). Pure: no session, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# The local tool that answers yes; its record names the app in ``arguments``.
CONFIRM_TOOL = "confirm"


class AppConfirm(BaseModel):
    """What conductor keeps of an app's ``confirm_required``."""

    app: str
    thread_id: int
    token: str
    tool: str
    question: str
    expires_at: str | None = None


@dataclass(frozen=True)
class PendingConfirm:
    """The live question, if the person's newest message may answer it."""

    app: str
    thread_id: int
    token: str
    tool: str
    question: str


def record_app(tool: str, arguments: dict[str, Any]) -> str | None:
    """The fleet app a stored call spoke to: ``ask_<app>``, or ``confirm(app=…)``."""
    if tool.startswith("ask_"):
        return tool.removeprefix("ask_")
    if tool == CONFIRM_TOOL:
        app = arguments.get("app")
        return str(app) if app else None
    return None


def pending(history: Sequence[dict[str, Any]]) -> PendingConfirm | None:
    """The question the last turn left waiting, from stored turns (oldest first)."""
    if not history or history[-1].get("role") != "assistant":
        return None
    live: PendingConfirm | None = None
    for stored in history[-1].get("tool_calls") or []:
        app = record_app(str(stored.get("tool", "")), stored.get("arguments") or {})
        if live is not None and app == live.app:
            live = None  # the app's thread heard another message: its question is void
        raw = stored.get("app_confirm")
        if raw is None:
            continue
        try:
            kept = AppConfirm.model_validate(raw)
        except ValidationError:
            continue
        live = PendingConfirm(
            app=kept.app,
            thread_id=kept.thread_id,
            token=kept.token,
            tool=kept.tool,
            question=kept.question,
        )
    return live
