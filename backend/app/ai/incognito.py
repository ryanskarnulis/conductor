"""Incognito turns (#82): the apps still see the request; conductor keeps none of the words.

A per-request switch on the turn request. On an incognito turn the person's
words and conductor's reply are never written: the stored messages and the
ledger turn hold :data:`PLACEHOLDER`. What the apps did (steps, receipts,
entities) is kept, because the apps really changed things, but memory,
recall, the planner's context, the state block and timeline search skip it,
and no fact is remembered or attached.

The words don't reach diagnostics either: while a turn is incognito
(:func:`active`, a context variable bound around the turn and copied into the
threads its reads run in) the turn trace and the context capture write
nothing, and :func:`redact_log` blanks the fields of every log line that can
carry words.

Pure: no session, no I/O.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

# What an incognito turn's words and reply are, wherever they would be written.
PLACEHOLDER = "(incognito)"

# What the person reads when they switch it on (the UI says the same).
NOTICE = "Incognito: this turn's words aren't kept. Apps still see the request."

_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("conductor_incognito", default=False)


def active() -> bool:
    """Whether the turn running in this context is incognito."""
    return _ACTIVE.get()


@contextmanager
def turn(on: bool) -> Iterator[None]:
    """Mark the code inside as one (incognito or not) turn's."""
    token = _ACTIVE.set(on)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


# --- what is kept of an incognito turn's calls ---------------------------------------------

# Arguments that name an app or a flag, never words.
_KEPT_ARGUMENTS = frozenset({"app", "destructive"})


def arguments(values: Mapping[str, Any]) -> dict[str, Any]:
    """A call's arguments without the words it carried (the request, intent,
    question, answer or attached facts)."""
    return {key: value for key, value in values.items() if key in _KEPT_ARGUMENTS}


def tool_call(stored: Mapping[str, Any]) -> dict[str, Any]:
    """A stored call as an incognito turn keeps it: the tool, the app's
    receipts and an app's waiting question (for the person's yes), but no
    words and no answer text."""
    return {
        **stored,
        "arguments": arguments(stored.get("arguments") or {}),
        "result": None if stored.get("result") is None else PLACEHOLDER,
    }


def plan_body(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A plan as an incognito turn keeps it once nothing waits on it: each
    step's id, kind and app, or the direct answer's kind."""
    if body is None:
        return None
    if "steps" not in body:
        return {key: body[key] for key in ("answer",) if key in body} or None
    steps = []
    for step in body.get("steps") or []:
        if isinstance(step, Mapping):
            steps.append({key: step[key] for key in ("id", "do", "app") if key in step})
    return {"steps": steps}


# --- logs ----------------------------------------------------------------------------------

# Log fields that can carry the person's words, a reply or an app's answer.
_WORDY = frozenset(
    {
        "problems",
        "error",
        "words",
        "utterance",
        "request",
        "reply",
        "question",
        "text",
        "content",
        "intent",
        "fact",
        "answer",
        "detail",
    }
)


def redact_log(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> Any:
    """A structlog processor: on an incognito turn, every field that can carry
    words reads :data:`PLACEHOLDER`, at every level."""
    if active():
        for key in _WORDY & event_dict.keys():
            event_dict[key] = PLACEHOLDER
        event_dict["incognito"] = True
    return event_dict
