"""Direct answers (#70): what conductor says when a plan runs nothing.

The compiler can answer a turn without steps (:class:`app.ai.plan.Direct`):

- ``decline``: code's line, never the model's. Nothing in the fleet can do
  it, so there's nothing to embellish (decision 4: anything current is
  declined).
- ``knowledge``: the model answers from general knowledge, and code labels it
  as that (decision 4: "answered and labelled"). Its prompt declines anything
  current, since the planner sometimes files weather or news here (D1). The
  planner also files some questions about conductor itself here ("what did you
  just do?", #70's eval), so the call sees the same facts as ``self`` and says,
  under a schema, whether the question was about it: then the reply is from
  the facts and carries no general-knowledge label.
- ``self``: the model answers from facts code gathered: the ledger's last
  turns and what's running (S16, "what are you doing / what did you just
  do?"), and the app list. It says it doesn't know rather than guess.
- ``chat``: one short line, in the persona. The planner also files "what did
  you just do?" here some of the time (which glance lines sit in its prompt
  moves it: S16 fell to 9/15 once the apps declared real ones, #73), so chat
  gets the same facts and the same about-you check as ``knowledge``; a
  question about conductor is answered from the facts, small talk is not.

A provider failure falls back to a code line, so the turn always has a reply.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import structlog

from app.ai import narrator
from app.ai.narrator import PERSONA, Chat
from app.ai.plan import AnswerKind
from app.ai.provider import ChatResult

logger = structlog.get_logger(__name__)

MAX_TOKENS = 300

DECLINE = "I can't do that one: nothing I run handles it."
KNOWLEDGE_LABEL = "From general knowledge, not looked up:"
_FALLBACK: dict[str, str] = {
    "knowledge": "I couldn't answer that just now.",
    "self": "I couldn't put that together just now.",
    "chat": "Noted.",
}

Path = Literal["code", "spoken", "fallback"]

_SYSTEM: dict[str, str] = {
    "knowledge": (
        "You are the household's local assistant. Answer the person's question plainly, in "
        "one to three sentences. If it is about you (what you can do, are doing, just did, "
        'or why), set "about_you" to true and answer only from the facts below, written by '
        "code from your own records, in plain verbs (added, moved, trashed, found), never "
        "naming tools or saying something was completed unless a fact says so; if they "
        "don't say, say you don't know. Otherwise set it "
        "to false and answer from general knowledge, only facts that don't change; if "
        "answering needs anything current (weather, news, prices, scores, schedules, what's "
        "happening today), say you can't check that and stop. Don't mention apps, steps or "
        "tools."
    ),
    "self": (
        "You are the household's local assistant, answering a question about yourself: what "
        "you can do, what you're doing or what you just did. Answer only from the facts "
        "below, written by code from your own records; if they don't say, say you don't "
        "know. Say what the apps did in plain verbs (added, moved, trashed, found), naming "
        "things by their titles in quotes; never say something was completed or finished "
        "unless a fact says so, and never name tools. One to three sentences."
    ),
    "chat": (
        "You are the household's local assistant. Reply to the person's small talk in one "
        "short sentence. Don't offer to do anything and don't ask a question. If what they "
        "said is really a question about you (what you can do, are doing, just did, or why), "
        'set "about_you" to true and answer only from the facts below, written by code from '
        "your own records, in plain verbs (added, moved, trashed, found), naming things by "
        "their titles in quotes; if they don't say, say you don't know. Otherwise set it to "
        "false and use the facts for nothing."
    ),
}


# The knowledge call's reply, under json_schema: was it about conductor, and the words.
_KNOWLEDGE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"about_you": {"type": "boolean"}, "reply": {"type": "string"}},
            "required": ["about_you", "reply"],
            "additionalProperties": False,
        },
    },
}

# A model call under a response format: messages, a cap and the format in.
StructuredChat = Callable[[list[dict[str, Any]], int, dict[str, Any]], ChatResult]


def answer(
    kind: AnswerKind,
    words: str,
    *,
    surface: str | None,
    chat: Chat,
    facts: str | None = None,
    structured: StructuredChat | None = None,
) -> tuple[str, Path]:
    """The reply to a turn the compiler answered directly, and how it was made."""
    if kind == "decline":
        return DECLINE, "code"
    system = f"{_SYSTEM[kind]}\n\nVoice: {PERSONA}"
    user = f"The person said: {words}"
    if kind in ("self", "knowledge", "chat"):
        user = f"Facts:\n{facts or '(none)'}\n\n{user}"
    if surface in narrator.SPOKEN:
        user += "\n\nYour reply is spoken aloud: at most two short sentences, no markdown."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    about_you = False
    try:
        if kind in ("knowledge", "chat") and structured is not None:
            result = structured(messages, MAX_TOKENS, dict(_KNOWLEDGE_FORMAT))
            parsed = json.loads(result.content or "{}")
            about_you = bool(parsed.get("about_you"))
            text = str(parsed.get("reply") or "").strip()
        else:
            result = chat(messages, MAX_TOKENS)
            text = (result.content or "").strip()
    except Exception as exc:
        logger.warning("speaker_failed", kind=kind, error=str(exc), error_type=type(exc).__name__)
        return _FALLBACK[kind], "fallback"
    if result.finish_reason == "length" or not text:
        return _FALLBACK[kind], "fallback"
    text = narrator.fit(text, surface)
    if kind == "knowledge" and not about_you:
        text = f"{KNOWLEDGE_LABEL} {text}"
    return text, "spoken"
