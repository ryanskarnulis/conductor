"""The speaker's prompts (#253): every direct answer knows who it is, and the
facts-only rule covers the record, not the name."""

from __future__ import annotations

from typing import Any

import pytest

from app.ai import speaker
from app.ai.provider import ChatResult


def _system(kind: str) -> str:
    sent: list[list[dict[str, Any]]] = []

    def chat(messages: list[dict[str, Any]], cap: int) -> ChatResult:
        sent.append(messages)
        return ChatResult(content="Glitch.", tool_calls=[], finish_reason="stop", usage=None)

    speaker.answer(kind, "what's your name?", surface=None, chat=chat, facts=None)  # type: ignore[arg-type]
    return str(sent[0][0]["content"])


@pytest.mark.parametrize("kind", ["self", "chat", "knowledge"])
def test_every_direct_answer_carries_the_character_sheet(kind: str) -> None:
    system = _system(kind)
    assert "Your name is Glitch" in system
    assert "About you:" in system


@pytest.mark.parametrize("kind", ["self", "chat", "knowledge"])
def test_the_record_rule_names_what_it_covers(kind: str) -> None:
    system = _system(kind)
    assert "nswer only from the facts below" in system
    assert "The one exception: a question about who you are" in system
