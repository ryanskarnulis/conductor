"""A scripted stand-in for the llama.cpp provider, shared by the agent tests.

Plays the model without a GPU: each ``chat()`` call pops the next scripted turn
(a ``ChatResult``, or an exception to raise) and records the request, so loop
behavior can be asserted deterministically end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.ai.provider import ChatResult, ToolCall, ToolSpec


def text_turn(content: str) -> ChatResult:
    return ChatResult(content=content, tool_calls=[], finish_reason="stop", usage=None)


def tool_calls_turn(*calls: tuple[str, dict[str, Any]]) -> ChatResult:
    return ChatResult(
        content=None,
        tool_calls=[
            ToolCall(id=f"call_{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(calls)
        ],
        finish_reason="tool_calls",
        usage=None,
    )


class ScriptedProvider:
    """Pops one scripted turn per ``chat()`` call; records every request."""

    def __init__(self, turns: Sequence[ChatResult | Exception]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.requests.append({"messages": list(messages), "tools": list(tools or [])})
        assert self._turns, "provider called more times than scripted"
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn
