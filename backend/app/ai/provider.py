"""The provider seam: the chat contract and its transport-agnostic types.

Nothing above this line knows which model or backend answers a completion — the
turn runtime (``app/ai/runtime.py``) depends only on the :class:`ChatProvider` protocol and
these shared types; the concrete llama.cpp client lives in
``app/ai/providers/llamacpp.py`` behind it, and tests substitute a scripted fake
at the same shape. The types here are the wire-neutral vocabulary both sides
share: what a tool looks like (:class:`ToolSpec`), what the model asked for
(:class:`ToolCall`), what one turn produced (:class:`ChatResult`), and the typed
errors a completion attempt can raise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel


class ProviderError(Exception):
    """Base for everything a completion attempt can raise."""


class ProviderRequestError(ProviderError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class ProviderResponseError(ProviderError):
    """The server answered 200 but the body failed validation."""


class ToolCallArgumentsError(ProviderResponseError):
    """The model emitted tool-call arguments that aren't a JSON object.

    The tool name is carried for the log; conductor's own calls offer no tools
    since #75, so it only reaches the planner as a reply that isn't a plan.
    """

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(f"tool call {tool_name!r}: {detail}")
        self.tool_name = tool_name


@dataclass(frozen=True)
class Sampling:
    """A model's sampling profile, sent with every request (each model has its
    own: ``../agent-standard/model-profile.md``; the ``think`` lane's 26B, #78)."""

    temperature: float
    top_p: float
    top_k: int


# The roles a call can sample as (#150). ``think`` has its own profile
# (``app/ai/think.py``) on its own provider.
SamplingRole = Literal["planner", "narrator", "speaker"]


class ToolSpec(BaseModel):
    """One callable tool: name, description, JSON Schema for the arguments."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    """A tool call with its arguments already parsed from the wire's JSON string."""

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """What the server counted for one call. ``cached_tokens`` (prompt tokens
    served from the server's prompt cache) and ``prefill_ms`` (time spent
    reading the rest of the prompt) are ``None`` when the server didn't say,
    never a zero that reads as measured (#152)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None
    prefill_ms: float | None = None


class ChatResult(BaseModel):
    """One validated completion turn. ``content`` is answer text only."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage | None

    def to_message(self) -> dict[str, Any]:
        """This turn as an assistant message for the next request's history.

        Tool arguments are re-serialized to the wire's JSON-string form; the
        model's ``reasoning_content`` deliberately never round-trips (it is
        dropped at the wire boundary and has no home on this type).
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "id": call.id,
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


def tool_result_message(tool_call_id: str, content: str) -> dict[str, Any]:
    """The ``role: tool`` message answering one :class:`ToolCall`."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ChatProvider(Protocol):
    """What a turn needs from a provider — matched by ``LlamaCppProvider``."""

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        role: SamplingRole | None = None,
    ) -> ChatResult: ...


__all__ = [
    "ChatProvider",
    "ChatResult",
    "ProviderError",
    "ProviderRequestError",
    "ProviderResponseError",
    "Sampling",
    "SamplingRole",
    "ToolCall",
    "ToolCallArgumentsError",
    "ToolSpec",
    "Usage",
    "tool_result_message",
]
