"""The provider seam: the chat contract and its transport-agnostic types.

Nothing above this line knows which model or backend answers a completion — the
turn runtime (``app/ai/runtime.py``) depends only on the :class:`ChatProvider` protocol and
these shared types; the concrete llama.cpp client lives in
``app/ai/providers/llamacpp.py`` behind it, and tests substitute a scripted fake
at the same shape. The shared types are the fleet's chat wire, vendored in
``app/ai/agentkit/chat.py`` and re-exported here: what a tool looks like
(:class:`ToolSpec`), what the model asked for (:class:`ToolCall`), what one
turn produced (:class:`ChatResult`), and the typed errors a completion attempt
can raise. Conductor's own are :data:`SamplingRole` and :class:`ChatProvider`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

# Re-exported: the fleet's chat wire, vendored (agent-standard/agentkit).
from app.ai.agentkit.chat import (
    ChatResult as ChatResult,
    ProviderError as ProviderError,
    ProviderFailure as ProviderFailure,
    ProviderRequestError as ProviderRequestError,
    ProviderResponseError as ProviderResponseError,
    ServerStamp as ServerStamp,
    ToolCall as ToolCall,
    ToolCallArgumentsError as ToolCallArgumentsError,
    ToolSpec as ToolSpec,
    Usage as Usage,
    tool_result_message as tool_result_message,
)

# The roles a call can sample as (#150): each is a phase of the model's
# profile (``app/ai/model_profile.py``). The ``think`` lane's call takes no
# role: it runs on its own provider, at its own model's sampling.
SamplingRole = Literal["planner", "narrator", "speaker"]


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
    "ProviderFailure",
    "ProviderRequestError",
    "ProviderResponseError",
    "SamplingRole",
    "ServerStamp",
    "ToolCall",
    "ToolCallArgumentsError",
    "ToolSpec",
    "Usage",
    "tool_result_message",
]
