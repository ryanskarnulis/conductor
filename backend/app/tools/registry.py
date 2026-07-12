"""Transport-agnostic tool registry: conductor's agent tool surface.

The single source of truth for the agent's tools — names, descriptions,
argument schemas, dispatch. Names come from ``__name__``, descriptions from the
docstring, and argument models plus JSON Schemas from the same ``func_metadata``
machinery FastMCP uses, so every consumer validates and advertises exactly the
same contract. The in-app loop (``app/ai/loop.py``) advertises :func:`tool_specs`
to the provider and dispatches via :func:`call_tool`; a later slice adds the MCP
server as a second consumer of the same registry.

The registry starts **empty**. Conductor's real tools are one delegate per
fleet app, built from each app's ``app.yaml`` block and registered in a later
slice; until then this module is just the decorator and dispatch machinery, and
tests register their own scratch tools against it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import structlog
from mcp.server.fastmcp.exceptions import ToolError as ToolError  # re-export
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata

from app.ai.provider import ToolSpec
from app.tools import runtime

logger = structlog.get_logger(__name__)


class UnknownToolError(Exception):
    """Dispatch was asked for a tool name the registry doesn't know."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown tool {name!r}")
        self.name = name


@dataclass(frozen=True)
class RegisteredTool:
    """One tool: callable body plus the metadata every consumer needs."""

    name: str
    description: str
    fn: Callable[..., Any]
    metadata: FuncMetadata

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the arguments — identical to the MCP inputSchema."""
        return self.metadata.arg_model.model_json_schema(by_alias=True)


_REGISTRY: dict[str, RegisteredTool] = {}

_F = TypeVar("_F", bound=Callable[..., Any])


def tool(fn: _F) -> _F:
    """Register a tool body: name from ``__name__``, description from the docstring."""
    description = inspect.getdoc(fn)
    if not description:
        raise ValueError(f"tool {fn.__name__} must have a docstring")
    _REGISTRY[fn.__name__] = RegisteredTool(
        name=fn.__name__,
        description=description,
        fn=fn,
        metadata=func_metadata(fn),
    )
    return fn


def all_tools() -> tuple[RegisteredTool, ...]:
    """Every registered tool, in registration order."""
    return tuple(_REGISTRY.values())


def get_tool(name: str) -> RegisteredTool:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownToolError(name) from None


def tool_specs() -> list[ToolSpec]:
    """The registry as provider tool specs, for ``LlamaCppProvider.chat(tools=…)``."""
    return [
        ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters)
        for tool in all_tools()
    ]


def call_tool(name: str, arguments: dict[str, Any], *, actor: str) -> Any:
    """Validate ``arguments`` against the tool's model, then run it as ``actor``.

    ``actor`` is bound to :data:`app.tools.runtime.current_tool_actor` for the
    duration of the call — the identity a tool body acts as. Raises
    :class:`UnknownToolError` for an unregistered name, ``pydantic.ValidationError``
    when the arguments fail the argument model (both schema-level — the caller
    can feed them back for self-correction), and :class:`ToolError` for domain
    rejections raised by the body.
    """
    tool = get_tool(name)
    validated = tool.metadata.arg_model.model_validate(tool.metadata.pre_parse_json(arguments))
    actor_token = runtime.current_tool_actor.set(actor)
    try:
        return tool.fn(**validated.model_dump_one_level())
    finally:
        runtime.current_tool_actor.reset(actor_token)
