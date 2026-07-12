"""Tool registry: schema derivation, argument validation, actor binding.

The registry ships empty (conductor's real delegate tools arrive in a later
slice), so every test registers its own scratch tool. The ``registry_isolation``
autouse fixture (conftest) restores the registry afterwards.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from pydantic import ValidationError

from app.ai.provider import ToolSpec
from app.tools import registry, runtime
from app.tools.registry import ToolError, UnknownToolError, call_tool, tool


def test_registry_starts_empty() -> None:
    assert registry.all_tools() == ()
    assert registry.tool_specs() == []


def test_name_description_and_schema_are_derived() -> None:
    @tool
    def sample_probe(count: int, label: str = "x") -> str:
        """A sample tool."""
        return label

    reg = registry.get_tool("sample_probe")
    assert reg.name == "sample_probe"  # name from __name__
    assert reg.description == "A sample tool."  # description from docstring

    schema = reg.parameters
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["label"]["default"] == "x"
    assert schema["required"] == ["count"]
    # Schema derivation is exactly func_metadata's — identical to PCC, including
    # its additionalProperties behavior (the arg model adds no such key).
    assert schema == func_metadata(sample_probe).arg_model.model_json_schema(by_alias=True)
    assert "additionalProperties" not in schema


def test_tool_specs_expose_registered_tools() -> None:
    @tool
    def another_probe(value: int) -> int:
        """Doubles a value."""
        return value * 2

    specs = registry.tool_specs()
    assert [spec.name for spec in specs] == ["another_probe"]
    spec = specs[0]
    assert isinstance(spec, ToolSpec)
    assert spec.description == "Doubles a value."
    assert spec.parameters["required"] == ["value"]


def test_tool_without_docstring_is_rejected() -> None:
    with pytest.raises(ValueError, match="must have a docstring"):

        @tool
        def undocumented(x: int) -> int:
            return x


def test_invalid_arguments_raise_schema_level_validation_error() -> None:
    @tool
    def needs_int(count: int) -> int:
        """Needs an int."""
        return count

    with pytest.raises(ValidationError):
        call_tool("needs_int", {}, actor="agent:loop")  # missing required arg
    with pytest.raises(ValidationError):
        call_tool("needs_int", {"count": "not-a-number"}, actor="agent:loop")


def test_unknown_tool_raises_unknown_tool_error() -> None:
    with pytest.raises(UnknownToolError):
        call_tool("nope", {}, actor="agent:loop")


def test_domain_rejection_surfaces_as_tool_error() -> None:
    @tool
    def always_fails(reason: str) -> str:
        """Always rejects."""
        raise ToolError(f"cannot: {reason}")

    with pytest.raises(ToolError, match="cannot: because"):
        call_tool("always_fails", {"reason": "because"}, actor="agent:loop")


def test_actor_is_bound_for_the_call_then_reset() -> None:
    @tool
    def whoami() -> str:
        """Report the acting actor."""
        return runtime.current_tool_actor.get()

    default_before = runtime.current_tool_actor.get()
    assert call_tool("whoami", {}, actor="agent:conductor") == "agent:conductor"
    assert call_tool("whoami", {}, actor="agent:loop") == "agent:loop"
    # The contextvar is reset after each dispatch — no leak.
    assert runtime.current_tool_actor.get() == default_before
