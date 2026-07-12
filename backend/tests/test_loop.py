"""Agent loop end-to-end against a scripted provider (no GPU).

``ScriptedProvider`` plays the model: each ``chat()`` call pops the next scripted
turn, so the full loop path — registry dispatch, argument validation, actor
threading, self-correction, termination — runs against scratch tools registered
per test. The live model is exercised by the eval harness (a later slice).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from app.ai.loop import LOOP_ACTOR, AgentLoop
from app.ai.provider import ToolCallArgumentsError
from app.tools import runtime
from app.tools.registry import ToolError, tool
from tests.scripted_provider import ScriptedProvider
from tests.scripted_provider import text_turn as _text
from tests.scripted_provider import tool_calls_turn as _calls


@pytest.fixture
def notes() -> Generator[list[str], None, None]:
    """Register scratch tools and hand back the store the writer tool appends to."""
    store: list[str] = []

    @tool
    def add_note(text: str) -> str:
        """Append a note and confirm it."""
        store.append(text)
        return f"added: {text}"

    @tool
    def boom(reason: str) -> str:
        """Always raise a domain rejection."""
        raise ToolError(f"cannot: {reason}")

    @tool
    def whoami() -> str:
        """Report the acting actor."""
        return runtime.current_tool_actor.get()

    yield store


def test_completes_on_a_text_turn() -> None:
    provider = ScriptedProvider([_text("hey, done.")])
    result = AgentLoop(provider).run("what's up")

    assert result.stop_reason == "completed"
    assert result.reply == "hey, done."
    assert result.tool_calls == []
    assert result.iterations == 1
    # The system prompt is always offered first.
    assert provider.requests[0]["messages"][0]["role"] == "system"


def test_dispatches_tool_calls_and_feeds_results_back(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("add_note", {"text": "buy milk"})),
            _text("bet, noted."),
        ]
    )
    result = AgentLoop(provider).run("note: buy milk")

    assert result.stop_reason == "completed"
    assert result.reply == "bet, noted."
    assert notes == ["buy milk"]
    assert result.tool_calls[0].tool == "add_note"
    assert result.tool_calls[0].error is None
    assert result.tool_calls[0].result == "added: buy milk"
    # The tool result went back to the model as a role:tool message.
    followup = provider.requests[1]["messages"]
    assert followup[-1]["role"] == "tool"
    assert "added: buy milk" in followup[-1]["content"]


def test_domain_error_is_an_ordinary_result_not_a_correction(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("boom", {"reason": "no reason"})),
            _text("that one didn't take."),
        ]
    )
    # max_corrections=0 proves the domain rejection isn't billed as a correction.
    result = AgentLoop(provider, max_corrections=0).run("cause an error")

    assert result.stop_reason == "completed"
    assert result.tool_calls[0].error is not None
    assert "cannot: no reason" in result.tool_calls[0].error
    feedback = provider.requests[1]["messages"][-1]
    assert feedback["role"] == "tool"
    assert "cannot: no reason" in feedback["content"]


def test_unknown_tool_is_a_schema_error(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("ghost_tool", {})), _text("no such tool.")])
    result = AgentLoop(provider).run("call a ghost")

    assert result.stop_reason == "completed"
    assert result.tool_calls[0].error is not None
    assert "Unknown tool" in result.tool_calls[0].error


def test_invalid_arguments_are_fed_back_and_corrected(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("add_note", {})),  # missing required 'text' -> ValidationError
            _calls(("add_note", {"text": "corrected"})),
            _text("fixed it."),
        ]
    )
    result = AgentLoop(provider).run("add a note")

    assert result.stop_reason == "completed"
    assert result.tool_calls[0].error is not None and "Invalid arguments" in (
        result.tool_calls[0].error
    )
    assert result.tool_calls[1].error is None
    assert notes == ["corrected"]  # only the corrected call landed
    retry = provider.requests[1]["messages"]
    assert retry[-1]["role"] == "tool"
    assert "Invalid arguments" in retry[-1]["content"]


def test_schema_errors_exhaust_the_correction_budget(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("add_note", {})),  # schema error 1
            _calls(("add_note", {})),  # schema error 2 -> over budget
        ]
    )
    result = AgentLoop(provider, max_corrections=1).run("add a note")

    assert result.stop_reason == "correction_limit"
    assert result.reply is None
    assert len(provider.requests) == 2
    assert notes == []


def test_unknown_tools_exhaust_the_correction_budget(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("ghost_tool", {})),  # schema error 1
            _calls(("ghost_tool", {})),  # schema error 2 -> over budget
        ]
    )
    result = AgentLoop(provider, max_corrections=1).run("call a ghost")

    assert result.stop_reason == "correction_limit"
    assert result.reply is None
    assert len(provider.requests) == 2
    assert notes == []


def test_unparseable_tool_call_corrected_via_user_message(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            ToolCallArgumentsError("add_note", "arguments are not valid JSON"),
            _text("never mind."),
        ]
    )
    result = AgentLoop(provider).run("add a note")

    assert result.stop_reason == "completed"
    assert result.tool_calls == []  # nothing dispatched
    retry = provider.requests[1]["messages"]
    assert retry[-1]["role"] == "user"  # correction is a user turn, not a tool result
    assert "add_note" in retry[-1]["content"]


def test_max_iterations_terminates_the_loop(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("add_note", {"text": "x"}))] * 3)
    result = AgentLoop(provider, max_iterations=3).run("loop forever")

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert len(provider.requests) == 3


def test_history_is_passed_through_verbatim_without_fabricated_transcripts() -> None:
    history = [
        {"role": "user", "content": "earlier ask"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    provider = ScriptedProvider([_text("ok")])
    AgentLoop(provider).run("now do this", history=history)

    messages = provider.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "now do this"}
    # The loop never invents tool transcripts into the model's context.
    assert not any(m.get("role") == "tool" for m in messages)


def test_default_actor_is_the_loop_identity(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("whoami", {})), _text("done")])
    result = AgentLoop(provider).run("who are you")
    assert result.tool_calls[0].result == LOOP_ACTOR


def test_actor_is_threaded_through_to_the_tool(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("whoami", {})), _text("done")])
    result = AgentLoop(provider).run("who are you", actor="agent:conductor")
    assert result.tool_calls[0].result == "agent:conductor"


def test_zero_max_iterations_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        AgentLoop(ScriptedProvider([]), max_iterations=0)
