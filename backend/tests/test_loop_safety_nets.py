"""STANDARD.md §3 safety nets in conductor's loop (#52).

The per-call token cap, ``truncated``, the stall rule, the per-run tool-call
cap, replies built from the code-kept record, and failure stops that carry the
partial run. All against a scripted provider and scratch tools; no GPU.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

import pytest

from app.ai import run_record
from app.ai.loop import FAILURE_STATUS, AgentLoop, AgentRunFailed
from app.ai.provider import ChatResult, ProviderRequestError, ToolSpec
from app.tools.registry import ToolError, tool
from tests.scripted_provider import ScriptedProvider
from tests.scripted_provider import text_turn as _text
from tests.scripted_provider import tool_calls_turn as _calls


@pytest.fixture
def notes() -> Generator[list[str], None, None]:
    store: list[str] = []

    @tool
    def add_note(text: str) -> str:
        """Append a note and confirm it."""
        store.append(text)
        return f"added: {text}"

    @tool
    def count_notes() -> str:
        """How many notes there are."""
        return str(len(store))

    @tool
    def refuse(reason: str) -> str:
        """Always refuse."""
        raise ToolError(f"cannot: {reason}")

    @tool
    def explode() -> str:
        """A bug: raises something that isn't a ToolError."""
        raise RuntimeError("kaboom")

    yield store


class _CapturingProvider:
    """Records the ``max_tokens`` of every call, then answers with ``turns``."""

    def __init__(self, turns: Sequence[ChatResult]) -> None:
        self._turns = list(turns)
        self.max_tokens: list[int | None] = []

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.max_tokens.append(max_tokens)
        return self._turns.pop(0)


def _cut_off(content: str | None, *calls: tuple[str, dict[str, Any]]) -> ChatResult:
    turn = _calls(*calls) if calls else _text(content or "")
    return ChatResult(
        content=content, tool_calls=turn.tool_calls, finish_reason="length", usage=None
    )


# --- max_tokens / truncated ---------------------------------------------------


def test_every_model_call_carries_max_tokens(notes: list[str]) -> None:
    provider = _CapturingProvider([_calls(("add_note", {"text": "a"})), _text("done")])
    AgentLoop(provider, max_tokens=321).run("note a")

    assert provider.max_tokens == [321, 321]


def test_a_cut_off_text_turn_is_dropped_and_stops_truncated() -> None:
    provider = ScriptedProvider([_cut_off("Sure, here is a very long answ")])
    result = AgentLoop(provider).run("tell me everything")

    assert result.stop_reason == "truncated"
    assert result.reply is None  # nothing ran: no record to speak from
    assert all("very long answ" not in str(m.get("content")) for m in result.messages)


def test_a_cut_off_turns_tool_calls_still_run(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [_cut_off("I'll add it and then expl", ("add_note", {"text": "milk"})), _text("added.")]
    )
    result = AgentLoop(provider).run("note milk")

    assert notes == ["milk"]
    assert result.stop_reason == "completed"
    # The fragment beside the calls never reaches history.
    assert all("then expl" not in str(m.get("content")) for m in result.messages)


def test_truncated_after_a_write_replies_from_the_record(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("add_note", {"text": "milk"})), _cut_off("Added mi")])
    result = AgentLoop(provider).run("note milk")

    assert result.stop_reason == "truncated"
    assert result.reply is not None
    assert "add_note" in result.reply and "added: milk" in result.reply
    assert "Added mi" not in result.reply
    assert run_record.LIMIT_NOTE in result.reply


# --- stall rule ------------------------------------------------------------------


def test_a_turn_that_only_repeats_seen_exchanges_stops_no_progress(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("count_notes", {}))] * 3)
    result = AgentLoop(provider).run("count")

    assert result.stop_reason == "no_progress"
    assert len(provider.requests) == 2
    assert result.reply is not None and "count_notes" in result.reply


def test_a_repeat_that_answers_differently_is_progress(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("count_notes", {})),
            _calls(("add_note", {"text": "a"})),
            _calls(("count_notes", {})),  # now answers "1": new work
            _text("one note."),
        ]
    )
    result = AgentLoop(provider).run("count, add, count")

    assert result.stop_reason == "completed"


def test_a_repeated_refusal_stops_no_progress(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(("refuse", {"reason": "x", "why": "y"})),
            # Same call, arguments in another order: still a repeat.
            _calls(("refuse", {"why": "y", "reason": "x"})),
        ]
    )
    result = AgentLoop(provider).run("refuse")

    assert result.stop_reason == "no_progress"
    assert result.reply is not None and "Failed:" in result.reply


# --- tool-call cap ---------------------------------------------------------------


def test_calls_past_the_cap_are_not_run_and_stop_tool_call_limit(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(
                ("add_note", {"text": "a"}),
                ("add_note", {"text": "b"}),
                ("add_note", {"text": "c"}),
            )
        ]
    )
    result = AgentLoop(provider, max_tool_calls=2).run("three notes")

    assert notes == ["a", "b"]
    assert result.stop_reason == "tool_call_limit"
    assert result.tool_calls[2].error is not None
    assert result.tool_calls[2].error.startswith(run_record.NOT_RUN_PREFIX)
    assert result.reply is not None and "Not run:" in result.reply
    # Every call in the transcript has exactly one answer.
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_messages) == 3


# --- record-built replies --------------------------------------------------------


def test_a_limit_stop_where_nothing_ran_has_no_reply(notes: list[str]) -> None:
    provider = ScriptedProvider([_calls(("nope", {}))] * 5)
    result = AgentLoop(provider, max_corrections=1).run("x")

    assert result.stop_reason == "correction_limit"
    assert result.reply is None


def test_a_limit_reply_never_carries_the_models_words(notes: list[str]) -> None:
    turn = _calls(("add_note", {"text": "a"}))
    turn = ChatResult(
        content="All done, everything is perfect!",
        tool_calls=turn.tool_calls,
        finish_reason="tool_calls",
        usage=None,
    )
    result = AgentLoop(ScriptedProvider([turn]), max_iterations=1).run("note a")

    assert result.stop_reason == "max_iterations"
    assert result.reply is not None
    assert "perfect" not in result.reply
    assert result.reply.count("Done:") == 1


# --- failure stops -----------------------------------------------------------------


def test_a_provider_death_after_a_write_raises_with_the_partial_run(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [_calls(("add_note", {"text": "milk"})), ProviderRequestError("llama-server timed out")]
    )
    with pytest.raises(AgentRunFailed) as caught:
        AgentLoop(provider).run("note milk")

    failed = caught.value
    assert failed.http_status == 502
    assert str(failed) == "agent run failed: llama-server timed out"
    assert failed.result.stop_reason == "provider_error"
    assert notes == ["milk"]
    assert failed.result.reply is not None and "added: milk" in failed.result.reply


def test_a_provider_death_before_anything_ran_has_no_reply() -> None:
    with pytest.raises(AgentRunFailed) as caught:
        AgentLoop(ScriptedProvider([ProviderRequestError("refused")])).run("hi")

    assert caught.value.result.stop_reason == "provider_error"
    assert caught.value.result.reply is None
    assert caught.value.result.tool_calls == []


def test_a_tool_that_raises_stops_internal_error_and_skips_the_rest(notes: list[str]) -> None:
    provider = ScriptedProvider(
        [
            _calls(
                ("add_note", {"text": "a"}),
                ("explode", {}),
                ("add_note", {"text": "b"}),
            )
        ]
    )
    with pytest.raises(AgentRunFailed) as caught:
        AgentLoop(provider).run("go")

    result = caught.value.result
    assert caught.value.http_status == 500
    assert result.stop_reason == "internal_error"
    assert notes == ["a"]
    assert [run_record.classify(r) for r in result.tool_calls] == ["done", "failed", "not_run"]


def test_failure_statuses_cover_exactly_the_failure_stops() -> None:
    assert FAILURE_STATUS == {"provider_error": 502, "timed_out": 504, "internal_error": 500}
