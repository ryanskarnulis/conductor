"""Delegate API v1.2 on conductor's side (#202): send ``context`` and
``mode: "read"`` to an app that declares ``agent.context``, and pause on its
typed ``needs_input``.

The contract is ``../agent-standard/delegate-api.md`` "v1.2", decided in
#145. All fakes: no live delegate call.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any

import pytest

from app.ai import dataflow, narrator, plan, run_record
from app.ai.runtime import ToolCallRecord
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import MessageExchange, MessageRead, NeedsInputRead
from app.fleet.manifests import AgentSpec, Fleet, FleetApp, _parse_agent
from app.fleet.tools import build_delegate_tools
from app.tools import registry

HONEY = {"app": "music", "kind": "album", "id": "rg-honey", "title": "Honey", "untrusted": True}


def _app(name: str, *, context: bool) -> FleetApp:
    return FleetApp(
        name=name,
        title=name.title(),
        upstream="127.0.0.1:8100",
        agent=AgentSpec(
            description=f"The {name} app.",
            api="/api/agent",
            examples=(f"ask {name}",),
            confirms=True,
            context=context,
        ),
    )


def _exchange(*, needs_input: dict[str, Any] | None = None) -> MessageExchange:
    when = datetime(2026, 11, 13)
    return MessageExchange(
        user_message=MessageRead(id=1, conversation_id=1, role="user", created_at=when),
        assistant_message=MessageRead(
            id=2,
            conversation_id=1,
            role="assistant",
            content="ok",
            stop_reason="completed",
            created_at=when,
            record=[],
            needs_input=None if needs_input is None else NeedsInputRead(**needs_input),
        ),
    )


class _Client:
    """Records every send, with its v1.2 fields."""

    def __init__(self, reply: MessageExchange | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._reply = reply or _exchange()

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        return 1

    def send_message(
        self,
        conversation_id: int,
        message: str,
        *,
        confirm_token: str | None = None,
        context: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> MessageExchange:
        self.sent.append({"message": message, "context": context, "mode": mode})
        return self._reply

    def cancel(self, conversation_id: int) -> bool:
        return False


@pytest.fixture
def bound() -> Generator[None, None, None]:
    context = DelegationContext(
        master_conversation_id="m1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        yield


def _tools(client: _Client) -> Callable[..., Any]:
    build_delegate_tools(
        Fleet(apps=(_app("music", context=True), _app("tasks", context=False))),
        lambda app: client,
    )
    return lambda name, **arguments: registry.get_tool(name).fn(**arguments)


# --- the manifest flag -----------------------------------------------------------


@pytest.mark.parametrize(("value", "flag"), [(True, True), ("yes", False), (None, False)])
def test_only_a_literal_true_declares_v1_2(value: object, flag: bool) -> None:
    block: dict[str, object] = {"description": "Music.", "api": "/api/agent", "examples": ["x"]}
    if value is not None:
        block["context"] = value
    spec = _parse_agent(block)
    assert spec is not None and spec.context is flag


# --- what the ask tool sends ------------------------------------------------------


def test_a_v1_2_app_gets_context_and_mode_beside_the_plain_words(bound: None) -> None:
    client = _Client()
    ask = _tools(client)

    ask(
        "ask_music",
        message="download their other album",
        facts=["I sort music on Sundays"],
        refs=[HONEY],
        read=True,
    )

    [sent] = client.sent
    assert sent["message"] == "download their other album"
    assert sent["context"] == {"refs": [HONEY], "facts": ["I sort music on Sundays"]}
    assert sent["mode"] == "read"


def test_a_v1_1_app_gets_its_facts_in_the_words_and_no_v1_2_fields(bound: None) -> None:
    client = _Client()
    ask = _tools(client)

    ask("ask_tasks", message="what's due?", facts=["I sort music on Sundays"], refs=[HONEY])

    [sent] = client.sent
    assert sent["context"] is None and sent["mode"] is None
    assert "I sort music on Sundays" in sent["message"]


def test_nothing_to_carry_sends_no_context(bound: None) -> None:
    client = _Client()
    _tools(client)("ask_music", message="how are downloads?")
    assert client.sent[0]["context"] is None and client.sent[0]["mode"] is None


def test_the_refs_and_read_mark_stay_out_of_a_models_view() -> None:
    shown = run_record.shown({"message": "m", "facts": ["f"], "refs": [HONEY], "read": True})
    assert shown == {"message": "m"}


# --- needs_input: the typed question ----------------------------------------------


def _asked(asks: dict[str, Any] | None, result: str = "ok") -> ToolCallRecord:
    record = ToolCallRecord(tool="ask_music", arguments={"message": "m"}, result=result)
    record.app_record = []
    record.app_asks = asks
    return record


def test_a_v1_2_apps_question_is_its_needs_input() -> None:
    asked = {"question": "Which one do you want, or do you have a link?", "candidates": [HONEY]}
    assert narrator.app_asks(_asked({"needs_input": asked})) == asked["question"]


def test_a_v1_2_app_that_says_no_needs_input_is_not_asking_whatever_its_words() -> None:
    worded = "Music's record…\n\nThe agent's note: \"Which album did you mean?\""
    assert narrator.app_asks(_asked({"needs_input": None}, worded)) is None


def test_a_v1_1_app_is_still_read_from_its_words() -> None:
    assert narrator.app_asks(_asked(None, "Which album did you mean?")) is not None


def test_the_ask_tool_records_a_v1_2_apps_needs_input(bound: None) -> None:
    from app.tools import runtime

    question = {"question": "Which folder does Caribou go in?", "candidates": []}
    client = _Client(_exchange(needs_input=question))
    token = runtime.delegated_asks.set(None)
    try:
        _tools(client)("ask_music", message="sort Caribou")
        assert runtime.delegated_asks.get() == {"needs_input": question}
    finally:
        runtime.delegated_asks.reset(token)


# --- the refs a step carries ------------------------------------------------------


def test_a_ref_value_keeps_its_entity_beside_the_quoted_title() -> None:
    value = dataflow.from_record(
        [{"tool": "download_album", "outcome": "done", "touched": [HONEY]}]
    )
    assert value is not None and value.text == "“Honey”"
    assert value.refs == ({**HONEY},)


def test_a_step_that_reads_is_the_one_sent_read() -> None:
    reads = plan.Ask(id="s1", do="ask", app="music", request="what's queued?", destructive=False)
    writes = plan.Ask(id="s1", do="ask", app="music", request="download Honey", destructive=False)
    assert plan.reads_only(reads) and not plan.reads_only(writes)
