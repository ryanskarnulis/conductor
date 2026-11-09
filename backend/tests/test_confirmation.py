"""Structural confirmation, conductor's side (#68).

An app with ``agent.confirms`` answers a gated write ``confirm_required``.
Conductor relays its question, keeps the token on the turn's record, and on
the person's yes next turn sends it back on the same thread with ``confirm``.
All fakes: no live delegate calls.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from datetime import date, datetime
from typing import Any

import httpx
import pytest

from app.ai import confirmation, state_block
from app.ai.interpreter import Interpreter
from app.ai.state_block import Now
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import (
    CONFIRM_HEADER,
    ConfirmRequired,
    DelegateClient,
    DelegateConfirmRefused,
    DelegateThreadGone,
    MessageExchange,
    MessageRead,
)
from app.fleet.manifests import AgentSpec, Fleet, FleetApp, _parse_agent
from app.fleet.tools import build_delegate_tools
from app.tools import registry
from app.tools.registry import ToolError
from tests.scripted_provider import ScriptedProvider
from tests.scripted_provider import text_turn as _text

QUESTION = "Trash Kitchen and its 12 tasks?"
WHEN = datetime(2026, 11, 6, 9, 0, 0)


def _app(name: str, *, confirms: bool) -> FleetApp:
    return FleetApp(
        name=name,
        title=name.title(),
        upstream="127.0.0.1:8100",
        agent=AgentSpec(
            description=f"The {name} app.",
            api="/api/agent",
            examples=(),
            confirms=confirms,
        ),
    )


def _fleet() -> Fleet:
    return Fleet(apps=(_app("tasks", confirms=True), _app("notes", confirms=False)))


def _exchange(
    content: str,
    *,
    stop_reason: str = "completed",
    confirm: ConfirmRequired | None = None,
    record: list[dict[str, Any]] | None = None,
) -> MessageExchange:
    return MessageExchange(
        user_message=MessageRead(
            id=1, conversation_id=7, role="user", content="msg", created_at=WHEN
        ),
        assistant_message=MessageRead.model_validate(
            {
                "id": 2,
                "conversation_id": 7,
                "role": "assistant",
                "content": content,
                "stop_reason": stop_reason,
                "created_at": WHEN,
                "record": record,
                "confirm_required": confirm.model_dump() if confirm else None,
            }
        ),
    )


def _asked() -> MessageExchange:
    return _exchange(
        QUESTION,
        stop_reason="confirm_required",
        confirm=ConfirmRequired(token="tok-1", tool="trash_project", question=QUESTION),
        record=[{"tool": "trash_project", "outcome": "not_run", "reason": "needs confirmation"}],
    )


def _trashed() -> MessageExchange:
    return _exchange(
        "Trashed Kitchen and its 12 tasks.",
        record=[
            {
                "tool": "trash_project",
                "outcome": "done",
                "count": 13,
                "touched": [{"app": "tasks", "kind": "project", "id": "7", "title": "Kitchen"}],
            }
        ],
    )


class FakeClient:
    """Records every create/send (with its token) and plays queued effects."""

    def __init__(self, effects: list[object]) -> None:
        self.log: list[tuple[object, ...]] = []
        self._effects = effects

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        self.log.append(("create",))
        return 7

    def send_message(
        self, conversation_id: int, message: str, *, confirm_token: str | None = None
    ) -> MessageExchange:
        self.log.append(("send", conversation_id, message, confirm_token))
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        assert isinstance(effect, MessageExchange)
        return effect

    def cancel(self, conversation_id: int) -> bool:
        return False


def _factory(client: FakeClient) -> Callable[[FleetApp], FakeClient]:
    return lambda app: client


@pytest.fixture
def context() -> Generator[DelegationContext, None, None]:
    bound = DelegationContext(
        master_conversation_id="m1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(bound):
        yield bound


def _pending(token: str = "tok-1") -> confirmation.PendingConfirm:
    return confirmation.PendingConfirm(
        app="tasks",
        thread_id=7,
        token=token,
        tool="trash_project",
        question=QUESTION,
        request="wipe the kitchen project",
    )


def _confirm(app: str = "tasks") -> str:
    result = registry.get_tool("confirm").fn(app=app)
    assert isinstance(result, str)
    return result


# --- the manifest flag -----------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [(True, True), ("yes", False), (None, False)])
def test_only_a_literal_true_declares_confirms(raw: object, expected: bool) -> None:
    spec = _parse_agent({"description": "d", "api": "/api/agent", "confirms": raw})
    assert spec is not None and spec.confirms is expected


def test_without_a_confirming_app_there_is_no_confirm_tool() -> None:
    fleet = Fleet(apps=(_app("notes", confirms=False),))
    names = build_delegate_tools(fleet, _factory(FakeClient([])))
    assert "confirm" not in names


# --- the delegate client --------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> DelegateClient:
    return DelegateClient(
        "http://pcc/api/agent", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_client_sends_the_token_and_reads_the_question() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=_asked().model_dump_json())

    exchange = _client(handler).send_message(7, "yes", confirm_token="tok-1")
    assert seen[0].headers[CONFIRM_HEADER] == "tok-1"
    asked = exchange.assistant_message.confirm_required
    assert asked is not None and asked.token == "tok-1" and asked.question == QUESTION


def test_client_sends_no_token_header_on_an_ordinary_message() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=_trashed().model_dump_json())

    _client(handler).send_message(7, "what's due")
    assert CONFIRM_HEADER not in seen[0].headers


def test_client_maps_a_refused_token_to_its_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "stale", "confirm": "stale"})

    with pytest.raises(DelegateConfirmRefused) as refused:
        _client(handler).send_message(7, "yes", confirm_token="tok-1")
    assert refused.value.kind == "stale"


def test_an_unreadable_confirmation_drops_only_itself() -> None:
    body = json.loads(_asked().model_dump_json())
    body["assistant_message"]["confirm_required"] = {"question": 3}
    exchange = MessageExchange.model_validate(body)
    assert exchange.assistant_message.confirm_required is None
    assert exchange.assistant_message.content == QUESTION


# --- asking: the question goes to the person, the token onto the record ---------


def test_an_armed_call_relays_the_question_and_keeps_the_token(
    context: DelegationContext,
) -> None:
    fleet = _fleet()
    build_delegate_tools(fleet, _factory(FakeClient([_asked()])))
    step = {
        "id": "s1",
        "do": "ask",
        "app": "tasks",
        "request": "wipe the kitchen project",
        "destructive": True,
    }
    provider = ScriptedProvider([_text(json.dumps({"steps": [step]}))])

    run = Interpreter(provider, fleet, narrate=True).run("wipe the kitchen project")

    (record,) = run.tool_calls
    assert record.app_confirm is not None
    assert record.app_confirm["token"] == "tok-1"
    assert record.app_confirm["thread_id"] == 7
    # tasks confirms its own writes: no gate of code's; its question goes out
    # word for word, with no narrator call, and the token reaches no model.
    assert run.reply == QUESTION
    assert len(provider.requests) == 1
    assert "tok-1" not in json.dumps(provider.requests, default=str)
    # A question asked this run can't be answered this run.
    assert context.pending_confirm is None


# --- what is pending -------------------------------------------------------------


def _turn(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": QUESTION, "tool_calls": list(calls)}


def _kept(app: str = "tasks") -> dict[str, Any]:
    return {
        "tool": f"ask_{app}",
        "arguments": {"message": "wipe the kitchen project"},
        "result": "…",
        "app_confirm": {
            "app": app,
            "thread_id": 7,
            "token": "tok-1",
            "tool": "trash_project",
            "question": QUESTION,
        },
    }


def test_the_last_turns_question_is_pending() -> None:
    history = [{"role": "user", "content": "wipe it"}, _turn(_kept())]
    assert confirmation.pending(history) == _pending()


def test_a_question_older_than_the_last_turn_has_lapsed() -> None:
    history = [_turn(_kept()), {"role": "user", "content": "actually no"}]
    assert confirmation.pending(history) is None
    history.append({"role": "assistant", "content": "ok", "tool_calls": None})
    assert confirmation.pending(history) is None


def test_another_message_to_the_same_app_voids_its_question() -> None:
    later = {"tool": "ask_tasks", "arguments": {"message": "what's due"}, "result": "…"}
    assert confirmation.pending([_turn(_kept(), later)]) is None
    other = {"tool": "ask_music", "arguments": {"message": "status"}, "result": "…"}
    assert confirmation.pending([_turn(_kept(), other)]) == _pending()


def test_the_state_block_says_how_to_answer_it() -> None:
    block = state_block.render(date(2026, 11, 6), Now(), [_turn(_kept())])
    assert f'tasks asked "{QUESTION}"' in block
    assert 'call confirm(app="tasks")' in block
    assert "tok-1" not in block


# --- answering yes ---------------------------------------------------------------


def test_yes_sends_the_token_on_the_thread_that_asked(context: DelegationContext) -> None:
    client = FakeClient([_trashed()])
    build_delegate_tools(_fleet(), _factory(client))
    context.pending_confirm = _pending()

    reply = _confirm()

    assert client.log == [("send", 7, "yes", "tok-1")]
    assert reply.startswith("Trashed Kitchen")
    # Used once: a second yes has nothing to answer and never leaves conductor.
    with pytest.raises(ToolError, match="Nothing from tasks is waiting"):
        _confirm()
    assert len(client.log) == 1


def test_yes_with_nothing_pending_makes_no_call(context: DelegationContext) -> None:
    client = FakeClient([])
    build_delegate_tools(_fleet(), _factory(client))
    with pytest.raises(ToolError, match="nothing was done"):
        _confirm()
    assert client.log == []


def test_yes_to_the_wrong_app_keeps_the_question(context: DelegationContext) -> None:
    build_delegate_tools(_fleet(), _factory(FakeClient([])))
    context.pending_confirm = _pending()
    with pytest.raises(ToolError):
        _confirm("notes")
    assert context.pending_confirm == _pending()


@pytest.mark.parametrize(
    ("error", "says"),
    [
        (DelegateConfirmRefused("409", kind="expired"), "waited too long"),
        (DelegateConfirmRefused("409", kind="stale"), "has changed since"),
        (DelegateConfirmRefused("409", kind="used"), "already used that yes"),
        (DelegateThreadGone("404"), "no longer has that question open"),
    ],
)
def test_a_refused_yes_did_nothing_and_never_recreates_the_thread(
    context: DelegationContext, error: Exception, says: str
) -> None:
    client = FakeClient([error])
    build_delegate_tools(_fleet(), _factory(client))
    context.pending_confirm = _pending()

    with pytest.raises(ToolError, match=says) as refused:
        _confirm()
    assert "nothing" in str(refused.value)
    assert [entry[0] for entry in client.log] == ["send"]
