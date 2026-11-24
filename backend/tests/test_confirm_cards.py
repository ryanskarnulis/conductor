"""Several app questions at once: one per gated call, each answered on its own.

delegate-api "Confirmation": an app that arms several calls in a turn sends
``confirms_required``. Conductor keeps every one on the turn's record, the page
shows a card each, and ``POST …/cards`` answers one card, or the group, with
no model call. No words approve one (#251): a yes gets "tap the card", other
turns leave the cards open, and "never mind" right after drops the group.
All fakes: no live delegate calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import confirmation, interpreter, narrator, state_block
from app.ai.interpreter import Interpreter
from app.ai.state_block import Now
from app.api import routes_agent
from app.db.models import LedgerPlan, LedgerReceipt, LedgerStep
from app.fleet.delegate import DelegateConfirmRefused, MessageExchange, MessageRead
from app.fleet.tools import build_delegate_tools
from app.main import app
from tests.scripted_provider import ScriptedProvider, text_turn
from tests.test_confirmation import QUESTION, WHEN, FakeClient, _asked, _factory, _fleet

TITLES = ("Test 1", "Test 2", "Test 3")
QUESTIONS = [f"Trash the task “{title}”?" for title in TITLES]
TOKENS = ["tok-a", "tok-b", "tok-c"]

ASK = text_turn(
    json.dumps(
        {
            "steps": [
                {
                    "id": "s1",
                    "do": "ask",
                    "app": "tasks",
                    "request": "delete the test tasks",
                    "destructive": True,
                }
            ]
        }
    )
)
DUE = text_turn(
    json.dumps(
        {
            "steps": [
                {
                    "id": "s1",
                    "do": "ask",
                    "app": "tasks",
                    "request": "what's due",
                    "destructive": False,
                }
            ]
        }
    )
)


def _assistant(body: dict[str, Any]) -> MessageExchange:
    return MessageExchange(
        user_message=MessageRead(
            id=1, conversation_id=7, role="user", content="msg", created_at=WHEN
        ),
        assistant_message=MessageRead.model_validate(
            {"id": 2, "conversation_id": 7, "role": "assistant", "created_at": WHEN, **body}
        ),
    )


def _armed(n: int, *, expires: datetime | None = None) -> dict[str, Any]:
    return {
        "token": TOKENS[n],
        "tool": "trash_task",
        "question": QUESTIONS[n],
        "preview": {
            "count": 1,
            "touched": [{"app": "tasks", "kind": "task", "id": str(n + 1), "title": TITLES[n]}],
        },
        "expires_at": (expires or datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    }


def _asked_three(*, expires: datetime | None = None) -> MessageExchange:
    armed = [_armed(n, expires=expires) for n in range(3)]
    return _assistant(
        {
            "content": "\n".join(QUESTIONS),
            "stop_reason": "confirm_required",
            "record": [{"tool": "trash_task", "outcome": "not_run", "reason": "needs confirmation"}]
            * 3,
            "confirm_required": armed[0],
            "confirms_required": armed,
        }
    )


def _trashed(n: int) -> MessageExchange:
    return _assistant(
        {
            "content": f"Trashed “{TITLES[n]}”.",
            "stop_reason": "completed",
            "record": [
                {
                    "tool": "trash_task",
                    "outcome": "done",
                    "touched": [
                        {"app": "tasks", "kind": "task", "id": str(n + 1), "title": TITLES[n]}
                    ],
                    "undo": {"handle": f"u{n}", "how": "restore_task"},
                }
            ],
        }
    )


def _due() -> MessageExchange:
    return _assistant({"content": "Nothing is due.", "stop_reason": "completed", "record": []})


def _start(client: TestClient, effects: list[object], script: list[Any]) -> tuple[int, FakeClient]:
    """A timeline whose first turn the tasks app answered with ``effects[0]``."""
    fake = FakeClient(effects)
    fleet = _fleet()
    build_delegate_tools(fleet, _factory(fake))
    provider = ScriptedProvider([ASK, *script])
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        provider, fleet, narrate=True
    )
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    said = client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": "delete the test tasks"},
    )
    assert said.status_code == 200, said.text
    return conversation, fake


def _answer(client: TestClient, conversation: int, token: str, answer: str) -> Any:
    return client.post(
        f"/api/agent/conversations/{conversation}/cards",
        json={"cards": [f"confirm:{token}"], "answer": answer},
    )


def _outcome(response: Any) -> dict[str, Any]:
    (one,) = response.json()["answers"]
    return dict(one)


def _open(client: TestClient, conversation: int) -> list[str]:
    cards = client.get(f"/api/agent/conversations/{conversation}/cards").json()
    return [card["id"].removeprefix("confirm:") for card in cards]


def _say(client: TestClient, conversation: int, words: str, **extra: Any) -> dict[str, Any]:
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": words, **extra}
    )
    assert response.status_code == 200, response.text
    return dict(response.json()["assistant_message"])


def _sent(fake: FakeClient) -> list[object]:
    """The tokens that went to the app, in order (``None``: a plain message)."""
    return [entry[3] for entry in fake.log if entry[0] == "send"]


def _asking(client: TestClient, conversation: int) -> dict[str, Any]:
    messages = client.get(f"/api/agent/conversations/{conversation}").json()["messages"]
    return next(m for m in messages if m["role"] == "assistant")


# --- what is kept ------------------------------------------------------------------------


def test_every_armed_question_is_kept_and_asked(client: TestClient) -> None:
    conversation, _ = _start(client, [_asked_three()], [])
    asking = _asking(client, conversation)
    (call,) = asking["tool_calls"]
    kept = call["app_confirms"]
    assert [one["token"] for one in kept] == TOKENS
    assert [one["question"] for one in kept] == QUESTIONS
    assert all(one["thread_id"] == 7 and one["app"] == "tasks" for one in kept)
    assert kept[1]["preview"]["touched"][0]["title"] == "Test 2"
    assert kept[0]["expires_at"]
    # The first stays where one-question readers look.
    assert call["app_confirm"] == kept[0]
    # Read, the reply is the app's questions, one per line, word for word.
    assert asking["content"] == "\n".join(QUESTIONS)


def test_an_app_that_sends_one_question_keeps_one(client: TestClient) -> None:
    conversation, fake = _start(client, [_asked(), _trashed(0)], [])
    (call,) = _asking(client, conversation)["tool_calls"]
    assert [one["token"] for one in call["app_confirms"]] == ["tok-1"]
    assert _asking(client, conversation)["content"] == QUESTION
    done = _answer(client, conversation, "tok-1", "yes")
    assert done.status_code == 200 and _outcome(done)["outcome"] == "done"
    assert _sent(fake) == [None, "tok-1"]


def test_a_confirmation_conductor_cant_read_drops_only_itself() -> None:
    body = json.loads(_asked_three().model_dump_json())
    body["assistant_message"]["confirms_required"][1] = {"question": 3}
    read = MessageExchange.model_validate(body).assistant_message
    assert [one.token for one in read.armed()] == ["tok-a", "tok-c"]
    body["assistant_message"]["confirms_required"] = "nope"
    assert [
        one.token for one in MessageExchange.model_validate(body).assistant_message.armed()
    ] == ["tok-a"]


# --- a card's answer ---------------------------------------------------------------------


def test_cards_are_answered_on_their_own_in_any_order(
    client: TestClient, db_session: Session
) -> None:
    stale = DelegateConfirmRefused("409", kind="stale")
    conversation, fake = _start(client, [_asked_three(), _trashed(1), stale], [])
    cards = client.get(f"/api/agent/conversations/{conversation}/cards").json()
    assert [(c["kind"], c["app"], c["question"]) for c in cards] == [
        ("app_confirm", "tasks", question) for question in QUESTIONS
    ]
    # One request's cards are one group, asked in the turn right before.
    assert len({c["group"] for c in cards}) == 1 and all(c["latest"] for c in cards)

    # The second card first: only its token goes out, and no model runs.
    second = _answer(client, conversation, "tok-b", "yes")
    assert second.status_code == 200, second.text
    assert _outcome(second)["outcome"] == "done"
    assert second.json()["exchanges"] == []
    assert _sent(fake) == [None, "tok-b"]
    answer = _asking(client, conversation)["tool_calls"][-1]
    assert (answer["tool"], answer["answers"]) == ("confirm", "tok-b")
    assert answer["app_record"][0]["undo"] == {"handle": "u1", "how": "restore_task"}

    # Not now on the first: nothing is sent, and that card alone closes.
    first = _answer(client, conversation, "tok-a", "no")
    assert _outcome(first)["outcome"] == "declined"
    assert _sent(fake) == [None, "tok-b"]
    declined = _asking(client, conversation)["tool_calls"][-1]
    assert (declined["answers"], declined["error"]) == ("tok-a", confirmation.DECLINED)
    assert _open(client, conversation) == ["tok-c"]

    # The third: the app refuses the token, and nothing is claimed done.
    third = _answer(client, conversation, "tok-c", "yes")
    assert third.status_code == 200
    assert _outcome(third)["outcome"] == "failed"
    assert "has changed since" in _outcome(third)["detail"]
    assert _asking(client, conversation)["tool_calls"][-1]["app_record"] is None

    # A reload reads the same record: three answers on the turn that asked.
    calls = _asking(client, conversation)["tool_calls"]
    assert [call["answers"] for call in calls] == [None, "tok-b", "tok-a", "tok-c"]
    # The ledger has the one write, on the asking turn's plan, undoable.
    receipts = (
        db_session.execute(select(LedgerReceipt).where(LedgerReceipt.outcome == "done"))
        .scalars()
        .all()
    )
    assert [(r.tool, r.undo_handle) for r in receipts] == [("trash_task", "u1")]
    steps = db_session.execute(select(LedgerStep).order_by(LedgerStep.seq)).scalars().all()
    assert [(step.kind, step.outcome) for step in steps][1:] == [
        ("confirm", "done"),
        ("confirm", "not_run"),
        ("confirm", "failed"),
    ]
    assert len({step.plan_id for step in steps}) == 1
    plan = db_session.get(LedgerPlan, steps[0].plan_id)
    assert plan is not None and plan.status != "paused"

    # Each is answered once.
    again = _answer(client, conversation, "tok-b", "yes")
    assert again.status_code == 409
    assert _answer(client, conversation, "tok-zzz", "yes").status_code == 409
    assert len(fake.log) == 4


def test_a_stale_card_does_nothing_and_says_what_changed(client: TestClient) -> None:
    """The app re-checks on approve (#251): the task was finished since."""
    said = "task “Test 1” was finished since you were asked, so nothing was trashed"
    stale = DelegateConfirmRefused("409", kind="stale", said=said)
    conversation, fake = _start(client, [_asked_three(), stale], [])
    tapped = _answer(client, conversation, "tok-a", "yes")
    assert tapped.status_code == 200
    assert _outcome(tapped) == {
        "id": "confirm:tok-a",
        "answer": "yes",
        "outcome": "failed",
        "detail": f"tasks did nothing: {said}.",
    }
    # Nothing is claimed done, and the card is closed: the others still wait.
    assert _asking(client, conversation)["tool_calls"][-1]["app_record"] is None
    assert _open(client, conversation) == ["tok-b", "tok-c"]


def test_an_expired_card_is_not_open(client: TestClient) -> None:
    past = datetime.now(UTC) - timedelta(minutes=1)
    conversation, fake = _start(client, [_asked_three(expires=past)], [])
    assert _open(client, conversation) == []
    refused = _answer(client, conversation, "tok-a", "yes")
    assert refused.status_code == 409 and "no longer open" in refused.json()["detail"]
    assert _sent(fake) == [None]


def test_a_card_is_open_for_the_tokens_life_whatever_is_said(client: TestClient) -> None:
    hours = datetime.now(UTC) + timedelta(hours=3)
    conversation, fake = _start(
        client,
        [_asked_three(expires=hours), _due(), _trashed(0)],
        [DUE, text_turn("Nothing is due.")],
    )
    _say(client, conversation, "what's due?")
    assert _open(client, conversation) == TOKENS
    assert _outcome(_answer(client, conversation, "tok-a", "yes"))["outcome"] == "done"
    assert _sent(fake) == [None, None, "tok-a"]


# --- approve all ---------------------------------------------------------------------------


def test_approve_all_answers_the_groups_cards_in_order(client: TestClient) -> None:
    expired = DelegateConfirmRefused("409", kind="expired")
    conversation, fake = _start(client, [_asked_three(), _trashed(0), expired, _trashed(2)], [])
    response = client.post(
        f"/api/agent/conversations/{conversation}/cards",
        json={"cards": [f"confirm:{token}" for token in TOKENS], "answer": "yes"},
    )
    assert response.status_code == 200, response.text
    assert _sent(fake) == [None, "tok-a", "tok-b", "tok-c"]
    # One the app refuses doesn't stop the others.
    assert [one["outcome"] for one in response.json()["answers"]] == ["done", "failed", "done"]
    assert response.json()["cards"] == []


# --- a spoken or typed answer ------------------------------------------------------------


def test_a_yes_approves_none_of_them(client: TestClient) -> None:
    conversation, fake = _start(client, [_asked_three()], [])  # no planner call is scripted
    for words in ("yes", "yes to all", "Yes, delete them."):
        reply = _say(client, conversation, words)
        assert reply["content"] == interpreter.TAP_THE_CARD
    assert _sent(fake) == [None]
    assert _open(client, conversation) == TOKENS


def test_a_refusal_right_after_dismisses_the_group(client: TestClient) -> None:
    conversation, fake = _start(client, [_asked_three()], [])  # code reads it: no planner
    reply = _say(client, conversation, "no thanks")
    assert reply["content"] == ("Okay, cancelled: tasks won't get a yes to any of its 3 questions.")
    assert _sent(fake) == [None]
    # They are closed on the turn that asked them: a card pressed late answers nothing.
    assert _open(client, conversation) == []
    assert _answer(client, conversation, "tok-a", "yes").status_code == 409
    closed = _asking(client, conversation)["tool_calls"][1:]
    assert [(call["answers"], call["error"]) for call in closed] == [
        (token, confirmation.DISMISSED) for token in TOKENS
    ]
    assert _sent(fake) == [None]


def test_another_request_leaves_them_open(client: TestClient) -> None:
    conversation, fake = _start(
        client, [_asked_three(), _due(), _trashed(0)], [DUE, text_turn("Nothing is due.")]
    )
    _say(client, conversation, "what's due?")
    assert _sent(fake) == [None, None]
    cards = client.get(f"/api/agent/conversations/{conversation}/cards").json()
    assert [card["latest"] for card in cards] == [False, False, False]
    # Out of their words' reach now: "never mind" later is not about them.
    history = [
        {"role": m["role"], "content": m["content"], "tool_calls": m["tool_calls"]}
        for m in client.get(f"/api/agent/conversations/{conversation}").json()["messages"]
    ]
    assert confirmation.open_confirms(history) == []
    assert _outcome(_answer(client, conversation, "tok-a", "yes"))["outcome"] == "done"


# --- what is open, from a turn's stored calls --------------------------------------------


def _calls() -> list[dict[str, Any]]:
    armed = [
        {"app": "tasks", "thread_id": 7, "token": token, "tool": "trash_task", "question": q}
        for token, q in zip(TOKENS, QUESTIONS, strict=True)
    ]
    return [
        {
            "tool": "ask_tasks",
            "arguments": {"message": "delete the test tasks"},
            "app_confirm": armed[0],
            "app_confirms": armed,
        }
    ]


def test_an_answer_closes_its_own_question_only() -> None:
    answered = {"tool": "confirm", "arguments": {"app": "tasks"}, "answers": "tok-b"}
    still = confirmation.open_in([*_calls(), answered])
    assert [one.token for one in still] == ["tok-a", "tok-c"]
    assert all(one.several and one.request == "delete the test tasks" for one in still)


def test_a_message_without_a_token_leaves_the_apps_questions_open() -> None:
    later = {"tool": "ask_tasks", "arguments": {"message": "what's due"}}
    assert len(confirmation.open_in([*_calls(), later])) == 3
    closed = confirmation.closing(confirmation.open_in(_calls())[0], confirmation.DISMISSED)
    assert [one.token for one in confirmation.open_in([*_calls(), closed])] == ["tok-b", "tok-c"]


def test_the_state_block_names_every_open_question() -> None:
    turn = {"role": "assistant", "content": "…", "tool_calls": _calls()}
    block = state_block.render(datetime(2026, 11, 6).date(), Now(), [turn])
    assert "tasks asked 3 questions" in block
    assert all(question in block for question in QUESTIONS)
    assert state_block.ON_CARDS in block and "confirm(" not in block
    assert "tok-a" not in block


# --- the spoken reply --------------------------------------------------------------------


def test_spoken_the_shared_words_are_said_once_and_every_item_named() -> None:
    said = narrator.several(QUESTIONS, "voice")
    assert said == "Trash the task “Test 1”, “Test 2” and “Test 3”? Each waits on its card."
    # Read, they stay word for word, a line each.
    assert narrator.several(QUESTIONS, None) == "\n".join(QUESTIONS)


def test_spoken_a_title_is_never_cut_and_unlike_questions_are_said_whole() -> None:
    alike = ["Trash the task “Test one”?", "Trash the task “Test two”?"]
    assert narrator.several(alike, "voice").startswith("Trash the task “Test one” and “Test two”?")
    unlike = ["Trash Kitchen and its 12 tasks?", "Reset the board?"]
    assert narrator.several(unlike, "voice").startswith(" ".join(unlike))


def test_an_incognito_turns_answered_card_is_not_open_again(client: TestClient) -> None:
    fake = FakeClient([_asked_three(), _trashed(1), _trashed(0), _trashed(2)])
    fleet = _fleet()
    build_delegate_tools(fleet, _factory(fake))
    provider = ScriptedProvider([ASK])
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
        provider, fleet, narrate=True
    )
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    _say(client, conversation, "delete the test tasks", incognito=True)
    done = _answer(client, conversation, "tok-b", "yes")
    assert done.status_code == 200 and _outcome(done)["outcome"] == "done"
    # The stored answer keeps its receipt and none of the app's words.
    assert _asking(client, conversation)["tool_calls"][-1]["result"] == "(incognito)"
    assert _open(client, conversation) == ["tok-a", "tok-c"]
    client.post(
        f"/api/agent/conversations/{conversation}/cards",
        json={"cards": ["confirm:tok-a", "confirm:tok-c"], "answer": "yes"},
    )
    assert _sent(fake) == [None, "tok-b", "tok-a", "tok-c"]
