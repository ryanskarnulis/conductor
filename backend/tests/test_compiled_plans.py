"""Compiled plans (#70): the schema, the validator, the compiler and interpreter v1.

The model is scripted (``ScriptedProvider``): the compiler's call returns plan
JSON, the narrator's and the speaker's return text. The apps are the session
evals' fake fleet (``fake_world.py``), reached through the real route, loop,
delegate tools and ledger.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai import compiler, speaker
from app.ai import plan as plans
from app.ai.loop import AgentLoop, LoopActivity
from app.ai.provider import ChatResult, ProviderRequestError
from app.api import routes_agent
from app.db.models import LedgerPlan, LedgerTurn
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.services import ledger
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn


def plan_turn(body: dict[str, Any]) -> ChatResult:
    return text_turn(json.dumps(body))


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]}


def ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


# --- the schema and the validator ----------------------------------------------------------


def test_the_schema_is_built_from_the_fleet_with_a_ceiling_not_a_limit(fleet: Fleet) -> None:
    schema = plans.schema(fleet)
    step_shape, answer_shape = schema["anyOf"]
    array = step_shape["properties"]["steps"]
    assert (array["minItems"], array["maxItems"]) == (1, plans.MAX_STEPS) == (1, 12)
    kinds = {item["properties"]["do"]["const"]: item for item in array["items"]["anyOf"]}
    assert set(kinds) == {
        "ask", "look", "open", "ask_user", "recall", "cancel", "amend", "confirm", "undo", "think"
    }  # fmt: skip
    assert kinds["ask"]["properties"]["app"]["enum"] == ["music", "tasks"]
    assert kinds["open"]["properties"]["app"]["enum"] == ["chess"]
    for item in kinds.values():  # llama-server's strict grammar: all required, nothing else
        assert set(item["required"]) == set(item["properties"])
        assert item["additionalProperties"] is False
    assert answer_shape["properties"]["answer"]["enum"] == ["knowledge", "decline", "self", "chat"]
    assert len(plans.schema(fleet, resumable=True)["anyOf"]) == 3


def _problems(body: dict[str, Any], fleet: Fleet, **checks: Any) -> list[str]:
    return plans.problems(plans.load(body), plans.Checks(fleet=fleet, **checks))


def test_the_validator_refuses_what_code_wont_run(fleet: Fleet) -> None:
    assert _problems(steps(ask("tasks", "add milk")), fleet) == []
    bad_ids = {"steps": [{"id": "s2", **ask("tasks", "x")}]}
    assert "ids go s1, s2" in _problems(bad_ids, fleet)[0]
    assert (
        "uses {s2} before s2 runs"
        in _problems(steps(ask("tasks", "add {s2}"), ask("music", "x")), fleet)[0]
    )
    assert _problems(
        steps({"do": "open", "app": "chess", "intent": "play"}, ask("tasks", "x")), fleet
    ) == ["open must be the last step"]
    assert _problems(
        steps(ask("tasks", "x"), {"do": "ask_user", "question": "which?", "options": []}), fleet
    ) == ["ask_user must be the first step"]
    assert _problems(steps({"do": "confirm", "answer": "yes"}), fleet) == [
        "confirm with nothing waiting on the person"
    ]
    assert _problems(steps({"do": "confirm", "answer": "yes"}), fleet, confirmable=True) == []
    assert "at most 3 per turn" in _problems(steps(*[ask("tasks", "x")] * 4), fleet)[0]
    invented = steps({"do": "ask_user", "question": "which?", "options": ["Call Dennis"]})
    assert "which State doesn't name" in _problems(invented, fleet, context="dentist")[0]
    assert _problems(invented, fleet, context='"Call Dennis" (pcc:task:9)') == []
    assert _problems({"resume": "the first"}, fleet) == ["resume with no question waiting"]
    assert _problems({"resume": "the first"}, fleet, resumable=True) == []


def test_twelve_steps_run_and_thirteen_are_a_runaway(fleet: Fleet) -> None:
    looks = [{"do": "look", "app": "tasks"}]
    assert _problems(steps(*looks * 12), fleet) == []
    assert "over the ceiling of 12" in _problems(steps(*looks * 13), fleet)[0]


def test_what_v1_runs_and_what_falls_back(fleet: Fleet) -> None:
    def reason(body: dict[str, Any]) -> str | None:
        return plans.fallback_reason(plans.load(body))

    assert reason(steps(ask("tasks", "add milk"), ask("tasks", "add eggs"))) is None
    assert reason({"answer": "chat"}) is None
    assert reason(steps({"do": "look", "app": "tasks"})) == "kind:look"
    assert reason(steps(ask("music", "grab Swim"), ask("tasks", "sort it"))) == "cross_app"
    assert reason(steps(ask("tasks", "add a"), ask("tasks", "move {s1}"))) == "data_flow"
    asked = steps(
        {"do": "ask_user", "question": "which?", "options": []}, ask("tasks", "move {s1}")
    )
    assert reason(asked) is None


def test_the_ack_says_what_is_about_to_happen() -> None:
    assert plans.ack(plans.load(steps(ask("music", "grab Swim")))) == "On it, asking music."
    assert plans.ack(plans.load({"answer": "chat"})) is None
    assert plans.ack(plans.load(steps({"do": "open", "app": "chess", "intent": "x"}))) is None


# --- the compiler ------------------------------------------------------------------------


def _compile(fleet: Fleet, *raws: str) -> tuple[compiler.Compiled, list[list[dict[str, Any]]]]:
    sent: list[list[dict[str, Any]]] = []
    replies = list(raws)

    def chat(messages: list[dict[str, Any]], _max: int, _fmt: dict[str, Any]) -> ChatResult:
        sent.append(list(messages))
        return text_turn(replies.pop(0))

    compiled = compiler.compile_turn(
        chat,
        system_prompt=compiler.planner_prompt(fleet),
        context="[State]",
        words="hi",
        fleet=fleet,
        checks=plans.Checks(fleet=fleet),
    )
    return compiled, sent


def test_one_repair_then_no_plan(fleet: Fleet) -> None:
    good = json.dumps(steps(ask("tasks", "add milk")))
    bad = json.dumps(steps({"do": "open", "app": "chess", "intent": "x"}, ask("tasks", "x")))
    compiled, sent = _compile(fleet, bad, good)
    assert compiled.plan is not None and compiled.repaired
    assert "open must be the last step" in sent[1][-1]["content"]
    compiled, _ = _compile(fleet, "not json", bad)
    assert compiled.plan is None and compiled.problems == ["open must be the last step"]


def test_the_planner_prompt_lists_the_fleet_and_never_caps_steps(fleet: Fleet) -> None:
    prompt = compiler.planner_prompt(fleet)
    assert "- tasks (ask, look): Manages projects and tasks." in prompt
    assert "- chess (open, look): Opens the chess board." in prompt
    assert "as many as the request needs" in prompt
    assert "0 to 5" not in prompt


# --- interpreter v1, through the route ---------------------------------------------------


@pytest.fixture
def run_with(client: TestClient, fleet: Fleet, world: World) -> Generator[Any, None, None]:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])

    def run(provider: ScriptedProvider, words: str) -> dict[str, Any]:
        app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(
            provider, fleet=fleet, narrate=True
        )
        response = client.post(
            f"/api/agent/conversations/{conversation}/messages", json={"content": words}
        )
        assert response.status_code == 200, response.text
        return dict(response.json()["assistant_message"])

    run.conversation = conversation  # type: ignore[attr-defined]
    yield run


def _plans(db: Session) -> list[LedgerPlan]:
    db.expire_all()
    return list(db.execute(select(LedgerPlan).order_by(LedgerPlan.id)).scalars())


def test_a_single_app_plan_runs_and_is_on_the_ledger(
    run_with: Any, world: World, db_session: Session
) -> None:
    provider = ScriptedProvider(
        [plan_turn(steps(ask("tasks", "add a task to water the plants"))), text_turn("Added it.")]
    )
    reply = run_with(provider, "add a task to water the plants")
    assert [w.kind for w in world.writes()] == ["create"]
    assert [call["tool"] for call in reply["tool_calls"]] == ["ask_tasks"]
    assert reply["stop_reason"] == "completed"
    planner_request = provider.requests[0]
    assert planner_request["response_format"]["type"] == "json_schema"
    assert planner_request["tools"] == []
    assert planner_request["messages"][-1]["content"].endswith(
        "The person said: add a task to water the plants"
    )
    (compiled,) = _plans(db_session)
    assert (compiled.source, compiled.status, compiled.acted) == ("compiled", "ran", True)
    assert [(s.kind, s.app, s.outcome) for s in compiled.steps] == [("ask", "tasks", "done")]
    assert compiled.steps[0].receipts[0].outcome == "done"


def test_a_plan_v1_cant_run_falls_back_to_the_loop(run_with: Any, db_session: Session) -> None:
    provider = ScriptedProvider(
        [plan_turn(steps({"do": "look", "app": "tasks"})), text_turn("Nothing due today.")]
    )
    reply = run_with(provider, "what's due today?")
    assert reply["content"] == "Nothing due today."
    assert provider.requests[1]["tools"]  # the loop's own call, with tools
    compiled, loop_plan = _plans(db_session)
    assert (compiled.status, compiled.note, compiled.acted) == ("fell_back", "kind:look", False)
    assert (loop_plan.source, loop_plan.acted) == ("loop", True)


def test_no_valid_plan_after_a_repair_falls_back(run_with: Any, db_session: Session) -> None:
    provider = ScriptedProvider([text_turn("nope"), text_turn("still nope"), text_turn("Hi.")])
    assert run_with(provider, "hello")["content"] == "Hi."
    compiled, _ = _plans(db_session)
    assert (compiled.status, compiled.acted, compiled.body) == ("invalid", False, None)


@pytest.mark.parametrize(
    ("answer", "spoken", "expected"),
    [
        ("decline", None, speaker.DECLINE),
        ("chat", "Any time.", "Any time."),
        (
            "knowledge",
            '{"about_you": false, "reply": "About nine minutes."}',
            f"{speaker.KNOWLEDGE_LABEL} About nine minutes.",
        ),
    ],
)
def test_direct_answers(
    run_with: Any, db_session: Session, answer: str, spoken: str | None, expected: str
) -> None:
    turns = [plan_turn({"answer": answer})] + ([text_turn(spoken)] if spoken else [])
    provider = ScriptedProvider(turns)
    assert run_with(provider, "words")["content"] == expected
    assert len(provider.requests) == len(turns)  # decline is code's line: no model call
    (compiled,) = _plans(db_session)
    assert (compiled.status, compiled.answer, compiled.steps) == ("answered", answer, [])


def test_self_answers_from_the_ledger(run_with: Any) -> None:
    run_with(
        ScriptedProvider(
            [plan_turn(steps(ask("tasks", "add a task to water the plants"))), text_turn("Ok.")]
        ),
        "add a task to water the plants",
    )
    provider = ScriptedProvider([plan_turn({"answer": "self"}), text_turn("I added a task.")])
    assert run_with(provider, "what did you just do?")["content"] == "I added a task."
    facts = provider.requests[1]["messages"][1]["content"]
    assert 'You were asked: "add a task to water the plants"' in facts
    assert "asked tasks: it answered" in facts
    assert 'the app did: create task "water the plants"' in facts
    assert "What I can do, through the apps:" in facts


def test_ask_user_pauses_and_the_answer_resumes_the_same_plan(
    run_with: Any, world: World, db_session: Session
) -> None:
    question = {"do": "ask_user", "question": "Which dentist thing?", "options": ["a", "b"]}
    first = ScriptedProvider([plan_turn(steps(question, ask("tasks", "push {s1} to friday")))])
    reply = run_with(first, "push the dentist thing to friday")
    assert reply["content"] == "Which dentist thing? (a / b)"
    assert world.writes() == []
    (paused,) = _plans(db_session)
    assert (paused.status, paused.pause, paused.resume_at) == ("paused", "ask_user", 1)
    assert ledger.paused_plan(db_session, run_with.conversation) is not None

    second = ScriptedProvider(
        [plan_turn({"resume": "the dentist appointment"}), text_turn("Moved.")]
    )
    run_with(second, "the appointment one")
    assert len(second.requests[0]["response_format"]["json_schema"]["schema"]["anyOf"]) == 3
    assert [w.kind for w in world.writes()] == ["update"]
    _, resumed = _plans(db_session)
    assert resumed.status == "ran" and resumed.body == paused.body
    (step,) = resumed.steps
    assert step.arguments == {"message": "push the dentist appointment to friday"}


def test_a_destructive_ask_to_an_app_that_doesnt_confirm_waits_at_the_gate(
    run_with: Any, world: World, db_session: Session
) -> None:
    gated = steps(ask("music", "delete the album download", destructive=True))
    first = ScriptedProvider([plan_turn(gated)])
    reply = run_with(first, "delete the album download")
    assert reply["content"] == 'Just to check: should I ask music to "delete the album download"?'
    assert reply["tool_calls"] is None
    (paused,) = _plans(db_session)
    assert (paused.pause, paused.resume_at, paused.pause_app) == ("gate", 0, "music")

    no = ScriptedProvider([plan_turn(steps({"do": "confirm", "answer": "no"}))])
    assert run_with(no, "no")["content"] == "Okay, I left it."
    assert ledger.paused_plan(db_session, run_with.conversation) is None


def test_a_yes_at_the_gate_runs_the_gated_step(run_with: Any, db_session: Session) -> None:
    gated = steps(ask("music", "delete the album download", destructive=True))
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    yes = ScriptedProvider(
        [plan_turn(steps({"do": "confirm", "answer": "yes"})), text_turn("Already gone.")]
    )
    reply = run_with(yes, "yes")
    assert [call["tool"] for call in reply["tool_calls"]] == ["ask_music"]
    assert _plans(db_session)[-1].status == "ran"


def test_an_apps_own_question_takes_a_confirm_step(
    run_with: Any, world: World, db_session: Session
) -> None:
    first = ScriptedProvider(
        [plan_turn(steps(ask("tasks", "delete the groceries task", destructive=True)))]
    )
    reply = run_with(first, "delete the groceries task")
    assert reply["content"] == 'Trash "groceries"?'  # tasks confirms: no gate, its question
    assert world.writes() == []
    yes = ScriptedProvider(
        [plan_turn(steps({"do": "confirm", "answer": "yes"})), text_turn("Trashed it.")]
    )
    reply = run_with(yes, "yes")
    assert [call["tool"] for call in reply["tool_calls"]] == ["confirm"]
    assert [w.kind for w in world.writes()] == ["trash"]


def test_the_ack_goes_out_before_any_app_is_asked(fleet: Fleet, world: World) -> None:
    beats: list[LoopActivity] = []
    provider = ScriptedProvider(
        [plan_turn(steps(ask("tasks", "add a task to water the plants"))), text_turn("Added.")]
    )
    loop = AgentLoop(provider, fleet=fleet, narrate=True)
    context = DelegationContext(
        master_conversation_id="1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )
    with use_delegation_context(context):
        result = loop.run("add a task to water the plants", on_activity=beats.append)
    acked = [i for i, beat in enumerate(beats) if beat.ack]
    asked = [i for i, beat in enumerate(beats) if beat.tool == "ask_tasks"]
    assert beats[acked[0]].ack == "On it, asking tasks."
    assert acked[0] < asked[0]
    assert result.plan is not None and result.plan.ack == "On it, asking tasks."


def test_a_dead_model_while_compiling_is_a_502_with_the_turn_kept(
    client: TestClient, fleet: Fleet, db_session: Session
) -> None:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    provider = ScriptedProvider([ProviderRequestError("llama-server unreachable")])
    app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(provider, fleet=fleet)
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": "hi"}
    )
    assert response.status_code == 502
    db_session.expire_all()
    turn = db_session.execute(select(LedgerTurn)).scalar_one()
    assert (turn.status, turn.stop_reason) == ("ended", "provider_error")


# --- the plan-level eval cases are well-formed -------------------------------------------


def test_every_eval_case_has_a_gated_group_and_a_bounded_state() -> None:
    from app.ai import state_block
    from tests.plan_cases import CASES, HOLDOUT, LONG
    from tests.test_plan_evals import FLOORS

    cases = CASES + HOLDOUT + LONG
    assert {case.group for case in cases} == set(FLOORS)
    assert len({case.id for case in cases}) == len(cases)
    assert all(len(case.state) <= state_block.MAX_CHARS for case in cases)
    assert len(LONG) >= 8


def test_the_eval_matchers_accept_a_right_plan_and_refuse_a_wrong_one() -> None:
    from tests.plan_cases import CASES, LONG

    by_id = {case.id: case for case in CASES + LONG}
    right = plans.load(steps(ask("tasks", "delete the groceries task", destructive=True)))
    wrong = plans.load(steps(ask("tasks", "delete the groceries task", destructive=False)))
    case = by_id["confirm-delete-task"]
    assert any(shape.matches(right) for shape in case.expect)
    assert not any(shape.matches(wrong) for shape in case.expect)
    long_right = plans.load(
        steps(
            ask("music", "download the new Bonobo album"),
            ask("music", "download the new Caribou album"),
            ask("music", "download the new Four Tet album"),
            ask("tasks", "add a task to sort {s1} on Saturday"),
            ask("tasks", "add a task to sort {s2} on Saturday"),
            ask("tasks", "add a task to sort {s3} on Saturday"),
            {"do": "open", "app": "chess", "intent": "let's play chess"},
        )
    )
    long_case = by_id["long-three-albums"]
    assert any(shape.matches(long_right) for shape in long_case.expect)
    no_chess = plans.load(steps(ask("music", "download bonobo, caribou and four tet albums")))
    assert not any(shape.matches(no_chess) for shape in long_case.expect)


def test_a_question_about_itself_filed_as_knowledge_answers_from_the_ledger(
    run_with: Any,
) -> None:
    run_with(
        ScriptedProvider(
            [plan_turn(steps(ask("tasks", "add a task to water the plants"))), text_turn("Ok.")]
        ),
        "add a task to water the plants",
    )
    provider = ScriptedProvider(
        [
            plan_turn({"answer": "knowledge"}),
            text_turn('{"about_you": true, "reply": "I added \\"water the plants\\"."}'),
        ]
    )
    reply = run_with(provider, "what did you just do?")["content"]
    assert reply == 'I added "water the plants".'  # no general-knowledge label
    request = provider.requests[1]
    assert request["response_format"]["json_schema"]["name"] == "answer"
    assert 'You were asked: "add a task to water the plants"' in request["messages"][1]["content"]
