"""The escalation lane (#78): ``think`` runs in the background on the 26B, and
its proposal waits for the person's yes.

The models are scripted (``ScriptedProvider``: the 12B's plans and the 26B's
proposals); the apps are the session evals' fake fleet, reached through the
real route, interpreter, delegate tools and ledger. The thinker is driven
through the job watcher's ``sweep`` (``force``: run now, to the end) or its
own ``tick`` on a fake clock, with the foreground signal driven by hand.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import clock as household
from app.ai import foreground, interpreter, narrator, think, turn_events
from app.ai import plan as plans
from app.ai.interpreter import Interpreter
from app.ai.provider import ChatProvider, ChatResult, ProviderRequestError
from app.api import routes_agent, turn_activity
from app.config import get_settings
from app.db.models import LedgerJob, LedgerPlan, LedgerReceipt
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.services import conversations as conversations_service
from app.services import ledger
from app.services.job_watcher import JobWatcher
from app.services.thinker import Thinker
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn

WEEK = "plan my week around the garage project"
GARAGE_REQUEST = (
    "Add these tasks to the Garage project:\n- buy the steel rack on monday\n"
    "- assemble the shelves on wednesday\n- sweep the garage floor on friday"
)


def plan_turn(body: dict[str, Any]) -> ChatResult:
    return text_turn(json.dumps(body))


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]}


def ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


def think_step(problem: str = WEEK, inputs: list[str] | None = None) -> dict[str, Any]:
    return {"do": "think", "problem": problem, "inputs": inputs or []}


def confirm(answer: str) -> dict[str, Any]:
    return steps({"do": "confirm", "answer": answer})


def proposal(*asks: dict[str, Any], summary: str = "3 garage tasks, Mon to Fri") -> ChatResult:
    body = {
        "summary": summary,
        "steps": [{"id": f"s{n}", **item} for n, item in enumerate(asks, start=1)],
    }
    return text_turn(json.dumps(body))


GOOD = proposal(ask("tasks", GARAGE_REQUEST))


class Clock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


@pytest.fixture
def events() -> Generator[list[tuple[int, dict[str, Any]]], None, None]:
    seen: list[tuple[int, dict[str, Any]]] = []
    turn_events.set_job_sink(lambda cid, payload: seen.append((cid, payload)))
    try:
        yield seen
    finally:
        turn_events.set_job_sink(turn_activity.publish_job)


@pytest.fixture
def run_with(client: TestClient, fleet: Fleet, world: World) -> Generator[Any, None, None]:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])

    def run(provider: ScriptedProvider, words: str) -> dict[str, Any]:
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            provider, fleet, narrate=True
        )
        response = client.post(
            f"/api/agent/conversations/{conversation}/messages", json={"content": words}
        )
        assert response.status_code == 200, response.text
        assert provider._turns == [], "the script wasn't fully consumed"
        return dict(response.json()["assistant_message"])

    def messages() -> list[dict[str, Any]]:
        detail = client.get(f"/api/agent/conversations/{conversation}")
        return list(detail.json()["messages"])

    run.conversation = conversation  # type: ignore[attr-defined]
    run.messages = messages  # type: ignore[attr-defined]
    yield run


def _thinker(
    session_factory: sessionmaker[Session],
    fleet: Fleet,
    *turns: ChatResult | Exception,
    clock: Clock | None = None,
) -> tuple[JobWatcher, Thinker, ScriptedProvider]:
    provider = ScriptedProvider(list(turns))
    thinker = Thinker(
        session_factory, fleet, provider=lambda: provider, clock=clock, watch_seconds=0.01
    )
    watcher = JobWatcher(session_factory, fleet, clock=clock, thinker=thinker)
    return watcher, thinker, provider


def _detach(run_with: Any, *plan_steps: dict[str, Any], words: str = WEEK) -> dict[str, Any]:
    body = steps(*plan_steps) if plan_steps else steps(think_step())
    reply: dict[str, Any] = run_with(ScriptedProvider([plan_turn(body)]), words)
    return reply


def _job(db: Session) -> LedgerJob:
    db.expire_all()
    return db.execute(select(LedgerJob).where(LedgerJob.app == think.APP)).scalar_one()


def _notes(run_with: Any) -> list[dict[str, Any]]:
    return [
        m for m in run_with.messages() if m["stop_reason"] in (narrator.JOB_NOTE, think.PROPOSAL)
    ]


def _proposed(
    run_with: Any, session_factory: sessionmaker[Session], fleet: Fleet, answer: ChatResult = GOOD
) -> dict[str, Any]:
    _detach(run_with)
    watcher, _, _ = _thinker(session_factory, fleet, answer)
    assert watcher.sweep(force=True) == 1
    notes = _notes(run_with)
    assert len(notes) == 1
    return notes[0]


# --- detaching -------------------------------------------------------------------------------


def test_a_think_detaches_at_once_with_codes_reply_and_a_job_conductor_owns(
    run_with: Any, world: World, db_session: Session
) -> None:
    reply = _detach(run_with)
    assert reply["stop_reason"] == "completed"
    assert reply["content"] == think.THINKING
    job = _job(db_session)
    assert (job.app, job.status, job.title, job.noted_at) == (think.APP, "submitted", WEEK, None)
    assert job.ref.startswith("think-")
    assert world.receipts == []
    # The state block shows it running, by its problem.
    lines = ledger.running_job_lines(db_session, run_with.conversation)
    assert lines == [f"conductor think {job.ref}, “{WEEK}”: submitted"]
    # "What are you doing?" says it from the ledger, claiming no proposal yet.
    report = ledger.self_report(db_session, run_with.conversation)
    assert f"- handed “{WEEK}” to the thinking model, in the background" in report
    assert f"Running: conductor think {job.ref}, “{WEEK}”: submitted" in report


def test_a_think_reads_what_its_inputs_found_and_the_reply_says_what_ran(
    run_with: Any, world: World, db_session: Session
) -> None:
    reply = run_with(
        ScriptedProvider(
            [
                plan_turn(
                    steps(ask("tasks", "what's in the garage project?"), think_step(inputs=["s1"]))
                )
            ]
        ),
        WEEK,
    )
    assert reply["content"].endswith(think.THINKING)
    calls = reply["tool_calls"]
    assert [c["tool"] for c in calls] == ["ask_tasks", "think"]
    given = calls[1]["arguments"]["given"]
    assert given[0]["step"] == "s1" and given[0]["asked"] == "what's in the garage project?"
    assert "answer" in given[0]
    job = ledger.think_job(db_session, _job(db_session).id)
    assert job is not None and job.given == given


def test_a_think_that_isnt_last_ends_the_plan_and_says_what_didnt_run(
    run_with: Any, world: World, db_session: Session
) -> None:
    # #75: nothing runs after a think; the steps after it are named as not run.
    plan = steps(think_step(), ask("tasks", "add them to the garage project"))
    assert plans.unrunnable(plans.load(plan)) == []
    reply = run_with(ScriptedProvider([plan_turn(plan)]), WEEK)
    assert [c["tool"] for c in reply["tool_calls"]] == ["think", "ask_tasks"]
    assert reply["tool_calls"][1]["error"].startswith("not run: nothing runs after")
    assert "didn't ask tasks to add them to the garage project" in reply["content"]
    assert think.THINKING in reply["content"]
    assert world.writes() == []
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).scalar_one().app == think.APP
    compiled = db_session.execute(
        select(LedgerPlan).where(LedgerPlan.source == "compiled")
    ).scalar_one()
    assert compiled.status == "ran"
    assert [s.outcome for s in compiled.steps] == ["done", "not_run"]


def test_with_the_lane_off_a_think_is_not_run_and_said(
    run_with: Any, world: World, monkeypatch: pytest.MonkeyPatch, db_session: Session
) -> None:
    monkeypatch.setattr(get_settings(), "conductor_think", False)
    plan = steps(ask("tasks", "add a task to clear the garage"), think_step(inputs=["s1"]))
    reply = run_with(ScriptedProvider([plan_turn(plan)]), WEEK)
    assert interpreter.THINK_OFF in reply["content"]
    # The steps before the think still ran; the think itself never left conductor.
    assert [w.kind for w in world.writes()] == ["create"]
    assert [c["tool"] for c in reply["tool_calls"]] == ["ask_tasks", "think"]
    assert reply["tool_calls"][1]["error"].startswith("not run: the thinking model")
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).first() is None
    compiled = db_session.execute(
        select(LedgerPlan).where(LedgerPlan.source == "compiled")
    ).scalar_one()
    assert compiled.status == "ran" and compiled.note == "think not run: the lane is off"


# --- from code (#146) ------------------------------------------------------------------------

STUCK = "sort out the garage, the bills and the dentist before the trip"


def _unplannable() -> ScriptedProvider:
    """The compile, its repair and the extra round (#75), none of them a plan."""
    return ScriptedProvider([text_turn("nope"), text_turn("still nope"), text_turn("no")])


def test_a_plan_no_round_can_assemble_is_filed_as_a_think_by_code(
    run_with: Any, world: World, db_session: Session
) -> None:
    provider = _unplannable()
    reply = run_with(provider, STUCK)
    assert len(provider.requests) == 3  # both repair rounds ran first
    assert reply["stop_reason"] == "completed"
    # Honest: it is thinking about it; nothing ran, nothing was proposed.
    assert reply["content"] == f"{interpreter.UNPLANNED_THINKS} {think.THINKING}"
    assert [c["tool"] for c in reply["tool_calls"]] == ["think"]
    assert world.writes() == [] and world.reads == []
    job = _job(db_session)
    assert (job.app, job.status, job.title, job.noted_at) == (think.APP, "submitted", STUCK, None)
    compiled = db_session.execute(
        select(LedgerPlan).where(LedgerPlan.source == "compiled")
    ).scalar_one()
    assert compiled.status == "invalid" and compiled.body is None
    assert compiled.note is not None and "not a plan" in compiled.note
    assert "thinking in the background" in compiled.note
    assert [(s.kind, s.outcome) for s in compiled.steps] == [("think", "done")]


def test_the_think_from_code_proposes_untried_and_waits_for_a_yes(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    run_with(_unplannable(), STUCK)
    watcher, _, provider = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    (note,) = _notes(run_with)
    assert note["stop_reason"] == think.PROPOSAL
    assert STUCK in provider.requests[0]["messages"][1]["content"]
    assert world.writes() == []  # a proposal, tainted: nothing runs without a yes
    reply = run_with(
        ScriptedProvider([plan_turn(confirm("yes")), text_turn("Added three garage tasks.")]),
        "yes",
    )
    assert [c["tool"] for c in reply["tool_calls"]] == ["ask_tasks"]
    assert len([r for r in world.receipts if r.kind == "create"]) == 3


def test_a_plan_that_stays_unrunnable_is_filed_too(
    run_with: Any, world: World, db_session: Session
) -> None:
    looked = plan_turn(steps({"do": "look", "app": "tasks"}, ask("tasks", "move {s1} to friday")))
    reply = run_with(ScriptedProvider([looked, looked]), "move what's due to friday")
    assert reply["content"].endswith(think.THINKING)
    assert world.writes() == [] and world.reads == []
    assert _job(db_session).title == "move what's due to friday"


def test_tries_that_disagree_on_the_app_keep_the_quick_question(
    run_with: Any, world: World, db_session: Session
) -> None:
    def misnumbered(app_name: str, request: str) -> ChatResult:
        return text_turn(json.dumps({"steps": [{"id": "s2", **ask(app_name, request)}]}))

    provider = ScriptedProvider(
        [
            misnumbered("music", "grab Odessa"),
            misnumbered("tasks", "add Odessa"),
            misnumbered("music", "grab Odessa"),
        ]
    )
    reply = run_with(provider, "Odessa")
    assert reply["content"] == 'Do you mean "grab Odessa" in music, or "add Odessa" in tasks?'
    assert db_session.execute(select(LedgerJob)).first() is None


def test_with_the_lane_off_an_unplannable_turn_is_as_it_always_was(
    run_with: Any, world: World, monkeypatch: pytest.MonkeyPatch, db_session: Session
) -> None:
    monkeypatch.setattr(get_settings(), "conductor_think", False)
    reply = run_with(_unplannable(), STUCK)
    assert reply["content"] == interpreter.UNPLANNED
    assert reply["tool_calls"] is None
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).first() is None
    compiled = db_session.execute(select(LedgerPlan)).scalar_one()
    assert compiled.status == "invalid" and compiled.steps == []


def test_a_dead_model_is_a_502_never_a_think(
    client: TestClient, fleet: Fleet, db_session: Session
) -> None:
    conversation = int(client.post("/api/agent/conversations", json={}).json()["id"])
    provider = ScriptedProvider([ProviderRequestError("llama-server unreachable")])
    app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(provider, fleet)
    response = client.post(
        f"/api/agent/conversations/{conversation}/messages", json={"content": STUCK}
    )
    assert response.status_code == 502
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).first() is None


def test_a_yes_with_nothing_waiting_is_not_worth_thinking_over(
    run_with: Any, world: World, db_session: Session
) -> None:
    # The planner read it as an answer to a question that isn't there (the
    # validator refuses it every round): there is no problem to think over.
    yes = plan_turn(confirm("yes"))
    reply = run_with(ScriptedProvider([yes, yes, yes]), "yes")
    assert reply["content"] == interpreter.UNPLANNED
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).first() is None


def test_an_answer_to_a_waiting_question_is_never_filed_as_a_think(
    run_with: Any, world: World, db_session: Session
) -> None:
    waiting = steps(ask("tasks", "delete the groceries task", destructive=True))
    asked = run_with(ScriptedProvider([plan_turn(waiting)]), "delete the groceries task")
    assert asked["content"] == 'Trash "groceries"?'
    reply = run_with(_unplannable(), "hmm, how about the other one")
    assert reply["content"] == interpreter.UNPLANNED
    db_session.expire_all()
    assert db_session.execute(select(LedgerJob)).first() is None


def test_only_a_request_the_planner_could_not_shape_is_worth_a_think() -> None:
    def raw(body: dict[str, Any]) -> str:
        return json.dumps(body)

    cancel = {"do": "cancel", "target": "that"}
    assert interpreter.answering([raw(confirm("yes")), "nope"]) is True
    assert interpreter.answering(["nope", raw(steps(ask("tasks", "x"), cancel))]) is True
    assert interpreter.answering([raw({"resume": "the first"})]) is True
    assert interpreter.answering(["nope", raw(steps(ask("tasks", "x")))]) is False
    assert interpreter.answering(["nope", raw({"answer": "chat"})]) is False
    assert interpreter.answering(["nope", "still nope"]) is False


# --- the proposal ----------------------------------------------------------------------------


def test_the_26b_thinks_under_json_schema_and_its_proposal_posts_untried(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
    events: list[tuple[int, dict[str, Any]]],
) -> None:
    _detach(run_with)
    watcher, _, provider = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    request = provider.requests[0]
    assert request["enable_thinking"] is True
    assert request["max_tokens"] == think.MAX_TOKENS
    assert request["response_format"]["json_schema"]["name"] == "proposal"
    assert request["messages"][0]["content"] == think.system_prompt(fleet)
    assert f"The problem: {WEEK}" in request["messages"][1]["content"]
    [note] = _notes(run_with)
    assert note["stop_reason"] == think.PROPOSAL
    assert "3 garage tasks" in note["content"] and "buy the steel rack" in note["content"]
    assert note["content"].endswith("Shall I go ahead?")
    # Never a write before the yes.
    assert world.receipts == []
    job = _job(db_session)
    assert (job.status, job.note_message_id) == ("completed", note["id"])
    paused = ledger.paused_plan(db_session, run_with.conversation)
    assert paused is not None and paused.reason == "proposal"
    assert paused.body == {"steps": [ask("tasks", GARAGE_REQUEST) | {"id": "s1"}]}
    ended = events[-1][1]
    assert ended["ended"] is True and ended["how"] == "proposed"
    assert ended["note_message_id"] == note["id"]
    # The proposal is an exchange: the next turn's state block asks about it.
    turns = conversations_service.stored_turns(db_session, run_with.conversation)
    assert turns[-1]["content"] == note["content"]


def test_the_26b_calendar_is_the_households_day_as_of_the_ask(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asked Sunday 18:30 in Denver, 01:30 UTC Monday (#130); thought about later.
    monkeypatch.setattr(household, "utc_now", lambda: datetime(2026, 11, 9, 1, 30, tzinfo=UTC))
    monkeypatch.setattr(get_settings(), "conductor_timezone", "America/Denver")
    _detach(run_with)
    later = Clock()
    later.now = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)
    monkeypatch.setattr(household, "utc_now", later)
    watcher, _, provider = _thinker(session_factory, fleet, GOOD, clock=later)
    assert watcher.sweep(force=True) == 1
    opening = provider.requests[0]["messages"][1]["content"]
    assert "Today: Sunday 2026-11-08" in opening and "Tomorrow: Monday 2026-11-09" in opening


def test_yes_runs_the_proposal_as_a_normal_plan(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _proposed(run_with, session_factory, fleet)
    reply = run_with(
        ScriptedProvider(
            [plan_turn(confirm("yes")), text_turn("Added three garage tasks for Mon, Wed and Fri.")]
        ),
        "yes",
    )
    assert [c["tool"] for c in reply["tool_calls"]] == ["ask_tasks"]
    created = [r for r in world.receipts if r.kind == "create"]
    assert len(created) == 3
    assert all(r.detail.get("project") == "Garage" for r in created)
    db_session.expire_all()
    kept = db_session.execute(
        select(LedgerPlan).where(LedgerPlan.source == think.KIND)
    ).scalar_one()
    assert kept.status == "accepted"
    assert ledger.paused_plan(db_session, run_with.conversation) is None


def test_no_sets_the_proposal_aside_and_nothing_runs(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _proposed(run_with, session_factory, fleet)
    reply = run_with(ScriptedProvider([plan_turn(confirm("no"))]), "no thanks")
    assert "set that plan aside" in reply["content"]
    assert world.receipts == []
    db_session.expire_all()
    kept = db_session.execute(
        select(LedgerPlan).where(LedgerPlan.source == think.KIND)
    ).scalar_one()
    assert kept.status == "set_aside"


def test_anything_else_lets_the_proposal_lapse(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _proposed(run_with, session_factory, fleet)
    run_with(ScriptedProvider([plan_turn({"answer": "chat"}), text_turn("Any time!")]), "thanks")
    assert ledger.paused_plan(db_session, run_with.conversation) is None
    # A later yes has nothing to answer: the validator refuses a confirm.
    checks = plans.Checks(fleet=fleet, confirmable=False)
    (found,) = plans.problems(plans.StepPlan.model_validate(confirm("yes")), checks)
    assert found.startswith("confirm with nothing waiting on the person")
    assert world.receipts == []


def test_an_amend_sets_the_proposal_aside_rather_than_editing_it(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _proposed(run_with, session_factory, fleet)
    reply = run_with(
        ScriptedProvider(
            [plan_turn(steps({"do": "amend", "target": "pending", "change": "skip friday"}))]
        ),
        "make it skip friday",
    )
    assert "can't edit a plan I proposed" in reply["content"]
    assert world.receipts == []


def test_a_tainted_proposal_still_stops_at_codes_gate_for_a_destructive_ask(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    risky = proposal(ask("music", "delete the Honey download", destructive=True), summary="tidy")
    _proposed(run_with, session_factory, fleet, risky)
    reply = run_with(ScriptedProvider([plan_turn(confirm("yes"))]), "yes")
    # music doesn't confirm its own deletes: code asks again, nothing written.
    assert reply["content"].startswith("Just to check")
    assert world.receipts == []


def test_an_invalid_proposal_gets_one_repair(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
) -> None:
    _detach(run_with)
    bad = proposal(ask("tasks", ""), summary="nothing")
    watcher, _, provider = _thinker(session_factory, fleet, bad, GOOD)
    assert watcher.sweep(force=True) == 1
    repair = provider.requests[1]["messages"][-1]["content"]
    assert repair.startswith("Code can't run that proposal: s1 asks tasks nothing")
    assert _notes(run_with)[0]["stop_reason"] == think.PROPOSAL


def test_no_runnable_plan_after_the_repair_is_said_honestly(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    watcher, _, _ = _thinker(session_factory, fleet, text_turn("not json"), text_turn("{}"))
    assert watcher.sweep(force=True) == 1
    [note] = _notes(run_with)
    assert note["stop_reason"] == narrator.JOB_NOTE
    assert "couldn't turn it into a plan" in note["content"]
    assert ledger.paused_plan(db_session, run_with.conversation) is None
    assert _job(db_session).status == "failed"


def test_an_answer_with_no_steps_is_a_plain_note(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with, think_step("should I paint or stain the shelves?"))
    answer = proposal(summary="Stain them: it holds up better to paint cans")
    watcher, _, _ = _thinker(session_factory, fleet, answer)
    assert watcher.sweep(force=True) == 1
    [note] = _notes(run_with)
    assert note["stop_reason"] == narrator.JOB_NOTE
    assert "Stain them" in note["content"]
    assert ledger.paused_plan(db_session, run_with.conversation) is None


def test_a_proposal_is_the_newest_question_over_one_left_waiting(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    gated = steps(ask("music", "delete the album download", destructive=True))
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    assert ledger.paused_plan(db_session, run_with.conversation).reason == "gate"  # type: ignore[union-attr]
    watcher, _, _ = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    paused = ledger.paused_plan(db_session, run_with.conversation)
    assert paused is not None and paused.reason == "proposal"


# --- yielding the GPU ------------------------------------------------------------------------


def test_an_interrupted_try_waits_for_the_quiet_then_gives_up_honestly(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    clock = Clock()
    cut = ProviderRequestError("llama-server returned 502: upstream exited")
    watcher, thinker, provider = _thinker(session_factory, fleet, cut, cut, cut, clock=clock)
    # The first try starts at once and is cut off (the 26B was stopped).
    assert watcher.sweep() == 0
    thinker.join(5)
    assert _job(db_session).status == "interrupted"
    assert len(provider.requests) == 1
    # Not again until the GPU has been free of turns for a minute.
    clock.advance(seconds=30)
    assert watcher.sweep() == 0
    assert len(provider.requests) == 1
    foreground.entered()
    clock.advance(seconds=40)
    watcher.sweep()
    assert len(provider.requests) == 1, "tried while a turn ran"
    foreground.left()
    foreground._reset()  # as if that turn ended a minute ago
    for expected in (2, 3):
        clock.advance(seconds=61)
        watcher.sweep()
        thinker.join(5)
        assert len(provider.requests) == expected
    clock.advance(seconds=61)
    assert watcher.sweep() == 1
    [note] = _notes(run_with)
    assert "couldn't finish thinking" in note["content"] and "3 times" in note["content"]
    assert "Nothing was changed" in note["content"]
    assert ledger.paused_plan(db_session, run_with.conversation) is None
    job = _job(db_session)
    assert job.status == "failed"
    cutoffs = (
        db_session.execute(
            select(LedgerReceipt).where(
                LedgerReceipt.job_id == job.id, LedgerReceipt.outcome == "not_run"
            )
        )
        .scalars()
        .all()
    )
    assert len(cutoffs) == 3 and all(r.reason.startswith("interrupted") for r in cutoffs)  # type: ignore[union-attr]
    assert world.receipts == []


class BlockingProvider:
    """A 26B call that runs until it's aborted (or released)."""

    def __init__(self, answer: ChatResult) -> None:
        self.answer = answer
        self.started = threading.Event()
        self.release = threading.Event()
        self.aborted = False

    def chat(self, messages: Any, **kwargs: Any) -> ChatResult:
        self.started.set()
        self.release.wait(10)
        if self.aborted:
            raise ProviderRequestError("llama-server request failed: client closed")
        return self.answer

    def abort(self) -> None:
        self.aborted = True
        self.release.set()


def test_a_turn_starting_cuts_the_try_off_and_it_retries_after_the_quiet(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    clock = Clock()
    blocking = BlockingProvider(GOOD)
    providers: list[ChatProvider] = [blocking, ScriptedProvider([GOOD])]
    thinker = Thinker(
        session_factory,
        fleet,
        provider=lambda: providers.pop(0),
        clock=clock,
        watch_seconds=0.01,
    )
    watcher = JobWatcher(session_factory, fleet, clock=clock, thinker=thinker)
    watcher.sweep()
    assert blocking.started.wait(5)
    foreground.entered()  # the person says something: their turn needs the GPU
    thinker.join(5)
    assert blocking.aborted
    assert _job(db_session).status == "interrupted"
    foreground.left()
    foreground._reset()
    clock.advance(seconds=61)
    watcher.sweep()
    thinker.join(5)
    assert _notes(run_with)[0]["stop_reason"] == think.PROPOSAL


def test_a_try_never_starts_while_a_turn_runs(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    watcher, thinker, provider = _thinker(session_factory, fleet, GOOD)
    foreground.entered()
    watcher.sweep()
    thinker.join(1)
    assert provider.requests == []
    foreground.left()
    watcher.sweep()
    thinker.join(5)
    assert len(provider.requests) == 1


# --- restarts and cancel ---------------------------------------------------------------------


def test_a_try_left_working_by_a_dead_process_is_a_cutoff_and_the_note_posts_once(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    job = _job(db_session)
    with session_factory() as db:
        ledger.think_started(db, job.id, datetime.now(UTC))
        db.commit()
    # A new process: a fresh thinker finds the row "working" with no try running.
    watcher, _, provider = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    reloaded = ledger.think_job(db_session, job.id)
    assert reloaded is not None and reloaded.interruptions == 1
    # And another restart never posts it again, nor calls the model.
    again, _, later = _thinker(session_factory, fleet)
    assert again.sweep(force=True) == 0
    assert later.requests == []
    assert len(_notes(run_with)) == 1
    assert len(provider.requests) == 1


def test_two_thinkers_racing_post_one_note(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    job = ledger.think_job(db_session, _job(db_session).id)
    assert job is not None
    _, first, _ = _thinker(session_factory, fleet, GOOD)
    _, second, _ = _thinker(session_factory, fleet, GOOD)
    first._try(job)
    assert second._try(job) == "skipped"
    assert len(_notes(run_with)) == 1


def test_cancel_stops_a_running_think_and_nothing_is_proposed(
    run_with: Any,
    world: World,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    reply = run_with(
        ScriptedProvider([plan_turn(steps({"do": "cancel", "target": "the job"}))]), "stop that"
    )
    assert reply["content"] == f"Stopped thinking over “{WEEK}”. Nothing was changed."
    watcher, _, provider = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    assert provider.requests == []
    [note] = _notes(run_with)
    assert note["stop_reason"] == narrator.JOB_NOTE and "stopped thinking" in note["content"]
    assert _job(db_session).status == "canceled"
    assert ledger.paused_plan(db_session, run_with.conversation) is None
    assert world.receipts == []


def test_a_cancel_cuts_a_try_in_flight(
    run_with: Any,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    blocking = BlockingProvider(GOOD)
    thinker = Thinker(session_factory, fleet, provider=lambda: blocking, watch_seconds=0.01)
    watcher = JobWatcher(session_factory, fleet, thinker=thinker)
    watcher.sweep()
    assert blocking.started.wait(5)
    think.request_cancel(_job(db_session).ref)
    thinker.join(5)
    assert blocking.aborted
    [note] = _notes(run_with)
    assert "stopped thinking" in note["content"]
    assert _job(db_session).status == "canceled"


def test_a_job_cards_cancel_stops_a_think_with_no_app_to_ask(
    run_with: Any,
    client: TestClient,
    fleet: Fleet,
    session_factory: sessionmaker[Session],
    db_session: Session,
) -> None:
    _detach(run_with)
    ref = _job(db_session).ref

    def no_app(*_: Any) -> Any:
        raise AssertionError("a think's cancel never reaches an app")

    app.dependency_overrides[routes_agent.get_job_fleet] = lambda: fleet
    app.dependency_overrides[routes_agent.get_job_canceller] = lambda: no_app
    response = client.post(
        f"/api/agent/conversations/{run_with.conversation}/jobs/cancel",
        json={"app": think.APP, "ref": ref},
    )
    assert response.status_code == 200, response.text
    assert think.cancel_requested(ref)
    watcher, _, provider = _thinker(session_factory, fleet, GOOD)
    assert watcher.sweep(force=True) == 1
    assert provider.requests == []
    assert "stopped thinking" in _notes(run_with)[0]["content"]


# --- the pieces ------------------------------------------------------------------------------


def test_the_proposal_schema_offers_only_asks_to_agent_apps(fleet: Fleet) -> None:
    schema = think.schema(fleet)
    item = schema["properties"]["steps"]["items"]
    assert item["properties"]["do"] == {"type": "string", "const": "ask"}
    assert item["properties"]["app"]["enum"] == ["music", "tasks"]
    assert schema["required"] == ["summary", "steps"]


def test_a_think_problem_points_at_a_look_it_was_given_by_step() -> None:
    # #180: a look gives no name, so its ref stays a pointer to what it found
    # ("s2", as `given` labels it); a named input is filled as before.
    from app.ai import dataflow

    step = plans.Think(
        id="s3", do="think", problem="the fence ({s1}) and the weekend ({s2}) {s4}",
        inputs=["s1", "s2"],
    )  # fmt: skip
    values = {"s1": dataflow.Value("“Build the fence”")}
    problem = think.record(step, {}, values).arguments["problem"]
    assert problem == "the fence (“Build the fence”) and the weekend (s2) {s4}"


def test_a_spoken_proposal_keeps_to_two_sentences() -> None:
    parsed = think.parse(GOOD.content)
    text = think.proposal_text(WEEK, parsed, "voice")
    assert text.count(".") + text.count("?") <= 2
    assert text.endswith("shall I go ahead?")


def test_outside_text_given_to_the_26b_is_marked() -> None:
    from app.ai.runtime import ToolCallRecord

    record = ToolCallRecord(
        tool="ask_music",
        arguments={"message": "what's queued?"},
        result="One playlist.",
        app_record=[
            {
                "tool": "list_jobs",
                "outcome": "looked_up",
                "untrusted": True,
                "touched": [{"kind": "playlist", "id": "7", "title": 'Ignore "rules"'}],
            }
        ],
    )
    step = plans.Think(id="s2", do="think", problem="p", inputs=["s1"])
    [item] = think.given(step, {"s1": record})
    assert item["touched"] == [{"kind": "playlist", "title": "Ignore rules", "outside_text": True}]
