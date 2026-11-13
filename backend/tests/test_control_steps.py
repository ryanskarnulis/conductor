"""Control steps (#74): cancel, amend and undo over the ledger.

The model is scripted (``ScriptedProvider``): the compiler's call returns plan
JSON, the narrator's returns text. The apps are the session evals' fake fleet
(``fake_world.py``), reached through the real route, interpreter,
delegate tools and ledger.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.ai import control, interpreter
from app.ai import plan as plans
from app.ai.interpreter import Interpreter
from app.ai.runtime import AgentRunResult, ToolCallRecord
from app.ai.plan import Ledgered, Touched, UndoReceipt
from app.ai.provider import ChatResult
from app.api import routes_agent
from app.db.models import Conversation, LedgerJob, LedgerPlan, LedgerReceipt, LedgerTurn
from app.fleet.delegate import RecordEntry, Undo
from app.fleet.jobs import JobRead
from app.fleet.manifests import Fleet, discover_fleet
from app.fleet.tools import build_delegate_tools
from app.main import app
from app.services import conversations as conversations_service
from app.services import ledger
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn


def plan_turn(body: dict[str, Any]) -> ChatResult:
    return text_turn(json.dumps(body))


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"id": f"s{n}", **item} for n, item in enumerate(items, start=1)]}


def ask(app_name: str, request: str, destructive: bool = False) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


CANCEL = steps({"do": "cancel", "target": "pending"})
UNDO = steps({"do": "undo", "target": "last"})
YES = steps({"do": "confirm", "answer": "yes"})
S3 = steps(
    ask("music", "download the new Caribou album"),
    ask("tasks", "add a task to sort {s1} this weekend"),
)


def amend(change: str) -> dict[str, Any]:
    return steps({"do": "amend", "target": "pending", "change": change})


@pytest.fixture
def fleet(tmp_path: Path) -> Fleet:
    return discover_fleet(write_fixture_fleet(tmp_path / "fleet"))


@pytest.fixture
def world(fleet: Fleet) -> World:
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    return world


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

    run.conversation = conversation  # type: ignore[attr-defined]
    yield run


def _plans(db: Session) -> list[LedgerPlan]:
    db.expire_all()
    return list(db.execute(select(LedgerPlan).order_by(LedgerPlan.id)).scalars())


def _receipts(db: Session) -> list[LedgerReceipt]:
    db.expire_all()
    return list(db.execute(select(LedgerReceipt).order_by(LedgerReceipt.id)).scalars())


def _tools(reply: dict[str, Any]) -> list[str]:
    return [call["tool"] for call in reply["tool_calls"] or []]


def _s3(run_with: Any) -> None:
    run_with(
        ScriptedProvider([plan_turn(S3), text_turn("Queued “Honey” and added the task.")]),
        "download the new Caribou album and add a task to sort it this weekend",
    )


# --- the plan: what runs ----------------------------------------------------------------


def test_a_control_step_needs_no_repair_alone_or_in_a_longer_plan() -> None:
    # #75: a control step inside a longer plan is split off and runs first
    # (tests/test_compiled_plans.py); only a {sN} on it can't run.
    for body in (CANCEL, UNDO, amend("the first ten")):
        assert plans.unrunnable(plans.load(body)) == []
    mixed = steps({"do": "undo", "target": "last"}, ask("tasks", "add milk"))
    assert plans.unrunnable(plans.load(mixed)) == []
    leaning = steps({"do": "undo", "target": "last"}, ask("tasks", "add {s1} again"))
    assert "s1 is an undo" in plans.unrunnable(plans.load(leaning))[0]
    # A running job's cancel/amend are the jobs runtime's (#76, tests/test_jobs.py).


def _receipt(ident: int, plan_id: int, title: str, ref: str = "1", **undo: Any) -> UndoReceipt:
    return UndoReceipt(
        id=ident,
        plan_id=plan_id,
        app="tasks",
        tool="create_task",
        outcome="done",
        touched=(Touched(kind="task", ref=ref, title=title),),
        **undo,
    )


def test_undo_picks_the_last_plan_unless_the_target_names_an_older_one() -> None:
    older = _receipt(1, 5, "Call the vet", ref="41", how="trash it")
    last = _receipt(2, 9, "Sort Honey", ref="52", how="trash it")
    other = _receipt(3, 9, "Pay rent", ref="53", how="trash it")
    view = Ledgered(last_plan=9, receipts=(last, other, older))
    assert control.pick(view, "last") == [last, other]
    assert control.pick(view, "that") == [last, other]
    assert control.pick(view, "the task Call the vet") == [older]
    assert control.pick(view, "tasks:task:41") == [older]
    # The planner naming one thing of the last plan doesn't narrow "undo that"...
    assert control.pick(view, "task 53", "undo that") == [last, other]
    # ...the person naming it does.
    assert control.pick(view, "task 53", "undo the pay rent task") == [other]
    # A word that is no name never reaches back: a read last means nothing to undo.
    assert control.pick(Ledgered(last_plan=10, receipts=(older,)), "last") == []


# --- undo ---------------------------------------------------------------------------------


def test_undo_trashes_the_task_and_says_the_download_cant_be_undone(
    run_with: Any, world: World, db_session: Session
) -> None:
    _s3(run_with)
    created = world.writes()[-1]
    asked = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    # tasks confirms its own writes: its question is the reply, after what can't be undone.
    assert _tools(asked) == ["ask_music", "ask_tasks"]
    assert asked["tool_calls"][0]["error"].startswith("not run: ")
    assert "Undo what you did earlier" in asked["tool_calls"][1]["arguments"]["message"]
    assert "undo handle: create_task:" in asked["tool_calls"][1]["arguments"]["message"]
    assert "can't be undone" in asked["content"] and "“Honey”" in asked["content"]
    assert asked["content"].endswith("?")
    assert [w.kind for w in world.writes()] == ["queue", "create"]
    undo_plan = _plans(db_session)[-1]
    assert (undo_plan.status, undo_plan.pause, undo_plan.pause_app) == (
        "paused",
        "app_confirm",
        "tasks",
    )

    # The narrator leaves out the download: code says it anyway.
    done = run_with(
        ScriptedProvider([plan_turn(YES), text_turn(f"Trashed “{created.title}”.")]), "yes"
    )
    assert [w.kind for w in world.writes()] == ["queue", "create", "trash"]
    assert world.tasks[int(created.ref.split(":")[1])].trashed
    assert "can't be undone" in done["content"] and "Honey" in done["content"]
    assert "say undo" not in done["content"].lower()  # an undo's inverse is never offered
    marked = [r for r in _receipts(db_session) if r.undone_by_step_id is not None]
    assert [(r.app, r.tool) for r in marked] == [("tasks", "create_task")]


def test_undo_twice_never_undoes_a_receipt_again(
    run_with: Any, world: World, db_session: Session
) -> None:
    _s3(run_with)
    run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    run_with(ScriptedProvider([plan_turn(YES), text_turn("Trashed it.")]), "yes")
    writes = len(world.writes())
    again = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert again["content"].startswith("Already undone: “")
    assert "can't be undone" in again["content"]
    assert [call["tool"] for call in again["tool_calls"] or [] if call.get("result")] == []
    assert len(world.writes()) == writes


def _album_only(run_with: Any) -> None:
    run_with(
        ScriptedProvider(
            [plan_turn(steps(ask("music", "download the new Caribou album"))), text_turn("Ok.")]
        ),
        "download the new Caribou album",
    )


@pytest.fixture
def stops(world: World, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The jobs the world's music was asked to stop."""
    asked: list[str] = []

    def cancel_job(fleet_app: Any, ref: str) -> JobRead:
        asked.append(ref)
        return JobRead.model_validate(world.job_cancel(ref))

    monkeypatch.setattr(interpreter, "cancel_job", cancel_job)
    return asked


def test_a_running_download_alone_is_stopped_by_an_undo(
    run_with: Any, world: World, stops: list[str]
) -> None:
    # #185: the job is all the last plan did, so undoing it is stopping it.
    _album_only(run_with)
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert stops == ["1"]
    assert _tools(reply) == ["cancel_job_music"]
    assert [w.kind for w in world.writes()] == ["queue", "cancel"]


def test_a_finished_download_alone_cant_be_undone_and_no_app_is_asked(
    run_with: Any, world: World, db_session: Session, stops: list[str]
) -> None:
    _album_only(run_with)
    db_session.execute(update(LedgerJob).values(status="completed"))
    db_session.commit()
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert reply["content"].startswith(control.CANT_UNDO)
    assert "a download can't be undone" in reply["content"]
    assert all(call.get("result") is None for call in reply["tool_calls"])
    assert [w.kind for w in world.writes()] == ["queue"] and stops == []


@pytest.mark.parametrize("compiled", [UNDO, steps({"do": "cancel", "target": "the download"})])
def test_undo_that_after_a_task_and_a_download_undoes_the_task_only(
    run_with: Any, world: World, stops: list[str], compiled: dict[str, Any]
) -> None:
    # S11 (#185): "undo that" read as a cancel went to the running download.
    _s3(run_with)
    created = world.writes()[-1]
    asked = run_with(ScriptedProvider([plan_turn(compiled)]), "undo that")
    assert stops == []
    assert _tools(asked) == ["ask_music", "ask_tasks"]
    assert "still running, so say stop to end it" in asked["content"]
    # The narrator says the task and the download's reason but not that it runs:
    # it gets one more try, then code says it.
    done = run_with(
        ScriptedProvider(
            [
                plan_turn(YES),
                text_turn(f"Trashed “{created.title}”; the Honey download can't be undone."),
                text_turn(f"Trashed “{created.title}”; the Honey download can't be undone."),
            ]
        ),
        "yes",
    )
    assert [w.kind for w in world.writes()] == ["queue", "create", "trash"] and stops == []
    assert "it is still running" in done["content"]


def test_stop_that_read_as_an_undo_stops_the_download_and_keeps_the_task(
    run_with: Any, world: World, stops: list[str]
) -> None:
    _s3(run_with)
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "stop that")
    assert stops == ["1"]
    assert _tools(reply) == ["cancel_job_music"]
    assert [w.kind for w in world.writes()] == ["queue", "create", "cancel"]


def _journal_by_hand(db: Session, conversation_id: int, *records: ToolCallRecord) -> None:
    """A turn the ledger closed with these calls, as if a plan had run them."""
    conversation = db.get(Conversation, conversation_id)
    assert conversation is not None
    message = conversations_service.append_user_message(db, conversation, "words")
    turn = ledger.open_turn(db, conversation, message)
    run = AgentRunResult(
        reply="ok",
        stop_reason="completed",
        tool_calls=list(records),
        messages=[],
    )
    ledger.close_turn(
        db, turn, run, conversations_service.append_assistant_message(db, conversation, run)
    )
    db.commit()


def test_an_unknown_receipt_is_not_undoable(
    run_with: Any, world: World, db_session: Session
) -> None:
    # A v1 answer: the app ran and left no record conductor could read.
    v1 = ToolCallRecord(tool="ask_tasks", arguments={"message": "tidy up"}, result="Done.")
    _journal_by_hand(db_session, run_with.conversation, v1)
    assert [r.outcome for r in _receipts(db_session)] == ["unknown"]
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert reply["content"].startswith(control.CANT_UNDO)
    assert "no record of what tasks did" in reply["content"]
    assert world.writes() == []


def test_undo_to_an_app_without_confirms_waits_at_codes_gate(
    run_with: Any, fleet: Fleet, world: World, db_session: Session
) -> None:
    filed = ToolCallRecord(
        tool="ask_music",
        arguments={"message": "file Odessa under chill"},
        result="Filed it.",
        app_record=[
            {
                "tool": "file_song",
                "outcome": "done",
                "touched": [{"app": "music", "kind": "song", "id": "12", "title": "Odessa"}],
                "undo": {"handle": "unfile:12", "how": "move it back to unsorted"},
            }
        ],
    )
    _journal_by_hand(db_session, run_with.conversation, filed)
    sent: list[str] = []
    inner = client_factory(world)

    def factory(fleet_app: Any) -> Any:
        client = inner(fleet_app)
        original = client.send_message

        def send(conversation_id: int, message: str, **kwargs: Any) -> Any:
            sent.append(message)
            exchange = original(conversation_id, message, **kwargs)
            exchange.assistant_message.record = [
                RecordEntry(tool="unfile_song", outcome="done", undo=Undo(none="-"))
            ]
            return exchange

        client.send_message = send  # type: ignore[method-assign]
        return client

    build_delegate_tools(fleet, factory)
    gated = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert gated["content"].startswith('Just to check: should I ask music to "Undo what')
    assert "(undo handle: unfile:12)" in gated["content"]
    assert sent == []
    paused = _plans(db_session)[-1]
    assert (paused.pause, paused.resume_at, paused.pause_app) == ("gate", 0, "music")

    run_with(ScriptedProvider([plan_turn(YES), text_turn("Moved “Odessa” back.")]), "yes")
    assert len(sent) == 1 and "song “Odessa”: move it back to unsorted" in sent[0]
    (receipt,) = [r for r in _receipts(db_session) if r.tool == "file_song"]
    assert receipt.undone_by_step_id is not None


def test_nothing_to_undo(run_with: Any, world: World) -> None:
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert reply["content"] == control.NOTHING_TO_UNDO
    run_with(
        ScriptedProvider(
            [plan_turn(steps(ask("tasks", "what's due today?"))), text_turn("Nothing.")]
        ),
        "what's due today?",
    )
    reply = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert reply["content"] == control.CHANGED_NOTHING


# --- cancel -------------------------------------------------------------------------------


def test_cancel_lapses_the_apps_token_and_says_what_wont_happen(
    run_with: Any, world: World, db_session: Session
) -> None:
    trash = steps(ask("tasks", "delete the groceries task", destructive=True))
    asked = run_with(ScriptedProvider([plan_turn(trash)]), "delete the groceries task")
    assert asked["content"] == 'Trash "groceries"?'
    cancelled = run_with(ScriptedProvider([plan_turn(CANCEL)]), "actually, cancel that")
    assert cancelled["content"] == (
        'Okay, cancelled: tasks won\'t get a yes to "Trash "groceries"?".'
    )
    assert cancelled["tool_calls"] is None
    # The token lapsed: a later yes has nothing to answer (the validator
    # refuses it every round), code says so, and nothing is trashed.
    late = ScriptedProvider([plan_turn(YES), plan_turn(YES), plan_turn(YES)])
    assert run_with(late, "yes")["content"] == interpreter.UNPLANNED
    assert world.writes() == []
    assert ledger.paused_plan(db_session, run_with.conversation) is None


def test_cancel_at_the_gate_ends_the_plan_on_the_ledger(
    run_with: Any, world: World, db_session: Session
) -> None:
    gated = steps(
        ask("music", "delete the album download", destructive=True),
        ask("tasks", "add a task to restock {s1}"),
    )
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    waiting = _plans(db_session)[-1]
    reply = run_with(ScriptedProvider([plan_turn(CANCEL)]), "no wait, stop")
    assert reply["content"] == (
        "Okay, cancelled: I won't ask music to \"delete the album download\" and I won't ask "
        'tasks to "add a task to restock that".'
    )
    db_session.refresh(waiting)
    assert (waiting.status, waiting.note) == ("cancelled", waiting.note)
    assert waiting.note is not None and waiting.note.startswith("cancelled by the next turn")
    assert ledger.paused_plan(db_session, run_with.conversation) is None
    assert world.writes() == []


def test_cancel_with_nothing_waiting_says_so(run_with: Any, world: World) -> None:
    reply = run_with(ScriptedProvider([plan_turn(CANCEL)]), "cancel")
    assert reply["content"] == control.NOTHING_TO_CANCEL
    assert world.writes() == []


# --- amend --------------------------------------------------------------------------------


def test_amend_rewrites_the_gated_step_and_resumes_the_same_plan(
    run_with: Any, world: World, db_session: Session
) -> None:
    gated = steps(ask("music", "delete the album download", destructive=True))
    run_with(ScriptedProvider([plan_turn(gated)]), "delete the album download")
    waiting = _plans(db_session)[-1]
    reply = run_with(ScriptedProvider([plan_turn(amend("just the first song"))]), "make it one")
    changed = "delete the album download (change: just the first song)"
    assert reply["content"] == f'Just to check: should I ask music to "{changed}"?'
    db_session.refresh(waiting)
    assert waiting.status == "amended"
    resumed = _plans(db_session)[-1]
    assert (resumed.status, resumed.pause, resumed.resume_at) == ("paused", "gate", 0)
    assert resumed.body is not None and resumed.body["steps"][0]["request"] == changed

    yes = run_with(ScriptedProvider([plan_turn(YES), text_turn("Done.")]), "yes")
    assert yes["tool_calls"][0]["arguments"] == {"message": changed}


def test_amend_resends_the_apps_question_changed(
    run_with: Any, world: World, db_session: Session
) -> None:
    trash = steps(
        ask("tasks", "delete the groceries task", destructive=True),
        ask("music", "download the new Caribou album"),
    )
    run_with(ScriptedProvider([plan_turn(trash)]), "delete the groceries task")
    assert _plans(db_session)[-1].pause == "app_confirm"
    reply = run_with(
        ScriptedProvider([plan_turn(amend("and only that one"))]),
        "only that one",
    )
    sent = reply["tool_calls"][0]
    assert sent["tool"] == "ask_tasks"
    assert sent["arguments"]["message"] == "delete the groceries task (change: and only that one)"
    # The app asks again, about the changed request; the plan waits on it.
    assert reply["content"].startswith("Trash ")
    assert _plans(db_session)[-1].pause == "app_confirm"
    assert world.writes() == []


def test_amend_while_asking_the_person_asks_again_with_the_change(
    run_with: Any, db_session: Session
) -> None:
    question = {"do": "ask_user", "question": "Which dentist thing?", "options": []}
    run_with(
        ScriptedProvider([plan_turn(steps(question, ask("tasks", "push {s1} to friday")))]),
        "push the dentist thing to friday",
    )
    reply = run_with(ScriptedProvider([plan_turn(amend("to thursday"))]), "make it thursday")
    assert reply["content"] == "Which dentist thing?"
    resumed = _plans(db_session)[-1]
    assert (resumed.pause, resumed.resume_at) == ("ask_user", 1)
    assert resumed.body is not None
    assert resumed.body["steps"][1]["request"] == "push {s1} to friday (change: to thursday)"


def test_a_correction_read_as_the_answer_to_which_one_amends_instead(
    run_with: Any, db_session: Session
) -> None:
    # #77: "no, Saturday" over "which one?" compiled as a resume picks none of
    # the options; filling {s1} with it would push a task called "Saturday".
    # Code reads the words, as #185 does: it is the amend.
    question = {
        "do": "ask_user",
        "question": "Which one?",
        "options": ["Dentist appointment", "Call the dentist"],
    }
    # The options must be ones State names: the last exchange does.
    run_with(
        ScriptedProvider(
            [plan_turn({"answer": "chat"}), text_turn('{"about_you": false, "reply": "Ok."}')]
        ),
        "I have Dentist appointment and Call the dentist on my list",
    )
    run_with(
        ScriptedProvider([plan_turn(steps(question, ask("tasks", "push {s1} to friday")))]),
        "push the dentist thing to friday",
    )
    resume = {"about": "the waiting question", "resume": "Saturday"}
    reply = run_with(ScriptedProvider([plan_turn(resume)]), "no, Saturday")
    assert reply["content"] == "Which one? (Dentist appointment / Call the dentist)"
    assert reply["tool_calls"] in (None, [])
    resumed = _plans(db_session)[-1]
    assert (resumed.pause, resumed.resume_at) == ("ask_user", 1)
    assert resumed.body is not None
    assert resumed.body["steps"][1]["request"] == "push {s1} to friday (change: no, Saturday)"
    # A real pick still resumes as picked.
    picked = run_with(
        ScriptedProvider([plan_turn({"resume": "the appointment"}), text_turn("Moved it.")]),
        "the appointment",
    )
    assert picked["tool_calls"][0]["tool"] == "ask_tasks"


def test_amend_with_nothing_waiting_says_so(run_with: Any, world: World) -> None:
    reply = run_with(ScriptedProvider([plan_turn(amend("ten"))]), "make it ten")
    assert reply["content"] == control.NOTHING_TO_CHANGE


def test_a_turns_plan_note_for_control_steps_is_on_the_ledger(
    run_with: Any, db_session: Session
) -> None:
    run_with(ScriptedProvider([plan_turn(CANCEL)]), "cancel")
    (turn,) = db_session.execute(select(LedgerTurn)).scalars()
    (compiled,) = turn.plans
    assert (compiled.status, compiled.note) == ("ran", "cancel: nothing was waiting")


def test_cancelling_an_undos_question_leaves_it_undoable(
    run_with: Any, world: World, db_session: Session
) -> None:
    _s3(run_with)
    run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    cancelled = run_with(ScriptedProvider([plan_turn(CANCEL)]), "no, cancel that")
    assert cancelled["content"].startswith("Okay, cancelled: tasks won't get a yes to ")
    assert all(r.undone_by_step_id is None for r in _receipts(db_session))
    # An undo can't be changed partway: amending one drops it, and says so.
    run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    dropped = run_with(ScriptedProvider([plan_turn(amend("only the task"))]), "just the task")
    assert dropped["content"] == control.CANT_AMEND_UNDO
    assert [w.kind for w in world.writes()] == ["queue", "create"]


def test_a_vague_line_about_the_undo_doesnt_stand_in_for_what_couldnt_be_undone(
    run_with: Any,
) -> None:
    _s3(run_with)
    run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    vague = "Moved it to the trash. The other undo was not performed."
    done = run_with(ScriptedProvider([plan_turn(YES), text_turn(vague)]), "yes")
    assert done["content"].startswith(vague)
    assert "didn't ask music to undo “Honey”, since a download can't be undone" in done["content"]


def test_undo_that_again_never_stops_the_download_left_running(
    run_with: Any, world: World, stops: list[str]
) -> None:
    _s3(run_with)
    run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    run_with(
        ScriptedProvider([plan_turn(YES), text_turn("Trashed it; Honey is still running.")]), "yes"
    )
    again = run_with(ScriptedProvider([plan_turn(UNDO)]), "undo that")
    assert again["content"].startswith("Already undone: “")
    assert stops == [] and [w.kind for w in world.writes()] == ["queue", "create", "trash"]


def test_an_undo_naming_an_older_task_is_no_stop_for_the_running_download(
    run_with: Any, world: World, stops: list[str]
) -> None:
    run_with(
        ScriptedProvider(
            [plan_turn(steps(ask("tasks", "add a task to water the plants"))), text_turn("Ok.")]
        ),
        "add a task to water the plants",
    )
    _album_only(run_with)
    asked = run_with(
        ScriptedProvider([plan_turn(steps({"do": "undo", "target": "water the plants"}))]),
        "undo the water the plants task",
    )
    assert stops == [] and _tools(asked) == ["ask_tasks"]


@pytest.mark.parametrize(
    ("sent", "read"),
    [
        ("push the dentist appointment to friday (change: no, Saturday)", "saturday"),
        ("wipe the kitchen project (change: just the paint cabinets task)", "paint cabinets"),
        (
            "wipe the kitchen project (change: only the paint cabinets task within the Kitchen "
            "project)",
            "paint cabinets",
        ),
        ("wipe the kitchen project", "kitchen project"),
    ],
)
def test_the_fake_tasks_app_reads_an_amends_change_as_pcc_does(sent: str, read: str) -> None:
    # #77: the session evals' fake app must read "(change: …)" like PCC's agent.
    from tests.fake_world import TasksApp, parse_date

    world = World()
    request = TasksApp(world)._changed(sent)
    if read == "saturday":
        saturday = parse_date("saturday", world.today)
        assert saturday is not None and request.endswith(f"to {saturday.isoformat()}")
        return
    assert read in request.lower() and "(change:" not in request
