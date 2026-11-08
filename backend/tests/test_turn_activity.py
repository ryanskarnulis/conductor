"""The turn-activity registry and the interpreter's on_activity seam."""

from __future__ import annotations

import json
from pathlib import Path

from app.ai.interpreter import Interpreter
from app.ai.runtime import Activity
from app.api import turn_activity
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import discover_fleet
from app.fleet.tools import build_delegate_tools
from tests.fake_world import World, client_factory, write_fixture_fleet
from tests.scripted_provider import ScriptedProvider, text_turn


def test_registry_begin_update_end() -> None:
    assert turn_activity.get(1) is None

    turn_activity.begin(1)
    started = turn_activity.get(1)
    assert started is not None
    assert (started.kind, started.tool, started.iteration) == ("model", None, 1)

    turn_activity.update(1, kind="tool", tool="ask_chess")
    updated = turn_activity.get(1)
    assert updated is not None
    # No iterations since #75: the wire field stays 1.
    assert (updated.kind, updated.tool, updated.iteration) == ("tool", "ask_chess", 1)
    # The turn's start time is preserved across beats — elapsed is turn-level.
    assert updated.started_at == started.started_at

    turn_activity.end(1)
    assert turn_activity.get(1) is None


def test_update_never_resurrects_an_ended_turn() -> None:
    turn_activity.begin(1)
    turn_activity.end(1)
    turn_activity.update(1, kind="tool", tool="ask_chess")
    assert turn_activity.get(1) is None


def _delegation() -> DelegationContext:
    return DelegationContext(
        master_conversation_id="1", thread_store=InMemoryThreadStore(), calls_per_turn_per_app=3
    )


def test_a_turn_emits_one_beat_per_model_call_and_step(tmp_path: Path) -> None:
    fleet = discover_fleet(write_fixture_fleet(tmp_path / "fleet"))
    build_delegate_tools(fleet, client_factory(World()))
    plan = {
        "steps": [
            {"id": "s1", "do": "ask", "app": "tasks", "request": "add milk", "destructive": False}
        ]
    }
    provider = ScriptedProvider([text_turn(json.dumps(plan)), text_turn("Added “milk”.")])
    beats: list[Activity] = []

    with use_delegation_context(_delegation()):
        run = Interpreter(provider, fleet, narrate=True).run("add milk", on_activity=beats.append)

    assert run.stop_reason == "completed"
    assert beats == [
        Activity(kind="model"),  # the compile
        Activity(kind="model", ack="On it, asking tasks."),
        Activity(kind="tool", tool="ask_tasks"),
        Activity(kind="model"),  # the narrator
    ]


def test_broken_activity_callback_never_breaks_the_run(tmp_path: Path) -> None:
    fleet = discover_fleet(write_fixture_fleet(tmp_path / "fleet"))

    def explode(_: Activity) -> None:
        raise RuntimeError("callback bug")

    provider = ScriptedProvider(
        [text_turn('{"answer": "chat"}'), text_turn('{"about_you": false, "reply": "Hi."}')]
    )
    with use_delegation_context(_delegation()):
        run = Interpreter(provider, fleet).run("hello", on_activity=explode)

    assert run.stop_reason == "completed"
    assert run.reply == "Hi."
