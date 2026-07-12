"""The turn-activity registry and the loop's on_activity seam."""

from __future__ import annotations

from app.ai.loop import AgentLoop, LoopActivity
from app.api import turn_activity
from app.tools import registry
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn


def test_registry_begin_update_end() -> None:
    assert turn_activity.get(1) is None

    turn_activity.begin(1)
    started = turn_activity.get(1)
    assert started is not None
    assert (started.kind, started.tool, started.iteration) == ("model", None, 1)

    turn_activity.update(1, kind="tool", tool="ask_chess", iteration=2)
    updated = turn_activity.get(1)
    assert updated is not None
    assert (updated.kind, updated.tool, updated.iteration) == ("tool", "ask_chess", 2)
    # The turn's start time is preserved across beats — elapsed is turn-level.
    assert updated.started_at == started.started_at

    turn_activity.end(1)
    assert turn_activity.get(1) is None


def test_update_never_resurrects_an_ended_turn() -> None:
    turn_activity.begin(1)
    turn_activity.end(1)
    turn_activity.update(1, kind="tool", tool="ask_chess", iteration=1)
    assert turn_activity.get(1) is None


def test_loop_emits_one_beat_per_provider_turn_and_tool_dispatch() -> None:
    @registry.tool
    def scratch_beat() -> str:
        """Scratch tool for beat ordering."""
        return "ok"

    provider = ScriptedProvider([tool_calls_turn(("scratch_beat", {})), text_turn("done")])
    beats: list[LoopActivity] = []

    run = AgentLoop(provider).run("go", on_activity=beats.append)

    assert run.stop_reason == "completed"
    assert beats == [
        LoopActivity(kind="model", iteration=1),
        LoopActivity(kind="tool", iteration=1, tool="scratch_beat"),
        LoopActivity(kind="model", iteration=2),
    ]


def test_broken_activity_callback_never_breaks_the_run() -> None:
    def explode(_: LoopActivity) -> None:
        raise RuntimeError("callback bug")

    run = AgentLoop(ScriptedProvider([text_turn("fine")])).run("go", on_activity=explode)

    assert run.stop_reason == "completed"
    assert run.reply == "fine"
