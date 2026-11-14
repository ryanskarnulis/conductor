"""Titles out of the words (#204, #145's decision 2): an app in
``dataflow.TITLES_OUT`` that takes v1.2's context gets ``{sN}`` as a pointer
("that album") and the title as a ``context.refs`` ref alone. PCC's evals pass
on that (project-command-center#531: ``conductor_s3_ref_only`` 20/20); a v1.1
app, or one not listed, still gets the quoted title.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import Any

import pytest

from app.ai import dataflow
from app.ai.interpreter import Interpreter
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet
from app.fleet.tools import build_delegate_tools
from tests.fake_world import FakeGlances, World, client_factory, without_context
from tests.scripted_provider import ScriptedProvider, text_turn
from tests.test_memory import ask, fleet, plan_turn, steps  # noqa: F401  (fixtures)

HONEY = {"app": "music", "kind": "album", "id": "rg-honey", "title": "Honey", "untrusted": True}
SWIM = {"app": "music", "kind": "album", "id": "rg-swim", "title": "Swim", "untrusted": True}
TASK = {"app": "tasks", "kind": "task", "id": "4", "title": "Sweep", "untrusted": False}


def _value(*refs: dict[str, Any]) -> dataflow.Value:
    return dataflow.Value("“Honey”", True, tuple(refs))


@pytest.mark.parametrize(
    ("refs", "said"),
    [
        ((HONEY,), "that album"),
        ((HONEY, SWIM), "those albums"),
        ((HONEY, TASK), "those"),
        ((), None),  # an answer or a recall's title: no refs, the title stays
        (({**HONEY, "kind": ""},), None),
    ],
)
def test_a_pointer_names_the_kind(refs: tuple[dict[str, Any], ...], said: str | None) -> None:
    assert dataflow.pointer(_value(*refs)) == said


@pytest.mark.parametrize(
    ("text", "filled"),
    [
        ("add a task to sort {s1} this weekend", "add a task to sort that album this weekend"),
        # Words before it that already say the kind go with it.
        ("sort the album {s1}", "sort that album"),
        ("sort the albums {s1}", "sort that album"),
        ("sort this album {s1}", "sort that album"),
        # Other words before it stay.
        ("sort the music {s1}", "sort the music that album"),
        ("{s1} needs sorting", "that album needs sorting"),
    ],
)
def test_pointers_fill_the_words(text: str, filled: str) -> None:
    got = dataflow.fill(text, {"s1": _value(HONEY)}, pointers=True)
    assert got.text == filled
    assert got.untrusted == ("s1",)  # still outside text, for the taint rules


def test_without_pointers_the_title_is_spliced_as_before() -> None:
    values = {"s1": _value(HONEY)}
    assert dataflow.fill("sort the album {s1}", values).text == "sort the album “Honey”"
    answer = {"s1": dataflow.answer("the blue one")}
    assert dataflow.fill("sort {s1}", answer, pointers=True).text == "sort the blue one"
    assert dataflow.fill("sort {s2}", values, pointers=True).missing == ("s2",)


# --- a plan --------------------------------------------------------------------------------


def _flag(fleet: Fleet, name: str) -> Fleet:  # noqa: F811
    apps = tuple(
        dataclasses.replace(app, agent=dataclasses.replace(app.agent, context=True))
        if app.name == name and app.agent is not None
        else app
        for app in fleet.apps
    )
    return dataclasses.replace(fleet, apps=apps)


def _s3(fleet: Fleet) -> World:  # noqa: F811
    world = World()
    build_delegate_tools(fleet, client_factory(world))
    provider = ScriptedProvider(
        [
            plan_turn(
                steps(
                    ask("music", "download the new Caribou album"),
                    ask("tasks", "add a task to sort {s1} this weekend"),
                )
            ),
            text_turn("Queued “Honey” and added a task to sort it."),
        ]
    )
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        result = conductor.run("download the new Caribou album and add a task to sort it")
    assert result.stop_reason == "completed", result
    return world


def test_a_v1_2_task_ask_says_that_album_and_sends_the_ref(fleet: Fleet) -> None:  # noqa: F811
    world = _s3(_flag(fleet, "tasks"))
    (sent,) = [message for app, message in world.sent if app == "tasks"]
    assert sent == "add a task to sort that album this weekend"
    ((context, _),) = [(c, m) for app, c, m in world.sent_v12 if app == "tasks"]
    assert context is not None
    first = context["refs"][0]
    assert (first["kind"], first["title"], first["untrusted"]) == ("album", "Honey", True)
    # The app named the task from the ref, as PCC's agent now does.
    (task,) = [t for t in world.tasks.values() if "sort" in t.title.lower()]
    assert "Honey" in task.title
    assert task.due is not None and task.due - world.today <= timedelta(days=7)


def test_a_v1_1_task_ask_still_carries_the_quoted_title(fleet: Fleet) -> None:  # noqa: F811
    world = _s3(without_context(fleet, "tasks"))
    (sent,) = [message for app, message in world.sent if app == "tasks"]
    assert sent == "add a task to sort “Honey” this weekend"
