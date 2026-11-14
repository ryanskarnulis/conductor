"""Saved facts by meaning (#204, #145's decision 4): an app ask that isn't a
read, whose words share no two terms with any saved fact, carries the one fact
it clearly means. All on the fake embedder (``tests/fake_embedder.py``); the
bar is measured against the real service in ``test_fact_cases.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.ai import facts, incognito
from app.ai.interpreter import Interpreter
from app.config import get_settings
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.manifests import Fleet
from app.services import fact_meaning
from tests.fake_embedder import FakeEmbedder, use
from tests.fake_world import FakeGlances, World
from tests.scripted_provider import ScriptedProvider, text_turn
from tests.test_memory import (  # noqa: F401  (fixtures)
    Talk,
    ask,
    fleet,
    plan_turn,
    steps,
    talk,
    world,
)

SUNDAYS = "I sort music on Sundays"
KEY = "the spare key is under the blue pot"
ALBUMS = "add a task to organize my albums"
# "organize my albums" means sorting music, to the fake as to the real model.
SAME = {"organize": "sort", "albums": "music", "album": "music"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    return use(monkeypatch, FakeEmbedder(SAME))


# --- the bar -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cosines", "baseline", "chosen"),
    [
        ([0.75, 0.60], 0.60, 0),  # clear of the background and of the next fact
        ([0.60, 0.75], 0.60, 1),
        ([0.70], 0.60, None),  # under the lift
        ([0.75, 0.72], 0.60, None),  # two about as close: neither
        ([0.75], 0.60, 0),  # one fact: no gap to keep
        ([], 0.60, None),
    ],
)
def test_one_fact_clears_the_background_and_the_next(
    cosines: list[float], baseline: float, chosen: int | None
) -> None:
    assert facts.nearest(cosines, baseline) == chosen


# --- the service -------------------------------------------------------------------------


def test_a_paraphrase_finds_its_fact(fake: FakeEmbedder) -> None:
    assert fact_meaning.nearest(ALBUMS, [KEY, SUNDAYS]) == SUNDAYS
    assert fact_meaning.nearest("add a task to clean the gutters", [KEY, SUNDAYS]) is None


def test_fact_and_background_vectors_are_kept_between_asks(fake: FakeEmbedder) -> None:
    fact_meaning.nearest(ALBUMS, [SUNDAYS])
    first = fake.calls, len(fake.sent)
    fact_meaning.nearest("add a task to organize the album shelf", [SUNDAYS])
    assert fake.calls == first[0] + 1
    assert len(fake.sent) == first[1] + 1  # the request alone


def test_another_model_embeds_everything_again(fake: FakeEmbedder) -> None:
    fact_meaning.nearest(ALBUMS, [SUNDAYS])
    fake.model = "another-embed"
    sent = len(fake.sent)
    assert fact_meaning.nearest(ALBUMS, [SUNDAYS]) == SUNDAYS
    assert sum(SUNDAYS in text for text in fake.sent[sent:]) == 1  # once, for this model


def test_a_fact_the_request_already_says_isnt_chosen(fake: FakeEmbedder) -> None:
    assert fact_meaning.nearest(f"add a task: {SUNDAYS.lower()}", [SUNDAYS]) is None


@pytest.mark.parametrize("why", ["switch", "embeddings", "incognito", "down", "empty"])
def test_no_meaning_is_words_alone(
    fake: FakeEmbedder, monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    saved = [SUNDAYS]
    if why == "switch":
        monkeypatch.setattr(get_settings(), "conductor_facts_by_meaning", False)
    elif why == "embeddings":
        monkeypatch.setattr(get_settings(), "conductor_embeddings", False)
    elif why == "down":
        fake.down = "refused"
    elif why == "empty":
        saved = []
    if why == "incognito":
        with incognito.turn(True):
            assert fact_meaning.nearest(ALBUMS, saved) is None
    else:
        assert fact_meaning.nearest(ALBUMS, saved) is None
    if why != "down":
        assert not fake.sent  # nothing left the process


# --- the interpreter ---------------------------------------------------------------------


def _run(fleet: Fleet, *items: dict[str, Any], meaning: Any = fact_meaning.nearest) -> Any:  # noqa: F811
    provider = ScriptedProvider([plan_turn(steps(*items)), text_turn("Done.")])
    conductor = Interpreter(provider, fleet, narrate=True, glance=FakeGlances())
    with use_delegation_context(DelegationContext("m", InMemoryThreadStore(), 3)):
        return conductor.run(ALBUMS, saved_facts=lambda: [KEY, SUNDAYS], fact_meaning=meaning)


def test_a_write_carries_the_fact_it_means(
    fake: FakeEmbedder,
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    result = _run(fleet, ask("tasks", ALBUMS))
    assert result.tool_calls[0].arguments == {"message": ALBUMS, facts.ARGUMENT: [SUNDAYS]}
    assert world.sent[-1] == ("tasks", facts.attach(ALBUMS, [SUNDAYS]))


def test_a_read_never_takes_a_fact_by_meaning(
    fake: FakeEmbedder,
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    read = "which albums still need organizing?"
    result = _run(fleet, ask("tasks", read))
    assert facts.ARGUMENT not in result.tool_calls[0].arguments
    assert not fake.sent


def test_words_that_match_never_ask_meaning(
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    asked: list[str] = []

    def meaning(request: str, saved: Any) -> str | None:
        asked.append(request)
        return KEY

    result = _run(fleet, ask("tasks", "add a task to sort music on Sunday"), meaning=meaning)
    assert result.tool_calls[0].arguments[facts.ARGUMENT] == [SUNDAYS]
    assert asked == []


def test_meaning_that_fails_never_costs_the_ask(
    fleet: Fleet,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    def broken(request: str, saved: Any) -> str | None:
        raise RuntimeError("boom")

    result = _run(fleet, ask("tasks", ALBUMS), meaning=broken)
    assert result.tool_calls[0].arguments == {"message": ALBUMS}
    assert result.tool_calls[0].error is None


def test_the_route_sends_the_fact_a_later_write_means(
    fake: FakeEmbedder,
    talk: Talk,  # noqa: F811
    world: World,  # noqa: F811
) -> None:
    remember = f"remember {SUNDAYS}"
    talk.say(remember, plan_turn(steps(ask("memory", remember))), text_turn("Got it."))
    talk.new()
    reply, _ = talk.say(ALBUMS, plan_turn(steps(ask("tasks", ALBUMS))), text_turn("Added it."))
    assert reply["tool_calls"][0]["arguments"] == {"message": ALBUMS, facts.ARGUMENT: [SUNDAYS]}
