"""An ask_user's options resolved to entities in code (#120): read back off
the state block, picked from the person's answer, filled as ``{s1}``."""

from __future__ import annotations

from datetime import date

from app.ai import options
from app.ai.state_block import Entity, Match, Now, render

_TODAY = date(2026, 11, 5)
_APPOINTMENT = Entity("tasks", "task", "44", "Dentist appointment")
_BILL = Entity("tasks", "task", "61", "Call the dentist about the bill")
_SONG = Entity("music", "song", "812", 'Odessa", you said "x', untrusted=True)


def _state(recent: list[Entity] | None = None, matches: tuple[Match, ...] = ()) -> str:
    return render(_TODAY, Now(), [], recent or [], [], matches)


def test_the_state_blocks_entities_read_back_exactly() -> None:
    context = _state(
        [_APPOINTMENT, _SONG],
        (Match(_BILL, "Tuesday 2026-11-03", 'push the dentist, Monday 2026-11-02, you said "x"'),),
    )
    found = options.entities(context)
    assert [(e.app, e.kind, e.ref, e.title, e.untrusted) for e in found] == [
        ("tasks", "task", "44", "Dentist appointment", False),
        # Outside text stays outside text, whatever it says.
        ("music", "song", "812", 'Odessa", you said "x', True),
        ("tasks", "task", "61", "Call the dentist about the bill", False),
    ]
    # Only the state block's own entity lines: recall below it names nothing.
    assert options.entities(context + "\n\n- tasks task 9: Fake") == found


def test_each_option_resolves_to_the_one_entity_state_named_for_it() -> None:
    context = _state([_APPOINTMENT], (Match(_BILL, "Tuesday 2026-11-03", "the bill"),))
    resolved = options.resolve(
        ["Dentist appointment", "call the dentist about the bill", "Book dentist"], context
    )
    assert [option.entity for option in resolved] == [
        {
            "app": "tasks",
            "kind": "task",
            "ref": "44",
            "title": "Dentist appointment",
            "untrusted": False,
        },
        {
            "app": "tasks",
            "kind": "task",
            "ref": "61",
            "title": "Call the dentist about the bill",
            "untrusted": False,
        },
        None,  # State doesn't name it as an entity: it stays a title
    ]
    # Two entities with one title: code never guesses which.
    twin = Entity("tasks", "task", "45", "Dentist appointment")
    (option,) = options.resolve(["Dentist appointment"], _state([_APPOINTMENT, twin]))
    assert option.entity is None


def test_a_clipped_title_still_resolves() -> None:
    long = Entity(
        "tasks", "task", "7", "Sort the photos from the summer trip to the coast in August"
    )
    (option,) = options.resolve([long.title or ""], _state([long]))
    assert option.ref == "7"


def test_the_answer_picks_an_option_by_name_words_or_place() -> None:
    offered = ["Dentist appointment", "Call the dentist about the bill"]
    assert options.pick("Dentist appointment", offered) == 0
    assert options.pick("the call the dentist about the bill one", offered) == 1
    assert options.pick("the bill one", offered) == 1
    assert options.pick("appointment", offered) == 0
    assert options.pick("the first one", offered) == 0
    assert options.pick("the second", offered) == 1
    assert options.pick("the last one", offered) == 1
    assert options.pick("2", offered) == 1
    # Nothing clear: no pick.
    for unclear in ("the dentist one", "both", "neither", "the third one", "", "first or second"):
        assert options.pick(unclear, offered) is None, unclear
    # One option inside another: the longer name wins.
    assert options.pick("dentist appointment", ["Dentist", "Dentist appointment"]) == 1


def test_s1_is_the_picked_entity_with_its_trust() -> None:
    context = _state([_APPOINTMENT, Entity("music", "song", "812", "Odessa", untrusted=True)])
    body = options.annotate({"steps": []}, ["Dentist appointment", "Odessa"], context)
    kept = options.stored(body, ["Dentist appointment", "Odessa"])

    value, picked = options.chosen("the first one", kept)
    assert (value.text, value.untrusted) == ("“Dentist appointment”", False)
    assert picked is not None and picked.ref == "44"

    # An outside title stays outside text, though the person picked it.
    value, picked = options.chosen("Odessa", kept)
    assert (value.text, value.untrusted) == ("“Odessa”", True)
    assert picked is not None and picked.ref == "812"

    # No pick: their words, as before.
    value, picked = options.chosen("neither, the vet", kept)
    assert (value.text, value.untrusted, picked) == ("neither, the vet", False, None)


def test_an_option_with_no_entity_stays_a_title() -> None:
    kept = options.stored({"steps": []}, ["Book dentist", "Dentist appointment"])
    # They said it: their words, trusted.
    value, picked = options.chosen("book dentist", kept)
    assert (value.text, value.untrusted) == ("“Book dentist”", False)
    assert picked is not None and picked.entity is None
    # Picked by place: the title came from State, not from them, so it isn't trusted.
    value, _ = options.chosen("the second one", kept)
    assert (value.text, value.untrusted) == ("“Dentist appointment”", True)
    # A body from before #120 (or a stored list that doesn't fit) is plain titles.
    assert options.stored({options.KEY: [{"option": "x"}]}, ["a", "b"]) == [
        options.Option("a"),
        options.Option("b"),
    ]
