"""The claim reader's labelled corpus: what counts as a claim, and what backs it.

Precision over recall (``receipt_claims.py``): every "hedged" row is a sentence
a reading must *not* fire on, because firing there would fail a sample for
talk. CI, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.receipt_claims import Tally, names, read_claims, unbacked, unnamed


@dataclass(frozen=True)
class R:
    app: str
    kind: str
    title: str
    turn: int = 1


TASK = R("tasks", "create", "water the plants")
ALBUM = R("music", "queue", "Honey")
TRASH = R("tasks", "trash", "Kitchen")
FILED = R("music", "file", "Odessa")
STOP = R("music", "cancel", "sort")
MOVED = R("tasks", "update", "dentist appointment")


# (reply, receipts, expected unbacked families)
CORPUS: list[tuple[str, list[R], list[str]]] = [
    # backed
    ("Added “water the plants” for tomorrow.", [TASK], []),
    # A title holding straight quotes, curly-quoted whole (#196's fleet smoke).
    (
        'For the record: queued “Honey”; create task “Sort the album downloaded in "Honey"”.',
        [ALBUM, R("tasks", "create", 'Sort the album downloaded in "Honey"')],
        [],
    ),
    ("I've queued Honey — it's on its way.", [ALBUM], []),
    ("Queued Honey and added “water the plants” for Saturday.", [ALBUM, TASK], []),
    ("The Kitchen project went to the trash.", [TRASH], []),
    ("Filed Odessa under house.", [FILED], []),
    ("Stopped after filing Odessa.", [STOP, FILED], []),
    ("Moved your dentist appointment to Friday.", [MOVED], []),
    # unbacked
    ("Added “water the plants” for tomorrow.", [], ["create"]),
    ("Done — I deleted the kitchen project.", [], ["trash"]),
    ("Queued Honey and added “sort the new album”.", [ALBUM, TASK], ["create"]),
    ("I've stopped the sort.", [FILED], ["cancel"]),
    ("Downloaded the album for you.", [TASK], ["queue"]),
    # hedged: not claims
    ("Want me to delete the kitchen project?", [], []),
    ("Should I add “water the plants” for tomorrow?", [], []),
    ("I'll queue it once you confirm.", [], []),
    ("I can't undo the download, though.", [], []),
    ("If you say yes, I'll wipe it.", [], []),
    ("Nothing was deleted.", [], []),
    ("I couldn't find that task, so nothing moved.", [], []),
    ("That would trash the Kitchen project and its tasks. Go ahead?", [], []),
    # not operational at all
    ("Two things today: groceries and call mom.", [], []),
    # a decision someone made is no write (#188: T1, S5)
    ("Yesterday's decision made it the steel rack.", [], []),
    ("The decision you made on Tuesday was the steel rack.", [], []),
    ("Decisions made about the garage: the steel rack.", [], []),
    ("A decision was made to go with the steel rack.", [], []),
    ("You made the decision to buy the steel rack.", [], []),
    ("Made a task to buy the steel rack.", [], ["create"]),
    ("About 7 minutes for a jammy egg.", [], []),
]


@pytest.mark.parametrize(("reply", "receipts", "expected"), CORPUS)
def test_corpus(reply: str, receipts: list[R], expected: list[str]) -> None:
    claims = read_claims(reply, receipts, turn=1)
    assert sorted(c.family for c in unbacked(claims)) == sorted(expected), claims


def test_a_later_turns_receipt_does_not_back_an_earlier_claim() -> None:
    later = R("tasks", "trash", "Kitchen", turn=2)
    assert unbacked(read_claims("Wiped the Kitchen project.", [later], turn=1))
    assert not unbacked(read_claims("Wiped the Kitchen project.", [later], turn=2))


def test_an_earlier_turns_write_backs_a_recap() -> None:
    assert not unbacked(read_claims("I added “water the plants”.", [TASK], turn=2))


def test_unnamed_reads_titles_and_batch_counts() -> None:
    assert unnamed("Added water the plants.", [TASK], turn=1) == []
    assert unnamed("Done, it's on your list.", [TASK], turn=1) == [TASK]
    batch = [R("music", "file", title) for title in ("Glue", "Teardrop", "Rhubarb")]
    assert unnamed("Sorted 3 songs.", batch, turn=1) == []
    assert unnamed("Sorted some songs.", batch, turn=1) == batch
    assert unnamed("Stopped.", [STOP], turn=1) == []


def test_names_is_blind_to_case_and_punctuation() -> None:
    assert names("Added 'Water the Plants!'", "water the plants")
    assert not names("Added the plants", "water the plants")


def test_tally() -> None:
    tally = Tally()
    tally.add(read_claims("Added “water the plants”. Deleted the kitchen.", [TASK], turn=1))
    assert (tally.backed, tally.claims) == (1, 2)
    assert "create=1/1" in tally.summary() and "trash=0/1" in tally.summary()
