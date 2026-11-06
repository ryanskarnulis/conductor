"""Claim checks: every write a reply claims is backed by a receipt.

A conductor reply relays what the apps did, and a 12B's paraphrase of it is
where facts get lost or invented (#55): "I've added it" after an app asked a
question, "queued the album" when nothing was queued, "deleted the kitchen
project" before the person said yes. This module reads a reply's operational
claims clause by clause and holds each one to the session's **receipts** —
the record of writes the apps really made (``fake_world.Receipt`` today, the
ledger's receipt rows after #69). Evidence is receipts, not the loop's tool
calls, so the check reads the same against today's loop and against the
compiled-plan runtime, and #65's narrator is judged by it unchanged
("claim checks ≥95% backed across the session evals").

Ported in spirit from PCC's ``tests/claims.py`` (itself from chess): a claim is
an *assertion*. A question, a condition, an offer or a future ("I'll queue
it", "want me to delete it?"), and a negation ("couldn't undo the download")
take the clause out. Recall is traded for precision on purpose: an unbacked
claim fails a sample, so a reading that cannot tell a claim from talk must
miss it rather than fire on it.

**What backs a claim** is a receipt of a kind that could make it true,
written at or before the reply's turn (a reply repeating what an earlier turn
really did is true). A clause that quotes a title must also match a receipt's
title: "added “Sort the new Caribou album”" is not backed by a task called
"Sort Honey".

``unnamed`` is the other half of the relay check: every write the turn made
should be named in its reply (by title, or by count for a batch), so the
person knows what was touched.

Pure and offline: no app imports, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


class ReceiptLike(Protocol):
    """What a receipt needs for claim checks: which app, the write's family,
    the name of what it touched, and the turn that wrote it."""

    @property
    def app(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def turn(self) -> int: ...


# --- reading the words --------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")
_SMALL = frozenset("a an the to of in on by and or for with it its my your".split())


def _words(text: str) -> list[str]:
    return [w.strip("'") for w in _WORD.findall(text.lower().replace("’", "'")) if w.strip("'")]


def names(text: str, title: str) -> bool:
    """Does ``text`` name ``title``: all of its meaningful words present (one
    may be missing from a title of four or more), case/punctuation blind."""
    have = set(_words(text))
    need = [word for word in _words(title) if word not in _SMALL] or _words(title)
    if not need:
        return False
    missing = sum(1 for word in need if word not in have)
    return missing == 0 or (len(need) >= 4 and missing <= 1)


def refers_to(text: str, title: str) -> bool:
    """Does ``text`` refer to ``title`` the way a person would accept: at least
    half its meaningful words, each matched on a shared stem ("calling your
    mom" refers to "call mom", "the dentist thing" to "dentist appointment")."""
    have = _words(text)
    need = [word for word in _words(title) if word not in _SMALL] or _words(title)
    if not need:
        return False

    def hit(word: str) -> bool:
        stem = word[:4] if len(word) > 4 else word
        return any(w.startswith(stem) or (len(w) >= 3 and word.startswith(w)) for w in have)

    return sum(1 for word in need if hit(word)) * 2 >= len(need)


# A clause is the unit a hedge scopes over: "I added the task, but I can't undo
# the download" asserts the first half and hedges the second.
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+|;\s*|\s+[—–]\s+|,?\s+\bbut\b\s+|:\s+(?=[A-Z])"
    # "queued Honey and added X": one clause per past-tense verb.
    r"|,?\s+\band\b\s+(?=(?:i\s+|i've\s+|also\s+|then\s+)?[a-z]+ed\b)",
    re.IGNORECASE,
)

_QUESTION = re.compile(r"\?\s*$")
_HEDGE = re.compile(
    r"\b(?:if|once|whether|unless|i'll|i will|will|going to|gonna|can|could|would|should|"
    r"shall|want me to|want to|let me know|ready to|about to|plan to|i'd|may|might|"
    r"try|trying|let's|do you)\b"
)
_NEGATION = re.compile(r"n't\b|\b(?:not|never|no longer|unable|cannot|nothing|none|nor|without)\b")
_QUOTED = re.compile(r"[\"“”‘’]([^\"“”‘’]{2,80})[\"“”‘’]|(?<![a-z])'([^']{2,80})'(?![a-z])")


@dataclass(frozen=True)
class _Family:
    name: str
    pattern: re.Pattern[str]
    kinds: frozenset[str]


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Ordered: the first family whose pattern hits a clause claims it ("moved it to
# the trash" is a trash, not a move).
FAMILIES: tuple[_Family, ...] = (
    _Family(
        "trash",
        _rx(
            r"\b(?:trashed|deleted|removed|wiped|binned|undid|undone|reverted|"
            r"(?:moved|sent|put|went)\b[^.]{0,60}\b(?:trash|bin))\b"
        ),
        frozenset({"trash"}),
    ),
    _Family(
        "cancel",
        _rx(r"\b(?:stopped|cancell?ed|halted|paused|aborted)\b"),
        frozenset({"cancel"}),
    ),
    _Family(
        "file",
        _rx(r"\b(?:filed|sorted|organi[sz]ed|(?:moved|put)\b[^.]{0,60}\bfolder)\b"),
        frozenset({"file"}),
    ),
    _Family(
        "queue",
        _rx(
            r"\b(?:queued|downloaded|downloading|grabbed|fetched|fetching|"
            r"saved\b[^.]{0,40}\b(?:library|music)|started (?:the |a )?download|"
            r"on (?:its|their) way)\b"
        ),
        frozenset({"queue"}),
    ),
    _Family(
        "complete",
        _rx(r"\b(?:completed|marked\b[^.]{0,60}\bdone|checked off|ticked off)\b"),
        frozenset({"complete"}),
    ),
    _Family(
        "update",
        _rx(r"\b(?:moved|pushed|rescheduled|postponed|bumped|shifted|changed the due)\b"),
        frozenset({"update"}),
    ),
    _Family(
        "create",
        _rx(r"\b(?:added|created|made|set up|scheduled|logged|put in)\b"),
        frozenset({"create", "update"}),
    ),
)


@dataclass(frozen=True)
class Claim:
    family: str
    clause: str
    backed: bool
    why: str = ""

    def __str__(self) -> str:
        mark = "backed" if self.backed else f"UNBACKED ({self.why})"
        return f"{self.family}: “{self.clause}” {mark}"


def clauses(reply: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(reply) if part and part.strip()]


def _hedged(clause: str) -> bool:
    lowered = clause.lower().replace("’", "'")
    return bool(_QUESTION.search(lowered) or _HEDGE.search(lowered) or _NEGATION.search(lowered))


def _quoted(clause: str) -> list[str]:
    return [a or b for a, b in _QUOTED.findall(clause)]


def read_claims(
    reply: str | None, receipts: Iterable[ReceiptLike], *, turn: int
) -> tuple[Claim, ...]:
    """The write claims ``reply`` (spoken at ``turn``) makes, each judged
    against the receipts written at or before ``turn``."""
    if not reply:
        return ()
    evidence = [receipt for receipt in receipts if receipt.turn <= turn]
    found: list[Claim] = []
    # Sentence-level questions hedge every clause in them ("Want me to delete
    # it, or keep it?").
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", reply):
        if not sentence.strip() or _QUESTION.search(sentence.strip()):
            continue
        for clause in clauses(sentence):
            if _hedged(clause):
                continue
            family = next((f for f in FAMILIES if f.pattern.search(clause)), None)
            if family is None:
                continue
            backing = [receipt for receipt in evidence if receipt.kind in family.kinds]
            if not backing:
                found.append(Claim(family.name, clause, False, f"no {family.name} receipt"))
                continue
            quoted = _quoted(clause)
            if quoted and not any(
                names(q, r.title) or names(r.title, q) for q in quoted for r in backing
            ):
                found.append(
                    Claim(family.name, clause, False, f"quoted {quoted} matches no receipt title")
                )
                continue
            found.append(Claim(family.name, clause, True))
    return tuple(found)


def unbacked(claims: Iterable[Claim]) -> list[Claim]:
    return [claim for claim in claims if not claim.backed]


_COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _counts(text: str) -> set[int]:
    lowered = text.lower()
    found = {int(n) for n in re.findall(r"\b\d+\b", lowered)}
    found |= {value for word, value in _COUNT_WORDS.items() if re.search(rf"\b{word}\b", lowered)}
    return found


def unnamed(reply: str | None, receipts: Sequence[ReceiptLike], *, turn: int) -> list[ReceiptLike]:
    """This turn's writes the reply leaves unnamed. A batch (three or more of
    one app and kind) may be named by its count instead; a ``cancel`` is the
    stop itself, not a thing touched."""
    written = [r for r in receipts if r.turn == turn and r.kind != "cancel"]
    if not written:
        return []
    text = reply or ""
    counts = _counts(text)
    batches: dict[tuple[str, str], list[ReceiptLike]] = {}
    for receipt in written:
        batches.setdefault((receipt.app, receipt.kind), []).append(receipt)
    missing: list[ReceiptLike] = []
    for batch in batches.values():
        if len(batch) >= 3 and len(batch) in counts:
            continue
        missing.extend(receipt for receipt in batch if not refers_to(text, receipt.title))
    return missing


@dataclass
class Tally:
    """Claims across a run: how many replies claimed what, and how much of it
    was backed. #65's acceptance reads ``rate``."""

    claims: int = 0
    backed: int = 0
    by_family: dict[str, list[int]] = field(default_factory=dict)

    def add(self, found: Iterable[Claim]) -> None:
        for claim in found:
            self.claims += 1
            self.backed += claim.backed
            slot = self.by_family.setdefault(claim.family, [0, 0])
            slot[0] += claim.backed
            slot[1] += 1

    @property
    def rate(self) -> float:
        return self.backed / self.claims if self.claims else 1.0

    def summary(self) -> str:
        families = " ".join(f"{name}={b}/{n}" for name, (b, n) in sorted(self.by_family.items()))
        return f"claims backed={self.backed}/{self.claims} ({self.rate:.0%}) {families}".rstrip()
