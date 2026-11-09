"""Memory over the ledger (#80): the words it matches on, and what recall found.

Recall answers "what did we decide about the garage shelves?" and "that
Caribou song" from the ledger alone (:mod:`app.services.memory` reads it):

- **The words.** :func:`terms` turns text into the words memory matches on:
  lower case, a light plural stem, and none of the words that say nothing
  about *which* thing ("that", "decide", "last week"). The same function
  indexes a turn and reads a query, so they always agree.
- **What it looks at.** The person's own words on each turn and the titles the
  apps' receipts touched. Never conductor's replies (a false reply would be
  remembered as a fact, #64) and never a model's plan text: no model call
  writes memory.
- **What it found.** A :class:`Recalled`: the matching turns, each cited by
  its day and conversation, with the entities they touched. The entity that
  best answers the query, when one clearly does, is what ``{sN}`` passes on;
  an outside title stays outside text, under :mod:`app.ai.dataflow`'s rules.

Pure: no session, no I/O. The record a recall step files is built here
(:func:`record`), so the ledger, the narrator and the live channel read the
same entries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app import clock

# A recall step's tool name on the record: conductor's own read of its ledger.
RECALL_TOOL = "recall"
# How many turns one recall cites.
MAX_HITS = 4
# How much of a cited turn's words the record quotes.
FACT_CHARS = 200

_WORD = re.compile(r"[a-z0-9][a-z0-9'’]+")

# Words that say nothing about which thing: a match on them is noise. Recall's
# list (:mod:`app.ai.recall`) plus the words people use to point at the past.
_STOP = frozenset(
    """the and that this those these with from into onto for are was were you your
    yours mine our ours its it's what which who whom when where how why can could would
    should will just put get got grab add make move set give take let all any some one
    thing things stuff please thanks again last next earlier then them they their there
    here have has had did does done not now new old about over under also like want need
    folder list show tell find
    decide decided decision remember remembered recall said say says talk talked talking
    discuss discussed week weeks yesterday today tomorrow day days ago time before
    previous other same back anything something everything conversation chat mention
    mentioned monday tuesday wednesday thursday friday saturday sunday morning evening
    night afternoon month year let's lets okay yes yeah sure""".split()
)


def stem(word: str) -> str:
    """A light plural stem, the same for an index and a query."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def terms(text: str) -> set[str]:
    """The words of ``text`` memory matches on."""
    words = {w.strip("'’") for w in _WORD.findall(text.lower())}
    words = {w.removesuffix("'s").removesuffix("’s") for w in words}
    return {stem(w) for w in words if len(w) >= 3 and w not in _STOP}


def merge(counts: Mapping[str, int] | None, words: Iterable[str], *, cap: int) -> dict[str, int]:
    """``counts`` plus one for each of ``words``, keeping the ``cap`` most
    frequent (ties by word, so the result is the same however it was built)."""
    merged = dict(counts or {})
    for word in words:
        merged[word] = merged.get(word, 0) + 1
    if len(merged) <= cap:
        return dict(sorted(merged.items()))
    kept = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:cap]
    return dict(sorted(kept))


# --- what recall found ------------------------------------------------------------------


@dataclass(frozen=True)
class Found:
    """An entity a cited turn touched, as the ledger has it (title as of then)."""

    app: str
    kind: str
    ref: str
    title: str | None = None
    untrusted: bool = False


@dataclass(frozen=True)
class Hit:
    """One turn recall cites: who said what, when, where, and what it touched."""

    turn_id: int
    conversation_id: int
    when: datetime
    words: str
    # The conversation's title (auto-derived from its first message).
    conversation: str | None = None
    # ``message`` (the person said it), ``note`` (they did it in the UI) or
    # ``job`` (a job reported back).
    kind: str = "message"
    found: tuple[Found, ...] = ()


@dataclass(frozen=True)
class Recalled:
    """What one recall step found: cited turns, best first, and the one entity
    that clearly answers the query (``None`` when none does, or several tie)."""

    query: str
    hits: tuple[Hit, ...] = ()
    best: Found | None = None


def day(when: datetime | date) -> str:
    """``Tuesday 2026-11-03``: a timestamp's day is the household's (#130)."""
    on = clock.local_day(when) if isinstance(when, datetime) else when
    return f"{on:%A} {on.isoformat()}"


def cues(when: datetime | date, today: date) -> list[str]:
    """Ways a reply can say when: the weekday, the date, and a relative word."""
    on = clock.local_day(when) if isinstance(when, datetime) else when
    found = [f"{on:%A}", on.isoformat(), f"{on:%B} {on.day}"]
    ago = (today - on).days
    if ago == 0:
        found.insert(0, "today")
        found.append("earlier")
    elif ago == 1:
        found.insert(0, "yesterday")
    elif 0 < ago < 14:
        found.append(f"{ago} days ago")
        found.append("last week" if ago >= 7 else "this week")
    return found


def record(recalled: Recalled, today: date) -> tuple[str, list[dict[str, Any]]]:
    """The recall step's record: a result line for the transcript, and one
    ``looked_up`` entry per cited turn (``source`` conductor) with the words
    quoted, where and when they were said, and what that turn touched."""
    entries: list[dict[str, Any]] = []
    for hit in recalled.hits:
        touched = []
        for found in hit.found:
            ref: dict[str, Any] = {"app": found.app, "kind": found.kind, "id": found.ref}
            if found.title:
                ref["title"] = found.title
            if found.untrusted:
                ref["untrusted"] = True
            if recalled.best is not None and found == recalled.best:
                ref["best"] = True
            touched.append(ref)
        entries.append(
            {
                "tool": RECALL_TOOL,
                "outcome": "looked_up",
                "source": "conductor",
                "fact": _clip(hit.words, FACT_CHARS),
                "from": _FROM.get(hit.kind, "your words"),
                "turn": hit.turn_id,
                "conversation": hit.conversation_id,
                "conversation_title": hit.conversation,
                "when": hit.when.isoformat(),
                "day": day(hit.when),
                "cues": cues(hit.when, today),
                "touched": touched,
            }
        )
    if not entries:
        return "Nothing on record matches.", entries
    lines = [
        f'{entry["day"]}, {entry["from"]}: "{entry["fact"]}"'
        + (f" [touched: {_names(entry['touched'])}]" if entry["touched"] else "")
        for entry in entries
    ]
    return "\n".join(lines), entries


_FROM = {"message": "your words", "note": "what you did in the app", "job": "a job's report"}


def is_recall(tool: str) -> bool:
    return tool == RECALL_TOOL


def best_ref(app_record: Sequence[Mapping[str, Any]] | None) -> Mapping[str, Any] | None:
    """The entity a recall resolved its query to, from its record."""
    for entry in app_record or []:
        for ref in entry.get("touched") or []:
            if ref.get("best"):
                return dict(ref)
    return None


def _names(refs: Sequence[Mapping[str, Any]]) -> str:
    names = []
    for ref in refs[:5]:
        title = ref.get("title")
        if not title:
            names.append(f"{ref.get('app')} {ref.get('kind')} {ref.get('id')}")
        elif ref.get("untrusted"):
            names.append(f'{ref.get("kind")} "{_clip(str(title), 60)}" (outside text)')
        else:
            names.append(f"{ref.get('kind')} {_clip(str(title), 60)}")
    return ", ".join(names)


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


# --- resolving a reference ---------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """An entity scored against the words of a query."""

    found: Found
    score: int
    seen: datetime


def score(query: set[str], entity_terms: Mapping[str, int] | None, kind: str) -> int:
    """How many of the query's words this entity was seen with. A match on
    the kind word alone ("that song") names no particular one: zero."""
    shared = query & set(entity_terms or {})
    if not shared - {stem(kind.lower())}:
        return 0
    return len(shared)


def best(candidates: Sequence[Candidate]) -> Found | None:
    """The one entity that clearly answers: a strictly higher score than any
    other titled one. A tie names nothing: code never guesses which."""
    titled = sorted(
        (c for c in candidates if c.found.title and not c.found.kind.endswith("job")),
        key=lambda c: -c.score,
    )
    if not titled or titled[0].score <= 0:
        return None
    if len(titled) > 1 and titled[1].score == titled[0].score:
        return None
    return titled[0].found
