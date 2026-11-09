"""Memory you can see (#81): the built-in ``memory`` app's words and receipts.

"Remember I sort music on Sundays", "what do you know about me?" and "forget
that" compile as asks to ``memory``, an app conductor runs itself: code, no
agent, no model, no delegate budget. It is one more value in the plan
schema's ``ask`` enum and one line in the planner prompt's app list
(:data:`PROMPT_LINE`), exactly as adding an app would be; the step kinds and
the schema's shape don't change.

What code does with a request (:func:`parse`):

- **remember**: the fact is kept *as the person said it*. Code takes it from
  their own words (:func:`own_words`): the planner's request is only a
  pointer, and a fact that isn't in what they said isn't saved. No model
  writes memory.
- **forget**: the target is matched against the saved facts by their words;
  "forget that" is the newest fact saved in this conversation. One clear
  match is forgotten; several, or none, is said, never guessed.
- anything else is a question about the list: the facts that share words
  with it, else all of them, newest first.

Saved facts also go with another app's ask (#134, :func:`related`,
:func:`attach`): when the request clearly shares terms with a fact ("add a
task for my music sorting day" and "I sort music on Sundays"), code sends the
fact along after the request, quoted and marked as context, for the app's
agent to read. The person's words are never rewritten; the app does the
interpreting. The step's arguments keep which facts went (:data:`ARGUMENT`),
for the ledger and the live channel; the record lines models read leave them
out (:func:`app.ai.run_record.shown`), so a forgotten fact never comes back.

The receipts are ordinary v1.1 entries (``done``/``looked_up``/``refused``)
touching ``fact`` refs titled with the person's words, so the narrator, the
ledger and the live channel read them like any app's. A fact has no undo
handle: "forget that" is how it goes. Pure: no session, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from app.ai.memory import day, stem, terms
from app.ai.run_record import ATTACHED_FACTS

APP = "memory"
# The step's tool on the record: an ask, filed like any app's.
TOOL = f"ask_{APP}"
REMEMBER = "remember"
FORGET = "forget"
LIST = "list_facts"
KIND = "fact"
# The longest fact kept; longer is refused, never cut.
MAX_FACT_CHARS = 300
# Facts one answer lists by name; the rest are counted.
MAX_LISTED = 8

# The planner prompt's line for memory, after the fleet's apps.
PROMPT_LINE = (
    f"- {APP} (ask): What the person asked you to remember about them, kept in their own "
    "words; they can see and change the list.\n"
    '  Asks that go here: "remember I sort music on Sundays"; "what do you know about '
    'me?"; "forget that"; "forget that I sort music on Sundays"'
)

NO_UNDO_REMEMBER = "say “forget that” to take it back"
NO_UNDO_FORGET = "a forgotten fact is gone; say “remember …” to save it again"

_LEAD = (
    r"^\s*(?:(?:hey|ok|okay|please|so|and|also|oh|conductor)[\s,]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?(?:please\s+)?"
)
_NOT_FORGET = re.compile(r"\b(?:don'?t|don’t|do not|never)\s+forget\b", re.IGNORECASE)
_REMEMBER = re.compile(
    _LEAD + r"(?:remember|don'?t forget|don’t forget|do not forget|never forget|keep in mind|"
    r"make a note|note|save)\b(?:\s+(?:that|this|for me))?[\s,:;-]*(?P<fact>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING = re.compile(
    r"^(?P<fact>.+?)[\s,;:.-]+(?:please\s+)?(?:remember|note|keep in mind)\s+(?:that|this|it)"
    r"\W*$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET = re.compile(
    _LEAD + r"(?:forget|delete|remove|erase|unremember)\b(?:\s+(?:about|that|this|it))?"
    r"[\s,:;-]*(?P<target>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_EVERYTHING = re.compile(
    r"^(?:everything|all|all of it|it all|all of them|everything you know)\b", re.IGNORECASE
)
_TAIL = re.compile(r"[\s,;]*(?:please|thanks|thank you)?[\s.!]*$", re.IGNORECASE)
_TOKEN = re.compile(r"[\w'’]+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'’]+")
# Words that say nothing about which fact. Weekdays stay: "Sundays" names one.
_STOP = frozenset(
    """the and that this those these with from into for are was were you your yours
    what which who when where how why can could would should will just please thanks
    about know knew tell told have has had did does don't dont not its it's thing things
    something anything everything fact facts saved remember remembered forget memory me
    mine my our all any some one also like want""".split()
)


@dataclass(frozen=True)
class Intent:
    """What a request to memory asks: ``text`` is the fact to remember, the
    target to forget ("" for "that"), or the question."""

    kind: Literal["remember", "forget", "list"]
    text: str


def parse(request: str) -> Intent:
    """The request's intent, by its leading words. A question that mentions
    remembering ("do you remember …?") is a question about the list."""
    text = " ".join(request.split())
    if not _NOT_FORGET.search(text):
        forget = _FORGET.match(text)
        if forget is not None:
            return Intent("forget", _clean(forget["target"]))
    for pattern in (_REMEMBER, _TRAILING):
        found = pattern.match(text)
        if found is not None and not found["fact"].rstrip().endswith("?"):
            return Intent("remember", _clean(found["fact"]))
    return Intent("list", text)


def _clean(text: str) -> str:
    text = _TAIL.sub("", text.strip())
    return text.strip().strip("\"“”'").strip()


def own_words(request: str, words: str) -> str | None:
    """The fact to keep, exactly as the person said it: the planner's version
    if it appears word for word in what they said, else what follows their
    own "remember". ``None`` when their words hold no fact."""
    candidates = []
    asked = parse(request)
    if asked.kind == "remember" and asked.text:
        candidates.append(asked.text)
    said = parse(words)
    if said.kind == "remember" and said.text:
        candidates.append(said.text)
    for candidate in candidates:
        found = literal(candidate, words)
        if found:
            return found
    return None


def literal(text: str, words: str) -> str | None:
    """``text`` as it appears in ``words`` (case, punctuation and spacing
    aside), cut from ``words`` so the person's own spelling is kept."""
    want = [_fold(token) for token in _TOKEN.findall(text)]
    if not want:
        return None
    found = list(_TOKEN.finditer(words))
    have = [_fold(match.group(0)) for match in found]
    for start in range(len(have) - len(want) + 1):
        if have[start : start + len(want)] == want:
            return words[found[start].start() : found[start + len(want) - 1].end()]
    return None


def _fold(token: str) -> str:
    return token.lower().replace("’", "'")


def words_of(text: str) -> set[str]:
    """The words a fact is matched on: lower case, a light plural stem."""
    found = {w.strip("'’") for w in _WORD.findall(text.lower())}
    found = {w.removesuffix("'s").removesuffix("’s") for w in found}
    return {stem(w) for w in found if len(w) >= 3 and w not in _STOP}


def same(a: str, b: str) -> bool:
    """Whether two facts say the same words."""
    return [_fold(t) for t in _TOKEN.findall(a)] == [_fold(t) for t in _TOKEN.findall(b)]


def everything(target: str) -> bool:
    return _EVERYTHING.match(target.strip()) is not None


# --- what the person saved, as code shows it --------------------------------------------


@dataclass(frozen=True)
class Saved:
    """One live fact, with where it came from in words."""

    id: int
    text: str
    saved_at: datetime
    # "you said it Tuesday 2026-11-03, in “Weekend plans”"
    source: str = ""
    # The turn it came from: receipts cite it, so forgetting can find them.
    turn: int | None = None


def source_line(
    when: datetime, *, kind: str = "message", thread: str | None = None, edited: bool = False
) -> str:
    """Where a fact came from, for the list and the receipts."""
    how = "you changed it in the memory list" if edited or kind == "note" else "you said it"
    line = f"{how} {day(when)}"
    return f"{line}, in “{thread}”" if thread else line


def ref(fact: Saved) -> dict[str, Any]:
    return {"app": APP, "kind": KIND, "id": str(fact.id), "title": fact.text}


def remembered(fact: Saved, *, already: bool) -> tuple[str, list[dict[str, Any]]]:
    if already:
        entry: dict[str, Any] = {
            "tool": REMEMBER,
            "outcome": "looked_up",
            "touched": [ref(fact)],
            "reason": "it was already in the memory list",
            "source_turn": fact.turn,
        }
        return f"Already remembered: “{fact.text}”.", [entry]
    entry = {
        "tool": REMEMBER,
        "outcome": "done",
        "touched": [ref(fact)],
        "undo": {"none": NO_UNDO_REMEMBER},
        "source_turn": fact.turn,
    }
    return f"Remembered “{fact.text}”. It's in the memory list.", [entry]


def forgot(fact: Saved) -> tuple[str, list[dict[str, Any]]]:
    entry: dict[str, Any] = {
        "tool": FORGET,
        "outcome": "done",
        "touched": [ref(fact)],
        "undo": {"none": NO_UNDO_FORGET},
    }
    return f"Forgot “{fact.text}”: gone from the memory list and from recall.", [entry]


def refused(tool: str, reason: str) -> tuple[str, list[dict[str, Any]]]:
    return f"Nothing changed: {reason}.", [{"tool": tool, "outcome": "refused", "reason": reason}]


def listed(shown: Iterable[Saved], *, total: int) -> tuple[str, list[dict[str, Any]]]:
    """A question about the list: one ``looked_up`` entry per fact shown, so
    the reply owes each by name, and where each came from in the result."""
    shown = list(shown)
    if not shown:
        return (
            "Nothing saved: the memory list is empty.",
            [{"tool": LIST, "outcome": "looked_up", "count": 0, "reason": "nothing saved yet"}],
        )
    entries = [
        {"tool": LIST, "outcome": "looked_up", "touched": [ref(fact)], "source_turn": fact.turn}
        for fact in shown
    ]
    lines = [
        f"“{fact.text}” ({fact.source})" if fact.source else f"“{fact.text}”" for fact in shown
    ]
    result = "Saved, in the person's words: " + "; ".join(lines) + "."
    if total > len(shown):
        result += f" And {total - len(shown)} more in the memory list."
    return result, entries


# --- saved facts in another app's ask (#134) -----------------------------------------------

# The ask's argument that names the facts sent with it.
ARGUMENT = ATTACHED_FACTS
# At most this many facts go with one request, each at most this long.
MAX_ATTACHED = 2
MAX_ATTACHED_CHARS = 200
# The terms a request must share with a fact for the fact to go with it.
MIN_SHARED = 2
# How the facts are marked in the message an app receives: context, not a request.
CONTEXT_LEAD = (
    "Context from conductor, not part of the request: what the person asked conductor to "
    "remember, in their own words:"
)
# Words that name an app's own things and say nothing about which fact.
_GENERIC = frozenset({"task", "todo", "reminder", "item", "note"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _root(word: str) -> str:
    """A light verb stem on top of :func:`app.ai.memory.stem`: "sorting" and
    "sorted" match "sort"."""
    for suffix in ("ing", "ed"):
        base = word.removesuffix(suffix)
        if base != word and len(base) >= 3:
            if len(base) > 3 and base[-1] == base[-2]:
                base = base[:-1]  # "running" -> "run"
            return base
    return word


def _match_terms(text: str) -> set[str]:
    return {_root(word) for word in terms(text)} - _GENERIC


def related(request: str, saved: Sequence[str]) -> list[str]:
    """The saved facts an app's request clearly points at, best first, at
    most :data:`MAX_ATTACHED`. Conservative: a fact needs two of its own terms
    in the request, so one shared word, a stop word or a weekday never pulls
    one in. A fact the request already says isn't sent again. ``saved`` is
    newest first; a tie keeps that order."""
    wanted = _match_terms(request)
    if len(wanted) < MIN_SHARED:
        return []
    scored: list[tuple[int, int, str]] = []
    for order, fact in enumerate(saved):
        shared = len(_match_terms(fact) & wanted)
        if shared < MIN_SHARED or literal(fact, request):
            continue
        scored.append((-shared, order, fact))
    return [fact for _, _, fact in sorted(scored)[:MAX_ATTACHED]]


def quotable(fact: str) -> str:
    """A fact as it is quoted to an app: one line, no double quotes, capped."""
    text = " ".join(_CONTROL.sub(" ", fact).replace('"', "'").split())
    if len(text) > MAX_ATTACHED_CHARS:
        text = text[: MAX_ATTACHED_CHARS - 1].rstrip() + "…"
    return text


def attach(message: str, attached: Sequence[str]) -> str:
    """The message an app receives: the request as the plan wrote it, then the
    facts on a line of their own, quoted and marked as context."""
    quoted = [f'"{text}"' for text in map(quotable, attached[:MAX_ATTACHED]) if text]
    if not quoted or not message.strip():
        return message
    return f"{message}\n\n({CONTEXT_LEAD} {'; '.join(quoted)})"
