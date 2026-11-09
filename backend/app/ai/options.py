"""An ``ask_user``'s options as entities, resolved in code (#120).

The planner offers options as titles (``"Dentist appointment"``), and the
validator only lets it offer what State names. Code then does the rest; the
plan schema stays as it is:

- **When the plan pauses**, each option is matched against the entities the
  state block listed for that turn (recent entities and the registry's
  matches, #80): the one entity whose title it is gives it an
  ``(app, kind, ref)`` and that title's trust. Two entities with that title,
  or none, leave it a plain title: code never guesses. The resolution rides
  the paused body (:data:`KEY`), beside the steps, so it outlives the pause.
- **When the person answers**, :func:`pick` finds the option they chose: one
  they named, the words only one option has ("the bill one"), or its place
  ("the first one"). ``{s1}`` becomes that option's entity title, quoted, with
  the entity's trust (an outside title stays outside text under
  :mod:`app.ai.dataflow`'s rules); an option with no entity stays its title,
  trusted only when the person said it themselves. An answer that picks no
  option goes in as they said it, as before.

Pure: no session, no I/O. The state block is code's own text, so reading it
back is exact: a title is one flattened line, and outside text always ends
``" (outside text)``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.ai import dataflow
from app.ai.memory import terms

# Where a paused body keeps its ask_user's resolved options.
KEY = "ask_user_options"

_OUTSIDE = " (outside text)"
_DAY = re.compile(
    r", (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d{4}-\d{2}-\d{2}"
    r"(?:, you said \".*)?$"
)
_MATCHES = "Matches for words"
_SECTIONS = ("Recent entities", _MATCHES)


@dataclass(frozen=True)
class Listed:
    """An entity line of the state block, read back."""

    app: str
    kind: str
    ref: str
    title: str
    untrusted: bool = False


@dataclass(frozen=True)
class Option:
    """One option as offered, and the entity it names, when exactly one does."""

    option: str
    app: str | None = None
    kind: str | None = None
    ref: str | None = None
    title: str | None = None
    untrusted: bool = False

    @property
    def entity(self) -> dict[str, Any] | None:
        if self.ref is None:
            return None
        return {
            "app": self.app,
            "kind": self.kind,
            "ref": self.ref,
            "title": self.title,
            "untrusted": self.untrusted,
        }


def entities(context: str) -> list[Listed]:
    """The entities the state block listed, titled ones only, in its order."""
    found: list[Listed] = []
    inside = False
    section: str | None = None
    for line in context.splitlines():
        if line.startswith("[State at the start"):
            inside = True
            continue
        if line.startswith("[End of state]"):
            break
        if not inside:
            continue
        if not line.startswith("- "):
            section = next((name for name in _SECTIONS if line.startswith(name)), None)
            continue
        if section is not None:
            entity = _entity(line[2:], match=section == _MATCHES)
            if entity is not None:
                found.append(entity)
    return found


def _entity(text: str, *, match: bool) -> Listed | None:
    head, sep, rest = text.partition(": ")
    parts = head.split(" ")
    if not sep or len(parts) != 3 or not rest:
        return None
    app, kind, ref = parts
    untrusted = _OUTSIDE in rest
    if rest.startswith('"') and untrusted:
        title = rest[1 : rest.find('"' + _OUTSIDE)]
    else:
        title = _DAY.sub("", rest) if match else rest
    title = title.strip()
    if not title:
        return None
    return Listed(app=app, kind=kind, ref=ref, title=title, untrusted=untrusted)


def resolve(options: Sequence[str], context: str) -> list[Option]:
    """Each option with the one entity in State whose title it is, if any."""
    listed = entities(context)
    resolved: list[Option] = []
    for option in options:
        named = [entity for entity in listed if _same_title(option, entity.title)]
        if len({(entity.app, entity.kind, entity.ref) for entity in named}) != 1:
            resolved.append(Option(option))
            continue
        entity = named[0]
        resolved.append(
            Option(
                option,
                app=entity.app,
                kind=entity.kind,
                ref=entity.ref,
                title=entity.title,
                # Listed twice (recent, and a match): outside text if either says so.
                untrusted=any(other.untrusted for other in named),
            )
        )
    return resolved


def annotate(body: dict[str, Any], options: Sequence[str], context: str) -> dict[str, Any]:
    """A pausing body with its ask_user's options resolved against State."""
    return {**body, KEY: [asdict(option) for option in resolve(options, context)]}


def stored(body: Mapping[str, Any], options: Sequence[str]) -> list[Option]:
    """The options a paused body resolved, or plain titles when it has none
    (a body paused before #120, or rebuilt by an amend)."""
    kept = body.get(KEY)
    if isinstance(kept, list) and len(kept) == len(options):
        try:
            return [Option(**item) for item in kept]
        except TypeError:
            pass
    return [Option(option) for option in options]


def chosen(answer: str, options: Sequence[Option]) -> tuple[dataflow.Value, Option | None]:
    """What ``{s1}`` becomes for the person's answer, and the option it picked."""
    index = pick(answer, [option.option for option in options])
    if index is None:
        return dataflow.answer(answer), None
    option = options[index]
    if option.ref is not None and option.title:
        return dataflow.Value(f"“{dataflow.clean(option.title)}”", option.untrusted), option
    said = _norm(option.option) in _norm(answer)
    return dataflow.Value(f"“{dataflow.clean(option.option)}”", not said), option


# --- which option the person picked --------------------------------------------------------

_ORDINALS = {
    "first": 0,
    "1st": 0,
    "former": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}
_LAST = {"last", "latter"}


def pick(answer: str, options: Sequence[str]) -> int | None:
    """The option the answer chose, or ``None`` when it doesn't clearly choose one."""
    said = _norm(answer)
    if not said or not options:
        return None
    # One they named outright ("the dentist appointment"); the longest wins
    # when one option is inside another.
    named = [i for i, option in enumerate(options) if _norm(option) and _within(option, said)]
    if named:
        longest = max(len(_norm(options[i])) for i in named)
        named = [i for i in named if len(_norm(options[i])) == longest]
        return named[0] if len(named) == 1 else None
    # Words only one option has ("the bill one").
    words = terms(answer)
    sets = [terms(option) for option in options]
    common = set.intersection(*sets) if sets else set()
    scores = [len((words & own) - common) for own in sets]
    best = max(scores)
    if best > 0 and scores.count(best) == 1:
        return scores.index(best)
    # Its place ("the first one", "the last", "2").
    places = {
        _ORDINALS[w] if w in _ORDINALS else len(options) - 1 if w in _LAST else int(w) - 1
        for w in said.split()
        if w in _ORDINALS or w in _LAST or (w.isdigit() and len(w) == 1)
    }
    if len(places) == 1:
        (place,) = places
        if 0 <= place < len(options):
            return place
    return None


def _same_title(option: str, title: str) -> bool:
    """Whether an option is this title, also when State clipped it ("…")."""
    want, have = _norm(option), _norm(title)
    if not want or not have:
        return False
    if have.endswith("…"):
        return want.startswith(have.rstrip("…").rstrip())
    return want == have


def _within(option: str, said: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(_norm(option))}(?!\w)", said) is not None


def _norm(text: str) -> str:
    flat = "".join(" " if char in "\"'“”‘’`.,!?;:()[]" else char for char in text.lower())
    return " ".join(flat.split())
