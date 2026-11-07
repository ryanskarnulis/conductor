"""Typed data flow between plan steps (#72): ``{sN}`` from step N's receipt.

A later step names an earlier one's result by writing ``{sN}`` (or
``{sN.title}``; any field reads the same). Code fills it in when the later step
runs, from what step N's app said it touched in its v1.1 record (never from
its agent's words):

- **The main title.** The titles of the ``done`` and ``looked_up`` entries'
  touched refs, in order, without job refs (a download job's "album
  download" is bookkeeping, not the thing) and without repeats. Each is
  cleaned (one line, no control characters, quotes or braces, at most
  :data:`narrator.MAX_TITLE_CHARS`) and quoted: ``“Honey”``.
- **Several titles:** up to :data:`narrator.MAX_NAMED` are all passed on,
  ``“A”, “B” and “C”``; more than that is a count, not a name, so there is
  nothing to pass on.
- **No title** (a v1 app with no record, a record that touched nothing
  titled, a step that answered without a receipt): nothing to pass on.
- **A failed step** ends the plan before anything could use it.

A step whose ``{sN}`` has nothing to pass on doesn't run, and the plan ends
there: the record says it didn't run and why. Code never guesses a name, and
never substitutes the app's reply text.

**Taint** (CaMeL; the lethal trifecta, #56): a title the app marks
``untrusted`` (outside text: a web title, a file's tags) is data, never
control. Control flow can't be steered by it by construction: the whole plan
(its steps, apps and order) is fixed before any step runs, and a title is
only ever spliced, quoted, into a request's text. Beyond that, an untrusted
title flows into another step only where whatever it might talk an agent
into still stops on the person:

- into a non-destructive ``ask`` to an app that confirms its own writes
  (``agent.confirms``), so a destructive action the text provokes still
  comes back as the app's own question;
- never into a ``destructive`` ask (outside text never picks what to
  delete), never into an ask to an app that doesn't confirm, and never into
  an ``open`` intent (a handoff's intent is an instruction to the page).

A refused flow is a step not run, on the record with its reason. The person's
own answer to an ``ask_user`` is trusted and goes in as they said it.

Values don't outlive a pause: when a plan pauses (gate, an app's question,
``ask_user``) the trusted values are written into the rest of the paused body;
an untrusted one is left as ``{sN}`` and the resumed step has nothing to pass
on, so it doesn't run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.ai.narrator import MAX_NAMED, MAX_TITLE_CHARS
from app.ai.plan import REF, STEP_TEXT_FIELDS

_CONTROL = re.compile(r"[\x00-\x1f\x7f  ]")
_STRIP = set("\"'“”‘’`{}")


@dataclass(frozen=True)
class Value:
    """What ``{sN}`` becomes: text ready to splice in, and whether it is outside text."""

    text: str
    untrusted: bool = False


@dataclass(frozen=True)
class Filled:
    """A step's text with its refs filled, or why it can't be."""

    text: str
    # Refs whose step had nothing to pass on.
    missing: tuple[str, ...] = ()
    # Refs that put outside text in.
    untrusted: tuple[str, ...] = ()


def clean(title: str) -> str:
    """A title made safe to splice into a request: one line, no control
    characters, quotes or braces, capped."""
    title = _CONTROL.sub(" ", title)
    title = "".join(" " if char in _STRIP else char for char in title)
    title = " ".join(title.split())
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 1].rstrip() + "…"
    return title


def from_record(app_record: Iterable[Mapping[str, Any]] | None) -> Value | None:
    """The main thing a step touched, from its app's v1.1 record; ``None`` when
    there's nothing to pass on."""
    titles: list[str] = []
    seen: set[str] = set()
    untrusted = False
    for entry in app_record or []:
        if entry.get("outcome") not in ("done", "looked_up"):
            continue
        for ref in entry.get("touched") or []:
            title = clean(str(ref.get("title") or ""))
            if not title or str(ref.get("kind") or "").endswith("job"):
                continue
            untrusted = untrusted or bool(ref.get("untrusted") or entry.get("untrusted"))
            key = title.casefold()
            if key not in seen:
                seen.add(key)
                titles.append(title)
    if not titles or len(titles) > MAX_NAMED:
        return None
    quoted = [f"“{title}”" for title in titles]
    text = quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"
    return Value(text, untrusted)


def answer(words: str) -> Value:
    """The person's answer to an ``ask_user``: trusted, as they said it."""
    return Value(words.strip().replace("{", "(").replace("}", ")"))


def fill(text: str, values: Mapping[str, Value]) -> Filled:
    """``text`` with every ``{sN}`` filled from ``values``, in one pass (a
    filled-in value is never read for refs)."""
    missing: list[str] = []
    untrusted: list[str] = []

    def one(match: re.Match[str]) -> str:
        ref = match.group(1)
        value = values.get(ref)
        if value is None:
            missing.append(ref)
            return match.group(0)
        if value.untrusted:
            untrusted.append(ref)
        return value.text

    filled = REF.sub(one, text)
    return Filled(filled, tuple(dict.fromkeys(missing)), tuple(dict.fromkeys(untrusted)))


def carry(body: dict[str, Any], start: int, values: Mapping[str, Value]) -> dict[str, Any]:
    """A paused plan's body with the trusted values written into the steps
    from ``start`` on, so they survive to the turn that resumes it."""
    trusted = {ref: value for ref, value in values.items() if not value.untrusted}
    if not trusted:
        return body
    steps = [dict(step) for step in body.get("steps") or []]
    for step in steps[start:]:
        for name in STEP_TEXT_FIELDS:
            if isinstance(step.get(name), str):
                step[name] = fill(step[name], trusted).text
    return {**body, "steps": steps}
