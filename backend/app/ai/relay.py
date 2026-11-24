"""The trusted channel for an app's answer to a read (#241).

An app's record of a read is refs: kind, id, title, a count. The narrator
retold "a report of my tasks" from those as "found 39 tasks; found 153
tasks", while the app's own grouped answer sat unused. Delegate v1.3 lets
an app send that answer beside the record (``assistant_message.answer``:
``text``, ``spoken``, ``untrusted``); this module decides when conductor
passes it on, and keeps it away from everything else.

**What passes** (:func:`passable`, all of it read by code):

- the step was a read: sent ``mode: "read"`` to a v1.2 app, or
  ``plans.reads_only``'s reading of the words for a v1.1 one. Asked for a
  change, conductor speaks from the record, whatever the app says;
- the call came back, and its record has at least one ``looked_up`` entry
  and nothing else: no ``done``, ``refused`` or ``not_run``;
- the app isn't asking: no confirm, no ``needs_input``.

So no write is ever claimed from an app's words: a passed answer rides only
a call whose code-built record says nothing changed, on a request that
asked for nothing to change.

**Where it goes.** To the person, and nowhere else. It stays on the call's
record (``ToolCallRecord.app_answer``, ``passed`` once it went out), which no
prompt reads; the reply conductor stores (``content``, which recall, the
state block and the self-report quote) is its own words: the narrator's for
the rest of the plan, then code's line that the app answered
(:func:`line`). The page shows the answer under the app's name; a voice
surface says its short form (:func:`spoken`). Nothing in it is ever spliced
into a request, matched against, or followed: the next turn knows the read
by the record's refs, as before.

**Outside text.** The answer is marked (``untrusted``) when the app says
so, when its record has an untrusted ref, or when it names a title this
window knows as outside text (:func:`mark_outside`). Marked or not, it is
cleaned (:func:`take`): no control characters, bounded, and its spoken form
is one line with no markup or links.

**Per step.** Each read that qualifies passes, whatever else the plan did:
the narrator speaks for the other steps and is told the app answered for
itself (:func:`stand_in`), so it neither repeats the read nor says it was
skipped. Only a plan that ends through the narrator passes anything; one
that ends on a question, a handoff or a stop replies as it always did.

Pure: no session, no I/O, no model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from app.ai import confirmation

if TYPE_CHECKING:
    from app.ai.runtime import ToolCallRecord

# The contract's bounds (delegate-api v1.3), enforced here whatever was sent.
MAX_TEXT_CHARS = 4000
MAX_SPOKEN_CHARS = 300
# An app's share of a spoken reply: two sentences when it alone answered,
# one each beside conductor's own words or another app's.
SPOKEN_SENTENCES = 2
SPOKEN_APPS = 2

SPOKEN_SURFACES: frozenset[str] = frozenset({"voice", "room_speaker"})

# What the narrator reads in place of a passed read's record and note.
STAND_IN = (
    "This app answered the person's question in its own words, which they are shown "
    "beside your reply. Say nothing about what it found and never that it wasn't checked."
)

_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")
_BLANK_RUN = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKUP = re.compile(r"[*_#`|>]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_QUOTES = "\"'“”‘’«»"


def take(answer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """An app's ``answer`` as conductor keeps it: cleaned and bounded, or
    ``None`` when there is nothing to show."""
    if not isinstance(answer, Mapping):
        return None
    text = _clean(str(answer.get("text") or ""))
    if not text:
        return None
    if len(text) > MAX_TEXT_CHARS:
        text = text[: MAX_TEXT_CHARS - 1].rstrip() + "…"
    said = _sayable(str(answer.get("spoken") or "")) or _sayable(_opening(text))
    return {"text": text, "spoken": said, "untrusted": bool(answer.get("untrusted"))}


def passable(record: ToolCallRecord) -> bool:
    """Whether a call's answer may go to the person as the app's words."""
    answer = record.app_answer
    if not answer or not answer.get("text") or not answer.get("read"):
        return False
    if not record.tool.startswith("ask_") or record.result is None or record.error is not None:
        return False
    if record.app_confirm or record.app_confirms:
        return False
    if (record.app_asks or {}).get("needs_input"):
        return False
    entries = record.app_record or []
    return bool(entries) and all(entry.get("outcome") == "looked_up" for entry in entries)


def mark(records: Sequence[ToolCallRecord]) -> list[ToolCallRecord]:
    """Mark each call whose answer passes (``passed``), and return them in
    order. An answer that doesn't pass stays on its record, unshown."""
    out = []
    for record in records:
        if passable(record):
            assert record.app_answer is not None
            record.app_answer["passed"] = True
            out.append(record)
    return out


def passed(record: Mapping[str, Any] | ToolCallRecord) -> dict[str, Any] | None:
    """A call's answer if it went to the person, from a live or stored call."""
    answer = (
        record.get("app_answer") if isinstance(record, Mapping) else record.app_answer
    ) or None
    return answer if answer and answer.get("passed") and answer.get("text") else None


def mark_outside(record: ToolCallRecord, outside_titles: Iterable[str]) -> None:
    """Mark an answer that may quote outside text: its own record has an
    untrusted ref, or it names a title this window knows came from outside."""
    answer = record.app_answer
    if not answer or answer.get("untrusted"):
        return
    entries = record.app_record or []
    if any(
        entry.get("untrusted") or any(ref.get("untrusted") for ref in entry.get("touched") or [])
        for entry in entries
    ):
        answer["untrusted"] = True
        return
    text = _folded(str(answer.get("text") or ""))
    for title in outside_titles:
        named = _folded(title)
        if len(named) >= 3 and named in text:
            answer["untrusted"] = True
            return


def stand_in(record: ToolCallRecord) -> ToolCallRecord:
    """A passed read as the narrator sees it: the app, that it answered, and
    nothing it said or found."""
    return record.model_copy(
        update={
            "result": STAND_IN,
            "app_record": [],
            "app_tools": None,
            "app_answer": None,
            "app_asks": {"needs_input": None},
        }
    )


def app_of(record: Mapping[str, Any] | ToolCallRecord) -> str:
    tool = record.get("tool") if isinstance(record, Mapping) else record.tool
    arguments = record.get("arguments") if isinstance(record, Mapping) else record.arguments
    return confirmation.record_app(str(tool or ""), dict(arguments or {})) or "the app"


def line(records: Sequence[Mapping[str, Any] | ToolCallRecord]) -> str:
    """Code's own line for the answers that passed: who answered, none of
    what they said. It is what the stored reply (and so recall) keeps."""
    apps: list[str] = []
    for record in records:
        if passed(record) is not None and app_of(record) not in apps:
            apps.append(app_of(record))
    if not apps:
        return ""
    names = apps[0] if len(apps) == 1 else ", ".join(apps[:-1]) + " and " + apps[-1]
    return f"{names[:1].upper()}{names[1:]} answered."


def reply(own: str | None, records: Sequence[ToolCallRecord]) -> str | None:
    """The stored reply: conductor's own words for the rest of the plan, then
    :func:`line`."""
    said = line(records)
    return " ".join(part for part in ((own or "").strip(), said) if part) or None


def spoken(content: str | None, calls: Sequence[Mapping[str, Any] | ToolCallRecord]) -> str | None:
    """What a voice surface says for a reply that passed answers on, or
    ``None`` when none did (the reply itself is said).

    Conductor's own words keep their two sentences. Each app is then named
    and its short form said: two sentences when it alone answered, one each
    beside conductor's words or another app's, at most :data:`SPOKEN_APPS`
    apps; anything left is on the page, and said to be. A lone answer about
    three things or fewer that names none of them gets code's sentence naming
    them from the record (:func:`_names`), in place of its second sentence."""
    answers = [(app_of(call), answer) for call in calls if (answer := passed(call)) is not None]
    if not answers:
        return None
    own = (content or "").strip()
    said = line(calls)
    if said and own.endswith(said):
        own = own[: -len(said)].strip()
    alone = not own and len(answers) == 1
    parts = [own] if own else []
    for call in calls:
        answer = passed(call)
        if answer is None or len(parts) - bool(own) >= SPOKEN_APPS:
            continue
        app = app_of(call)
        short = _sentences(str(answer.get("spoken") or ""))
        named = _names(call, " ".join(short)) if alone else ""
        # A short list is said by name, from the record, in place of the
        # app's second sentence.
        words = " ".join(short[: 1 if named or not alone else SPOKEN_SENTENCES])
        if words:
            parts.append(f"{app[:1].upper()}{app[1:]} says: {words}")
        if named:
            parts.append(named)
    if len(answers) > SPOKEN_APPS:
        parts.append("The rest is on the page.")
    return " ".join(parts) or None


# A read that found this few things has them said by name on voice.
SPOKEN_NAMES = 3


def _names(call: Mapping[str, Any] | ToolCallRecord, said: str) -> str:
    """Code's sentence naming what a small read found, from its record's
    refs (titles cleaned and quoted, as the narrator's are), when the app's
    spoken form names none of them: "you've got two things due" is no answer
    to someone who can't see the page. Empty for a longer list."""
    entries = (call.get("app_record") if isinstance(call, Mapping) else call.app_record) or []
    titles: list[str] = []
    total = 0
    for entry in entries:
        refs = [
            ref
            for ref in entry.get("touched") or []
            if ref.get("title") and not str(ref.get("kind", "")).endswith("job")
        ]
        total += max(int(entry.get("count") or 0), len(refs))
        for ref in refs:
            title = " ".join("".join(" " if c in _QUOTES else c for c in str(ref["title"])).split())
            title = _CONTROL.sub(" ", title)[:80]
            if title and title not in titles:
                titles.append(title)
    if not titles or total > SPOKEN_NAMES or len(titles) > SPOKEN_NAMES:
        return ""
    heard = _folded(said)
    if any(_folded(title) in heard for title in titles):
        return ""
    quoted = [f"“{title}”" for title in titles]
    listed = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    return f"That's {listed}."


def shown(content: str | None, calls: Sequence[Mapping[str, Any] | ToolCallRecord]) -> str:
    """The whole reply as plain text, as a screen shows it: conductor's
    words, then each answer under its app's name, outside text marked."""
    parts = [(content or "").strip()]
    for call in calls:
        answer = passed(call)
        if answer is None:
            continue
        app = app_of(call)
        note = " (includes outside text)" if answer.get("untrusted") else ""
        parts.append(f"{app[:1].upper()}{app[1:]} says{note}:\n{answer['text']}")
    return "\n\n".join(part for part in parts if part)


def blank(call: Mapping[str, Any]) -> dict[str, Any]:
    """A stored call without its answer's words (an incognito turn keeps none)."""
    return {**call, "app_answer": None} if call.get("app_answer") else dict(call)


def _clean(text: str) -> str:
    text = _CONTROL.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in text.split("\n")]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def _opening(text: str) -> str:
    """An answer's first paragraph without its list lines: what can be said."""
    first = text.split("\n\n", 1)[0]
    return " ".join(line for line in first.splitlines() if not _BULLET.match(line))


def _sayable(text: str) -> str:
    """Words safe to say aloud: one line, no markup, links or quote marks
    of their own, at most two sentences and :data:`MAX_SPOKEN_CHARS`."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _MARKUP.sub("", _CONTROL.sub(" ", text.replace("\n", " ")))
    text = "".join("'" if char in '"“”«»' else char for char in text)
    text = " ".join(_sentences(" ".join(text.split()))[:SPOKEN_SENTENCES])
    if len(text) > MAX_SPOKEN_CHARS:
        text = _sentences(text)[0]
    if len(text) > MAX_SPOKEN_CHARS:
        text = text[: MAX_SPOKEN_CHARS - 1].rstrip() + "…"
    return text


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_END.split(text.strip()) if sentence]


def _folded(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


__all__ = [
    "MAX_SPOKEN_CHARS",
    "MAX_TEXT_CHARS",
    "SPOKEN_SURFACES",
    "STAND_IN",
    "app_of",
    "blank",
    "line",
    "mark",
    "mark_outside",
    "passable",
    "passed",
    "reply",
    "shown",
    "spoken",
    "stand_in",
    "take",
]
