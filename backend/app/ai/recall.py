"""What came before this turn, as data in the turn's opening message (#64).

Conductor used to replay a conversation as chat turns, which is how an agent
poisons itself on its own false replies (PCC's ``poisoned_*`` frontier, chess)
and why a turn's cost grew with the thread. Now the model is handed a
:class:`Recall` instead, rendered as data under "Before this turn:", in PCC's
three parts (``project-command-center/backend/app/ai/recall.py``, from chess's
``conversation.Recall``):

1. **The record of earlier turns** — every delegate call that got an answer
   or failed partway after doing something (:mod:`app.ai.run_record` sorts
   them), with what the person had asked, and what the app's own record
   (delegate-api v1.1) says it touched.
2. **What the person asked earlier, in their own words** — every person turn
   but the last exchange's, newest kept under a size cap with a count of what
   went.
3. **The last exchange, whole** — what "it", "that" and "yes" point at. The
   reply is labelled as words, not a record.

Plus, across conversations, :func:`hits`: the earlier conversations' calls
that share words with what the person just said ("that Caribou song", days
later in a new conversation), each labelled with its day and "as of then".

No model writes any of it: the record is code's, the requests and the last
exchange are copied words. Pure: no session, no I/O. Every part is capped, so
the block is bounded however long the history (:data:`MAX_CHARS`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.ai import run_record
from app.ai.runtime import ToolCallRecord

# The person's earlier requests: each cut to this many characters, and the
# newest kept until this many in all (PCC's and chess's numbers).
REQUEST_CHARS = 160
REQUESTS_MAX_CHARS = 1500
# The record of earlier turns: the newest lines kept until this many characters.
RECORD_MAX_CHARS = 2000
# One record line's answer, and the last exchange's two sides.
ANSWER_CHARS = 240
LAST_EXCHANGE_CHARS = 1200
# Hits from earlier conversations: how many, and each line's length.
MAX_HITS = 3
HIT_CHARS = 360
# The whole recall section is never longer than this (asserted in the tests).
MAX_CHARS = (
    RECORD_MAX_CHARS + REQUESTS_MAX_CHARS + 2 * LAST_EXCHANGE_CHARS + MAX_HITS * HIT_CHARS + 600
)

REPLY_LABEL = "What you said then (your words, not a record)"


@dataclass(frozen=True)
class Recall:
    """This conversation before the turn, in the three parts code keeps."""

    record: tuple[str, ...] = ()
    requests: tuple[str, ...] = ()
    last: tuple[str, str | None] | None = None

    @classmethod
    def of(cls, history: Sequence[dict[str, Any]]) -> Recall:
        """Built from stored turns: ``{"role", "content"}`` plus, on an
        assistant turn, its ``tool_calls`` as stored. ``content`` may be
        ``None``: a turn that ended without a reply still did what its record
        shows."""
        at = _last_exchange_at(history)
        last = None
        if at is not None:
            reply = history[at].get("content")
            last = (str(history[at - 1].get("content") or ""), None if reply is None else reply)
        return cls(
            record=tuple(_earlier_record(history)),
            requests=tuple(_requests(history, skip=None if at is None else at - 1)),
            last=last,
        )

    def render(self) -> str:
        parts = []
        if self.record:
            parts.append(
                "The record of earlier turns in this conversation (what the apps did, "
                "oldest first):\n" + "\n".join(f"- {line}" for line in self.record)
            )
        if self.requests:
            parts.append(
                "What the person asked earlier, in their own words, oldest first:\n"
                + "\n".join(f"- {r}" if r.startswith("(") else f'- "{r}"' for r in self.requests)
            )
        if self.last is not None:
            said, reply = self.last
            line = (
                f'{REPLY_LABEL}: "{_truncate(_flat(reply), LAST_EXCHANGE_CHARS)}"'
                if reply
                else f"{REPLY_LABEL}: nothing."
            )
            parts.append(
                "The last exchange:\n"
                f'The person said: "{_truncate(_flat(said), LAST_EXCHANGE_CHARS)}"\n{line}'
            )
        return "\n\n".join(parts)


@dataclass(frozen=True)
class EarlierCall:
    """One delegate call from an earlier conversation, as a recall candidate."""

    when: datetime
    asked: str
    record: ToolCallRecord


def hits(words: str, calls: Iterable[EarlierCall], *, limit: int = MAX_HITS) -> list[str]:
    """The earlier calls sharing the most words with ``words``, newest first
    among equals, as lines. A call that never reached an app is left out.

    Only when the person points back at something ("that song", "the same",
    "it again"): cross-conversation recall exists to resolve a reference, and
    shown on a fresh request it gets read as the state of things ("you
    already trashed that yesterday") instead of asking the app (#64's live
    run: S9, S3).
    """
    if not points_back(words):
        return []
    wanted = _terms(words)
    if not wanted:
        return []
    scored: list[tuple[int, datetime, str]] = []
    for call in calls:
        if run_record.classify(call.record) not in ("done", "looked_up"):
            continue
        shared = wanted & _terms(_searchable(call))
        if shared:
            scored.append((len(shared), call.when, _hit_line(call)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [line for _, _, line in scored[:limit]]


def section(recall: Recall, earlier: Sequence[str] = ()) -> str | None:
    """The "Before this turn:" block, or ``None`` when there is nothing before it."""
    parts = [recall.render()]
    if earlier:
        parts.append(
            "Possibly what the person is pointing back at, from earlier conversations. "
            "Leads for naming it to the app, not facts about now: use the name in your "
            "message to the app, and never tell the person something is done or exists "
            "from these lines alone:\n" + "\n".join(f"- {line}" for line in earlier)
        )
    rendered = "\n\n".join(part for part in parts if part)
    return f"Before this turn:\n\n{rendered}" if rendered else None


# --- parts -------------------------------------------------------------------------


def _last_exchange_at(history: Sequence[dict[str, Any]]) -> int | None:
    for index in range(len(history) - 1, 0, -1):
        if history[index]["role"] == "assistant" and history[index - 1]["role"] == "user":
            return index
    return None


def _requests(history: Sequence[dict[str, Any]], *, skip: int | None) -> list[str]:
    requests = [
        _truncate(_flat(str(message["content"])), REQUEST_CHARS)
        for index, message in enumerate(history)
        if message["role"] == "user" and message.get("content") and index != skip
    ]
    return _newest_within(requests, REQUESTS_MAX_CHARS, "earlier requests")


def _earlier_record(history: Sequence[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    asked = ""
    for message in history:
        if message["role"] == "user":
            asked = str(message.get("content") or "")
            continue
        for stored in message.get("tool_calls") or []:
            record = ToolCallRecord.model_validate(stored)
            line = _record_line(record)
            if line is not None:
                lines.append(f'{line} (asked: "{_truncate(_flat(asked), 80)}")')
    return _newest_within(lines, RECORD_MAX_CHARS, "earlier lines")


def _record_line(record: ToolCallRecord) -> str | None:
    outcome = run_record.classify(record)
    if outcome not in ("done", "looked_up"):
        return None  # never ran, or failed with nothing done: nothing to point at
    call = f"{record.tool} {run_record.clip(_compact(record.arguments))}"
    answer = _truncate(_flat(record.result or record.error or ""), ANSWER_CHARS)
    line = f"{call} -> {answer}"
    touched = _touched(record)
    if touched:
        line += f" [touched: {touched}]"
    return line


def _touched(record: ToolCallRecord) -> str:
    """What the app's own record says the call touched, by title (outside text
    quoted and marked), at most five."""
    names: list[str] = []
    for entry in record.app_record or []:
        if entry.get("outcome") not in ("done", "looked_up"):
            continue
        for ref in entry.get("touched") or []:
            title = ref.get("title")
            if not title:
                names.append(f"{ref.get('kind')} {ref.get('id')}")
            elif ref.get("untrusted") or entry.get("untrusted"):
                names.append(
                    f'{ref.get("kind")} {ref.get("id")} "{_truncate(_flat(title), 60)}" '
                    "(outside text)"
                )
            else:
                names.append(f"{ref.get('kind')} {_truncate(_flat(title), 60)}")
    return ", ".join(names[:5]) + (f", +{len(names) - 5} more" if len(names) > 5 else "")


def _hit_line(call: EarlierCall) -> str:
    line = _record_line(call.record) or ""
    when = f"{call.when:%A} {call.when.date().isoformat()}"
    return _truncate(f'{when}, asked "{_truncate(_flat(call.asked), 80)}": {line}', HIT_CHARS)


def _searchable(call: EarlierCall) -> str:
    titles = " ".join(
        str(ref.get("title") or "")
        for entry in call.record.app_record or []
        for ref in entry.get("touched") or []
    )
    return " ".join(
        [call.asked, json.dumps(call.record.arguments), call.record.result or "", titles]
    )


# Words that point back at something said or done before.
_POINTS_BACK = re.compile(
    r"\b(that|those|it|its|this|these|them|same|again|last|earlier|before|previous|"
    r"other|the one|from (?:yesterday|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday))\b",
    re.IGNORECASE,
)


def points_back(words: str) -> bool:
    """Whether the person's words refer back to something not named in them."""
    return _POINTS_BACK.search(words) is not None


# Words that say nothing about *which* thing: a match on them is noise.
_STOP = frozenset(
    """the and that this those these with from into onto for are was were you your
    yours mine our ours its it's what which who whom when where how why can could would
    should will just put get got grab add make move set give take let all any some one
    thing things stuff please thanks again last next earlier then them they their there
    here have has had did does done not now new old about over under also like want need
    folder list show tell find""".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9'’]+")


def _terms(text: str) -> set[str]:
    words = {w.strip("'’") for w in _WORD.findall(text.lower())}
    return {w for w in words if len(w) >= 3 and w not in _STOP}


def _newest_within(items: list[str], max_chars: int, what: str) -> list[str]:
    kept: list[str] = []
    total = 0
    for item in reversed(items):
        total += len(item)
        if total > max_chars and kept:
            break
        kept.append(item)
    kept.reverse()
    dropped = len(items) - len(kept)
    return ([f"({dropped} {what} not listed)"] if dropped else []) + kept


def _flat(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    """Cut on a word boundary, marked: half a title is worse than a shorter line."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = head.rfind(" ")
    return f"{text[:cut].rstrip() if cut > 0 else text[:limit]}…"


def _compact(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, separators=(",", ":"), sort_keys=True, default=str)
