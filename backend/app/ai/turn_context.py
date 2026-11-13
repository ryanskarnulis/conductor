"""A turn's context: the state block, then recall, then the person's words (#64).

The one place a turn's opening user message is composed. Conductor no longer
replays chat turns: the model sees the system prompt and the tool schemas
(byte-identical turn to turn), then one user message holding

1. the state block (:mod:`app.ai.state_block`), with the entity registry's
   matches from earlier conversations (#80, :mod:`app.services.memory`) and
   the facts the person saved (#81, :mod:`app.services.facts`),
2. "Before this turn:" (:mod:`app.ai.recall`): this conversation's record,
   requests and last exchange, plus earlier conversations' calls that share
   words with what the person just said when the registry has no match,
3. the person's words, last, nearest the model's answer.

Building it never costs the turn: a lookup that fails is logged and the block
falls back to the calendar alone. Its size is bounded by the parts' caps
(:data:`MAX_CHARS`), however long the history. The planner reads it through
:func:`for_planner`, which leaves out an earlier request in the person's same
words (#141); everything else reads it whole.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import structlog

from app.ai import memory, recall, state_block
from app.ai.state_block import Now

logger = structlog.get_logger(__name__)

# The registry's matches and the word-matched calls never both appear (the
# calls are the fallback when the registry finds nothing), so the bound counts
# the matches' share once.
MAX_CHARS = (
    state_block.MAX_CHARS
    + recall.MAX_CHARS
    + 100
    - min(state_block.MATCHES_MAX_CHARS, recall.MAX_HITS * recall.HIT_CHARS)
)


def build(
    today: date,
    now: Now,
    utterance: str,
    history: Callable[[], Sequence[dict[str, Any]]],
    earlier: Callable[[], Sequence[recall.EarlierCall]],
    entities: Callable[[], Sequence[state_block.Entity] | None] | None = None,
    jobs: Callable[[], Sequence[str]] | None = None,
    matches: Callable[[], Sequence[state_block.Match]] | None = None,
    facts: Callable[[], Sequence[state_block.Fact]] | None = None,
    heard: Callable[[], state_block.Heard | None] | None = None,
) -> str:
    """The context that opens the turn (without the person's words, which the
    compiler adds last). ``history`` is this conversation's stored turns;
    ``earlier`` other conversations' delegate calls, newest first;
    ``entities`` the ledger's recent entities (#69), newest first — when it is
    absent, answers ``None`` or fails, the state block reads them from
    ``history``. ``jobs`` are the ledger's running jobs (#76), likewise.
    ``matches`` are the entity registry's leads from earlier conversations
    (#80); when it finds any, the word-matched calls from ``earlier`` are left
    out (they'd say the same thing twice). ``facts`` are what the person asked
    conductor to remember (#81). ``heard`` is the last reply when the person
    spoke over it (#77)."""
    try:
        turns = list(history())
    except Exception as exc:
        logger.warning("turn_context_history_failed", error=str(exc), error_type=type(exc).__name__)
        turns = []
    leads: list[state_block.Match] = []
    if matches is not None:
        try:
            leads = list(matches())
        except Exception as exc:
            logger.warning(
                "turn_context_matches_failed", error=str(exc), error_type=type(exc).__name__
            )
    try:
        hits = [] if leads else recall.hits(utterance, earlier())
    except Exception as exc:
        logger.warning("turn_context_recall_failed", error=str(exc), error_type=type(exc).__name__)
        hits = []
    recent: Sequence[state_block.Entity] | None = None
    if entities is not None:
        try:
            found = entities()
            recent = None if found is None else list(found)
        except Exception as exc:
            logger.warning(
                "turn_context_entities_failed", error=str(exc), error_type=type(exc).__name__
            )
    running: Sequence[str] | None = None
    if jobs is not None:
        try:
            running = list(jobs())
        except Exception as exc:
            logger.warning(
                "turn_context_jobs_failed", error=str(exc), error_type=type(exc).__name__
            )
    saved: list[state_block.Fact] = []
    if facts is not None:
        try:
            saved = list(facts())
        except Exception as exc:
            logger.warning(
                "turn_context_facts_failed", error=str(exc), error_type=type(exc).__name__
            )
    cut: state_block.Heard | None = None
    if heard is not None:
        try:
            cut = heard()
        except Exception as exc:
            logger.warning(
                "turn_context_heard_failed", error=str(exc), error_type=type(exc).__name__
            )
    try:
        block = state_block.render(today, now, turns, recent, running, leads, saved, cut)
        before = recall.section(recall.Recall.of(turns), hits)
    except Exception as exc:
        # A stored turn conductor can't read: the calendar alone, never no turn.
        logger.warning("turn_context_failed", error=str(exc), error_type=type(exc).__name__)
        return state_block.render(today, now, [])
    return "\n\n".join(part for part in (block, before) if part)


# Where the person's earlier words sit in the opening, as recall and the
# state block render them: a record line's `(asked: "…")`, an earlier
# conversation's hit, an earlier request, the last exchange's first side, and
# a registry match's `you said "…"`.
_RECORD_ASKED = re.compile(r'\(asked: "(?P<asked>.*)"\)$')
_HIT_ASKED = re.compile(r'^- \w+ \d{4}-\d{2}-\d{2} \(as of then\), asked "(?P<asked>.*?)": ')
_REQUEST = re.compile(r'^- "(?P<asked>.*)"$')
_SAID = re.compile(r'^The person said: "(?P<asked>.*)"$')
_MATCH_SAID = re.compile(r', you said "(?P<asked>.*)"$')
# Recall's parts that list lines, each by its heading.
_LISTS = (
    (recall.RECORD_HEADING, _RECORD_ASKED),
    (recall.HITS_HEADING, _HIT_ASKED),
    (recall.REQUESTS_HEADING, _REQUEST),
)


def for_planner(context: str, words: str) -> str:
    """The opening as the planner reads it: an earlier request in the
    person's same words left out (#141).

    Asked "what's due today?" again, the 12B files the question as already
    answered whenever it sees that request before this turn, with or without
    the app's answer, and asks again when the record holds a different
    question (#141's diagnosis). So the request is left out wherever it
    shows: its record lines, an earlier conversation's hit, the earlier
    requests, a registry match it touched, and the last exchange when it was
    that request. Same words means the terms :func:`app.ai.plan.recalled_answer`
    matches on, equal. Everything else stays byte for byte, and a part left
    empty goes with its heading. Only the planner's message reads this: the
    validator gets the whole opening, so the repair still catches an answer
    from memory; the narrator and speaker never read the opening."""
    wanted = memory.terms(words)
    if not wanted:
        return context

    def repeats(line: str, pattern: re.Pattern[str]) -> bool:
        found = pattern.search(line)
        if found is None:
            return False
        asked = found.group("asked").rstrip("…").strip()
        return bool(asked) and memory.terms(asked) == wanted

    kept: list[str] = []
    for block in context.split("\n\n"):
        lines = block.split("\n")
        head = lines[0]
        if head == recall.LAST_HEADING and len(lines) > 1 and repeats(lines[1], _SAID):
            continue
        pattern = next((p for h, p in _LISTS if head.startswith(h)), None)
        if pattern is not None:
            items = [line for line in lines[1:] if not repeats(line, pattern)]
            if items:
                kept.append("\n".join([head, *items]))
            continue
        kept.append("\n".join(_without_matches(lines, lambda line: repeats(line, _MATCH_SAID))))
    if kept and kept[-1] == recall.HEADING:
        kept.pop()  # nothing left before this turn
    return "\n\n".join(kept)


def _without_matches(lines: list[str], repeats: Callable[[str], bool]) -> list[str]:
    """The state block's lines with the registry matches the same request
    touched left out, and the matches' heading when none is left."""
    out: list[str] = []
    heading_at: int | None = None
    for line in lines:
        if line == state_block.MATCHES_HEADING:
            heading_at = len(out)
            out.append(line)
            continue
        if heading_at is not None:
            if line.startswith("- "):
                if not repeats(line):
                    out.append(line)
                continue
            if len(out) == heading_at + 1:
                out.pop()  # every match went
            heading_at = None
        out.append(line)
    if heading_at is not None and len(out) == heading_at + 1:
        out.pop()
    return out
