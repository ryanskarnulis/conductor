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
(:data:`MAX_CHARS`), however long the history.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import structlog

from app.ai import recall, state_block
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
    conductor to remember (#81)."""
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
    try:
        block = state_block.render(today, now, turns, recent, running, leads, saved)
        before = recall.section(recall.Recall.of(turns), hits)
    except Exception as exc:
        # A stored turn conductor can't read: the calendar alone, never no turn.
        logger.warning("turn_context_failed", error=str(exc), error_type=type(exc).__name__)
        return state_block.render(today, now, [])
    return "\n\n".join(part for part in (block, before) if part)
