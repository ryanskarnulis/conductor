"""A turn's context: the state block, then recall, then the person's words (#64).

The one place a turn's opening user message is composed. Conductor no longer
replays chat turns: the model sees the system prompt and the tool schemas
(byte-identical turn to turn), then one user message holding

1. the state block (:mod:`app.ai.state_block`),
2. "Before this turn:" (:mod:`app.ai.recall`): this conversation's record,
   requests and last exchange, plus earlier conversations' calls that share
   words with what the person just said,
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

MAX_CHARS = state_block.MAX_CHARS + recall.MAX_CHARS + 100


def build(
    today: date,
    now: Now,
    utterance: str,
    history: Callable[[], Sequence[dict[str, Any]]],
    earlier: Callable[[], Sequence[recall.EarlierCall]],
) -> str:
    """The context that opens the turn (without the person's words, which the
    loop adds last). ``history`` is this conversation's stored turns;
    ``earlier`` other conversations' delegate calls, newest first."""
    try:
        turns = list(history())
    except Exception as exc:
        logger.warning("turn_context_history_failed", error=str(exc), error_type=type(exc).__name__)
        turns = []
    try:
        hits = recall.hits(utterance, earlier())
    except Exception as exc:
        logger.warning("turn_context_recall_failed", error=str(exc), error_type=type(exc).__name__)
        hits = []
    try:
        block = state_block.render(today, now, turns)
        before = recall.section(recall.Recall.of(turns), hits)
    except Exception as exc:
        # A stored turn conductor can't read: the calendar alone, never no turn.
        logger.warning("turn_context_failed", error=str(exc), error_type=type(exc).__name__)
        return state_block.render(today, now, [])
    return "\n\n".join(part for part in (block, before) if part)
