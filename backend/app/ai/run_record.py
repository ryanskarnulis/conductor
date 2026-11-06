"""The run record: what a run did, built by code from its tool calls.

STANDARD.md §3: when a run stops at a limit, or fails after calls ran, its
reply is this record — never the model's words. Every
:class:`~app.ai.loop.ToolCallRecord` is sorted into

- **done**: a delegate or handoff call that came back with an answer, or a
  delegate call that failed but whose app's record lists ``done`` work (#67):
  that work happened, so it is never filed as a failure;
- **failed**: a call the app or the tool refused or that broke partway (its
  reason attached, so a "stopped partway" from #60 is kept word for word);
- **looked up**: a local read (``list_agents``) that came back;
- **not run**: a call that never left conductor: schema-invalid arguments, an
  unknown tool, the tool-call cap, or the rest of a turn after a crash.

The rendered record always says "Done: nothing." when nothing was, and a stop
adds a note: whatever the record does not show as done was not done. It is
what the next turn's history replays, so a write that happened is never lost
from the transcript (#52).

Mirrors PCC's ``app/ai/run_record.py``. A limit or failure stop replies with
the rendered record itself; a completed run's reply is the narrator's
(:mod:`app.ai.narrator`, #65), which sorts calls with :func:`classify` too.
Pure: no I/O, no registry
lookups (so a replayed record reads the same), and nothing here reads the
model's words.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.ai.loop import ToolCallRecord

Outcome = Literal["done", "failed", "looked_up", "not_run"]

# Conductor's only local, read-only tool. Everything else either hands a request
# to an app (which may write) or hands the user over, so it counts as done.
READ_TOOLS = frozenset({"list_agents"})

# The prefix of every error text the loop writes for a call it never
# dispatched. The loop builds those texts from it, so a reworded one cannot
# start reading as a refusal.
NOT_RUN_PREFIX = "not run: "
# The loop's prefix for schema-invalid arguments (``AgentLoop._dispatch``),
# and the registry's for a tool name it does not know (``UnknownToolError``).
INVALID_ARGUMENTS_PREFIX = "Invalid arguments: "
UNKNOWN_TOOL_PREFIX = "Unknown tool "

# How much of one call's arguments or answer the record carries: enough to name
# what was asked and answered, not a transcript.
_MAX_FIELD_CHARS = 200

LIMIT_NOTE = "Whatever the record above does not show as done was not done."

# Why the run stopped, keyed by stop reason. Limit stops and the failure stops
# that can follow calls that ran.
STOP_REASONS: dict[str, str] = {
    "max_iterations": "I ran out of steps",
    "correction_limit": "I kept making malformed tool calls",
    "truncated": "my reply ran past the length limit",
    "no_progress": "I kept repeating the same calls",
    "tool_call_limit": "I reached my limit of tool calls for one request",
    "provider_error": "my model stopped answering",
    "timed_out": "I ran out of time",
    "internal_error": "something broke inside conductor",
    "cancelled": "you stopped me",
}


def classify(record: ToolCallRecord) -> Outcome:
    """Sort one call by what its record shows happened."""
    if record.result is not None:
        return "looked_up" if record.tool in READ_TOOLS else "done"
    if _app_done(record):
        return "done"
    error = record.error or ""
    if error.startswith((NOT_RUN_PREFIX, INVALID_ARGUMENTS_PREFIX, UNKNOWN_TOOL_PREFIX)):
        return "not_run"
    return "failed"


def any_ran(records: Sequence[ToolCallRecord]) -> bool:
    """Whether any call reached a tool (done, failed or looked up)."""
    return any(classify(record) != "not_run" for record in records)


def render(records: Sequence[ToolCallRecord]) -> str:
    """The record grouped by outcome, one line per call, in run order (``#n``)."""
    groups: dict[Outcome, list[str]] = {"done": [], "failed": [], "looked_up": [], "not_run": []}
    for ref, record in enumerate(records, start=1):
        outcome = classify(record)
        line = f"#{ref} {record.tool} {_clip(_compact(record.arguments))}"
        if outcome in ("done", "looked_up"):
            line += f" -> {_clip(record.result or record.error or '')}"
            done = _app_done(record)
            if done:
                line += f" [app did: {', '.join(done)}]"
            elif record.app_tools:
                line += f" [app ran: {', '.join(record.app_tools)}]"
        else:
            line += f" -> {_clip(record.error or '')}"
        groups[outcome].append(line)
    lines = ["Done:"]
    lines += [f"- {line}" for line in groups["done"]] or ["- nothing"]
    for outcome, heading in (
        ("failed", "Failed"),
        ("looked_up", "Looked up"),
        ("not_run", "Not run"),
    ):
        if groups[outcome]:
            lines.append(f"{heading}:")
            lines += [f"- {line}" for line in groups[outcome]]
    return "\n".join(lines)


def stop_reply(stop_reason: str, records: Sequence[ToolCallRecord]) -> str | None:
    """The reply for a run that stopped short: the record, or ``None`` if nothing ran.

    A stop where no call reached a tool has nothing verified to say; the client
    shows its own stop line.
    """
    if not any_ran(records):
        return None
    reason = STOP_REASONS.get(stop_reason, stop_reason)
    return f"I stopped before finishing: {reason}. Here is what ran.\n\n{render(records)}\n\n{LIMIT_NOTE}"


def _app_done(record: ToolCallRecord) -> list[str]:
    """The tools the app's own record (v1.1) says it did, in order."""
    return [
        str(entry.get("tool"))
        for entry in record.app_record or []
        if entry.get("outcome") == "done"
    ]


def _compact(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, separators=(",", ":"), sort_keys=True, default=str)


def clip(text: str) -> str:
    """One field cut to the record's line length, whitespace flattened."""
    return _clip(text)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return text[: _MAX_FIELD_CHARS - 1].rstrip() + "…"
