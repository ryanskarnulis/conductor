"""The turn trace: one JSONL record per conductor turn (#62).

Conductor's runs used to leave ``delegate_call`` log lines and nothing else. A
multi-agent misfire is found by reading traces, and blame has to come by
construction, so with ``CONDUCTOR_TRACE_PATH`` set each turn appends one
record holding:

- the turn id (``run_id``), the conversation, the utterance, the opening
  (state block and recall) and the compiled plan;
- one entry per model call (status, wall clock, tokens, finish reason);
- every tool call with its arguments and answer;
- one **hop** per delegate call: its ``hop_id`` (sent to the app as
  ``X-Agent-Hop``, beside the turn as ``X-Agent-Turn``), the app, the
  subagent thread, the app's own request id (its ``X-Request-ID`` response
  header), what the app ran, its stop reason, and the error class if it
  failed. Turn → hops → app runs reads straight off one record;
- the stop reason, the reply, and the planner prompt's hash beside the
  model, so two baselines' configurations can be told apart.

The exact bytes of each model call are #62's other half,
:mod:`app.ai.context_capture`, keyed by the same ``run_id``; together they
replay a turn offline (:mod:`app.ai.replay`).

Diagnostics only: off unless a path is set, and a write that fails is logged
and swallowed (:meth:`app.ai.runtime.Runtime.write_trace`). Mirrors PCC's
``app/ai/trace.py`` (#400), minus its narrator, gather and state-block fields,
plus hops. Read records with ``backend/scripts/read_trace.py``; the layout is
in ``docs/agent-tracing.md``.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from app.config import get_settings

if TYPE_CHECKING:
    from app.ai.runtime import AgentRunResult
    from app.ai.provider import Usage

# The record layout's version; bump it on any change a reader must branch on.
# 2 (#75): the loop's ``history``, ``iterations``, ``corrections``, ``limits``
# and tool-schema fields are gone; the system prompt hash is the planner's.
TRACE_SCHEMA = 2
KIND_RUN = "run"

# How one model call ended: ``ok``, ``truncated`` (cut off at ``max_tokens``),
# ``bad_args`` (tool-call arguments that were not a JSON object), ``failed``
# (no usable answer).
CallStatus = Literal["ok", "truncated", "bad_args", "failed"]
CallRole = Literal["planner", "narrator", "compiler", "speaker"]


def new_run_id() -> str:
    """A fresh turn id: full uuid4 hex, unique across processes."""
    return uuid.uuid4().hex


def new_hop_id() -> str:
    return uuid.uuid4().hex[:16]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ModelCall:
    """One model round trip. Token counts are ``None`` when unknown, never a
    zero that reads as measured. ``role``: the planner's turn or the
    narrator's (#65)."""

    role: CallRole = "planner"
    status: CallStatus = "ok"
    ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    tool_calls: int = 0
    failure: str = ""

    def meter(self, usage: Usage | None) -> None:
        if usage is not None:
            self.prompt_tokens = usage.prompt_tokens
            self.completion_tokens = usage.completion_tokens

    def as_trace(self, seq: int) -> dict[str, Any]:
        return {"seq": seq, **asdict(self)}


@dataclass
class Hop:
    """One delegate call: conductor → one app's agent.

    ``app_request_id`` is the app's ``X-Request-ID`` response header: the id
    its own logs carry for that request. ``None`` when the app never answered
    (connect error) or sent no such header.
    """

    hop_id: str
    app: str
    # The turn this hop belongs to (sent as ``X-Agent-Turn``).
    run_id: str = ""
    thread_id: int | None = None
    ms: int = 0
    app_request_id: str | None = None
    app_stop_reason: str | None = None
    app_tools: list[str] = field(default_factory=list)
    error: str | None = None
    # delegate-api v1.1 (#67): the key every message of this hop carries, so a
    # retry is answered from the app's store rather than run again; and the
    # app's typed record of what it did (its receipts), on a 200 or in a
    # failure body. ``None`` when the app sent none (v1).
    idempotency_key: str | None = None
    app_record: list[dict[str, Any]] | None = None


@dataclass
class RunTrace:
    """What one turn learned as it went, filled in by the interpreter and the
    delegate tools. Always kept; :meth:`record` turns it into the JSONL line."""

    run_id: str
    actor: str
    utterance: str
    conversation_id: str | None = None
    model: str | None = None
    # The state block and recall that opened the turn's user message (#64).
    context: str | None = None
    # Where the person is (``Now.surface``) and how the reply was made (#65).
    surface: str | None = None
    narration: dict[str, Any] | None = None
    # The compiled plan (#70).
    plan: dict[str, Any] | None = None
    # The planner prompt's hash: what the fleet's manifests made of it.
    system_prompt_sha256: str | None = None
    calls: list[ModelCall] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)
    started: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record(self, result: AgentRunResult, *, error: str = "", total_ms: int) -> dict[str, Any]:
        calls = self.calls
        return {
            "schema": TRACE_SCHEMA,
            "kind": KIND_RUN,
            "ts": datetime.now(UTC).isoformat(),
            "started": self.started.isoformat(),
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "actor": self.actor,
            "utterance": self.utterance,
            "context": self.context,
            "surface": self.surface,
            "narration": self.narration,
            "plan": self.plan,
            "stop_reason": result.stop_reason,
            "error": error,
            "reply": result.reply,
            "tools": [record.model_dump() for record in result.tool_calls],
            "hops": [asdict(hop) for hop in self.hops],
            "model_calls": len(calls),
            "prompt_tokens": sum(call.prompt_tokens or 0 for call in calls),
            "completion_tokens": sum(call.completion_tokens or 0 for call in calls),
            "model_ms": sum(call.ms for call in calls),
            "total_ms": total_ms,
            "calls": [call.as_trace(seq) for seq, call in enumerate(calls)],
            "serving": {
                "model": self.model,
                "system_prompt_sha256": self.system_prompt_sha256,
            },
        }


# The turn being traced, for the delegate tools to add hops to. Set by the
# interpreter around the turn; ``None`` outside one (a tool test, the MCP server's
# direct calls).
_CURRENT: contextvars.ContextVar[RunTrace | None] = contextvars.ContextVar(
    "conductor_run_trace", default=None
)


# The delegate call in flight, for the client to stamp its headers from.
_CURRENT_HOP: contextvars.ContextVar[Hop | None] = contextvars.ContextVar(
    "conductor_hop", default=None
)


@contextmanager
def tracing(trace: RunTrace) -> Iterator[RunTrace]:
    token = _CURRENT.set(trace)
    try:
        yield trace
    finally:
        _CURRENT.reset(token)


def current() -> RunTrace | None:
    return _CURRENT.get()


@contextmanager
def hop(app: str) -> Iterator[Hop]:
    """One delegate call as a hop: attached to the current turn's trace when
    there is one, and current for the client while the block runs. Outside a
    traced turn the hop still gets ids, so the headers always go out."""
    trace = current()
    new = Hop(hop_id=new_hop_id(), app=app, run_id=trace.run_id if trace is not None else "")
    if trace is not None:
        trace.hops.append(new)
    token = _CURRENT_HOP.set(new)
    try:
        yield new
    finally:
        _CURRENT_HOP.reset(token)


def current_hop() -> Hop | None:
    return _CURRENT_HOP.get()


class Tracer(Protocol):
    """Whatever records a finished turn. Free to fail: the runtime swallows it."""

    def write(self, record: dict[str, Any]) -> None: ...


# Turns on different conversations finish concurrently in worker threads; two
# appends must never interleave inside one line.
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class JsonlTracer:
    """Appends each turn to a JSONL file, newest last, one line per turn."""

    path: Path

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def tracer_from_settings() -> Tracer | None:
    """A :class:`JsonlTracer` at ``CONDUCTOR_TRACE_PATH``, or ``None`` (off)."""
    path = get_settings().conductor_trace_path
    return None if path is None else JsonlTracer(path)
