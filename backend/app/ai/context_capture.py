"""Context capture: the exact bytes of every model call (#62).

The turn trace (:mod:`app.ai.trace`) records what a turn decided and what each
call cost. It does not record what the model was shown, which is the question
a misfire leaves open. With ``CONDUCTOR_CONTEXT_PATH`` set, every model call
appends one JSONL line with:

- the request body byte for byte as it went to llama-server;
- the response body byte for byte (``reasoning_content`` and raw tool-call
  argument strings included);
- the typed error the call raised;
- the prompt the server's chat template rendered from the same bytes
  (llama-swap's ``/upstream/<model>/apply-template``).

Lines are keyed ``(run_id, seq)``: ``run_id`` is the turn id the interpreter binds in
structlog's context, and ``seq`` counts the turn's calls from 0, so it names
the same call as the trace record's ``calls[seq]``. A capture plus its trace
record replays the turn offline (:mod:`app.ai.replay`).

Off unless the path is set; heavy (every call is kept twice), so it is a
debugging switch, not something to leave on. A capture that fails is logged
(``context_capture_failed``) and dropped: it never costs the turn. Mirrors
PCC's ``app/ai/context_capture.py`` (#401).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

from app.config import get_settings

KIND_MODEL_CALL = "model_call"
CAPTURE_SCHEMA = 1

# How many turns' call counters one capture remembers; bounds memory only.
_SEQ_MEMORY = 256


def current_run_id() -> str | None:
    """The turn id :meth:`Interpreter.run` bound in structlog's context."""
    run_id = structlog.contextvars.get_contextvars().get("run_id")
    return None if run_id is None else str(run_id)


@dataclass(frozen=True)
class CallStamp:
    """Who a call belongs to, fixed when it was sent."""

    run_id: str | None
    seq: int
    started_at: str


class ContextCapture(Protocol):
    """Whatever keeps captured calls. Free to fail: the provider swallows it."""

    def begin(self) -> CallStamp: ...

    def record(self, stamp: CallStamp, call: dict[str, Any]) -> None: ...


# Providers are built per request and turns finish together in worker threads;
# two appends must never interleave inside one line.
_WRITE_LOCK = threading.Lock()
_SEQ_LOCK = threading.Lock()
# Per-turn call counters, shared across provider instances (one per request).
_SEQ: OrderedDict[str | None, int] = OrderedDict()


class JsonlContextCapture:
    """Appends one JSONL line per model call, newest last."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def begin(self) -> CallStamp:
        run_id = current_run_id()
        with _SEQ_LOCK:
            seq = _SEQ.pop(run_id, -1) + 1
            _SEQ[run_id] = seq
            while len(_SEQ) > _SEQ_MEMORY:
                _SEQ.popitem(last=False)
        return CallStamp(run_id=run_id, seq=seq, started_at=_now())

    def record(self, stamp: CallStamp, call: dict[str, Any]) -> None:
        line = json.dumps(
            {
                "schema": CAPTURE_SCHEMA,
                "kind": KIND_MODEL_CALL,
                "run_id": stamp.run_id,
                "seq": stamp.seq,
                "started_at": stamp.started_at,
                "ended_at": _now(),
                **call,
            },
            ensure_ascii=False,
        )
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def context_capture_from_settings() -> ContextCapture | None:
    """A :class:`JsonlContextCapture` at ``CONDUCTOR_CONTEXT_PATH``, or ``None``."""
    path = get_settings().conductor_context_path
    return None if path is None else JsonlContextCapture(path)


def read_captures(path: Path, run_id: str) -> list[dict[str, Any]]:
    """One turn's captured calls, in ``seq`` order."""
    lines = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            line = json.loads(raw)
            if line.get("kind") == KIND_MODEL_CALL and line.get("run_id") == run_id:
                lines.append(line)
    return sorted(lines, key=lambda line: int(line["seq"]))


def _now() -> str:
    return datetime.now(UTC).isoformat()
