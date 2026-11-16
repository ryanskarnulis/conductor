# vendored — canonical: agent-standard/agentkit/trace.py; re-copy, never edit
"""The trace writer: how a finished run reaches its JSONL file, whatever the app.

Canonical in ``../agent-standard/agentkit/trace.py``; every app vendors it
verbatim as ``agentkit/trace.py`` inside its own package, with one
``# vendored …`` first line. Standard library only, and no import from the app.

What a record *holds* is the app's: its fields, its schema number and its
reader live in the app's own trace module. This file is what never varies:

- :func:`new_run_id` and the hashes a record names its prompt and tool schemas
  by (:func:`sha256`, :func:`tool_schemas_sha256`);
- :class:`Tracer`, whatever records a finished run, and :class:`JsonlTracer`,
  the one that appends a line to a file;
- :func:`write_record`, which builds and writes a record and never raises.

Tracing is diagnostics and nothing more: losing a record must never cost the
run it describes.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def new_run_id() -> str:
    """A fresh id for one run: what its record and its log lines are keyed by.
    Full uuid4 hex — unique across processes."""
    return uuid.uuid4().hex


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class WireSchema(Protocol):
    """A tool as the model is offered it."""

    def to_wire(self) -> dict[str, Any]: ...


def tool_schemas_sha256(specs: Sequence[WireSchema]) -> str:
    """One hash over every offered tool's wire schema, canonicalised (sorted
    keys, no whitespace) so it changes with a name, a docstring or an argument,
    never with dict ordering."""
    wire = [spec.to_wire() for spec in specs]
    return sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")))


class Tracer(Protocol):
    """Whatever records a finished run. Free to fail: :func:`write_record`
    swallows it."""

    def write(self, record: dict[str, Any]) -> None: ...


# One lock for every JsonlTracer in the process: runs finish concurrently in
# worker threads, and two appends must never interleave inside one line.
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class JsonlTracer:
    """Appends each record to a JSONL file, newest last, one line per record.

    JSONL because records are appended forever and read back a few at a time:
    a crash mid-write costs at most the last line, ``tail -f`` works during a
    live session, and a file of records loads with a one-line comprehension.
    The line is serialised before the lock is taken, so a record that cannot
    be serialised leaves no half line behind.
    """

    path: Path

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def write_record(
    tracer: Tracer | None,
    build: Callable[[], dict[str, Any]],
    *,
    failed: Callable[[Exception], None],
) -> bool:
    """Build a record and hand it to ``tracer``; never raises.

    The run's own outcome stands whether or not its record can be built or
    written: a failure is handed to ``failed`` (the app logs it its own way)
    and swallowed. Says whether a record was written; ``False`` with no tracer
    too, and then ``build`` is never called.
    """
    if tracer is None:
        return False
    try:
        tracer.write(build())
    except Exception as exc:
        failed(exc)
        return False
    return True
