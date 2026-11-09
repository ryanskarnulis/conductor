"""Replay a captured turn offline (#62).

A turn's trace record (:mod:`app.ai.trace`) and its captured model calls
(:mod:`app.ai.context_capture`) hold everything the turn saw: the utterance,
the opening (state block and recall), every model request and response byte
for byte, and every step's answer. :func:`replay` runs today's interpreter
over them with no model and no apps:

- the real :class:`~app.ai.providers.llamacpp.LlamaCppProvider` is served the
  captured response bodies in ``seq`` order through an ``httpx.MockTransport``,
  so parsing, validation and errors take the live code path;
- each step's call (an ask, a look, a recall) is answered from the trace's
  recorded calls, in order;
- the planner prompt is today's, built from the ``fleet`` passed in, so a
  manifest change since the turn shows up as a request that drifted.

It then reports whether today's code rebuilt each request the turn sent,
byte for byte as JSON, and ended the same way (stop reason, reply, step
trajectory). A divergence names the first call where the requests differ. A
traced misfire is a ready-made regression case: replay it before and after a
fix. A turn that resumed a paused plan, acted on the ledger or thought in the
background reads state the trace doesn't keep, and diverges.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.ai import memory
from app.ai import plan as plans
from app.ai.context_capture import read_captures
from app.ai.interpreter import Interpreter, _Run
from app.ai.provider import ToolCall
from app.ai.providers.llamacpp import LlamaCppProvider
from app.ai.run_record import NOT_RUN_PREFIX, shown
from app.ai.runtime import (
    ACTOR,
    INTERNAL_ERROR,
    AgentRunFailed,
    AgentRunResult,
    ToolCallRecord,
)
from app.ai.trace import KIND_RUN
from app.fleet.manifests import Fleet

_REPLAY_BASE_URL = "http://replay.invalid/v1"


class ReplayError(Exception):
    """The trace or capture can't be replayed (missing, or ran out of calls)."""


@dataclass
class ReplayReport:
    run_id: str
    calls_captured: int
    calls_replayed: int = 0
    # Seq of the first replayed request that differs from the captured one.
    first_request_divergence: int | None = None
    divergence_detail: str = ""
    stop_reason: tuple[str, str] = ("", "")
    reply_matches: bool = False
    tools_match: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def faithful(self) -> bool:
        return (
            self.first_request_divergence is None
            and self.calls_replayed == self.calls_captured
            and self.stop_reason[0] == self.stop_reason[1]
            and self.reply_matches
            and self.tools_match
        )


def load_trace(path: Path, run_id: str) -> dict[str, Any]:
    """The trace record of one turn."""
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("kind") == KIND_RUN and record.get("run_id") == run_id:
                return dict(record)
    raise ReplayError(f"no trace record for run {run_id} in {path}")


class _ReplayInterpreter(Interpreter):
    """The live interpreter, with each step's answer from the capture."""

    def __init__(self, provider: LlamaCppProvider, fleet: Fleet, *, record: dict[str, Any]) -> None:
        # A narrated turn (#65) captured the narrator's call too; a turn that
        # replied from the record never called it.
        path = (record.get("narration") or {}).get("path")
        super().__init__(provider, fleet, narrate=path != "code:record")
        # Only calls that were sent answer from the record; the interpreter
        # writes the "not run" ones itself.
        self._answers = [
            ToolCallRecord.model_validate(tool)
            for tool in record.get("tools") or []
            if not (tool.get("error") or "").startswith(NOT_RUN_PREFIX)
        ]

    def _answer_for(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        if not self._answers:
            raise ReplayError(f"the trace has no recorded answer for {name}")
        recorded = self._answers.pop(0)
        # A replay has no memory list: the saved facts an ask carried (#134)
        # aren't asked of it, and the recorded call answers with them.
        if recorded.tool != name or shown(recorded.arguments) != shown(arguments):
            raise ReplayError(
                f"replay asked for {name}({arguments}); the turn ran "
                f"{recorded.tool}({recorded.arguments})"
            )
        if recorded.error == INTERNAL_ERROR:
            raise RuntimeError("replayed crash")
        return recorded

    def _call(self, call: ToolCall, actor: str) -> ToolCallRecord:
        return self._answer_for(call.name, call.arguments)

    def _recall(self, state: _Run, step: plans.Recall) -> ToolCallRecord:
        record = self._answer_for(memory.RECALL_TOOL, {"query": step.query})
        state.records.append(record)
        return record


def replay(trace_path: Path, context_path: Path, run_id: str, fleet: Fleet) -> ReplayReport:
    """Replay one captured turn over ``fleet`` and report whether today's
    interpreter reproduces it."""
    record = load_trace(trace_path, run_id)
    captures = read_captures(context_path, run_id)
    if not captures or captures[0].get("request") is None:
        raise ReplayError(f"no captured model calls for run {run_id} in {context_path}")
    report = ReplayReport(run_id=run_id, calls_captured=len(captures))
    first_request = json.loads(captures[0]["request"])

    def serve(request: httpx.Request) -> httpx.Response:
        seq = report.calls_replayed
        if seq >= len(captures):
            raise ReplayError(f"replay made call {seq}; the turn made {len(captures)}")
        captured = captures[seq]
        report.calls_replayed += 1
        if report.first_request_divergence is None:
            sent, original = json.loads(request.content), json.loads(captured["request"])
            if sent != original:
                report.first_request_divergence = seq
                report.divergence_detail = _first_difference(sent, original)
        if captured.get("response") is None:
            raise httpx.ConnectError(captured.get("error") or "no response captured")
        return httpx.Response(int(captured["status_code"]), text=captured["response"])

    model = (record.get("serving") or {}).get("model") or first_request.get("model", "")
    provider = LlamaCppProvider(
        _REPLAY_BASE_URL, model, client=httpx.Client(transport=httpx.MockTransport(serve))
    )
    interpreter = _ReplayInterpreter(provider, fleet, record=record)
    try:
        result: AgentRunResult = interpreter.run(
            str(record["utterance"]),
            context=record.get("context"),
            surface=record.get("surface"),
            actor=str(record.get("actor") or ACTOR),
        )
    except AgentRunFailed as exc:
        result = exc.result
    finally:
        provider.close()

    report.stop_reason = (str(record.get("stop_reason")), result.stop_reason)
    report.reply_matches = record.get("reply") == result.reply
    report.tools_match = _trajectory(record.get("tools") or []) == _trajectory(
        [tool.model_dump() for tool in result.tool_calls]
    )
    return report


def _trajectory(tools: Sequence[dict[str, Any]]) -> list[tuple[str, str, Any, Any]]:
    return [
        (t["tool"], json.dumps(t.get("arguments"), sort_keys=True), t.get("result"), t.get("error"))
        for t in tools
    ]


def _first_difference(sent: Any, original: Any, path: str = "$") -> str:
    """Where two JSON values first differ, as a path and both sides (clipped)."""
    if isinstance(sent, dict) and isinstance(original, dict):
        for key in sorted(set(sent) | set(original)):
            if sent.get(key) != original.get(key):
                return _first_difference(sent.get(key), original.get(key), f"{path}.{key}")
    if isinstance(sent, list) and isinstance(original, list):
        for index, (a, b) in enumerate(zip(sent, original, strict=False)):
            if a != b:
                return _first_difference(a, b, f"{path}[{index}]")
        if len(sent) != len(original):
            return f"{path}: length {len(sent)} vs captured {len(original)}"
    return f"{path}: {json.dumps(sent)[:200]} vs captured {json.dumps(original)[:200]}"
