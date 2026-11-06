"""Replay a captured turn offline (#62).

A turn's trace record (:mod:`app.ai.trace`) and its captured model calls
(:mod:`app.ai.context_capture`) hold everything the turn saw: the utterance,
the history, the system prompt and tool schemas as sent, every model response
byte for byte, and every tool answer. :func:`replay` runs today's loop over
them with no model and no apps:

- the real :class:`~app.ai.providers.llamacpp.LlamaCppProvider` is served the
  captured response bodies in ``seq`` order through an ``httpx.MockTransport``,
  so parsing, validation and errors take the live code path;
- tool calls are answered from the trace's recorded calls, in order;
- the system prompt and tool schemas are the captured ones, not today's.

It then reports whether today's code rebuilt each request the turn sent,
byte for byte as JSON, and ended the same way (stop reason, reply, tool
trajectory). A divergence names the first call where the requests differ. A
traced misfire is a ready-made regression case: replay it before and after a
fix.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.ai.context_capture import read_captures
from app.ai.loop import (
    _INTERNAL_ERROR,
    AgentLoop,
    AgentRunFailed,
    AgentRunResult,
    ToolCallRecord,
)
from app.ai.provider import ToolCall, ToolSpec
from app.ai.providers.llamacpp import LlamaCppProvider
from app.ai.run_record import NOT_RUN_PREFIX
from app.ai.trace import KIND_RUN

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


class _ReplayLoop(AgentLoop):
    """The live loop, with the tool answers, prompt and tool list from the capture."""

    def __init__(
        self,
        provider: LlamaCppProvider,
        *,
        record: dict[str, Any],
        system_prompt: str,
        specs: list[ToolSpec],
    ) -> None:
        limits = record.get("limits") or {}
        super().__init__(
            provider,
            max_iterations=int(limits.get("max_iterations", 6)),
            max_corrections=int(limits.get("max_corrections", 3)),
            max_tokens=int(limits.get("max_tokens", 2048)),
            max_tool_calls=int(limits.get("max_tool_calls", 12)),
        )
        self._captured_prompt = system_prompt
        self._captured_specs = specs
        # Only calls that were dispatched answer from the record; the loop
        # writes the "not run" ones itself.
        self._answers = [
            ToolCallRecord.model_validate(tool)
            for tool in record.get("tools") or []
            if not (tool.get("error") or "").startswith(NOT_RUN_PREFIX)
        ]

    def _tool_specs(self) -> list[ToolSpec]:
        return self._captured_specs

    def _system_prompt(self) -> str:
        return self._captured_prompt

    def _dispatch(self, call: ToolCall, actor: str) -> tuple[ToolCallRecord, str, bool]:
        if not self._answers:
            raise ReplayError(f"the trace has no recorded answer for {call.name}")
        recorded = self._answers.pop(0)
        if recorded.tool != call.name or recorded.arguments != call.arguments:
            raise ReplayError(
                f"replay asked for {call.name}({call.arguments}); the turn ran "
                f"{recorded.tool}({recorded.arguments})"
            )
        if recorded.error == _INTERNAL_ERROR:
            raise RuntimeError("replayed crash")
        if recorded.error is not None:
            schema = recorded.error.startswith(("Invalid arguments: ", "Unknown tool "))
            return recorded, f"Error: {recorded.error}", schema
        return recorded, recorded.result or "", False


def replay(trace_path: Path, context_path: Path, run_id: str) -> ReplayReport:
    """Replay one captured turn and report whether today's loop reproduces it."""
    record = load_trace(trace_path, run_id)
    captures = read_captures(context_path, run_id)
    if not captures or captures[0].get("request") is None:
        raise ReplayError(f"no captured model calls for run {run_id} in {context_path}")
    report = ReplayReport(run_id=run_id, calls_captured=len(captures))
    first_request = json.loads(captures[0]["request"])
    system_prompt = str(first_request["messages"][0]["content"])
    specs = [
        ToolSpec(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            parameters=tool["function"]["parameters"],
        )
        for tool in first_request.get("tools") or []
    ]

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
    loop = _ReplayLoop(provider, record=record, system_prompt=system_prompt, specs=specs)
    try:
        result: AgentRunResult = loop.run(
            str(record["utterance"]),
            history=record.get("history") or [],
            actor=str(record.get("actor") or "agent:loop"),
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
