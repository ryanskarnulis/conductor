"""Replay one traced turn offline against today's code (#62).

    python scripts/replay_turn.py <trace.jsonl> <context.jsonl> <run_id>

Needs both files from a turn run with CONDUCTOR_TRACE_PATH and
CONDUCTOR_CONTEXT_PATH set; the planner prompt is built from today's fleet
(FLEET_MANIFEST_DIR). No model and no apps are called. Exits 0 when the
replay reproduces the turn (same requests, stop, reply and tool calls), 1 when
it diverges.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.ai.replay import ReplayError, replay
from app.fleet.manifests import fleet_from_settings


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    trace_path, context_path, run_id = Path(argv[0]), Path(argv[1]), argv[2]
    try:
        report = replay(trace_path, context_path, run_id, fleet_from_settings())
    except ReplayError as exc:
        print(f"cannot replay: {exc}", file=sys.stderr)
        return 2
    print(f"run {report.run_id}: {report.calls_replayed}/{report.calls_captured} model calls")
    if report.first_request_divergence is not None:
        print(f"  request {report.first_request_divergence} differs: {report.divergence_detail}")
    print(f"  stop: captured {report.stop_reason[0]}, replayed {report.stop_reason[1]}")
    print(f"  reply matches: {report.reply_matches}; tool calls match: {report.tools_match}")
    print("FAITHFUL" if report.faithful else "DIVERGED")
    return 0 if report.faithful else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
