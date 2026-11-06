"""Print conductor's turn traces (#62): turn → model calls → hops → app request ids.

    python scripts/read_trace.py <trace.jsonl> [run_id] [--last N]

With a run id, prints that turn; otherwise the last N turns (default 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _turns(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _show(turn: dict[str, Any]) -> None:
    print(
        f"turn {turn['run_id']}  conv={turn.get('conversation_id')}  "
        f"{turn['stop_reason']}  {turn['total_ms']} ms  "
        f"tokens {turn['prompt_tokens']}+{turn['completion_tokens']}"
    )
    print(f"  user: {turn['utterance']}")
    for call in turn.get("calls", []):
        print(
            f"  model #{call['seq']}: {call['status']} {call['ms']} ms "
            f"finish={call['finish_reason']} tool_calls={call['tool_calls']}"
        )
    for hop in turn.get("hops", []):
        outcome = hop.get("error") or hop.get("app_stop_reason")
        print(
            f"  hop {hop['hop_id']} → {hop['app']} thread={hop['thread_id']} "
            f"app_request={hop['app_request_id']} {outcome} {hop['ms']} ms "
            f"app ran {hop.get('app_tools') or []}"
        )
    print(f"  reply: {(turn.get('reply') or '')[:200]}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    turns = _turns(Path(argv[0]))
    rest = argv[1:]
    if rest and not rest[0].startswith("--"):
        matches = [turn for turn in turns if turn.get("run_id") == rest[0]]
        for turn in matches:
            _show(turn)
        return 0 if matches else 1
    last = int(rest[1]) if len(rest) == 2 and rest[0] == "--last" else 5
    for turn in turns[-last:]:
        _show(turn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
