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


def _cached(cached: int | None, prompt: int | None) -> str:
    """`` (1200 cached, 97%)``; nothing when the record has no cache count (an
    older turn) or no prompt. A 0 is a measured miss and is shown."""
    if cached is None or not prompt:
        return ""
    return f" ({cached} cached, {cached / prompt:.0%})"


def _prompt(call: dict[str, Any]) -> str:
    """One call's prompt: tokens, how many the cache served, prefill ms. Each
    part appears only when the record has it (#152)."""
    parts = []
    if call.get("prompt_tokens") is not None:
        parts.append(f"prompt={call['prompt_tokens']}")
    if call.get("cached_tokens") is not None:
        parts.append(f"cached={call['cached_tokens']}")
    if call.get("prefill_ms") is not None:
        parts.append(f"prefill={call['prefill_ms']:.0f} ms")
    return f" {' '.join(parts)}" if parts else ""


def _show(turn: dict[str, Any]) -> None:
    print(
        f"turn {turn['run_id']}  conv={turn.get('conversation_id')}  "
        f"{turn['stop_reason']}  {turn['total_ms']} ms  "
        f"tokens {turn['prompt_tokens']}+{turn['completion_tokens']}"
        f"{_cached(turn.get('cached_tokens'), turn['prompt_tokens'])}"
    )
    print(f"  user: {turn['utterance']}")
    for call in turn.get("calls", []):
        print(
            f"  model #{call['seq']}: {call['status']} {call['ms']} ms "
            f"finish={call['finish_reason']} tool_calls={call['tool_calls']}"
            f"{_prompt(call)}"
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
