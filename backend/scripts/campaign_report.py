"""Join an eval campaign's blocks into one paired table (#53).

    python scripts/campaign_report.py <campaign-out-dir>

Reads ``block-<n>-<arm>.jsonl`` (one ``evalstats.scenario_record`` per
scenario per block, each with its scored ``samples``: seed and verdict) and
prints, per scenario, each arm's pooled count and the paired verdict: samples
of the two arms with the same seed are a pair, and the exact McNemar test
reads the pairs the change flipped. Unseeded samples are counted, never
paired.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.evalstats import format_paired, paired_counts, wilson_interval  # noqa: E402

_BLOCK = re.compile(r"block-(\d+)-([ab])\.jsonl$")


def load(out: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """scenario → arm → samples (each tagged with its block)."""
    joined: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(out.glob("block-*-*.jsonl")):
        match = _BLOCK.search(path.name)
        if not match:
            continue
        block, arm = int(match.group(1)), match.group(2)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for sample in record.get("samples", []):
                joined[record["scenario"]][arm].append({**sample, "block": block})
    return joined


def table(joined: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    lines = [f"{'scenario':<24} {'A':>12} {'B':>12}  paired"]
    for scenario in sorted(joined):
        arms = joined[scenario]
        cells = []
        for arm in ("a", "b"):
            samples = arms.get(arm, [])
            passed = sum(1 for s in samples if s["pass"])
            low, high = wilson_interval(passed, len(samples)) if samples else (0.0, 1.0)
            cells.append(f"{passed}/{len(samples)} [{low:.2f},{high:.2f}]")
        by_seed = {
            arm: {s["seed"]: s["pass"] for s in arms.get(arm, []) if s.get("seed") is not None}
            for arm in ("a", "b")
        }
        seeds = sorted(set(by_seed["a"]) & set(by_seed["b"]))
        if seeds:
            _, a_only, b_only, _ = paired_counts(
                [by_seed["a"][s] for s in seeds], [by_seed["b"][s] for s in seeds]
            )
            paired = format_paired(a_only, b_only, len(seeds))
        else:
            paired = "unpaired (no shared seeds)"
        lines.append(f"{scenario:<24} {cells[0]:>12} {cells[1]:>12}  {paired}")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    joined = load(Path(argv[1]))
    if not joined:
        print("no block records found", file=sys.stderr)
        return 1
    print("\n".join(table(joined)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
