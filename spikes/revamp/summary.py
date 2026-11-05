"""Spike summary tables for #56."""
import json
from collections import defaultdict

from common import load, OUT, pct, wilson

rows = load(OUT / "d1.jsonl")
SUBSET = {"cross", "ref", "control", "memory", "hard", "self", "holdout"}
arms = ["12b", "12b-t03", "12b-v2", "12b-t03-v2", "12b-v2c", "26b-v2", "26b-think-v2"]
groups = ["golden", "cross", "ref", "control", "memory", "knowledge", "hard", "self", "holdout"]
print("| arm | " + " | ".join(groups) + " | all | valid | p50 s |")
print("|---" * (len(groups) + 4) + "|")
for arm in arms:
    xs = [r for r in rows if r["arm"] == arm]
    if not xs:
        continue
    cells = []
    for g in groups:
        gs = [r for r in xs if r["group"] == g]
        cells.append(f"{sum(r['correct'] for r in gs)}/{len(gs)}" if gs else "–")
    ok = sum(r["correct"] for r in xs)
    lo, _ = wilson(ok, len(xs))
    valid = sum(r["valid"] for r in xs)
    secs = [r["seconds"] for r in xs]
    print(f"| {arm} | " + " | ".join(cells)
          + f" | {ok}/{len(xs)} ({ok/len(xs):.0%}, lo {lo:.2f}) | {valid}/{len(xs)} | {pct(secs, .5):.2f} |")
print()
print("Same subset as the thinking arm (cross, ref, control, memory, hard, self, holdout):")
for arm in ["12b-v2", "12b-t03-v2", "26b-v2", "26b-think-v2"]:
    xs = [r for r in rows if r["arm"] == arm and r["group"] in SUBSET]
    if xs:
        ok = sum(r["correct"] for r in xs)
        secs = [r["seconds"] for r in xs]
        print(f"- {arm}: {ok}/{len(xs)} ({ok/len(xs):.0%}); p50 {pct(secs, .5):.2f} s, p90 {pct(secs, .9):.2f} s")
