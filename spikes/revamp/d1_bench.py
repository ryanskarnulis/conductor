"""D1: can the planner compile typed plans? One call per sample.

    python d1_bench.py ARM [samples] [group,group...]
    python d1_bench.py --report

Arms: 12b, 12b-t03 (temperature 0.3), 26b, 26b-think (no json_schema: the
grammar would also bind the thought, so JSON is parsed from the content).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from pydantic import ValidationError

from common import OUT, append, chat, load, pct, wilson
from d1_cases import CASES, HOLDOUT
from plan_schema import PLANNER_PROMPT, SCHEMA, Plan, user_message, validate
from plan_v2 import PROMPT2, SCHEMA2, Plan2, validate2

# v2c: v2 with the chess glance line no longer promising "who's ahead".
PROMPT2C = PROMPT2.replace(
    "the current game (whose move, who's ahead) and the last finished game's result",
    "where the current game stands (whose move, move count) and the last finished "
    "game's result; it doesn't analyze positions",
)
assert PROMPT2C != PROMPT2

ALL = {c.id: c for c in CASES + HOLDOUT}

ARMS = {
    "12b": {"model": "gemma-4-12b"},
    "12b-t03": {"model": "gemma-4-12b", "temperature": 0.3},
    "26b": {"model": "gemma-4-26b-a4b"},
    "26b-think": {"model": "gemma-4-26b-a4b", "thinking": True, "free": True},
    "12b-v2": {"model": "gemma-4-12b", "v2": True},
    "12b-t03-v2": {"model": "gemma-4-12b", "temperature": 0.3, "v2": True},
    "26b-v2": {"model": "gemma-4-26b-a4b", "v2": True},
    "26b-think-v2": {"model": "gemma-4-26b-a4b", "thinking": True, "free": True, "v2": True},
    "12b-v2c": {"model": "gemma-4-12b", "v2": True, "prompt": "v2c"},
    "26b-think-v2s": {"model": "gemma-4-26b-a4b", "thinking": True, "v2": True},
}
PATH = OUT / "d1.jsonl"


def run(arm: str, samples: int, groups: set[str] | None) -> None:
    spec = ARMS[arm]
    v2 = spec.get("v2", False)
    prompt, schema = (PROMPT2, SCHEMA2) if v2 else (PLANNER_PROMPT, SCHEMA)
    if spec.get("prompt") == "v2c":
        prompt = PROMPT2C
    done = {(r["case"], r["sample"]) for r in load(PATH) if r["arm"] == arm}
    # Warm the model and the cached prefix so the first sample isn't a load.
    chat(spec["model"], [{"role": "system", "content": prompt},
                         {"role": "user", "content": "warm-up"}], max_tokens=1)
    for sample in range(samples):
        for case in CASES + HOLDOUT:
            if groups and case.group not in groups:
                continue
            if (case.id, sample) in done:
                continue
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message(case.text, case.state or _empty())},
            ]
            free = spec.get("free", False)
            reply, _ = chat(
                spec["model"],
                messages,
                schema=None if free else schema,
                thinking=spec.get("thinking", False),
                temperature=spec.get("temperature"),
                max_tokens=6000 if spec.get("thinking") else 700,
            )
            row = {"arm": arm, "case": case.id, "group": case.group, "sample": sample,
                   "seconds": round(reply.seconds, 2), "out_tokens": reply.completion_tokens,
                   "raw": reply.content, "think_chars": len(reply.reasoning)}
            row.update(grade(case, reply.content, v2))
            append(PATH, row)
            mark = "ok " if row["correct"] else ("inv" if not row["valid"] else "BAD")
            print(f"{arm} {sample} {mark} {reply.seconds:5.2f}s {case.id}: {_short(row)}",
                  flush=True)


def _empty() -> str:
    from d1_cases import state  # noqa: PLC0415

    return state()


def _short(row: dict) -> str:
    return row.get("plan_short") or row.get("error", "")[:120]


def grade(case, content: str, v2: bool = False) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        plan = (Plan2 if v2 else Plan).model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        return {"parsed": False, "valid": False, "correct": False, "error": str(exc)[:300]}
    if v2:
        problems = validate2(plan, set(case.refs))
        say = getattr(plan.root, "answer", "result")
        steps = getattr(plan.root, "steps", [])
    else:
        problems = validate(plan, set(case.refs))
        say, steps = plan.say, plan.steps
    short = f"say={say} " + " | ".join(
        f"{s.do}:{getattr(s, 'app', '')}:{json.dumps(s.model_dump(exclude={'id', 'do', 'app'}))}"
        for s in steps
    )
    correct = not problems and any(shape.matches(plan) for shape in case.expect)
    return {"parsed": True, "valid": not problems, "problems": problems, "correct": correct,
            "plan_short": short[:400], "steps": len(steps)}


def regrade() -> None:
    rows = load(PATH)
    for row in rows:
        row.update(grade(ALL[row["case"]], row["raw"], ARMS[row["arm"]].get("v2", False)))
        row["group"] = ALL[row["case"]].group
    PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))


def report() -> None:
    rows = load(PATH)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    for arm, arm_rows in by_arm.items():
        print(f"\n=== {arm}: {len(arm_rows)} samples")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in arm_rows:
            groups[row["group"]].append(row)
        for group, group_rows in sorted(groups.items()):
            n = len(group_rows)
            ok = sum(r["correct"] for r in group_rows)
            valid = sum(r["valid"] for r in group_rows)
            lo, hi = wilson(ok, n)
            print(f"  {group:9} correct {ok}/{n} ({ok / n:.0%}, ci [{lo:.2f}, {hi:.2f}])"
                  f"  valid {valid}/{n}")
        n = len(arm_rows)
        ok = sum(r["correct"] for r in arm_rows)
        valid = sum(r["valid"] for r in arm_rows)
        parsed = sum(r["parsed"] for r in arm_rows)
        lo, _ = wilson(ok, n)
        one_step = [r["seconds"] for r in arm_rows if r.get("steps") == 1]
        all_secs = [r["seconds"] for r in arm_rows]
        print(f"  ALL       correct {ok}/{n} ({ok / n:.0%}, lower {lo:.2f})  valid {valid}/{n}"
              f"  parsed {parsed}/{n}")
        print(f"  latency   one-step p50 {pct(one_step, .5):.2f}s p90 {pct(one_step, .9):.2f}s;"
              f" all p50 {pct(all_secs, .5):.2f}s p90 {pct(all_secs, .9):.2f}s")
        per_case: dict[str, list[dict]] = defaultdict(list)
        for row in arm_rows:
            per_case[row["case"]].append(row)
        weak = [(c, sum(r["correct"] for r in rs), len(rs)) for c, rs in per_case.items()
                if sum(r["correct"] for r in rs) < len(rs)]
        print("  misses    " + ", ".join(f"{c} {k}/{n}" for c, k, n in sorted(weak)))


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    elif sys.argv[1] == "--regrade":
        regrade()
    else:
        arm = sys.argv[1]
        samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        groups = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
        run(arm, samples, groups)
