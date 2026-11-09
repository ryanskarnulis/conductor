"""Plan-level evals (#70): does the 12B compile the right plan? (opt-in, the real model)

Spike D1's bench in the repo (``plan_cases.py``): every routing golden as a
plan, the S1–S16 scenarios, D1's old held-out set (now the ``mixed`` group,
#142) and the long-plan cases. Each sample is one compile as the interpreter
does it (prompt, schema, validator, its repair, and since #75 one more round
when the plan still can't run) against the real llama-server, with the real
workspace manifests; nothing runs. Since #75 (per #59) this suite and the
session evals are the eval gate. A sample passes when the plan is valid and
matches one of the case's shapes. Each group also reports its first-try rate:
samples right on the planner's first reply, with no repair (#142).

Gated per group under the vendored Wilson rule (``evalstats.decide``): red
takes evidence (the upper bound below the floor). Floors are set from the
spike's 12B v2 run (#56, results in the master issue), a little under each
group's observed rate; ``long`` from this suite's first run.

    cd backend
    CONDUCTOR_PLAN_EVALS=1 .venv/bin/pytest tests/test_plan_evals.py -v -s

``CONDUCTOR_PLAN_SAMPLES`` (default 20, #142) samples each case; ``-k <group>``
runs one group; ``CONDUCTOR_EVAL_REPORT=path`` appends one JSON line per sample.

The held-out set (``held_out_cases.py``) runs only at milestones and is never
tuned against: ``CONDUCTOR_HELDOUT_EVALS=1`` runs it, gated as one pool.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from app.ai import compiler
from app.ai import plan as plans
from app.ai.provider import ChatResult, ProviderError
from app.ai.providers.llamacpp import provider_from_settings
from app.fleet.manifests import Fleet, GlanceSpec, discover_fleet
from tests.evalstats import Decision, decide, wilson_interval
from tests.held_out_cases import HELD_OUT, KINDS
from tests.plan_cases import CASES, LONG, MIXED, Case, empty_state

plan_evals = pytest.mark.skipif(
    os.environ.get("CONDUCTOR_PLAN_EVALS") != "1",
    reason="plan evals run the real model: set CONDUCTOR_PLAN_EVALS=1",
)
held_out_evals = pytest.mark.skipif(
    os.environ.get("CONDUCTOR_HELDOUT_EVALS") != "1",
    reason="the held-out set runs only at milestones: set CONDUCTOR_HELDOUT_EVALS=1",
)

WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)
# n=5 was too noisy to gate on (#142: chess-resign read 5/20 to 15/20 on one day).
SAMPLES = int(os.environ.get("CONDUCTOR_PLAN_SAMPLES", "20"))
_REPORT = os.environ.get("CONDUCTOR_EVAL_REPORT")
_INFRA_RETRIES = 3

# Per group, a little under the spike's 12B v2 rate (golden 89/105, cross 32/35,
# ref 20/20, control 23/25, memory 10/10, knowledge 12/15, hard 4/5, self
# 10/15, held-out 46/50). Cross keeps the master plan's 80% bar.
FLOORS: dict[str, float] = {
    "golden": 0.8,
    "cross": 0.8,
    "ref": 0.85,
    "control": 0.8,
    "memory": 0.8,
    "knowledge": 0.65,
    "hard": 0.5,
    "self": 0.55,
    # D1's old held-out set (#142): tuned against since 2026-11-06, so ordinary.
    "mixed": 0.85,
    "long": 0.7,
    # A question about now over a recalled answer (#127): it reads the app.
    "stale": 0.8,
}

ALL: tuple[Case, ...] = CASES + MIXED + LONG

# The held-out set's one pooled floor, fixed before its first run (#142).
HELD_OUT_FLOOR = 0.8


@pytest.fixture(scope="module")
def fleet() -> Fleet:
    found = discover_fleet(WORKSPACE_ROOT)
    names = {app.name for app in found.apps}
    if not {"tasks", "music", "chess"} <= names:
        pytest.skip(
            f"fleet discovery found {sorted(names)}; is {WORKSPACE_ROOT} the workspace root? "
            "From a worktree, set CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace"
        )
    return _with_glances(found)


# The glance lines the planner prompt carried as a placeholder until apps
# declared their own (#73). An app whose manifest has no `glance:` yet (PCC#504,
# music#82, chess#536 are building them) gets its line here, so the plan cases
# that expect a `look` keep measuring; a manifest that declares one wins.
EVAL_GLANCES: dict[str, str] = {
    "tasks": "today's and overdue tasks, and today's Focus",
    "music": "running and recent download jobs with progress, and how many songs wait to be sorted",
    "chess": (
        "where the current game stands (whose move, move count) and the last finished "
        "game's result; it doesn't analyze positions"
    ),
}


def _with_glances(found: Fleet) -> Fleet:
    return Fleet(
        apps=tuple(
            dataclasses.replace(app, glance=GlanceSpec(holds=EVAL_GLANCES[app.name]))
            if app.glance is None and app.name in EVAL_GLANCES
            else app
            for app in found.apps
        )
    )


def _compile(case: Case, fleet: Fleet, prompt: str) -> tuple[compiler.Compiled, float]:
    provider = provider_from_settings()
    calls: list[float] = []

    def chat(
        messages: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any]
    ) -> ChatResult:
        started = time.monotonic()
        try:
            return provider.chat(messages, max_tokens=max_tokens, response_format=response_format)
        finally:
            calls.append(time.monotonic() - started)

    state = case.state or empty_state()
    checks = plans.Checks(fleet=fleet, confirmable=case.confirmable, context=state)
    try:
        compiled = compiler.compile_turn(
            chat, system_prompt=prompt, context=state, words=case.text, fleet=fleet, checks=checks
        )
        # The interpreter's one more round (#75), as shipped.
        found = compiled.problems if compiled.plan is None else plans.unrunnable(compiled.plan)
        if found:
            compiled = compiler.repair_turn(
                chat,
                system_prompt=prompt,
                context=state,
                words=case.text,
                fleet=fleet,
                checks=checks,
                previous=compiled,
                found=found,
            )
    finally:
        provider.close()
    return compiled, calls[0] if calls else 0.0


def _sample(case: Case, fleet: Fleet, prompt: str) -> dict[str, Any]:
    for attempt in range(_INFRA_RETRIES + 1):
        try:
            compiled, seconds = _compile(case, fleet, prompt)
            break
        except ProviderError as exc:
            if attempt == _INFRA_RETRIES:
                pytest.fail(f"{case.id}: llama-server kept failing: {exc}")
            print(f"[plan-eval]   ⟳ infra ({case.id}): {exc}")
    plan = compiled.plan
    correct = plan is not None and any(shape.matches(plan) for shape in case.expect)
    first_valid = not compiled.repaired and plan is not None
    return {
        "case": case.id,
        "group": case.group,
        "correct": correct,
        "valid": plan is not None,
        "first_valid": first_valid,
        # Right on the planner's first reply: no repair, nudge or extra round.
        "first_right": correct and first_valid,
        "repaired": compiled.repaired,
        # Planner calls: 1; 2 after the compiler's repair; 3 after #75's round.
        "calls": len(compiled.raw),
        "steps": len(plan.steps) if isinstance(plan, plans.StepPlan) else 0,
        "plan": plans.describe(plan) if plan is not None else None,
        "problems": compiled.problems,
        "seconds": round(seconds, 2),
        "raw": compiled.raw,
    }


@dataclasses.dataclass(frozen=True)
class Summary:
    """One group's samples, counted: the score, and the first-try rate beside it."""

    label: str
    passed: int
    runs: int
    first_right: int
    valid: int
    first_valid: int
    third_call: int
    right_within_two: int
    p50_seconds: float
    max_steps: int

    @classmethod
    def of(cls, label: str, rows: list[dict[str, Any]]) -> Summary:
        return cls(
            label=label,
            passed=sum(row["correct"] for row in rows),
            runs=len(rows),
            first_right=sum(row["first_right"] for row in rows),
            valid=sum(row["valid"] for row in rows),
            first_valid=sum(row["first_valid"] for row in rows),
            third_call=sum(row["calls"] >= 3 for row in rows),
            right_within_two=sum(row["correct"] and row["calls"] <= 2 for row in rows),
            p50_seconds=statistics.median(row["seconds"] for row in rows),
            max_steps=max(row["steps"] for row in rows),
        )

    def line(self, floor: float) -> str:
        low, high = wilson_interval(self.passed, self.runs)
        return (
            f"[plan-eval] GROUP {self.label}: {self.passed}/{self.runs} "
            f"({self.passed / self.runs:.0%}, Wilson {low:.2f}–{high:.2f}, floor {floor}) "
            f"first-try {self.first_right}/{self.runs} ({self.first_right / self.runs:.0%}) "
            f"valid {self.valid}/{self.runs} first-valid {self.first_valid}/{self.runs} "
            f"third-call {self.third_call}/{self.runs} "
            f"right-within-two {self.right_within_two}/{self.runs} "
            f"p50 {self.p50_seconds:.2f}s max steps {self.max_steps}"
        )

    def verdict(self, floor: float) -> str | None:
        """Why the group is red, or None: the Wilson rule on the final score."""
        if decide(self.passed, self.runs, floor) is not Decision.BELOW_FLOOR:
            return None
        return (
            f"{self.label} is below its floor with evidence: {self.passed}/{self.runs} vs "
            f"{floor} (first-try {self.first_right}/{self.runs})"
        )


def _run(cases: list[Case], fleet: Fleet, prompt: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        verdicts = []
        for _ in range(SAMPLES):
            row = _sample(case, fleet, prompt)
            rows.append(row)
            verdicts.append("✓" if row["correct"] else ("✗" if row["valid"] else "∅"))
            if _REPORT:
                with open(_REPORT, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
        misses = Counter(
            row["plan"] or "invalid"
            for row in rows
            if row["case"] == case.id and not row["correct"]
        )
        miss = f"  misses: {dict(misses)}" if misses else ""
        print(f"[plan-eval] {label} {case.id}: {''.join(verdicts)}{miss}")
    return rows


@plan_evals
@pytest.mark.parametrize("group", list(FLOORS))
def test_plan_group(group: str, fleet: Fleet) -> None:
    prompt = compiler.planner_prompt(fleet)
    cases = [case for case in ALL if case.group == group]
    assert cases, group
    summary = Summary.of(group, _run(cases, fleet, prompt, group))
    print(summary.line(FLOORS[group]))
    red = summary.verdict(FLOORS[group])
    assert red is None, red


@held_out_evals
def test_held_out(fleet: Fleet) -> None:
    """Milestones only; never tuned against (#142). Read per kind, gated as one pool."""
    prompt = compiler.planner_prompt(fleet)
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        cases = [case for case in HELD_OUT if case.group == kind]
        kind_rows = _run(cases, fleet, prompt, f"held-out {kind}")
        print(Summary.of(f"held-out {kind}", kind_rows).line(HELD_OUT_FLOOR))
        rows += kind_rows
    summary = Summary.of("held-out", rows)
    print(summary.line(HELD_OUT_FLOOR))
    red = summary.verdict(HELD_OUT_FLOOR)
    assert red is None, red


# --- the summary itself (no model: runs in the ordinary suite) -----------------------------


def _row(
    *, correct: bool, valid: bool = True, repaired: bool = False, calls: int = 1
) -> dict[str, Any]:
    first_valid = valid and not repaired
    return {
        "correct": correct,
        "valid": valid,
        "first_valid": first_valid,
        "first_right": correct and first_valid,
        "repaired": repaired,
        "calls": calls,
        "seconds": 0.5,
        "steps": 1,
    }


def test_a_repaired_plan_counts_in_the_score_but_not_the_first_try_rate() -> None:
    rows = [
        _row(correct=True),
        _row(correct=True, repaired=True, calls=2),
        _row(correct=True, repaired=True, calls=3),
        _row(correct=False),
        _row(correct=False, valid=False, repaired=True, calls=3),
    ]
    summary = Summary.of("self", rows)
    assert (summary.passed, summary.first_right, summary.runs) == (3, 1, 5)
    assert (summary.valid, summary.first_valid) == (4, 2)
    assert (summary.third_call, summary.right_within_two) == (2, 2)
    line = summary.line(0.55)
    assert "GROUP self: 3/5" in line
    assert "first-try 1/5 (20%)" in line


def test_the_gate_reads_the_final_score_and_names_the_first_try_rate() -> None:
    nudged = [_row(correct=True, repaired=True, calls=2)] * 20
    assert Summary.of("self", nudged).verdict(0.55) is None
    missed = [_row(correct=False)] * 20
    red = Summary.of("self", missed).verdict(0.55)
    assert red is not None and "first-try 0/20" in red
