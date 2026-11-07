"""Plan-level evals (#70): does the 12B compile the right plan? (opt-in, the real model)

Spike D1's bench in the repo (``plan_cases.py``): every routing golden as a
plan, the S1–S16 scenarios, the held-out set, and the long-plan cases. Each
sample is one compile through the shipped compiler (prompt, schema, validator
and its one repair) against the real llama-server, with the real workspace
manifests; nothing runs. A sample passes when the plan is valid and matches one
of the case's shapes.

Gated per group under the vendored Wilson rule (``evalstats.decide``): red
takes evidence (the upper bound below the floor). Floors are set from the
spike's 12B v2 run (#56, results in the master issue), a little under each
group's observed rate; ``long`` from this suite's first run.

    cd backend
    CONDUCTOR_PLAN_EVALS=1 .venv/bin/pytest tests/test_plan_evals.py -v -s

``CONDUCTOR_PLAN_SAMPLES`` (default 5) samples each case; ``-k <group>`` runs
one group; ``CONDUCTOR_EVAL_REPORT=path`` appends one JSON line per sample.
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
from tests.plan_cases import CASES, HOLDOUT, LONG, Case, empty_state

pytestmark = pytest.mark.skipif(
    os.environ.get("CONDUCTOR_PLAN_EVALS") != "1",
    reason="plan evals run the real model: set CONDUCTOR_PLAN_EVALS=1",
)

WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)
SAMPLES = int(os.environ.get("CONDUCTOR_PLAN_SAMPLES", "5"))
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
    "holdout": 0.85,
    "long": 0.7,
}

ALL: tuple[Case, ...] = CASES + HOLDOUT + LONG


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

    try:
        compiled = compiler.compile_turn(
            chat,
            system_prompt=prompt,
            context=case.state or empty_state(),
            words=case.text,
            fleet=fleet,
            checks=plans.Checks(
                fleet=fleet, confirmable=case.confirmable, context=case.state or empty_state()
            ),
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
        "repaired": compiled.repaired,
        "steps": len(plan.steps) if isinstance(plan, plans.StepPlan) else 0,
        "plan": plans.describe(plan) if plan is not None else None,
        "problems": compiled.problems,
        "seconds": round(seconds, 2),
        "raw": compiled.raw,
    }


@pytest.mark.parametrize("group", list(FLOORS))
def test_plan_group(group: str, fleet: Fleet) -> None:
    prompt = compiler.planner_prompt(fleet)
    cases = [case for case in ALL if case.group == group]
    assert cases, group
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
        print(f"[plan-eval] {group} {case.id}: {''.join(verdicts)}{miss}")
    passed = sum(row["correct"] for row in rows)
    runs = len(rows)
    floor = FLOORS[group]
    low, high = wilson_interval(passed, runs)
    seconds = [row["seconds"] for row in rows]
    print(
        f"[plan-eval] GROUP {group}: {passed}/{runs} ({passed / runs:.0%}, Wilson "
        f"{low:.2f}–{high:.2f}, floor {floor}) valid {sum(r['valid'] for r in rows)}/{runs} "
        f"first-try {sum(r['first_valid'] for r in rows)}/{runs} "
        f"p50 {statistics.median(seconds):.2f}s max steps {max(r['steps'] for r in rows)}"
    )
    assert decide(passed, runs, floor) is not Decision.BELOW_FLOOR, (
        f"{group} is below its floor with evidence: {passed}/{runs} vs {floor}"
    )
