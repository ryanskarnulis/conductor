"""The eval harness's infra classifier and budget, run on every commit (#25)."""

from __future__ import annotations

import pytest

from tests.evalstats import InfraBudget, Outcome, classify


@pytest.mark.parametrize(
    ("status_code", "detail", "expected"),
    [
        (200, None, Outcome.SCORED),
        # The #25 failure: llama-swap reloading the model mid-suite.
        (502, "agent run failed: llama-server request failed: timed out", Outcome.INFRA),
        (
            502,
            "agent run failed: llama-server request failed: [Errno 111] Connection refused",
            Outcome.INFRA,
        ),
        # llama-swap answers 5xx while it swaps.
        (502, "agent run failed: llama-server returned 503: Loading model", Outcome.INFRA),
        (502, "agent run failed: llama-server returned 500: oops", Outcome.INFRA),
        # A death that doesn't name itself is treated as transient.
        (502, "agent run failed: something new", Outcome.INFRA),
        (502, None, Outcome.INFRA),
        # Deterministic refusals: same request, same answer.
        (
            502,
            "agent run failed: llama-server returned 400: context overrun",
            Outcome.PROVIDER_REJECTED,
        ),
        (
            502,
            "agent run failed: llama-server response failed wire validation: …",
            Outcome.PROVIDER_REJECTED,
        ),
        # Anything else is conductor's own bug, never a retry.
        (500, "Internal Server Error", Outcome.HARNESS),
        (422, "validation error", Outcome.HARNESS),
    ],
)
def test_classify(status_code: int, detail: str | None, expected: Outcome) -> None:
    assert classify(status_code=status_code, detail=detail) is expected


def test_budget_caps_retries_per_scenario() -> None:
    budget = InfraBudget(retries=2, budget=10)
    assert budget.spend("a")
    assert budget.spend("a")
    assert not budget.spend("a")
    # Another scenario still has its own allowance.
    assert budget.spend("b")


def test_budget_caps_retries_across_the_suite() -> None:
    budget = InfraBudget(retries=5, budget=3)
    assert budget.spend("a")
    assert budget.spend("b")
    assert budget.spend("c")
    assert not budget.spend("d")


def test_summary_says_where_the_budget_went() -> None:
    budget = InfraBudget(retries=2, budget=6)
    assert budget.summary() == "infra_spent=0/6"
    budget.spend("chess-reset")
    budget.spend("chess-reset")
    budget.spend("tasks-due")
    assert budget.summary() == "infra_spent=3/6 (chess-reset=2, tasks-due=1)"
