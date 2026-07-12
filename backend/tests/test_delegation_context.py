"""DelegationContext internals: thread store seam, budget, and binding."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from app.fleet.context import (
    DelegationContext,
    InMemoryThreadStore,
    NoDelegationContextError,
    current_delegation_context,
    set_process_delegation_context,
    use_delegation_context,
)
from app.tools.registry import ToolError


@pytest.fixture(autouse=True)
def _clear_process_default() -> Generator[None, None, None]:
    set_process_delegation_context(None)
    try:
        yield
    finally:
        set_process_delegation_context(None)


def _context(budget: int = 3) -> DelegationContext:
    return DelegationContext(
        master_conversation_id="m1",
        thread_store=InMemoryThreadStore(),
        calls_per_turn_per_app=budget,
    )


def test_thread_store_is_keyed_by_master_and_app() -> None:
    store = InMemoryThreadStore()
    store.set("m1", "chess", 5)
    store.set("m2", "chess", 9)
    assert store.get("m1", "chess") == 5
    assert store.get("m2", "chess") == 9
    assert store.get("m1", "tasks") is None
    store.forget("m1", "chess")
    assert store.get("m1", "chess") is None


def test_charge_call_enforces_the_per_turn_budget() -> None:
    context = _context(budget=2)
    context.charge_call("chess")
    context.charge_call("chess")
    with pytest.raises(ToolError, match="per-turn limit"):
        context.charge_call("chess")
    # A different app has its own budget.
    context.charge_call("tasks")


def test_reset_turn_clears_the_counters() -> None:
    context = _context(budget=1)
    context.charge_call("chess")
    context.reset_turn()
    context.charge_call("chess")  # allowed again after reset


def test_zero_budget_never_charges() -> None:
    context = _context(budget=0)
    for _ in range(50):
        context.charge_call("chess")


def test_no_bound_context_raises() -> None:
    with pytest.raises(NoDelegationContextError):
        current_delegation_context()


def test_bound_context_wins_over_process_default() -> None:
    process = _context()
    request_scoped = _context()
    set_process_delegation_context(process)

    assert current_delegation_context() is process
    with use_delegation_context(request_scoped):
        assert current_delegation_context() is request_scoped
    assert current_delegation_context() is process
