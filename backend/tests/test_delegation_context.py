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


@pytest.fixture(autouse=True)
def _clear_process_default() -> Generator[None, None, None]:
    set_process_delegation_context(None)
    try:
        yield
    finally:
        set_process_delegation_context(None)


def _context() -> DelegationContext:
    return DelegationContext(
        master_conversation_id="m1",
        thread_store=InMemoryThreadStore(),
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
