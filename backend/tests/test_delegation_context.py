"""DelegationContext internals: thread store seam, budget, and binding."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from app.fleet.context import (
    DelegationContext,
    InMemoryThreadStore,
    NoDelegationContextError,
    answering,
    answering_thread,
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


def test_opened_threads_are_recorded_under_the_window_and_used() -> None:
    store = InMemoryThreadStore()
    context = DelegationContext(master_conversation_id="m1", thread_store=store)
    context.opened("chess", 5)
    context.opened("chess", 9)  # #262: many threads per window and app
    context.sent_to("tasks", 3)  # an answer on a thread opened earlier
    assert store.threads == [("m1", "chess", 5), ("m1", "chess", 9)]
    assert context.used == {("chess", 5), ("chess", 9), ("tasks", 3)}
    assert context.keeps_threads
    assert not DelegationContext(master_conversation_id="mcp", thread_store=None).keeps_threads


def test_answering_pins_one_app_to_one_thread_inside_the_block() -> None:
    assert answering_thread("music") is None
    with answering("music", 7):
        assert answering_thread("music") == 7
        assert answering_thread("tasks") is None
    assert answering_thread("music") is None


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
