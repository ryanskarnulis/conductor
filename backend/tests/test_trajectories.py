"""Seeded trajectory walks, in CI (#53, STANDARD.md §6).

``trajectory.py`` is the machinery: a seeded walk of user messages through the
real route, loop, registry and delegate tools over the fake fleet, a scripted
provider, injected faults, and every invariant checked after every step. This
file is what runs:

- **The corpus.** ``CONDUCTOR_TRAJ_SEEDS`` walks (default 32) of
  ``CONDUCTOR_TRAJ_LENGTH`` messages (default 8) from seed
  ``CONDUCTOR_TRAJ_SEED_BASE`` (default 0). A soak:
  ``CONDUCTOR_TRAJ_SEEDS=2000 CONDUCTOR_TRAJ_SEED_BASE=10000 .venv/bin/pytest
  tests/test_trajectories.py -q -k corpus``.
- **Named cases.** Fixed step sequences: a ``known bug`` reproduces a
  ``KNOWN_BUGS`` entry strictly (xfail until the fix lands, then it XPASSes and
  fails until the entry and the mark go), a ``pending`` case waits on a
  capability (Stop, #66), the rest pin combinations the corpus reaches only
  now and then.
- **The instrument's own checks.** Every invariant must be able to fire, every
  registered tool must be walked, and a walk must be a pure function of its
  seed — a checker that cannot fire reads green for having looked at nothing.
- **Replay.** ``CONDUCTOR_TRAJ_SEED=<n> CONDUCTOR_TRAJ_LENGTH=<k> .venv/bin/pytest
  tests/test_trajectories.py -k replay -s`` reruns one walk and prints it — the
  command every breach report carries.
"""

from __future__ import annotations

import os
import random
from dataclasses import replace
from pathlib import Path

import pytest

from tests.trajectory import (
    CORPUS_LENGTH,
    INVARIANTS,
    Fault,
    InvariantBreach,
    Observed,
    Step,
    Turn,
    check_delegate_budget,
    check_every_app_write_is_recorded,
    check_every_call_has_one_answer,
    check_fault_ends_the_run_as_documented,
    check_handoffs_ask_nothing,
    check_history_is_text_only,
    check_nothing_starts_after_stop,
    check_saved_turn_is_what_ran,
    describe,
    fixture_fleet,
    harness,
    next_step,
    run_steps,
    walk,
)

CORPUS_SEEDS = int(os.environ.get("CONDUCTOR_TRAJ_SEEDS", "32"))
SEED_BASE = int(os.environ.get("CONDUCTOR_TRAJ_SEED_BASE", "0"))
LENGTH = int(os.environ.get("CONDUCTOR_TRAJ_LENGTH", str(CORPUS_LENGTH)))


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    return fixture_fleet(tmp_path / "fleet")


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + CORPUS_SEEDS))
def test_corpus_walk_keeps_every_invariant(seed: int, fleet: Path) -> None:
    walk(fleet, seed, LENGTH)


def test_replay(fleet: Path) -> None:
    seed = os.environ.get("CONDUCTOR_TRAJ_SEED")
    if seed is None:
        pytest.skip("set CONDUCTOR_TRAJ_SEED to replay one walk")
    log = walk(fleet, int(seed), LENGTH)
    print("\n".join(describe(log)))


# --- named cases -----------------------------------------------------------------------


def calls(*batch: tuple[str, dict[str, str]]) -> Turn:
    return Turn("calls", calls=tuple(batch))


DONE = Turn("text", text="Done.")
ADD = ("ask_tasks", {"message": "add a task to water the plants tomorrow"})
SORT = ("ask_music", {"message": "sort my whole library"})


def test_a_write_before_a_provider_death_is_kept(fleet: Path) -> None:
    # Found by the walks before #52: the route saved only the person's words,
    # so the app's write was in no turn. Now the partial record is saved.
    for kind in ("provider_dies", "provider_crashes"):
        step = Step("add it", (calls(ADD), DONE), Fault(kind, 1))
        log = run_steps(fleet, [step], tolerate_known_bugs=False)
        saved = log[0].saved_assistant
        assert log[0].receipts and saved is not None
        assert saved["stop_reason"] in ("provider_error", "internal_error")


def test_partial_write_is_reported_as_outcome_unknown(fleet: Path) -> None:
    log = run_steps(fleet, [Step("add it", (calls(ADD), DONE), Fault("app_partial", 0))])
    assert log[0].receipts and log[0].status == 200


def test_app_down_is_a_domain_error_and_the_run_completes(fleet: Path) -> None:
    log = run_steps(fleet, [Step("sort", (calls(SORT), DONE), Fault("app_down", 0))])
    assert log[0].run is not None and log[0].run.stop_reason == "completed"
    assert not log[0].receipts


def test_over_budget_asks_never_reach_the_app(fleet: Path) -> None:
    ask = ("ask_tasks", {"message": "what's due today?"})
    log = run_steps(fleet, [Step("go", (calls(ask, ask), calls(ask, ask), DONE))])
    assert log[0].sends == {"tasks": 3}


def test_history_stays_text_after_tool_turns(fleet: Path) -> None:
    steps = [Step("add it", (calls(ADD), DONE)), Step("and sort", (calls(SORT), DONE))]
    log = run_steps(fleet, steps)
    # No chat turns replayed (#64): system, then one user message whose recall
    # carries the earlier turn as data.
    assert len(log[1].requests[0]) == 2  # system, user
    assert "The record of earlier turns" in log[1].requests[0][1]["content"]


def test_stop_mid_delegate(fleet: Path) -> None:
    log = run_steps(fleet, [Step("sort", (calls(SORT), DONE), Fault("stop", 0))])
    run = log[0].run
    assert run is not None and str(run.stop_reason) == "cancelled"
    # The record replies: the delegate call that finished, nothing after it.
    assert run.reply is not None and run.reply.startswith("I stopped before finishing")
    assert [record.tool for record in run.tool_calls] == ["ask_music"]
    # The app got conductor's Stop, and no model call started after it.
    assert ("music", "cancel") in log[0].tool_events
    assert log[0].chat_calls == 1


# --- the instrument's own checks ---------------------------------------------------------


def _one_clean_step(fleet: Path) -> Observed:
    return run_steps(fleet, [Step("add it", (calls(ADD), DONE))])[0]


def _breaches(check: object, observed: Observed) -> bool:
    try:
        check(observed)  # type: ignore[operator]
    except InvariantBreach:
        return True
    return False


def test_every_invariant_can_fire(fleet: Path) -> None:
    clean = _one_clean_step(fleet)
    for invariant in INVARIANTS:
        assert not _breaches(invariant, clean), invariant.__name__
    assert clean.run is not None
    run = clean.run
    unanswered = run.model_copy(
        update={"messages": [m for m in run.messages if m["role"] != "tool"]}
    )
    mutants = {
        check_every_call_has_one_answer: replace(clean, run=unanswered),
        check_saved_turn_is_what_ran: replace(clean, run=run.model_copy(update={"reply": "x"})),
        check_history_is_text_only: replace(
            clean,
            requests=[[{"role": "system"}, {"role": "tool", "content": "x"}, {"role": "user"}]],
        ),
        check_delegate_budget: replace(clean, sends={"tasks": 4}),
        check_handoffs_ask_nothing: replace(clean, sends={"chess": 1}),
        check_every_app_write_is_recorded: replace(clean, saved_after=clean.saved_before),
        check_fault_ends_the_run_as_documented: replace(clean, status=502),
        check_nothing_starts_after_stop: replace(clean, stopped_at=0),
    }
    assert set(mutants) == set(INVARIANTS), "an invariant has no mutant"
    for invariant, mutant in mutants.items():
        assert _breaches(invariant, mutant), f"{invariant.__name__} cannot fire"


def test_a_walk_is_a_pure_function_of_its_seed(fleet: Path, tmp_path: Path) -> None:
    first = list(describe(walk(fleet, 7, 6)))
    second = list(describe(walk(fixture_fleet(tmp_path / "again"), 7, 6)))
    assert first == second


def test_the_corpus_walks_every_tool_and_fault(fleet: Path) -> None:
    with harness(fleet) as h:
        tools, faults = set(h.tools), set(h.faults)
    seen_tools: set[str] = set()
    seen_faults: set[str] = set()
    for seed in range(SEED_BASE, SEED_BASE + CORPUS_SEEDS):
        rng = random.Random(seed)
        for _ in range(LENGTH):
            step = next_step(rng, sorted(tools), sorted(faults))
            seen_tools |= {name for turn in step.turns for name, _ in turn.calls}
            if step.fault:
                seen_faults.add(step.fault.kind)
    assert tools <= seen_tools, f"never walked: {sorted(tools - seen_tools)}"
    assert faults <= seen_faults, f"never injected: {sorted(faults - seen_faults)}"
