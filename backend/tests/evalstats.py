# vendored — canonical: agent-standard/evalstats.py; re-copy, never edit
"""The eval gate's arithmetic: how many samples, and what the count means.

Canonical in ``../agent-standard/evalstats.py``; every app vendors it verbatim
as ``backend/tests/evalstats.py`` with one ``# vendored …`` first line, beside
the vendored spec ``test_evalstats.py``. ``check-sync.sh`` diffs the copies. Fix
a bug here and re-copy, never in a copy. App-specific drift guards (that the
loop's stop reasons and the provider's error text are the ones read below) live
in each app's own ``test_evalstats_app.py``. Grown in chess, cut down and
adapted in PCC (#403), canonical since ryanskarnulis/agent-standard#16.

The live suite in ``test_agent_evals.py`` cannot be unit-tested — every
assertion there needs the model on the GPU — so its *verdict* never was. This
module is that verdict: pure functions over counts, stop reasons and failure
strings, with no I/O, no model and no import of the app, plus ``run_gate``,
the block-sampling driver, which takes the app's one-sample function.

Why it exists (chess measured all of this first):

- **"N clean runs in a row" is weak evidence either way.** At a true rate of
  exactly 0.80, a fixed 5-sample ``rate >= 0.8`` gate goes green 74% of the
  time and a fixed 20-sample one **63%**. ``decide`` replaces the comparison
  with a one-sided Wilson bound and block-sequential escalation, which at n=5
  costs a healthy scenario exactly five samples.
- **Infrastructure was scored as behavior.** A llama-swap reload, a dead
  socket or a 5xx is not the model choosing badly. ``classify`` separates the
  two so the harness retries instead of scoring — and splits the death itself:
  a request the server *rejected* (a 4xx) answers the same way every time and
  is ``Outcome.PROVIDER_REJECTED`` on the first one. ``InfraBudget`` bounds the
  retries per scenario and per suite.
- **Consecutive samples on one llama-server are correlated** (chess: one
  unchanged prompt read 5/20, 19/40 and 31/40 in three batches), so comparing
  two arms needs pairing: sample *j* of every arm sends the same sampling seed
  (``sample_seed``), and ``mcnemar_exact`` reads the pairs the change flipped.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# The **one-sided** 95% normal quantile. The red test asks a single-tailed
# question — is the interval's *upper* end below the floor? — so 1.645 is what
# makes it a 95% test; 1.96 would overstate the evidence by a whole tail.
Z_ONE_SIDED_95 = 1.645

# The floor every scenario is held to unless it says otherwise. Lowering one
# for quiet hollows out the gate: power depends on it, so 0.8 → 0.7 halves the
# gate's chance of catching a real drop to 0.6.
DEFAULT_FLOOR = 0.8

# A block stops being "the same run" when its rate differs from another
# block's by this much (chess's record: 0.6 flags a 5/5-then-0/5 pair and stays
# quiet on ordinary 5/5-vs-3/5 spread at a true rate near the floor).
UNSTABLE_SPREAD = 0.6

# The standard's failure stops (STANDARD.md §3) that mean the model was never
# (fully) asked: the provider died, or the run's wall-clock budget ran out
# waiting on it. Literals rather than imports so this module stays importable
# with nothing installed; each app's ``test_evalstats_app`` pins them against
# its loop.
INFRA_STOPS = frozenset({"provider_error", "timed_out"})

# The statuses the delegate route answers those stops with (delegate-api.md):
# 502 for a provider failure, 504 for the deadline. A harness that reads only
# the HTTP response (no stop reason on an error body) classifies on these.
INFRA_STATUSES = frozenset({502, 504})

# The agent route's rate limit (delegate-api.md). A 429 is the harness
# outrunning the limiter, never the model: infrastructure.
STATUS_RATE_LIMITED = 429

# Provider-failure kinds (``provider_failure_kind``) a retry cannot fix. Anything
# else — a transport error, a 5xx, a timeout, or no kind at all — is retried:
# spending a few samples on a deterministic failure is cheaper than aborting a
# suite on a server that was only reloading.
DETERMINISTIC_FAILURES = frozenset({"rejected", "malformed_response"})

_RETURNED = re.compile(r"llama-server returned (\d{3})")
# 4xx statuses that are about timing, not the request: worth asking again.
_TRANSIENT_4XX = frozenset({408, 409, 425, 429})


class Outcome(StrEnum):
    """What one sample was, before it is allowed to count as anything."""

    PASS = "PASS"
    FAIL = "FAIL"
    # The provider died (or the route rate-limited us). Retried, never scored.
    INFRA = "INFRA"
    # The provider refused in a way a retry cannot fix (a 4xx, an unparseable
    # body). Never retried, never scored: fails the item on the first one.
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    # A bug in this repo's harness (a KeyError in a check). Never retried,
    # never scored, fails the item immediately.
    HARNESS = "HARNESS"


class Decision(StrEnum):
    """What a scenario's accumulated count says about the floor."""

    # The point estimate reaches the floor. Green.
    ABOVE_FLOOR = "ABOVE_FLOOR"
    # The one-sided upper bound is *below* the floor: evidence of regression,
    # not a bad sample. Red.
    BELOW_FLOOR = "BELOW_FLOOR"
    # The interval straddles the floor and the point estimate is under it.
    # Take another block; at the cap this is green, flagged.
    UNDECIDED = "UNDECIDED"


class Stability(StrEnum):
    """Whether the blocks looked like samples of one thing."""

    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"


def wilson_interval(passed: int, runs: int, z: float = Z_ONE_SIDED_95) -> tuple[float, float]:
    """The Wilson score interval for ``passed``/``runs``, clamped to [0, 1].

    Wilson rather than the normal approximation because the counts are tiny
    and the rates near 1, exactly where ``p ± z·√(pq/n)`` breaks. ``runs == 0``
    is the whole unit interval: a scenario whose every sample was an infra
    death knows nothing.
    """
    if runs <= 0:
        return (0.0, 1.0)
    point = passed / runs
    z2 = z * z
    denominator = 1 + z2 / runs
    centre = (point + z2 / (2 * runs)) / denominator
    half = (z / denominator) * math.sqrt(point * (1 - point) / runs + z2 / (4 * runs * runs))
    return (max(0.0, centre - half), min(1.0, centre + half))


def decide(passed: int, runs: int, floor: float, *, z: float = Z_ONE_SIDED_95) -> Decision:
    """Is this count below the floor, at it, or not yet resolved?

    Deliberately asymmetric: **red** takes evidence (the upper bound below the
    floor), **green** takes only the point estimate (demanding the lower bound
    clear the floor would fail every 5-sample scenario — 5/5's is 0.65), and
    everything else is UNDECIDED, which the sampler answers with another block.
    At n=5 and floor 0.8: 0–2 red, 3 escalates, 4–5 green.
    """
    upper = wilson_interval(passed, runs, z)[1]
    if upper < floor:
        return Decision.BELOW_FLOOR
    if runs > 0 and passed / runs >= floor:
        return Decision.ABOVE_FLOOR
    return Decision.UNDECIDED


def provider_failure_kind(message: str) -> str:
    """Name a provider death from its error text, for ``classify``.

    Read off the message the reference ``LlamaCppProvider`` raises (and the
    route returns as its 502 detail), because the provider keeps no structured
    status on the error; each app's ``test_evalstats_app`` drives its real
    provider over a fake server to pin these strings.
    """
    returned = _RETURNED.search(message)
    if returned is not None:
        status = int(returned.group(1))
        if 400 <= status < 500 and status not in _TRANSIENT_4XX:
            return "rejected"
        return f"http_{status}"
    if "failed wire validation" in message:
        return "malformed_response"
    return "transport"


def classify(
    *,
    status_code: int,
    stop_reason: str | None,
    error: BaseException | None,
    provider_failure: str | None = None,
) -> Outcome:
    """What kind of sample this was. Precedence is the point of the function.

    A harness bug outranks everything (a retry cannot fix a typo, and the infra
    budget must not be spent on one). Infrastructure outranks a failed
    assertion: when the provider died, the check failed *because* it died.
    A provider death then splits on ``provider_failure`` — a kind in
    ``DETERMINISTIC_FAILURES`` is PROVIDER_REJECTED, everything else INFRA.
    Only then is a failed assertion, or a non-200 that isn't infrastructure (a
    500 from a tool that raised), a real miss. ``stop_reason`` may be ``None``
    when the harness only has an error response: the status then decides.
    """
    if error is not None and not isinstance(error, AssertionError):
        return Outcome.HARNESS
    if (
        status_code == STATUS_RATE_LIMITED
        or status_code in INFRA_STATUSES
        or stop_reason in INFRA_STOPS
    ):
        if provider_failure in DETERMINISTIC_FAILURES:
            return Outcome.PROVIDER_REJECTED
        return Outcome.INFRA
    if error is not None or status_code != 200:
        return Outcome.FAIL
    return Outcome.PASS


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")


def failure_signature(message: str) -> str:
    """One failure message, normalised so identical *modes* collapse.

    Five failures naming five different ids or dates are one finding; five
    different messages are five. Normalised rather than hashed, because the
    signature goes in the report and has to stay recognisable.
    """
    return _WHITESPACE.sub(" ", _NUMBER.sub("<n>", message)).strip()


def block_stability(blocks: Sequence[tuple[int, int]]) -> Stability:
    """Did the blocks look like samples of one thing?

    Consecutive samples of one ask are correlated, so the Wilson width
    understates the truth; this makes the confound *visible* where escalation
    bought a second block. A count that decides on its first block has one
    rate and no spread, so it is STABLE by construction. Blocks with no scored
    samples (every one retried for infra) have no rate and are skipped.
    """
    rates = [passed / runs for passed, runs in blocks if runs > 0]
    if len(rates) < 2:
        return Stability.STABLE
    return Stability.UNSTABLE if max(rates) - min(rates) >= UNSTABLE_SPREAD else Stability.STABLE


@dataclass(frozen=True)
class RateResult:
    """What sampling one scenario measured, and what it means.

    Every derived field is built by ``from_blocks`` from the counts, never
    passed in: a record whose verdict disagrees with its counts is worse than
    no record.
    """

    passed: int
    runs: int
    blocks: tuple[tuple[int, int], ...]
    failures: tuple[str, ...]
    # Signature → count on every scenario: a 4/5 whose one failure is the same
    # mode every time is a different animal from one that fails differently.
    failure_modes: Mapping[str, int] = field(default_factory=dict)
    # Samples thrown away and re-taken because the provider died. Printed even
    # at 1: a rate taken over a crashing server is worth less than a clean one.
    infra: int = 0
    interval: tuple[float, float] = (0.0, 1.0)
    decision: Decision = Decision.UNDECIDED
    stability: Stability = Stability.STABLE
    # 0 passes and exactly one failure signature: not sampling noise but a
    # broken build, a changed tool offer, or a deterministic server refusal.
    deterministic_suspect: bool = False

    @classmethod
    def from_blocks(
        cls,
        *,
        blocks: Sequence[tuple[int, int]],
        failures: Sequence[str],
        floor: float,
        infra: int = 0,
        z: float = Z_ONE_SIDED_95,
    ) -> RateResult:
        passed = sum(block_passed for block_passed, _ in blocks)
        runs = sum(block_runs for _, block_runs in blocks)
        modes = Counter(failure_signature(failure) for failure in failures)
        return cls(
            passed=passed,
            runs=runs,
            blocks=tuple((block_passed, block_runs) for block_passed, block_runs in blocks),
            failures=tuple(failures),
            failure_modes=dict(modes),
            infra=infra,
            interval=wilson_interval(passed, runs, z),
            decision=decide(passed, runs, floor, z=z),
            stability=block_stability(blocks),
            deterministic_suspect=passed == 0 and len(modes) == 1,
        )

    @property
    def rate(self) -> float:
        """0.0 for an unsampled scenario, so a report line still serialises."""
        return self.passed / self.runs if self.runs else 0.0

    def summary(self) -> str:
        """The one-line ``[eval]`` verdict a human reads off the terminal."""
        low, high = self.interval
        blocks = "+".join(f"{p}/{r}" for p, r in self.blocks) or "none"
        line = (
            f"{self.passed}/{self.runs} ({self.rate:.0%}) "
            f"ci=[{low:.2f},{high:.2f}] {self.decision} {self.stability} "
            f"blocks={blocks} infra={self.infra}"
        )
        if self.deterministic_suspect:
            line += " DETERMINISTIC_SUSPECT"
        return line


def scenario_record(scenario: str, result: RateResult, floor: float, **meta: Any) -> dict[str, Any]:
    """One scenario's verdict as a JSON-serialisable report line. ``**meta``
    carries what the call site knows and this module must not (per-sample
    records, wall clock)."""
    return {
        "scenario": scenario,
        "floor": floor,
        "passed": result.passed,
        "runs": result.runs,
        "rate": result.rate,
        "blocks": [list(block) for block in result.blocks],
        "interval": list(result.interval),
        "decision": str(result.decision),
        "stability": str(result.stability),
        "infra": result.infra,
        "deterministic_suspect": result.deterministic_suspect,
        "failure_modes": dict(result.failure_modes),
        "failures": list(result.failures),
        **meta,
    }


# --- paired arms --------------------------------------------------------------
#
# An A/B that seeds sample *j* of every arm alike (common random numbers) turns
# two independent counts into pairs. For a small change most pairs agree, and
# the verdict lives in the ones that do not: McNemar's test on the discordant
# pairs resolves an effect an unpaired comparison needs several times the
# samples for. Pairing is approximate — llama.cpp is not bit-reproducible
# across batch sizes or slots — but never worse than unpaired.


def sample_seed(base: int | None, index: int) -> int | None:
    """The seed of a scenario's ``index``-th *scored* sample (0-based), or
    ``None`` when the run is unseeded. Keyed on the scored position, not the
    attempt: an infra retry re-takes the same sample and must draw the same
    stream, or every later pair of the scenario is misaligned."""
    return None if base is None else base + index


def paired_counts(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, int, int]:
    """(both pass, only A passes, only B passes, neither) over aligned pairs."""
    if len(a) != len(b):
        raise ValueError(f"unaligned pairs: {len(a)} against {len(b)}")
    both = sum(x and y for x, y in zip(a, b, strict=True))
    a_only = sum(x and not y for x, y in zip(a, b, strict=True))
    b_only = sum(y and not x for x, y in zip(a, b, strict=True))
    return both, a_only, b_only, len(a) - both - a_only - b_only


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p: under no effect each discordant pair is a
    fair coin, so the smaller side is Binomial(a_only + b_only, ½). No
    discordant pairs is no evidence either way, p = 1."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def format_paired(a_only: int, b_only: int, pairs: int) -> str:
    """One paired verdict as the report prints it."""
    return (
        f"pairs {pairs} · A-only {a_only} · B-only {b_only} · p={mcnemar_exact(a_only, b_only):.3g}"
    )


# --- infra budget -------------------------------------------------------------


@dataclass
class InfraBudget:
    """The per-scenario and suite-wide ceilings on infra retries.

    Suite-wide as well as per-scenario because the failure it guards against is
    not one unlucky scenario but a server that has stopped coming back: without
    a ceiling, a dead llama-server turns the suite into an unbounded retry loop
    that never reports anything. ``per_scenario`` records where the budget
    went, so a scenario that keeps straddling a model swap shows up by name.
    """

    retries: int
    budget: int
    spent: int = 0
    per_scenario: dict[str, int] = field(default_factory=dict)

    def spend(self, scenario: str) -> bool:
        """Charge one thrown-away sample; False when ``scenario`` may not retry.

        ``retries`` is re-takes per scenario and ``budget`` re-takes across
        the suite.
        """
        self.spent += 1
        used = self.per_scenario.get(scenario, 0) + 1
        self.per_scenario[scenario] = used
        return used <= self.retries and self.spent <= self.budget

    def summary(self) -> str:
        """The suite line: total spent, and which scenarios spent it."""
        where = ", ".join(f"{name}={n}" for name, n in self.per_scenario.items() if n)
        return f"infra_spent={self.spent}/{self.budget}" + (f" ({where})" if where else "")


# --- the block-sampling driver ------------------------------------------------


@dataclass(frozen=True)
class Taken:
    """One sample as the app's ``take`` function saw it.

    ``failure`` says why a FAIL failed (it feeds the failure modes);
    ``detail`` is the server's words on a provider death; ``error`` is the
    exception behind a HARNESS outcome, re-raised by ``run_gate``.
    """

    outcome: Outcome
    failure: str | None = None
    detail: str | None = None
    error: BaseException | None = None


class GateAborted(Exception):
    """A scenario that measured no rate: the server rejected the request
    (``PROVIDER_REJECTED``) or would not stay up (``INFRA_ABORTED``)."""

    def __init__(self, scenario: str, decision: str, message: str, infra: int) -> None:
        super().__init__(f"{scenario}: {decision} — {message}")
        self.scenario = scenario
        self.decision = decision
        self.infra = infra


def run_gate(
    scenario: str,
    take: Callable[[int | None], Taken],
    *,
    budget: InfraBudget,
    floor: float = DEFAULT_FLOOR,
    block_runs: int = 5,
    max_runs: int = 20,
    seed_base: int | None = None,
    on_infra: Callable[[int, Taken], None] | None = None,
) -> RateResult:
    """Sample one scenario in blocks until its count decides against ``floor``.

    ``take(seed)`` runs one fresh sample and classifies it (``classify``). An
    INFRA sample is thrown away and re-taken with the same seed, charged to
    ``budget``; running out raises ``GateAborted("INFRA_ABORTED")``. A
    PROVIDER_REJECTED sample raises ``GateAborted`` at once, a HARNESS one
    re-raises its error. Otherwise blocks of ``block_runs`` scored samples
    accumulate until the decision is not UNDECIDED or ``max_runs`` is reached.
    The caller holds the result to the verdict: red is
    ``Decision.BELOW_FLOOR``; an UNDECIDED at the cap passes, flagged.
    """
    if block_runs < 1 or max_runs < 1:
        raise ValueError("block_runs and max_runs must be >= 1")
    blocks: list[tuple[int, int]] = []
    failures: list[str] = []
    infra = 0
    while True:
        block_passed = block_count = 0
        while block_count < block_runs:
            scored = sum(runs for _, runs in blocks) + block_count
            sample = take(sample_seed(seed_base, scored))
            if sample.outcome is Outcome.HARNESS:
                if sample.error is not None:
                    raise sample.error
                raise RuntimeError(f"{scenario}: harness error with no exception")
            if sample.outcome is Outcome.PROVIDER_REJECTED:
                raise GateAborted(
                    scenario,
                    "PROVIDER_REJECTED",
                    f"llama-server refused the request ({sample.detail}); "
                    "a retry would get the same refusal",
                    infra,
                )
            if sample.outcome is Outcome.INFRA:
                infra += 1
                if on_infra is not None:
                    on_infra(infra, sample)
                if not budget.spend(scenario):
                    raise GateAborted(
                        scenario,
                        "INFRA_ABORTED",
                        f"{infra} provider deaths ({budget.summary()}); "
                        "no rate was measured: llama-server is not staying up",
                        infra,
                    )
                continue
            block_count += 1
            if sample.outcome is Outcome.PASS:
                block_passed += 1
            else:
                failures.append(f"run {scored + 1}: {sample.failure or 'failed'}")
        blocks.append((block_passed, block_count))
        result = RateResult.from_blocks(blocks=blocks, failures=failures, floor=floor, infra=infra)
        if result.decision is not Decision.UNDECIDED or result.runs >= max_runs:
            return result
