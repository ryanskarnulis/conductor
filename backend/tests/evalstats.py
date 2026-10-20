"""The routing eval's verdict on one attempt: was the model actually asked?

Conductor's cut of chess's ``backend/tests/evalstats.py`` (#25). The live suite
in ``test_agent_evals.py`` needs the 12B on the GPU, so its own logic is never
exercised in CI; this module is that logic, extracted as pure functions with no
I/O and no model, and ``test_evalstats.py`` runs it on every commit.

Only chess's infrastructure split is ported. Conductor's goldens are single
assertion-based samples (expected tool vs. acted), not Wilson-bounded pass
rates, so the sampling machinery has nothing to act on here. What does carry
over is the rule chess's docs state directly: **infrastructure is retried, not
scored.** llama-swap evicting or reloading the model mid-suite surfaces as a
502 ``agent run failed: llama-server request failed: timed out`` — the routing
decision was never made, so failing the golden on it reports GPU scheduling as
a routing regression.

A provider death splits two ways, because retrying is only worth anything
against a server that might answer differently:

- **INFRA** — the transport died (timeout, refused or reset connection) or the
  server answered 5xx (llama-swap mid-swap). Retried, bounded per scenario and
  across the suite.
- **PROVIDER_REJECTED** — the server answered 4xx (a context overrun), or 200
  with a body that fails wire validation. Same answer every time, so the
  golden fails on the first one instead of spending the infra budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

# ``routes_agent.post_message`` maps every ``ProviderError`` to this status.
PROVIDER_FAILURE_STATUS = 502

# ``LlamaCppProvider._post``'s message for an HTTP status other than 200.
_RETURNED = re.compile(r"llama-server returned (\d{3})")
# …and for a 200 whose body could not be parsed. Deterministic: same bytes back.
_WIRE_INVALID = "failed wire validation"


class Outcome(StrEnum):
    """What one attempt was, before it is allowed to count as anything."""

    # The turn ran; the golden's assertions decide pass or fail.
    SCORED = "SCORED"
    # The provider died. Retried, never scored: a crash is not a routing miss.
    INFRA = "INFRA"
    # The provider refused in a way a retry cannot fix. Never retried, never
    # scored as routing — reported, and the golden fails on the first one.
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    # Any other non-200: a bug in conductor or this harness. Never retried.
    HARNESS = "HARNESS"


def classify(*, status_code: int, detail: str | None) -> Outcome:
    """Sort one ``POST …/messages`` response by whether the model was asked.

    ``detail`` is the error body's ``detail`` (``None`` on success). An
    unrecognised 502 detail stays INFRA, as in chess: a provider death that
    doesn't name itself is more likely transient than a stable refusal, and the
    budget bounds the cost of being wrong.
    """
    if status_code == 200:
        return Outcome.SCORED
    if status_code != PROVIDER_FAILURE_STATUS:
        return Outcome.HARNESS
    text = detail or ""
    if _WIRE_INVALID in text:
        return Outcome.PROVIDER_REJECTED
    returned = _RETURNED.search(text)
    if returned is not None and 400 <= int(returned.group(1)) < 500:
        return Outcome.PROVIDER_REJECTED
    return Outcome.INFRA


@dataclass
class InfraBudget:
    """The per-scenario and suite-wide ceilings on infra retries.

    Suite-wide as well as per-scenario because the failure it guards against is
    not one unlucky golden but a server that has stopped coming back: without a
    ceiling, a dead llama-server turns the suite into an unbounded retry loop
    that never reports anything. ``per_scenario`` records where the budget went,
    so a scenario that keeps straddling a model swap shows up by name rather
    than hiding inside one suite total.
    """

    retries: int
    budget: int
    spent: int = 0
    per_scenario: dict[str, int] = field(default_factory=dict)

    def spend(self, scenario: str) -> bool:
        """Charge one thrown-away attempt; False when ``scenario`` may not retry.

        ``retries`` is re-runs per scenario (so ``retries + 1`` attempts) and
        ``budget`` is re-runs across the suite.
        """
        self.spent += 1
        used = self.per_scenario.get(scenario, 0) + 1
        self.per_scenario[scenario] = used
        return used <= self.retries and self.spent <= self.budget

    def summary(self) -> str:
        """The suite line: total spent, and which scenarios spent it."""
        where = ", ".join(f"{name}={n}" for name, n in self.per_scenario.items() if n)
        return f"infra_spent={self.spent}/{self.budget}" + (f" ({where})" if where else "")
