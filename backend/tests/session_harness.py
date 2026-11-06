"""Session evals' machinery: scripted multi-turn sessions over the fake world.

A scenario (``session_scenarios.py``) is a short script of what a person does —
say this, then "yes", open a new conversation days later, say "stop" partway
through — and a set of checks on what came of it: the end state of each app
(``fake_world.World``), and the replies held to the receipts
(``receipt_claims``). One *sample* is one whole session on a fresh world; the
vendored Wilson gate (``evalstats.run_gate``) samples a scenario like any
routing golden.

**The driver is the seam between the harness and the runtime.** Scenarios
never touch conductor's code; they talk to a :class:`Driver` — begin a session
on a world, say something, start a new conversation, Stop, read what arrived
unprompted. :class:`LoopDriver` drives today's loop through the same HTTP
route the web UI uses; the compiled-plan runtime (#67, #68) gets its own
driver over its own routes, and every scenario and check runs on it unchanged.

**Capabilities a runtime lacks fail the sample, they don't skip it.** A
scenario declares what it ``needs`` beyond saying things — ``"stop"`` (a
cancel route, #66), ``"tool:ask_chess"`` — and a driver without it scores the
sample a FAIL with an ``unsupported:`` failure mode. A frontier scenario is
then measured at 0 for a reason the report names, and turns into a real
measurement the day the runtime grows the capability, with no harness change:
``LoopDriver`` finds a ``…/cancel`` route by looking at the app's routes.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tests.evalstats import Outcome, Taken, classify, provider_failure_kind
from tests.fake_world import Receipt, World
from tests.receipt_claims import Claim, Tally, read_claims, unbacked, unnamed

# --- what a turn left behind ------------------------------------------------------


@dataclass
class TurnRecord:
    """One said thing and what came back. ``index`` is the session turn
    (1-based) the world stamped on the receipts it wrote."""

    index: int
    text: str
    surface: str
    status: int
    reply: str | None = None
    stop_reason: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    detail: str | None = None
    seconds: float = 0.0
    label: str | None = None
    # The person pressed Stop during this turn: it ends ``cancelled``.
    stopped: bool = False

    @property
    def acted(self) -> list[str]:
        """Tools that acted on an app: ``ask_`` delegates, ``open_`` hands over."""
        return [call["tool"] for call in self.tools if call["tool"].startswith(("ask_", "open_"))]


class Driver(Protocol):
    """What a runtime must offer for sessions to be run against it."""

    def begin(self, world: World) -> None:
        """Start a sample: bind the world behind the delegate seam, open a
        fresh conversation."""

    def new_conversation(self) -> None:
        """A later session (another day, another surface): a new conversation
        over the same world. Whatever the runtime remembers across it is its
        memory."""

    def say(self, text: str, *, surface: str) -> TurnRecord:
        """One person turn through the runtime's real entry point."""

    def stop(self) -> None:
        """The person's Stop on the in-flight turn (``"stop"`` capability)."""

    def unprompted(self) -> list[str]:
        """Assistant messages that arrived since the last turn without one
        being asked for (a job reporting back, S4)."""

    def supports(self, capability: str) -> bool: ...


# --- steps --------------------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    text: str
    label: str | None = None
    surface: str = "desktop"


@dataclass(frozen=True)
class ConfirmIfAsked:
    """Answer ``text`` only if the previous turn wrote nothing and asked a
    question — a confirmation the runtime may or may not ask for."""

    text: str = "yes"
    label: str | None = None


@dataclass(frozen=True)
class NewConversation:
    pass


@dataclass(frozen=True)
class Advance:
    """Time passes in the world: queued downloads finish."""


@dataclass(frozen=True)
class StopAfter:
    """Say ``text``, and press Stop once ``app`` has made ``writes`` writes in
    this turn — mid-delegate, from another thread, as a person would."""

    text: str
    app: str
    writes: int
    label: str | None = None


Step = Say | ConfirmIfAsked | NewConversation | Advance | StopAfter


# --- the session a check reads ------------------------------------------------------


@dataclass
class Session:
    world: World
    turns: list[TurnRecord] = field(default_factory=list)
    unprompted: list[str] = field(default_factory=list)
    claims: list[tuple[int, Claim]] = field(default_factory=list)

    def turn(self, label: str) -> TurnRecord:
        for record in self.turns:
            if record.label == label:
                return record
        raise AssertionError(f"no turn labelled {label!r} (the session was cut short)")

    @property
    def last(self) -> TurnRecord:
        assert self.turns, "no turns ran"
        return self.turns[-1]

    def writes(self, turn: TurnRecord | None = None) -> list[Receipt]:
        return self.world.writes(turn=turn.index if turn else None)


Check = Callable[[Session], None]


@dataclass(frozen=True)
class Scenario:
    """One product scenario. ``checks`` raise ``AssertionError`` with a short
    reason; the generic checks (every turn completed, claims backed, writes
    named) run on every scenario first. ``script`` is the golden model side
    for the scripted CI run: the provider turns that should pass."""

    id: str
    summary: str
    steps: tuple[Step, ...]
    checks: tuple[Check, ...]
    tier: str = "frontier"  # "gate" | "frontier"
    needs: frozenset[str] = frozenset()
    script: Callable[[World], list[Any]] | None = None
    known_red: str | None = None


# --- generic checks -----------------------------------------------------------------


def check_turns_completed(session: Session) -> None:
    for record in session.turns:
        want = "cancelled" if record.stopped else "completed"
        assert record.stop_reason == want, f"turn {record.index} stopped {record.stop_reason}"
        assert record.reply and record.reply.strip(), f"turn {record.index}: empty reply"


def check_claims_backed(session: Session) -> None:
    bad = [f"turn {turn}: {claim}" for turn, claim in session.claims if not claim.backed]
    assert not bad, "unbacked claim: " + "; ".join(bad)


def check_writes_named(session: Session) -> None:
    for record in session.turns:
        missing = unnamed(record.reply, session.world.receipts, turn=record.index)
        assert not missing, (
            f"turn {record.index}'s reply doesn't name what it touched: "
            + ", ".join(f"{r.kind} {r.title!r}" for r in missing)
        )


GENERIC_CHECKS: tuple[Check, ...] = (
    check_turns_completed,
    check_claims_backed,
    check_writes_named,
)


# --- running one sample -------------------------------------------------------------


class Unsupported(Exception):
    pass


@dataclass
class Sample:
    taken: Taken
    session: Session
    seconds: float
    seed: int | None = None


def run_sample(scenario: Scenario, driver: Driver, world: World | None = None) -> Sample:
    """One whole session of ``scenario`` on a fresh world, classified."""
    world = world or World()
    session = Session(world)
    started = time.monotonic()

    def done(taken: Taken) -> Sample:
        return Sample(taken, session, time.monotonic() - started)

    missing = sorted(need for need in scenario.needs if not driver.supports(need))
    if missing:
        return done(Taken(Outcome.FAIL, failure=f"unsupported: {', '.join(missing)}"))
    driver.begin(world)
    for step in scenario.steps:
        try:
            record = _run_step(step, driver, session)
        except Unsupported as exc:
            return done(Taken(Outcome.FAIL, failure=f"unsupported: {exc}"))
        if record is None:
            continue
        if record.status != 200:
            outcome = classify(
                status_code=record.status,
                stop_reason=None,
                error=None,
                provider_failure=provider_failure_kind(record.detail or ""),
            )
            return done(
                Taken(
                    outcome,
                    failure=f"turn {record.index} HTTP {record.status}: {record.detail}",
                    detail=record.detail,
                )
            )
        session.claims.extend(
            (record.index, claim)
            for claim in read_claims(record.reply, world.receipts, turn=record.index)
        )
    session.unprompted = driver.unprompted()
    # Scenario checks first: the end state is the headline, the generic reply
    # checks the footnote. Every failing check is reported, not just the first.
    failed: list[str] = []
    for check in (*scenario.checks, *GENERIC_CHECKS):
        try:
            check(session)
        except AssertionError as exc:
            failed.append(str(exc) or check.__name__)
        except Exception as exc:  # a broken check is the harness's fault, not the model's
            return done(Taken(Outcome.HARNESS, failure=repr(exc), error=exc))
    if failed:
        return done(Taken(Outcome.FAIL, failure=" · ".join(failed)))
    return done(Taken(Outcome.PASS))


def _say(
    driver: Driver, session: Session, text: str, label: str | None, surface: str
) -> TurnRecord:
    session.world.turn = len(session.turns) + 1
    record = driver.say(text, surface=surface)
    record.label = label
    session.turns.append(record)
    return record


def _run_step(step: Step, driver: Driver, session: Session) -> TurnRecord | None:
    world = session.world
    if isinstance(step, Say):
        return _say(driver, session, step.text, step.label, step.surface)
    if isinstance(step, ConfirmIfAsked):
        last = session.last
        if session.writes(last) or not (last.reply and "?" in last.reply):
            return None
        return _say(driver, session, step.text, step.label, "desktop")
    if isinstance(step, NewConversation):
        driver.new_conversation()
        return None
    if isinstance(step, Advance):
        world.advance()
        return None
    if isinstance(step, StopAfter):
        if not driver.supports("stop"):
            raise Unsupported("stop")
        fired = threading.Event()

        def hook(receipt: Receipt) -> None:
            this_turn = world.writes(turn=world.turn, app=step.app)
            if len(this_turn) >= step.writes and not fired.is_set():
                fired.set()
                # Stop arrives on its own request, as the person's would; the
                # app's write loop waits for it so the stop lands mid-delegate.
                stopper = threading.Thread(target=driver.stop, daemon=True)
                stopper.start()
                stopper.join(timeout=30)

        world.on_write = hook
        try:
            record = _say(driver, session, step.text, step.label, "desktop")
            record.stopped = fired.is_set()
            return record
        finally:
            world.on_write = None
    raise TypeError(f"unknown step {step!r}")


# --- today's loop ---------------------------------------------------------------------


class WorldHolder:
    """The delegate seam's indirection: the app's lifespan builds its tools
    once, with one client factory, and every sample swaps the world behind
    it."""

    def __init__(self) -> None:
        self.world: World | None = None
        self._factory: Callable[[Any], Any] | None = None

    def bind(self, world: World) -> None:
        from tests.fake_world import client_factory

        self.world = world
        self._factory = client_factory(world)

    def factory(self, fleet_app: Any) -> Any:
        assert self._factory is not None, "no world bound: call Driver.begin first"
        return self._factory(fleet_app)


_CANCEL_ROUTE = re.compile(r"^/api/agent/conversations/\{[^}]+\}/(?:cancel|stop)$")


class LoopDriver:
    """Today's loop, driven through ``POST /api/agent/conversations/{id}/messages``.

    ``client`` is a started ``TestClient`` whose lifespan built the delegate
    tools over ``holder.factory``; ``reset`` is called before every turn
    (the route's rate limiter allows 10 messages a minute, and a session
    sampled 20 times sends far more). ``fresh_db`` is called at the start of
    every sample: samples share one app and one database, and a sample's
    "new conversation" must not find another world's conversations (#64's
    cross-conversation recall reads them).
    """

    def __init__(
        self,
        client: Any,
        holder: WorldHolder,
        *,
        tool_names: Callable[[], set[str]],
        reset: Callable[[], None] = lambda: None,
        fresh_db: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client
        self._holder = holder
        self._tool_names = tool_names
        self._reset = reset
        self._fresh_db = fresh_db
        self._conversation: int | None = None
        self._seen: int = 0
        # The OpenAPI paths, not ``app.routes``: included routers nest there.
        routes = set(client.app.openapi()["paths"])
        self._cancel_path = next((path for path in routes if _CANCEL_ROUTE.match(path)), None)

    def supports(self, capability: str) -> bool:
        if capability == "stop":
            return self._cancel_path is not None
        if capability.startswith("tool:"):
            return capability.removeprefix("tool:") in self._tool_names()
        return False

    def begin(self, world: World) -> None:
        self._fresh_db()
        self._holder.bind(world)
        self.new_conversation()

    def new_conversation(self) -> None:
        created = self._client.post("/api/agent/conversations", json={})
        assert created.status_code == 201, created.text
        self._conversation = created.json()["id"]
        self._seen = 0

    def say(self, text: str, *, surface: str) -> TurnRecord:
        # A voice turn reaches the loop as the transcribed text plus its
        # surface, as the voice client sends it (#64's ``Now``).
        assert self._conversation is not None
        self._reset()
        index = self._holder.world.turn if self._holder.world else 0
        started = time.monotonic()
        response = self._client.post(
            f"/api/agent/conversations/{self._conversation}/messages",
            json={"content": text, "surface": surface},
        )
        seconds = time.monotonic() - started
        if response.status_code != 200:
            try:
                detail = str(response.json().get("detail"))
            except ValueError:
                detail = response.text
            return TurnRecord(
                index, text, surface, response.status_code, detail=detail, seconds=seconds
            )
        assistant = response.json()["assistant_message"]
        self._seen = max(self._seen, int(assistant["id"]))
        return TurnRecord(
            index,
            text,
            surface,
            200,
            reply=assistant["content"],
            stop_reason=assistant["stop_reason"],
            tools=list(assistant["tool_calls"] or []),
            seconds=seconds,
        )

    def stop(self) -> None:
        assert self._cancel_path is not None and self._conversation is not None
        path = re.sub(r"\{[^}]+\}", str(self._conversation), self._cancel_path)
        self._client.post(path)

    def unprompted(self) -> list[str]:
        assert self._conversation is not None
        detail = self._client.get(f"/api/agent/conversations/{self._conversation}")
        if detail.status_code != 200:
            return []
        messages = detail.json().get("messages", [])
        last_user = max((m["id"] for m in messages if m["role"] == "user"), default=0)
        return [
            m["content"] or ""
            for m in messages
            if m["role"] == "assistant" and m["id"] > max(last_user, self._seen)
        ]


# --- reporting ------------------------------------------------------------------------


def sample_line(scenario: Scenario, sample: Sample) -> str:
    session = sample.session
    turns = " | ".join(f"{t.index}:{'+'.join(t.acted) or '-'}" for t in session.turns) or "-"
    writes = ",".join(f"{r.kind}:{r.title}" for r in session.world.receipts) or "-"
    line = (
        f"[session]   {scenario.id} {sample.taken.outcome} turns={turns} writes={writes} "
        f"{sample.seconds:.1f}s"
    )
    if os.environ.get("CONDUCTOR_SESSION_VERBOSE") == "1":
        for t in session.turns:
            sent = [str(c["arguments"])[:120] for c in t.tools]
            line += f"\n[session]     {t.index}> {t.text!r} sent={sent}\n[session]     {t.index}< {t.reply!r}"
    if sample.taken.failure and sample.taken.outcome is not Outcome.PASS:
        line += f"\n[session]     ✗ {sample.taken.failure[:300]}"
    return line


def tally_claims(tally: Tally, samples: Sequence[Sample]) -> None:
    for sample in samples:
        tally.add(claim for _, claim in sample.session.claims)


__all__ = [
    "Advance",
    "Check",
    "ConfirmIfAsked",
    "Driver",
    "LoopDriver",
    "NewConversation",
    "Sample",
    "Say",
    "Scenario",
    "Session",
    "StopAfter",
    "TurnRecord",
    "WorldHolder",
    "run_sample",
    "sample_line",
    "unbacked",
]
