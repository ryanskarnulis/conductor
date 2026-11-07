"""Seeded random walks through conductor's route, loop and delegate tools (#53).

The loop tests drive one feature each: one provider death, one over-budget
delegate, one schema correction. The combinations nobody wrote a test for — an
app that dies mid-write in the second call of a batch after a correction, a
provider death on the turn after a handoff — are what this drives
(STANDARD.md §6; after PCC's ``tests/trajectory.py``).

A walk is a sequence of user messages posted through the real route
(``POST /api/agent/conversations/{id}/messages``) to the real
:class:`~app.ai.loop.AgentLoop`, the real registry and the real delegate
tools, whose network hop lands on the session evals' fake fleet
(``fake_world.py``: apps that really write, and leave a receipt per write). A
scripted provider (:class:`WalkProvider`) stands in for llama-server. Each
message's script — tool batches across every registered tool, unknown tools,
bad arguments, an unparseable turn, a final reply — is drawn from a seeded
:class:`random.Random`, and so is the one fault a message may carry:

- ``provider_dies``: llama-server goes away on a given model call;
- ``provider_crashes``: the provider raises something that is not a
  ``ProviderError`` (a bug in it);
- ``app_down``: an app's agent is unreachable on a given delegate call;
- ``app_partial``: an app writes, then fails the response (``outcome
  unknown``, #92);
- ``app_bug``: the delegate client raises something that is not a
  ``DelegateError`` — "a tool that raises";
- ``stop``: the person presses Stop during a given delegate call. Drawn only
  when the app has a cancel route (#66); until then the named case for it is a
  strict xfail.

There is no run deadline in conductor yet (#52 brings ``timed_out``); the
fault joins the list when it lands.

After every message every invariant in :data:`INVARIANTS` is checked against
what the step observed. The one that matters most for the revamp is the
ledger invariant (#69): **every app write has a receipt row** — no write in
the world that the turn's ledger receipts don't account for, by the entity it
touched, or as conductor's ``unknown`` when the app's answer was lost.

The model is scripted and the world is fresh, so a walk is a pure function of
its seed: a breach is reported with the seed, the steps so far and the command
that replays it (:func:`replay_command`).
"""

from __future__ import annotations

import random
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main_module
from app.ai.loop import AgentLoop, AgentRunFailed, AgentRunResult
from app.ai.provider import (
    ChatResult,
    ProviderRequestError,
    ToolCall,
    ToolCallArgumentsError,
    ToolSpec,
)
from app.api import rate_limit, routes_agent, turn_activity
from app.config import get_settings
from app.db.models import Base, LedgerTurn
from app.db.session import get_db
from app.fleet.delegate import DelegateOutcomeUnknown, DelegateUnavailable, MessageExchange
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.services import ledger
from app.tools import registry
from tests.fake_world import Receipt, World, client_factory, write_fixture_fleet

# --- the step ----------------------------------------------------------------------

Call = tuple[str, dict[str, Any]]
FaultSite = Literal["chat", "tool"]
FaultKind = Literal[
    "provider_dies", "provider_crashes", "app_down", "app_partial", "app_bug", "stop"
]
_SITE: dict[FaultKind, FaultSite] = {
    "provider_dies": "chat",
    "provider_crashes": "chat",
    "app_down": "tool",
    "app_partial": "tool",
    "app_bug": "tool",
    "stop": "tool",
}


@dataclass(frozen=True)
class Turn:
    """One scripted model turn: a batch of calls, an unparseable turn, or text."""

    kind: Literal["calls", "malformed", "text"]
    calls: tuple[Call, ...] = ()
    text: str = ""

    def describe(self) -> str:
        if self.kind == "calls":
            return "calls[" + ", ".join(f"{name}({_args(args)})" for name, args in self.calls) + "]"
        if self.kind == "malformed":
            return "malformed"
        return f"text({self.text[:30]!r})"


@dataclass(frozen=True)
class Fault:
    kind: FaultKind
    at: int  # the n-th (0-based) model call, or delegate call, of the step

    def describe(self) -> str:
        return f"{self.kind}@{_SITE[self.kind]}{self.at}"


@dataclass(frozen=True)
class Step:
    message: str
    turns: tuple[Turn, ...]
    fault: Fault | None = None

    def describe(self) -> str:
        fault = f" !{self.fault.describe()}" if self.fault else ""
        return f"{self.message!r}: " + " → ".join(t.describe() for t in self.turns) + fault


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={str(v)[:24]!r}" for k, v in args.items())


class InvariantBreach(AssertionError):
    def __init__(self, invariant: str, message: str) -> None:
        super().__init__(f"{invariant}: {message}")
        self.invariant = invariant


# --- drawing a step -------------------------------------------------------------------

_TASK_ASKS = (
    "what's due today?",
    "add a task to water the plants tomorrow",
    "push the dentist appointment to friday",
    "mark the groceries task done",
    "delete the call mom task",
    "what's overdue?",
    "add a task called “fix the fence” due next week",
)
_MUSIC_ASKS = (
    "download the new Caribou album",
    "grab Odessa by Caribou",
    "what songs still need sorting?",
    "sort my whole library",
    "how are my downloads going?",
    "put Glue in the house folder",
)
_USER = (
    "what's due today?",
    "add a task to water the plants",
    "grab Odessa by Caribou",
    "let's play chess",
    "what can you do?",
    "sort my music",
    "yes",
    "undo that",
)


def _valid_call(rng: random.Random, tools: Sequence[str]) -> Call:
    name = rng.choice(tools)
    if name == "ask_tasks":
        return name, {"message": rng.choice(_TASK_ASKS)}
    if name == "ask_music":
        return name, {"message": rng.choice(_MUSIC_ASKS)}
    if name.startswith("open_"):
        return name, {"intent": rng.choice(("let's play", "", "how is the game going?"))}
    return name, {}


def _call(rng: random.Random, tools: Sequence[str]) -> Call:
    roll = rng.random()
    if roll < 0.08:
        return "ask_weather", {"message": "tomorrow?"}  # unknown tool: a schema error
    if roll < 0.16:
        return "ask_tasks", {}  # missing argument: a schema error
    if roll < 0.20:
        return "ask_tasks", {"message": "   "}  # a domain refusal (empty message)
    return _valid_call(rng, tools)


def next_step(rng: random.Random, tools: Sequence[str], faults: Sequence[FaultKind]) -> Step:
    """One user message and its script, and maybe one fault."""
    turns: list[Turn] = []
    # Mostly short runs; now and then one long enough to hit the iteration cap.
    for _ in range(rng.choice((0, 1, 1, 2, 2, 3, 3, 7))):
        if rng.random() < 0.1:
            turns.append(Turn("malformed"))
        else:
            batch = tuple(_call(rng, tools) for _ in range(rng.choice((1, 1, 2))))
            turns.append(Turn("calls", calls=batch))
    turns.append(
        Turn("text", text=rng.choice(("Done.", "Here you go.", "All set — nothing else.")))
    )
    fault: Fault | None = None
    if faults and rng.random() < 0.35:
        kind = rng.choice(list(faults))
        if _SITE[kind] == "chat":
            fault = Fault(kind, rng.randrange(len(turns)))
        else:
            delegates = sum(
                1 for t in turns for name, _ in t.calls if name in ("ask_tasks", "ask_music")
            )
            if delegates:
                fault = Fault(kind, rng.randrange(delegates))
    return Step(rng.choice(_USER), tuple(turns), fault)


# --- what a step observed ---------------------------------------------------------------


@dataclass
class LedgerView:
    """The step's ledger turn as the walk reads it back (#69)."""

    status: str
    stop_reason: str | None
    steps: list[tuple[str, str]]  # (tool, outcome), in order
    receipts: list[tuple[str, str | None, str, frozenset[str]]]  # app, tool, outcome, touched


@dataclass
class Observed:
    step: Step
    number: int
    status: int
    conversation: int
    run: AgentRunResult | None
    saved_before: list[dict[str, Any]]
    saved_after: list[dict[str, Any]]
    receipts: list[Receipt]
    sends: dict[str, int]
    requests: list[list[dict[str, Any]]]
    chat_calls: int = 0
    tool_events: list[tuple[str, str]] = field(default_factory=list)  # (app, outcome)
    fault_fired: bool = False
    stopped_at: int | None = None  # chat_calls when Stop landed
    ledger: LedgerView | None = None

    @property
    def new_messages(self) -> list[dict[str, Any]]:
        seen = {m["id"] for m in self.saved_before}
        return [m for m in self.saved_after if m["id"] not in seen]

    @property
    def saved_assistant(self) -> dict[str, Any] | None:
        return next((m for m in self.new_messages if m["role"] == "assistant"), None)


# --- the harness -------------------------------------------------------------------------


class WalkProvider:
    """Plays the current step's script; fires its chat fault."""

    def __init__(self, harness: Harness) -> None:
        self._harness = harness

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        run = self._harness.current
        assert run is not None, "a model call outside a step"
        index = run.chat_calls
        run.chat_calls += 1
        run.requests.append([dict(m) for m in messages])
        fault = run.step.fault
        if fault is not None and _SITE[fault.kind] == "chat" and fault.at == index:
            run.fault_fired = True
            if fault.kind == "provider_dies":
                raise ProviderRequestError("llama-server request failed: connection refused")
            raise RuntimeError("provider bug: unexpected payload")
        turns = run.step.turns
        if index >= len(turns):
            # The loop asked for more than the script holds (a correction
            # re-ask): answer with the final text again.
            turn = turns[-1]
        else:
            turn = turns[index]
        if turn.kind == "malformed":
            raise ToolCallArgumentsError("ask_tasks", "Expecting value: line 1 column 1")
        if turn.kind == "text":
            return ChatResult(content=turn.text, tool_calls=[], finish_reason="stop", usage=None)
        return ChatResult(
            content=None,
            tool_calls=[
                ToolCall(id=f"s{run.number}c{index}t{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(turn.calls)
            ],
            finish_reason="tool_calls",
            usage=None,
        )


class _CapturingLoop(AgentLoop):
    """The real loop; keeps its result so invariants can compare the saved turn."""

    def __init__(self, harness: Harness) -> None:
        super().__init__(
            WalkProvider(harness),
            max_iterations=get_settings().conductor_max_iterations,
            fleet_section=getattr(app.state, "fleet_section", None),
        )
        self._harness = harness

    def run(self, user_message: str, **kwargs: Any) -> AgentRunResult:
        assert self._harness.current is not None
        try:
            result = super().run(user_message, **kwargs)
        except AgentRunFailed as failed:
            # A failure stop still carries what ran (#52): the route saves it.
            self._harness.current.run = failed.result
            raise
        self._harness.current.run = result
        return result


class _FaultyClient:
    """The fake fleet's client with the step's delegate fault spliced in."""

    def __init__(self, inner: Any, app_name: str, harness: Harness) -> None:
        self._inner = inner
        self._app = app_name
        self._harness = harness

    def __enter__(self) -> _FaultyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        return int(self._inner.create_conversation(title=title))

    def send_message(
        self, conversation_id: int, message: str, *, confirm_token: str | None = None
    ) -> MessageExchange:
        run = self._harness.current
        assert run is not None
        index = sum(run.sends.values())
        run.sends[self._app] = run.sends.get(self._app, 0) + 1
        fault = run.step.fault
        firing = fault is not None and _SITE[fault.kind] == "tool" and fault.at == index
        if firing:
            assert fault is not None
            run.fault_fired = True
            if fault.kind == "app_down":
                run.tool_events.append((self._app, "down"))
                raise DelegateUnavailable(f"{self._app} is down")
            if fault.kind == "app_bug":
                run.tool_events.append((self._app, "bug"))
                raise RuntimeError(f"{self._app} client bug")
            if fault.kind == "stop":
                self._harness.press_stop()
        exchange: MessageExchange = self._inner.send_message(conversation_id, message)
        if firing and fault is not None and fault.kind == "app_partial":
            run.tool_events.append((self._app, "partial"))
            raise DelegateOutcomeUnknown(f"{self._app} returned 502 after partial work")
        run.tool_events.append((self._app, "ok"))
        return exchange

    def cancel(self, conversation_id: int) -> bool:
        run = self._harness.current
        if run is not None:
            run.tool_events.append((self._app, "cancel"))
        return bool(self._inner.cancel(conversation_id))


_CANCEL_ROUTE = re.compile(r"^/api/agent/conversations/\{[^}]+\}/(?:cancel|stop)$")


class Harness:
    """A started app over an in-memory database and a fresh fake world."""

    def __init__(self, fleet_dir: Path) -> None:
        self.world = World()
        self.current: Observed | None = None
        self.current_message: str | None = None
        self._number = 0
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        self._db = self._factory()
        self._mp = pytest.MonkeyPatch()
        inner = client_factory(self.world)
        self._mp.setattr(get_settings(), "fleet_manifest_dir", fleet_dir)
        self._mp.setattr(
            main_module,
            "default_client_factory",
            lambda: lambda fleet_app: _FaultyClient(inner(fleet_app), fleet_app.name, self),
        )
        app.dependency_overrides[get_db] = lambda: self._db
        app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(
            self._factory
        )
        app.dependency_overrides[routes_agent.get_agent_loop] = lambda: _CapturingLoop(self)
        self._saved_registry = dict(registry._REGISTRY)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.client.__enter__()
        # The OpenAPI paths, not ``app.routes``: included routers nest there.
        routes = set(app.openapi()["paths"])
        self.cancel_path = next((p for p in routes if _CANCEL_ROUTE.match(p)), None)
        created = self.client.post("/api/agent/conversations", json={})
        self.conversation = int(created.json()["id"])

    @property
    def tools(self) -> list[str]:
        return sorted(tool.name for tool in registry.all_tools())

    @property
    def faults(self) -> list[FaultKind]:
        kinds: list[FaultKind] = [
            "provider_dies",
            "provider_crashes",
            "app_down",
            "app_partial",
            "app_bug",
        ]
        if self.cancel_path is not None:
            kinds.append("stop")
        return kinds

    def press_stop(self) -> None:
        assert self.cancel_path is not None, "Stop drawn with no cancel route"
        assert self.current is not None
        self.current.stopped_at = self.current.chat_calls
        path = self.cancel_path.replace(
            self.cancel_path[self.cancel_path.index("{") : self.cancel_path.index("}") + 1],
            str(self.conversation),
        )
        # On its own request, as the person's Stop would arrive.
        stopper = threading.Thread(target=lambda: self.client.post(path), daemon=True)
        stopper.start()
        stopper.join(timeout=30)

    def _saved(self) -> list[dict[str, Any]]:
        detail = self.client.get(f"/api/agent/conversations/{self.conversation}")
        return list(detail.json()["messages"])

    def take(self, step: Step) -> Observed:
        self._number += 1
        self.world.turn = self._number
        before = self._saved()
        observed = Observed(
            step=step,
            number=self._number,
            status=0,
            conversation=self.conversation,
            run=None,
            saved_before=before,
            saved_after=[],
            receipts=[],
            sends={},
            requests=[],
        )
        self.current = observed
        self.current_message = step.message
        rate_limit._reset()
        try:
            response = self.client.post(
                f"/api/agent/conversations/{self.conversation}/messages",
                json={"content": step.message},
            )
        finally:
            self.current = None
        observed.status = response.status_code
        observed.saved_after = self._saved()
        observed.receipts = self.world.writes(turn=self._number)
        observed.ledger = self._ledger()
        return observed

    def _ledger(self) -> LedgerView | None:
        self._db.expire_all()
        turn = self._db.execute(
            select(LedgerTurn)
            .where(LedgerTurn.conversation_id == self.conversation)
            .order_by(LedgerTurn.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if turn is None or turn.words != (self.current_message or turn.words):
            return None
        steps = [(step.tool, step.outcome) for plan in turn.plans for step in plan.steps]
        receipts = [
            (
                receipt.app,
                receipt.tool,
                receipt.outcome,
                frozenset(f"{t.entity.kind}:{t.entity.ref}" for t in receipt.touched),
            )
            for receipt in ledger.receipts_for(self._db, turn.id)
        ]
        return LedgerView(turn.status, turn.stop_reason, steps, receipts)

    def close(self) -> None:
        try:
            self.client.__exit__(None, None, None)
        finally:
            app.dependency_overrides.clear()
            registry._REGISTRY.clear()
            registry._REGISTRY.update(self._saved_registry)
            self._mp.undo()
            self._db.close()
            self._engine.dispose()
            rate_limit._reset()
            turn_activity._reset()


# --- invariants ---------------------------------------------------------------------------


def check_every_call_has_one_answer(observed: Observed) -> None:
    """Every tool call the model made got exactly one ``role: tool`` answer."""
    if observed.run is None:
        return
    asked: list[str] = []
    answered: list[str] = []
    for message in observed.run.messages:
        asked += [call["id"] for call in message.get("tool_calls") or []]
        if message.get("role") == "tool":
            answered.append(message["tool_call_id"])
    if sorted(asked) != sorted(answered):
        raise InvariantBreach("every_call_has_one_answer", f"asked {asked}, answered {answered}")


def check_saved_turn_is_what_ran(observed: Observed) -> None:
    """The saved assistant turn is exactly what the run did. A failed run
    (502/500) saves its partial record when anything ran (#52), and only the
    person's words when nothing did; it never invents a turn."""
    new = observed.new_messages
    roles = [m["role"] for m in new]
    run = observed.run
    if observed.status != 200 and (run is None or not run.tool_calls):
        if roles != ["user"]:
            raise InvariantBreach(
                "saved_turn_is_what_ran", f"HTTP {observed.status}, nothing ran, saved {roles}"
            )
        return
    if roles != ["user", "assistant"] or run is None:
        raise InvariantBreach("saved_turn_is_what_ran", f"HTTP {observed.status} saved {roles}")
    saved = new[1]
    ran = [(r.tool, r.arguments, r.result, r.error) for r in run.tool_calls]
    kept = [(c["tool"], c["arguments"], c["result"], c["error"]) for c in saved["tool_calls"] or []]
    if kept != ran or saved["content"] != run.reply or saved["stop_reason"] != run.stop_reason:
        raise InvariantBreach("saved_turn_is_what_ran", f"saved {kept!r}, ran {ran!r}")


def check_history_is_text_only(observed: Observed) -> None:
    """Earlier turns reach the model as text alone: no tool calls, no tool
    results, nothing of an app's transcript (CLAUDE.md, "history replays text
    turns only")."""
    for request in observed.requests[:1]:
        prior = request[1:-1]  # after the system prompt, before this user message
        for message in prior:
            if message.get("role") not in ("user", "assistant") or message.get("tool_calls"):
                raise InvariantBreach(
                    "history_is_text_only", f"history carried {message.get('role')} {message}"
                )


def check_delegate_budget(observed: Observed) -> None:
    """No app is asked more than the per-turn budget."""
    cap = get_settings().conductor_delegate_calls_per_turn_per_app
    over = {app_name: n for app_name, n in observed.sends.items() if n > cap}
    if over:
        raise InvariantBreach("delegate_budget", f"sends {over} over the cap {cap}")


def check_handoffs_ask_nothing(observed: Observed) -> None:
    """``open_`` apps are handed the person, never asked."""
    if observed.sends.get("chess"):
        raise InvariantBreach("handoffs_ask_nothing", f"chess was asked {observed.sends}")


def check_every_turn_is_on_the_ledger(observed: Observed) -> None:
    """Every turn is on the ledger, ended the way the saved turn says, with
    one step per call the saved turn shows, in order (#69)."""
    view = observed.ledger
    if view is None:
        raise InvariantBreach("every_turn_is_on_the_ledger", "no ledger turn for this message")
    if view.status != "ended":
        raise InvariantBreach("every_turn_is_on_the_ledger", f"turn left {view.status}")
    saved = observed.saved_assistant
    if saved is None:
        return
    if view.stop_reason != saved.get("stop_reason"):
        raise InvariantBreach(
            "every_turn_is_on_the_ledger",
            f"ledger says {view.stop_reason}, the saved turn {saved.get('stop_reason')}",
        )
    calls = [call["tool"] for call in saved.get("tool_calls") or []]
    if [tool for tool, _ in view.steps] != calls:
        raise InvariantBreach(
            "every_turn_is_on_the_ledger", f"ledger steps {view.steps} vs saved calls {calls}"
        )


def check_every_app_write_has_a_receipt_row(observed: Observed) -> None:
    """The ledger invariant (#69): every write an app made this step has a
    receipt row — a ``done`` receipt touching what it wrote, or conductor's
    ``unknown`` for a call to that app whose answer was lost. A write
    conductor has no record of can't be narrated, undone or repeated."""
    if not observed.receipts:
        return
    rows = observed.ledger.receipts if observed.ledger is not None else []
    for receipt in observed.receipts:
        if not any(
            app == receipt.app
            and (
                outcome == "unknown"
                or (
                    outcome == "done"
                    and (
                        _touches(touched, receipt.ref)
                        or (receipt.kind == "cancel" and tool == "cancel")
                    )
                )
            )
            for app, tool, outcome, touched in rows
        ):
            raise InvariantBreach(
                "every_app_write_has_a_receipt_row",
                f"{receipt.app} wrote {receipt.kind} {receipt.ref} {receipt.title!r} but the "
                f"ledger (HTTP {observed.status}) has no receipt for it: {rows}",
            )


def _touches(touched: frozenset[str], ref: str) -> bool:
    """The world's ``kind:id`` against the receipt's refs; the world calls a
    download job ``job`` where music's record says ``download_job``."""
    kind, _, ident = ref.partition(":")
    return any(
        key_ident == ident and (key_kind == kind or key_kind.endswith(f"_{kind}"))
        for key_kind, _, key_ident in (key.partition(":") for key in touched)
    )


def check_every_app_write_is_recorded(observed: Observed) -> None:
    """Every write an app made this step is accounted for by a delegate call
    in the saved turn — answered, or reported as "outcome unknown"."""
    if not observed.receipts:
        return
    saved = observed.saved_assistant
    calls = (saved or {}).get("tool_calls") or []
    for receipt in observed.receipts:
        tool = f"ask_{receipt.app}"
        if not any(
            call["tool"] == tool
            and (call["result"] is not None or "stopped partway" in (call["error"] or ""))
            for call in calls
        ):
            raise InvariantBreach(
                "every_app_write_is_recorded",
                f"{receipt.app} wrote {receipt.kind} {receipt.title!r} but the saved turn "
                f"(HTTP {observed.status}) has no {tool} call that ran",
            )


def check_fault_ends_the_run_as_documented(observed: Observed) -> None:
    """Each fault ends the turn the way the route and loop document."""
    fault = observed.step.fault
    if fault is None or not observed.fault_fired:
        if observed.status != 200:
            raise InvariantBreach("fault_ends_as_documented", f"no fault, HTTP {observed.status}")
        return
    expected = {
        "provider_dies": 502,
        "provider_crashes": 500,
        "app_bug": 500,  # unexpected exceptions propagate: a bug, not a domain error
        "app_down": 200,
        "app_partial": 200,
        "stop": 200,
    }[fault.kind]
    if observed.status != expected:
        raise InvariantBreach(
            "fault_ends_as_documented", f"{fault.describe()} → HTTP {observed.status}"
        )
    if fault.kind in ("app_down", "app_partial") and observed.run is not None:
        errors = [r.error or "" for r in observed.run.tool_calls if r.error]
        want = "unavailable" if fault.kind == "app_down" else "stopped partway"
        if not any(want in error for error in errors):
            raise InvariantBreach(
                "fault_ends_as_documented", f"{fault.kind}: no '{want}' error in {errors}"
            )
    if fault.kind == "stop":
        saved = observed.saved_assistant or {}
        if saved.get("stop_reason") != "cancelled":
            raise InvariantBreach(
                "fault_ends_as_documented", f"Stop ended {saved.get('stop_reason')}"
            )


def check_nothing_starts_after_stop(observed: Observed) -> None:
    """After Stop, no further model call starts."""
    if observed.stopped_at is not None and observed.chat_calls > observed.stopped_at:
        raise InvariantBreach(
            "nothing_starts_after_stop",
            f"{observed.chat_calls - observed.stopped_at} model call(s) after Stop",
        )


INVARIANTS: tuple[Callable[[Observed], None], ...] = (
    check_every_call_has_one_answer,
    check_saved_turn_is_what_ran,
    check_history_is_text_only,
    check_delegate_budget,
    check_handoffs_ask_nothing,
    check_every_app_write_is_recorded,
    check_every_turn_is_on_the_ledger,
    check_every_app_write_has_a_receipt_row,
    check_fault_ends_the_run_as_documented,
    check_nothing_starts_after_stop,
)

# Breaches a walk reaches on main, by invariant: tolerated in the corpus,
# pinned strictly in a named case, removed when fixed. Empty since #52 saves a
# failed run's partial record (the write-before-a-provider-death case the walks
# first found is now ``test_a_write_before_a_provider_death_is_kept``).
KNOWN_BUGS: dict[str, str] = {}


def _known(observed: Observed, breach: InvariantBreach) -> bool:
    fault = observed.step.fault
    return (
        breach.invariant in KNOWN_BUGS
        and fault is not None
        and fault.kind in ("provider_dies", "provider_crashes", "app_bug")
        and observed.status != 200
    )


# --- walking --------------------------------------------------------------------------------

CORPUS_LENGTH = 8


def replay_command(seed: int, length: int) -> str:
    return (
        f"CONDUCTOR_TRAJ_SEED={seed} CONDUCTOR_TRAJ_LENGTH={length} "
        ".venv/bin/pytest tests/test_trajectories.py -k replay -s"
    )


def fixture_fleet(root: Path) -> Path:
    return write_fixture_fleet(root)


@contextmanager
def harness(fleet_dir: Path) -> Iterator[Harness]:
    h = Harness(fleet_dir)
    try:
        yield h
    finally:
        h.close()


def run_steps(
    fleet_dir: Path,
    steps: Sequence[Step] | Callable[[Harness, int], Step],
    length: int | None = None,
    *,
    seed: int | None = None,
    tolerate_known_bugs: bool = True,
) -> list[Observed]:
    """Take ``steps`` (a list, or a function drawing step *n*), checking every
    invariant after each. A breach raises with the steps so far and, for a
    seeded walk, the replay command."""
    log: list[Observed] = []
    with harness(fleet_dir) as h:
        count = len(steps) if isinstance(steps, Sequence) else (length or CORPUS_LENGTH)
        for index in range(count):
            step = steps[index] if isinstance(steps, Sequence) else steps(h, index)
            observed = h.take(step)
            log.append(observed)
            for invariant in INVARIANTS:
                try:
                    invariant(observed)
                except InvariantBreach as breach:
                    if tolerate_known_bugs and _known(observed, breach):
                        continue
                    raise AssertionError(_report(log, breach, seed, count)) from breach
    return log


def walk(fleet_dir: Path, seed: int, length: int = CORPUS_LENGTH) -> list[Observed]:
    rng = random.Random(seed)
    return run_steps(
        fleet_dir,
        lambda h, _index: next_step(rng, h.tools, h.faults),
        length,
        seed=seed,
    )


def _report(log: Sequence[Observed], breach: InvariantBreach, seed: int | None, length: int) -> str:
    lines = [f"invariant breached at step {len(log)}: {breach}"]
    if seed is not None:
        lines.append(f"seed: {seed}")
        lines.append(f"replay: {replay_command(seed, length)}")
    lines.extend(describe(log))
    return "\n".join(lines)


def describe(log: Sequence[Observed]) -> Iterator[str]:
    for observed in log:
        stop = observed.run.stop_reason if observed.run else "-"
        writes = ",".join(f"{r.kind}:{r.title}" for r in observed.receipts) or "-"
        yield (
            f"  {observed.number}. {observed.step.describe()} ⇒ HTTP {observed.status} "
            f"stop={stop} writes={writes}"
        )
