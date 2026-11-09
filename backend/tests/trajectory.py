"""Seeded random walks through conductor's route, interpreter and delegate tools (#53).

The unit tests drive one feature each: one provider death, one over-budget
plan, one repair. The combinations nobody wrote a test for — an app that dies
mid-write in the second step of a cross-app plan, a provider death on the
narrator after a write, a Stop during an undo split off a longer plan — are
what this drives (STANDARD.md §6; after PCC's ``tests/trajectory.py``).

A walk is a sequence of user messages posted through the real route
(``POST /api/agent/conversations/{id}/messages``) to the real
:class:`~app.ai.interpreter.Interpreter` (planner → validator → interpreter →
narrator, #75), the real registry and the real delegate tools, whose network
hop lands on the session evals' fake fleet (``fake_world.py``: apps that
really write, and leave a receipt per write); a ``look`` reads the same
world's glance. A scripted provider (:class:`WalkProvider`) stands in for
llama-server and answers by request: the planner's calls (``json_schema``
named ``plan``) get the step's plan replies in order, the speaker's
structured call an answer, and the narrator plain text. Each message's script
is drawn from a seeded :class:`random.Random`: plans over every step kind
(asks to both apps with and without ``{sN}`` data flow, looks, a handoff,
``ask_user``, ``recall``, ``confirm``, the control steps alone or inside a
longer plan, ``think`` last or not), the direct answers, and replies that
aren't plans or can't run, so the repair rounds and code's unplanned and
clarifying answers are walked too. The retired loop's script kinds (unknown
tools, bad arguments, an unparseable tool turn, the iteration cap) are gone
with it. A message may also carry one fault:

- ``provider_dies``: llama-server goes away on a given model call;
- ``provider_crashes``: the provider raises something that is not a
  ``ProviderError`` (a bug in it);
- ``app_down``: an app's agent is unreachable on a given delegate call;
- ``app_partial``: an app writes, then fails the response (``outcome
  unknown``, #92);
- ``app_bug``: the delegate client raises something that is not a
  ``DelegateError`` — "a tool that raises";
- ``stop``: the person presses Stop during a given delegate call (#66).

A provider fault on a planner call fails the turn (502/500); on the narrator
or the speaker it falls back to code's reply and the turn stands.

After every message every invariant in :data:`INVARIANTS` is checked against
what the step observed. The one that matters most is the ledger invariant
(#69): **every app write has a receipt row** — no write in the world that the
turn's ledger receipts don't account for, by the entity it touched, or as
conductor's ``unknown`` when the app's answer was lost.

The model is scripted and the world is fresh, so a walk is a pure function of
its seed: a breach is reported with the seed, the steps so far and the command
that replays it (:func:`replay_command`).
"""

from __future__ import annotations

import json
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
from app.ai.interpreter import Interpreter
from app.ai.provider import ChatResult, ProviderRequestError, ToolSpec
from app.ai.runtime import AgentRunFailed, AgentRunResult
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
# The step kinds and direct answers a walk draws, for the instrument's own check.
KINDS = frozenset(
    {"ask", "look", "open", "ask_user", "recall", "cancel", "amend", "confirm", "undo", "think"}
)
ANSWERS = frozenset({"knowledge", "decline", "self", "chat"})


@dataclass(frozen=True)
class Turn:
    """One scripted planner reply: a plan (or direct answer), or text that isn't one."""

    kind: Literal["plan", "garbage"]
    body: str = ""

    @staticmethod
    def plan(body: dict[str, Any]) -> Turn:
        return Turn("plan", json.dumps(body))

    def describe(self) -> str:
        if self.kind == "garbage":
            return "garbage"
        body = json.loads(self.body)
        if "answer" in body:
            return f"answer:{body['answer']}"
        return "[" + " | ".join(_describe_step(step) for step in body["steps"]) + "]"


def _describe_step(step: dict[str, Any]) -> str:
    what = step.get("app") or step.get("target") or step.get("answer") or ""
    text = step.get("request") or step.get("intent") or step.get("query") or ""
    return f"{step['id']}:{step['do']}:{what}" + (f"({text[:24]!r})" if text else "")


@dataclass(frozen=True)
class Fault:
    kind: FaultKind
    at: int  # the n-th (0-based) model call, or delegate call, of the step

    def describe(self) -> str:
        return f"{self.kind}@{_SITE[self.kind]}{self.at}"


@dataclass(frozen=True)
class Step:
    message: str
    # The planner's replies, in order; a repair past the end gets the last again.
    turns: tuple[Turn, ...]
    # What the narrator (and the speaker) say.
    text: str = "Done."
    fault: Fault | None = None

    def describe(self) -> str:
        fault = f" !{self.fault.describe()}" if self.fault else ""
        return f"{self.message!r}: " + " → ".join(t.describe() for t in self.turns) + fault


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
    "actually, stop",
)
_TEXT = ("Done.", "Here you go.", "All set — nothing else.")


def _ask(rng: random.Random) -> dict[str, Any]:
    app_name = rng.choice(("tasks", "music"))
    request = rng.choice(_TASK_ASKS if app_name == "tasks" else _MUSIC_ASKS)
    destructive = "delete" in request or rng.random() < 0.05
    return {"do": "ask", "app": app_name, "request": request, "destructive": destructive}


def _control(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(("cancel", "amend", "undo", "undo"))
    if kind == "amend":
        return {"do": "amend", "target": "the pending plan", "change": "make it friday"}
    return {"do": kind, "target": rng.choice(("last", "the pending plan", "the job"))}


def _steps(rng: random.Random) -> list[dict[str, Any]]:
    """The body of one step plan: mostly asks, with the rest of the kinds mixed in."""
    roll = rng.random()
    if roll < 0.10:
        return [_control(rng)]
    if roll < 0.24:
        first = {"do": "confirm", "answer": rng.choice(("yes", "yes", "no"))}
        return [first] + ([_ask(rng)] if rng.random() < 0.3 else [])
    if roll < 0.29:
        asked = {"do": "ask_user", "question": "Which one?", "options": []}
        return [asked, {**_ask(rng), "request": "do it for {s1}"}]
    steps: list[dict[str, Any]] = []
    for _ in range(rng.choice((1, 1, 1, 2, 2, 3, 4))):
        kind = rng.random()
        if kind < 0.65:
            steps.append(_ask(rng))
        elif kind < 0.82:
            steps.append({"do": "look", "app": rng.choice(("tasks", "music", "chess"))})
        else:
            steps.append({"do": "recall", "query": rng.choice(("the fence", "that Caribou song"))})
    if steps and rng.random() < 0.2:
        # Data flow from an earlier step: an ask (or recall) names; a look doesn't.
        source = rng.randrange(len(steps))
        steps.append(
            {
                "do": "ask",
                "app": "tasks",
                "request": f"add a task to sort {{s{source + 1}}} this weekend",
                "destructive": False,
            }
        )
    if rng.random() < 0.08:
        steps.insert(rng.randrange(len(steps) + 1), _control(rng))
    if rng.random() < 0.08:
        inputs = [f"s{n}" for n in range(1, len(steps) + 1)][:2]
        think = {"do": "think", "problem": "plan my week", "inputs": inputs}
        steps.insert(rng.choice((len(steps), max(0, len(steps) - 1))), think)
    if rng.random() < 0.1:
        steps.append({"do": "open", "app": "chess", "intent": rng.choice(("let's play", ""))})
    return steps


def _numbered(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Ids s1, s2, ... in order; a think's inputs stay those before it."""
    numbered = []
    for n, step in enumerate(steps, start=1):
        step = {"id": f"s{n}", **step}
        if step["do"] == "think":
            step["inputs"] = [ref for ref in step["inputs"] if int(ref[1:]) < n]
        numbered.append(step)
    return {"steps": numbered}


def _misnumbered(app_name: str, request: str) -> Turn:
    """A one-app plan the validator refuses (its id isn't s1)."""
    step = {"id": "s2", "do": "ask", "app": app_name, "request": request, "destructive": False}
    return Turn("plan", json.dumps({"steps": [step]}))


def _replies(rng: random.Random) -> tuple[Turn, ...]:
    """The planner's replies for one message: usually one plan; now and then
    one that isn't a plan first, or none that can run at all."""
    roll = rng.random()
    if roll < 0.04:
        # Tries that disagree on the app: code asks which was meant.
        music = _misnumbered("music", rng.choice(_MUSIC_ASKS))
        return (music, _misnumbered("tasks", rng.choice(_TASK_ASKS)), music)
    if roll < 0.08:
        return (Turn("garbage"),)
    if roll < 0.16:
        return (Turn.plan({"answer": rng.choice(sorted(ANSWERS))}),)
    plan = Turn.plan(_numbered(_steps(rng)))
    if roll < 0.24:
        return (Turn("garbage"), plan)
    return (plan,)


def _delegates(turn: Turn) -> int:
    """Delegate calls a planner reply could make: asks, confirms and undos."""
    if turn.kind != "plan":
        return 0
    steps = json.loads(turn.body).get("steps") or []
    return sum(1 for step in steps if step["do"] in ("ask", "confirm", "undo"))


def next_step(rng: random.Random, faults: Sequence[FaultKind], *, waiting: bool = False) -> Step:
    """One user message and its script, and maybe one fault. ``waiting``: the
    last turn left a question (a pause, an app's confirm), so the person
    answers it, or cancels or changes it, more often than chance would."""
    if waiting and rng.random() < 0.6:
        answer = rng.random()
        if answer < 0.6:
            reply = [{"do": "confirm", "answer": rng.choice(("yes", "yes", "no"))}]
        elif answer < 0.8:
            reply = [{"do": "cancel", "target": "the pending plan"}]
        else:
            reply = [{"do": "amend", "target": "the pending plan", "change": "make it friday"}]
        turns: tuple[Turn, ...] = (Turn.plan(_numbered(reply)),)
    else:
        turns = _replies(rng)
    fault: Fault | None = None
    if faults and rng.random() < 0.35:
        kind = rng.choice(list(faults))
        if _SITE[kind] == "chat":
            # The planner's calls, then the narrator's or speaker's.
            fault = Fault(kind, rng.randrange(len(turns) + 1))
        else:
            delegates = max(_delegates(turn) for turn in turns)
            if delegates:
                fault = Fault(kind, rng.randrange(delegates))
    return Step(rng.choice(_USER), turns, rng.choice(_TEXT), fault)


def drawn_kinds(step: Step) -> set[str]:
    """The step kinds and direct answers a step's replies use."""
    kinds: set[str] = set()
    for turn in step.turns:
        if turn.kind != "plan":
            continue
        body = json.loads(turn.body)
        if "answer" in body:
            kinds.add(body["answer"])
        kinds |= {step["do"] for step in body.get("steps") or []}
    return kinds


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
    planner_calls: int = 0
    tool_events: list[tuple[str, str]] = field(default_factory=list)  # (app, outcome)
    fault_fired: bool = False
    # The model call a chat fault fired on: "plan", "answer" (the speaker's
    # structured call) or "text" (the narrator's or speaker's plain call).
    fault_role: str | None = None
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
    """Plays the current step's script by request; fires its chat fault."""

    def __init__(self, harness: Harness) -> None:
        self._harness = harness

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        role: str | None = None,
    ) -> ChatResult:
        run = self._harness.current
        assert run is not None, "a model call outside a step"
        index = run.chat_calls
        run.chat_calls += 1
        run.requests.append([dict(m) for m in messages])
        role = str((response_format or {}).get("json_schema", {}).get("name") or "text")
        fault = run.step.fault
        if fault is not None and _SITE[fault.kind] == "chat" and fault.at == index:
            run.fault_fired, run.fault_role = True, role
            if fault.kind == "provider_dies":
                raise ProviderRequestError("llama-server request failed: connection refused")
            raise RuntimeError("provider bug: unexpected payload")
        if role == "plan":
            turns = run.step.turns
            turn = turns[min(run.planner_calls, len(turns) - 1)]
            run.planner_calls += 1
            content = "this is not a plan" if turn.kind == "garbage" else turn.body
        elif role == "answer":
            content = json.dumps({"about_you": False, "reply": run.step.text})
        else:
            content = run.step.text
        return ChatResult(content=content, tool_calls=[], finish_reason="stop", usage=None)


class _CapturingInterpreter(Interpreter):
    """The real interpreter; keeps its result so invariants can compare the saved turn."""

    def __init__(self, harness: Harness) -> None:
        super().__init__(
            WalkProvider(harness),
            app.state.fleet,
            glance=harness.world.glance_reader(),
            narrate=True,
        )
        self._harness = harness

    def run(self, words: str, **kwargs: Any) -> AgentRunResult:
        assert self._harness.current is not None
        try:
            result = super().run(words, **kwargs)
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
        self.last: Observed | None = None
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
        app.dependency_overrides[routes_agent.get_conductor] = lambda: _CapturingInterpreter(self)
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
        self.last = observed
        return observed

    @property
    def waiting(self) -> bool:
        """Whether the last turn left the person a question to answer."""
        run = self.last.run if self.last is not None else None
        if run is None:
            return False
        paused = run.plan is not None and run.plan.status == "paused"
        return paused or any(record.app_confirm for record in run.tool_calls)

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


def check_every_record_has_one_answer(observed: Observed) -> None:
    """Every step on the turn's record has exactly one answer: what came back,
    or why it failed or didn't run. A step with neither claims nothing."""
    if observed.run is None:
        return
    bad = [
        record.tool
        for record in observed.run.tool_calls
        if (record.result is None) == (record.error is None)
    ]
    if bad:
        raise InvariantBreach("every_record_has_one_answer", f"records without one answer: {bad}")


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
    """No chat turns are replayed (#64): the turn's first model call is the
    system prompt and one user message (State, recall, the words), and no
    model call carries a tool call, a tool result or an app's transcript; a
    repair adds only the planner's own reply and code's objections."""
    for request in observed.requests[:1]:
        roles = [message.get("role") for message in request]
        if roles != ["system", "user"]:
            raise InvariantBreach("history_is_text_only", f"first call carried {roles}")
    for request in observed.requests:
        for message in request:
            if message.get("role") not in ("system", "user", "assistant") or message.get(
                "tool_calls"
            ):
                raise InvariantBreach(
                    "history_is_text_only", f"a call carried {message.get('role')} {message}"
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
    in the saved turn (its ask, or the confirm that answered its question) —
    answered, or reported as "outcome unknown"."""
    if not observed.receipts:
        return
    saved = observed.saved_assistant
    calls = (saved or {}).get("tool_calls") or []
    for receipt in observed.receipts:
        tool = f"ask_{receipt.app}"
        if not any(
            (
                call["tool"] == tool
                or (call["tool"] == "confirm" and call["arguments"].get("app") == receipt.app)
            )
            and (call["result"] is not None or "stopped partway" in (call["error"] or ""))
            for call in calls
        ):
            raise InvariantBreach(
                "every_app_write_is_recorded",
                f"{receipt.app} wrote {receipt.kind} {receipt.title!r} but the saved turn "
                f"(HTTP {observed.status}) has no call to {receipt.app} that ran",
            )


def check_fault_ends_the_run_as_documented(observed: Observed) -> None:
    """Each fault ends the turn the way the route and the interpreter
    document. A provider fault fails the turn only on the planner's call; the
    narrator and the speaker fall back to code's words and the turn stands."""
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
    if _SITE[fault.kind] == "chat" and observed.fault_role != "plan":
        expected = 200
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
    check_every_record_has_one_answer,
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
        lambda h, _index: next_step(rng, h.faults, waiting=h.waiting),
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
