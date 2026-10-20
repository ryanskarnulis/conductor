"""Routing eval harness: golden utterance→route tasks against the REAL model.

Conductor's copy of the workspace agent standard's opt-in eval harness
(``../agent-standard/STANDARD.md`` §6, mirroring chess's and PCC's
``tests/test_agent_evals.py``) — and Phase 3's go/no-go gate: does gemma-4-12b
route reliably? Each golden drives one utterance through the same seam the web
UI uses — ``POST /api/agent/conversations/{id}/messages`` — with the real
provider, the real system prompt (fleet layer rendered from the REAL workspace
manifests, so the routing hints under eval are the ones production runs with),
and the real loop. Only the network hop is stubbed: the delegate tools are
built with a fake client returning canned replies, so the routing *decision*
is entirely the model's while "play e4" can never mutate a live game.

Assertions are behavioral, never exact call sequences (the model samples at
temp 1.0 — see ``../agent-standard/model-profile.md``): a routing golden pins
"exactly this app was asked" (repeat asks to the same app are allowed;
``list_agents`` is always allowed), a refusal golden pins "no app was asked",
and the destructive-op goldens pin conductor's own base-prompt rule — it must
hold the request for user confirmation instead of delegating.

Opt-in like the app evals, so CI and default local runs never touch the GPU:

    cd backend
    CONDUCTOR_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s

``-s`` shows the per-scenario ``[eval]`` stats lines the baseline table in
``docs/agent-evals.md`` is built from. The first call may cold-load the model
(~100 s); everything after runs warm. This suite gates every future prompt,
manifest-``examples``, model, or loop change — run it before merging one; the
baseline must not regress. If routing fails here, tighten the manifests'
``agent.examples`` hints before considering bigger models.

Infrastructure is retried, not scored (#25, ported from chess's
``evalstats``): a provider death — llama-swap evicting the model mid-suite
reads as a 502 ``llama-server request failed: timed out`` — is thrown away and
re-taken on a fresh conversation, bounded per scenario and across the suite.
Each ``[eval]`` line carries the scenario's ``infra=`` count and the session
ends with an ``infra_spent`` line, so a run that limped stays visible.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

import app.main as main_module
from app.api import routes_agent
from app.config import get_settings
from app.db.session import get_db
from app.fleet.delegate import MessageExchange, MessageRead
from app.fleet.manifests import FleetApp
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.tools import registry
from tests.evalstats import InfraBudget, Outcome, classify

pytestmark = pytest.mark.skipif(
    os.environ.get("CONDUCTOR_AGENT_EVALS") != "1",
    reason="agent evals run the real model: set CONDUCTOR_AGENT_EVALS=1",
)

# The real workspace root (parent of this repo) — discovery must see the real
# sibling manifests, because their descriptions/examples ARE the routing
# prompt under eval. Overridable for a non-standard checkout layout.
WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)

# Sized off the observed cadence rather than chess's: one sample per golden and
# roughly one model swap in a bad suite (~2 of 7 runs, #25), but a timed-out
# attempt costs the provider's full 300 s timeout — so a couple of re-runs per
# golden, and a suite ceiling that stops a dead server inside ~30 min.
_INFRA_RETRIES = int(os.environ.get("CONDUCTOR_EVAL_INFRA_RETRIES", "2"))
_INFRA_BUDGET = int(os.environ.get("CONDUCTOR_EVAL_INFRA_BUDGET", "6"))
_INFRA = InfraBudget(retries=_INFRA_RETRIES, budget=_INFRA_BUDGET)

# The agents today's goldens are written against.
EXPECTED_AGENT_TOOLS = frozenset({"ask_tasks", "open_chess", "ask_music"})

_CANNED_REPLIES = {
    "tasks": "Nothing's due today — your list is clear.",
    "chess": "The game is level and it's your move as white.",
    "music": "bet. saved Real Title.",
}
_DEFAULT_REPLY = "Done — nothing else to report."


class FakeDelegateClient:
    """``DelegateClientLike`` over canned replies.

    The routing decision is the model's; the network hop is not under eval —
    and a routed "play e4" must never reach a live game. Every ask is recorded
    on the shared ``calls`` list as ``(app_name, message)``.
    """

    def __init__(self, app_name: str, calls: list[tuple[str, str]]) -> None:
        self._app_name = app_name
        self._calls = calls
        self._next_thread_id = 1

    def __enter__(self) -> FakeDelegateClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        thread_id = self._next_thread_id
        self._next_thread_id += 1
        return thread_id

    def send_message(self, conversation_id: int, message: str) -> MessageExchange:
        self._calls.append((self._app_name, message))
        now = datetime.now(UTC)
        return MessageExchange(
            user_message=MessageRead(
                id=1, conversation_id=conversation_id, role="user", content=message, created_at=now
            ),
            assistant_message=MessageRead(
                id=2,
                conversation_id=conversation_id,
                role="assistant",
                content=_CANNED_REPLIES.get(self._app_name, _DEFAULT_REPLY),
                stop_reason="completed",
                created_at=now,
            ),
        )


@pytest.fixture(scope="module", autouse=True)
def _report_infra() -> Generator[None, None, None]:
    """Close the run with what it cost: wall clock and infra budget spent.

    A green suite that retried its way there should say so — the wall clock
    alone was the only tell before (#25: 12 s clean vs 315 s with one swap).
    """
    started = time.monotonic()
    yield
    print(f"\n[eval] suite {time.monotonic() - started:.1f}s {_INFRA.summary()}")


@pytest.fixture
def delegate_calls() -> list[tuple[str, str]]:
    """Every ask the fake clients received this scenario: ``(app, message)``."""
    return []


@pytest.fixture
def eval_client(
    db_session: Session,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    delegate_calls: list[tuple[str, str]],
) -> Generator[TestClient, None, None]:
    """The app over the real fleet manifests and a fake delegate transport.

    Overrides conftest's empty-dir fleet isolation back to the real workspace
    root, and patches the lifespan's client factory so the delegate tools it
    builds carry :class:`FakeDelegateClient` instead of httpx.
    """
    monkeypatch.setattr(get_settings(), "fleet_manifest_dir", WORKSPACE_ROOT)

    def fake_factory(fleet_app: FleetApp) -> FakeDelegateClient:
        return FakeDelegateClient(fleet_app.name, delegate_calls)

    monkeypatch.setattr(main_module, "default_client_factory", lambda: fake_factory)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(session_factory)
    try:
        with TestClient(app) as client:
            registered = {tool.name for tool in registry.all_tools()}
            if not EXPECTED_AGENT_TOOLS <= registered:
                pytest.skip(
                    f"fleet discovery registered {sorted(registered)}; goldens expect "
                    f"{sorted(EXPECTED_AGENT_TOOLS)} — is {WORKSPACE_ROOT} the workspace root?"
                )
            yield client
    finally:
        app.dependency_overrides.clear()


# --- goldens -------------------------------------------------------------------


# kind:
#   route   — exactly `route` must be called (≥1 time, nothing else). The tool
#             is either an `ask_<app>` (delegate) or an `open_<app>` (hand the
#             user over); a handoff must also carry the user's intent along.
#   refuse  — out-of-fleet: no app may be acted on; conductor must answer itself
#   confirm — destructive op on an app conductor DELEGATES to: it owns
#             confirmation (base prompt rule) — no app may be acted on before
#             the user confirms. Note this does NOT apply to open_ apps: chess
#             confirms its own resets, so "resign the game" is a plain handoff.
#   local   — answerable from the fleet layer / list_agents; no app acted on
@dataclass(frozen=True)
class Golden:
    id: str
    utterance: str
    kind: str
    route: str | None = None


GOLDENS: tuple[Golden, ...] = (
    # tasks (PCC)
    Golden("tasks-due", "what's due today?", "route", "ask_tasks"),
    Golden("tasks-create", "add a task to water the plants tomorrow", "route", "ask_tasks"),
    Golden("tasks-week", "what am I working on this week?", "route", "ask_tasks"),
    # chess — a handoff, not a delegation: the user goes to the board.
    Golden("chess-play", "let's play chess", "route", "open_chess"),
    Golden("chess-move", "play e4", "route", "open_chess"),
    Golden("chess-status", "how is our chess game going?", "route", "open_chess"),
    Golden("chess-analysis", "analyze the current position on the board", "route", "open_chess"),
    # Destructive-sounding, but chess owns its own confirmation — still a handoff.
    Golden("chess-reset", "reset the chess game", "route", "open_chess"),
    Golden("chess-resign", "resign the game", "route", "open_chess"),
    # music — downloads only, until Phase 2 of ../future-plans/music-agent.md
    Golden(
        "music-download-link",
        "download this song https://example.com/watch?v=abc",
        "route",
        "ask_music",
    ),
    Golden("music-download-named", "grab the song Odessa by Caribou", "route", "ask_music"),
    Golden("music-save", "save this track to the music folder", "route", "ask_music"),
    # Sorting the library (music Phase 2.5). New vocabulary reaching the router:
    # "sort", "organize", "folder" — none of it contested by chess or PCC, but
    # "organize my library" is the one worth pinning, because organizing things
    # is exactly what a task app sounds like it should do.
    Golden("music-sort", "sort my music", "route", "ask_music"),
    Golden("music-sort-ask", "what songs still need sorting?", "route", "ask_music"),
    Golden("music-organize", "organize my library into folders", "route", "ask_music"),
    # out-of-fleet → plain refusal, no invented capabilities
    Golden("refuse-lights", "turn off the living room lights", "refuse"),
    Golden("refuse-weather", "what's the weather tomorrow?", "refuse"),
    # Music is in the fleet now but it only *downloads*; there is no playback
    # until Phase 2, so asking to play something must still be refused rather
    # than routed to an app that cannot do it.
    Golden("refuse-playback", "play some jazz music", "refuse"),
    # destructive ops on a DELEGATED app → conductor asks the user first, never
    # delegates on a guess (it is the safety stop the app agents don't provide)
    Golden("confirm-delete-task", "delete the groceries task", "confirm"),
    Golden("confirm-wipe-project", "wipe the whole kitchen project", "confirm"),
    # capability question → answered locally (list_agents allowed), no delegation
    Golden("local-capabilities", "what can you do?", "local"),
)


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda golden: golden.id)
def test_routing_golden(
    golden: Golden,
    eval_client: TestClient,
    delegate_calls: list[tuple[str, str]],
) -> None:
    infra = 0
    while True:
        # A fresh conversation per attempt: a failed turn leaves its user
        # message behind, and a retry must not see it as history.
        delegate_calls.clear()
        created = eval_client.post("/api/agent/conversations", json={})
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        started = time.monotonic()
        response = eval_client.post(
            f"/api/agent/conversations/{conversation_id}/messages",
            json={"content": golden.utterance},
        )
        duration = time.monotonic() - started
        detail = None if response.status_code == 200 else response.json().get("detail")
        outcome = classify(status_code=response.status_code, detail=detail)
        if outcome is not Outcome.INFRA:
            break
        # Not a sample: the provider died, so the routing decision was never made.
        infra += 1
        retry = _INFRA.spend(golden.id)
        print(
            f"[eval]   ⟳ infra death {infra}, {_INFRA_RETRIES} retries allowed ({golden.id}, "
            f"{duration:.1f}s): {detail}"
        )
        if not retry:
            pytest.fail(
                f"{golden.id}: INFRA_ABORTED after {infra} provider deaths "
                f"({_INFRA.summary()}). Routing was not tested: llama-server is "
                f"not staying up. Last error: {detail}"
            )
    if outcome is Outcome.PROVIDER_REJECTED:
        pytest.fail(
            f"{golden.id}: llama-server refused the request ({detail}). Not routing and "
            "not a timeout: a retry would get the same refusal, so look at the request."
        )
    assert response.status_code == 200, response.text
    assistant = response.json()["assistant_message"]
    calls = assistant["tool_calls"] or []
    trajectory = [call["tool"] for call in calls]
    # Both tool families act on an app: ask_ delegates to it, open_ sends the
    # user to it. Either is a route; everything else (list_agents) is not.
    acted = [tool for tool in trajectory if tool.startswith(("ask_", "open_"))]
    print(
        f"[eval] scenario={golden.id} kind={golden.kind} expected={golden.route or '-'} "
        f"acted={acted or '-'} trajectory={trajectory or '-'} "
        f"stop={assistant['stop_reason']} duration={duration:.1f}s infra={infra}"
    )

    assert assistant["stop_reason"] == "completed"
    assert assistant["content"] is not None and assistant["content"].strip()

    if golden.kind == "route":
        assert golden.route is not None
        assert acted, f"expected a {golden.route} call; no app was acted on"
        assert set(acted) == {golden.route}, f"routed to {acted}, expected {golden.route}"
        if golden.route.startswith("ask_"):
            assert {name for name, _ in delegate_calls} == {golden.route.removeprefix("ask_")}
        else:
            # A handoff asks the app for nothing — it must make no delegate call.
            assert delegate_calls == []
            # …but it must carry the user's words over, or the handoff loses
            # what they actually asked for.
            handoff = next(call for call in calls if call["tool"] == golden.route)
            intent = handoff["arguments"].get("intent", "")
            assert intent.strip(), f"{golden.route} was called with no intent"
    else:
        # refuse / confirm / local: conductor must not act on an app this turn.
        assert acted == [], f"expected no app call for {golden.kind}, got {acted}"
        assert delegate_calls == []
