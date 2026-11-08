"""Session evals (#63): scripted multi-turn scenarios S1–S16 over a fake world.

Two runs share every scenario, check and driver (``session_harness.py``,
``session_scenarios.py``, ``fake_world.py``, ``receipt_claims.py``):

- **Scripted (CI, no GPU).** Each scenario's golden ``script`` plays the
  model; the run must PASS — proof the checks are satisfiable, so a live 0/5
  is the model's. Deliberately wrong scripts must FAIL with the right reason —
  proof the checks bite. A scenario whose runtime capability is missing must
  FAIL ``unsupported:``.
- **Live (opt-in, the real model).** ``CONDUCTOR_AGENT_EVALS=1`` runs the
  ``gate`` tier under the vendored Wilson gate (``evalstats.run_gate``, floor
  0.8, known reds xfail); ``CONDUCTOR_FRONTIER_EVALS=1`` also measures the
  ``frontier`` tier — fixed samples, printed, never asserted. Each sample is a
  whole session on a fresh world through the real route, loop, prompt and
  workspace manifests; only the apps are fake. ``CONDUCTOR_EVAL_REPORT=path``
  appends one JSON line per scenario.

    cd backend
    CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1 \\
        .venv/bin/pytest tests/test_session_evals.py -v -s -k live

The baseline is in ``docs/agent-evals.md`` ("Session evals").
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

import app.main as main_module
from app.ai import interpreter as interpreter_module
from app.ai.loop import AgentLoop, loop_from_settings
from app.ai.provider import ChatProvider, ProviderRequestError
from app.ai.providers.llamacpp import provider_from_settings
from app.api import rate_limit, routes_agent
from app.config import get_settings
from app.db.session import get_db
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.services.job_watcher import JobWatcher
from app.services.thinker import Thinker, think_provider_from_settings
from app.tools import registry
from tests import eval_seed
from tests.eval_db import database_wiper
from tests.evalstats import (
    Decision,
    GateAborted,
    InfraBudget,
    Outcome,
    Taken,
    run_gate,
    scenario_record,
)
from tests.fake_world import World, write_fixture_fleet
from tests.receipt_claims import Tally
from tests.scripted_provider import ScriptedProvider, text_turn, tool_calls_turn
from tests.session_harness import LoopDriver, Sample, Scenario, WorldHolder, run_sample, sample_line
from tests.session_scenarios import (
    S2,
    S3,
    S4,
    S5,
    S6,
    S9,
    S10,
    S11,
    S12,
    S14,
    S15,
    S16,
    SCENARIOS,
)

_LIVE = os.environ.get("CONDUCTOR_AGENT_EVALS") == "1"
_FRONTIER = os.environ.get("CONDUCTOR_FRONTIER_EVALS") == "1"
_REPORT = os.environ.get("CONDUCTOR_EVAL_REPORT")

WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)
_BLOCK_RUNS = int(os.environ.get("CONDUCTOR_EVAL_RUNS", "5"))
_MAX_RUNS = int(os.environ.get("CONDUCTOR_EVAL_MAX_RUNS", "20"))
_FRONTIER_RUNS = int(os.environ.get("CONDUCTOR_FRONTIER_RUNS", "5"))
_INFRA = InfraBudget(
    retries=int(os.environ.get("CONDUCTOR_EVAL_INFRA_RETRIES", "3")),
    budget=int(os.environ.get("CONDUCTOR_EVAL_INFRA_BUDGET", "12")),
)
_CLAIMS = Tally()
# Paired A/B (#53): sample j of a scenario sends seed base + j (``eval_seed``).
_SEED_BASE = eval_seed.seed_base()


def _tool_names() -> set[str]:
    return {tool.name for tool in registry.all_tools()}


def _watching(
    session_factory: sessionmaker[DbSession],
    holder: WorldHolder,
    provider: Callable[[], ChatProvider | None],
    think_provider: Callable[[], ChatProvider | None] | None = None,
) -> Callable[[], None]:
    """The job watcher (#76) over the eval's database and the world's jobs
    endpoint, with the thinker (#78) when a ``think_provider`` is given; the
    returned ``settle`` is one pass of it with every job due (a think runs to
    its end), run when the harness reads what arrived unprompted or the
    person waits (time has passed)."""
    thinker = (
        Thinker(session_factory, app.state.fleet, provider=think_provider)
        if think_provider is not None and get_settings().conductor_think
        else None
    )
    watcher = JobWatcher(
        session_factory,
        app.state.fleet,
        reader=holder.read_job,
        provider=provider,
        thinker=thinker,
    )
    app.state.job_watcher = watcher

    def settle() -> None:
        watcher.sweep(force=True)

    return settle


def _started(
    fleet_dir: Path,
    db_session: DbSession,
    session_factory: sessionmaker[DbSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, WorldHolder], None, None]:
    holder = WorldHolder()
    monkeypatch.setattr(get_settings(), "fleet_manifest_dir", fleet_dir)
    monkeypatch.setattr(main_module, "default_client_factory", lambda: holder.factory)
    # A plan's `look` reads the world's glance, never the real app behind the
    # manifest (the interpreter resolves its default reader at construction).
    monkeypatch.setattr(interpreter_module, "read_glance", holder.glance)
    # A cancel on a running job reaches the world's jobs endpoint (#76).
    monkeypatch.setattr(interpreter_module, "cancel_job", holder.cancel_job)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(session_factory)
    try:
        with TestClient(app) as client:
            yield client, holder
    finally:
        app.dependency_overrides.clear()


# --- the scripted run (CI) -------------------------------------------------------------


class _ScriptedRun:
    """A started app whose loop plays a fresh script per sample. ``compiled``
    runs the compiler and interpreter (#70) with the narrator, as live does."""

    def __init__(
        self,
        client: TestClient,
        holder: WorldHolder,
        db_session: DbSession,
        session_factory: sessionmaker[DbSession],
        *,
        compiled: bool = False,
    ) -> None:
        self.client = client
        self.provider = ScriptedProvider([])
        # A job's note is narrated by the same scripted model, after the turns,
        # and a think's proposal (#78) comes from it too, when the person waits.
        settle = _watching(session_factory, holder, lambda: self.provider, lambda: self.provider)
        app.dependency_overrides[routes_agent.get_agent_loop] = lambda: AgentLoop(
            self.provider,
            fleet_section=getattr(app.state, "fleet_section", None),
            fleet=getattr(app.state, "fleet", None) if compiled else None,
            narrate=compiled,
        )
        self.driver = LoopDriver(
            client,
            holder,
            tool_names=_tool_names,
            reset=rate_limit._reset,
            fresh_db=database_wiper(db_session),
            settle=settle,
        )

    def run(self, scenario: Scenario, turns: list[Any] | None = None) -> Sample:
        world = World()
        script = turns if turns is not None else (scenario.script(world) if scenario.script else [])
        self.provider = ScriptedProvider(script)
        sample = run_sample(scenario, self.driver, world)
        print(sample_line(scenario, sample))
        return sample


@pytest.fixture
def scripted(
    tmp_path: Path,
    db_session: DbSession,
    session_factory: sessionmaker[DbSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_ScriptedRun, None, None]:
    fleet_dir = write_fixture_fleet(tmp_path / "fleet")
    for client, holder in _started(fleet_dir, db_session, session_factory, monkeypatch):
        yield _ScriptedRun(client, holder, db_session, session_factory)


@pytest.fixture
def compiled(
    tmp_path: Path,
    db_session: DbSession,
    session_factory: sessionmaker[DbSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_ScriptedRun, None, None]:
    fleet_dir = write_fixture_fleet(tmp_path / "fleet")
    for client, holder in _started(fleet_dir, db_session, session_factory, monkeypatch):
        yield _ScriptedRun(client, holder, db_session, session_factory, compiled=True)


def _plan(*steps: dict[str, Any]) -> Any:
    body = {"steps": [{"id": f"s{n}", **step} for n, step in enumerate(steps, start=1)]}
    return text_turn(json.dumps(body))


def _ask_step(app_name: str, request: str) -> dict[str, Any]:
    return {"do": "ask", "app": app_name, "request": request, "destructive": False}


_S3_PLAN = _plan(
    _ask_step("music", "download the new Caribou album"),
    _ask_step("tasks", "add a task to sort {s1} this weekend"),
)

# The compiled runtime's golden model side (#74): plans, then the narrator.
# S10's Stop lands mid-delegate, so nothing is narrated. S11's undo turn is
# tasks' own question, passed through; the yes turn is narrated.
_S4_NOTE = text_turn(
    "Your playlist is in: 38 of 40 downloaded; “Lost Signal” and “Paper Moons” couldn't be found."
)

COMPILED_SCRIPTS: dict[str, Callable[[World], list[Any]]] = {
    # The playlist detaches as a job; the watcher's note comes after (#76).
    S4.id: lambda w: [
        _plan(_ask_step("music", "download this playlist https://example.com/playlist?list=PL40")),
        text_turn("Queued your playlist: 40 tracks. I'll post a note when it's done."),
        _S4_NOTE,
    ],
    S10.id: lambda w: [_plan(_ask_step("music", "sort my whole library"))],
    # The think detaches with code's reply; the 26B's proposal posts while the
    # person waits; their yes runs it (#78).
    S14.id: lambda w: [
        _plan({"do": "think", "problem": "plan my week around the garage project", "inputs": []}),
        text_turn(
            json.dumps(
                {
                    "summary": "3 garage tasks, Monday to Friday",
                    "steps": [
                        {
                            "id": "s1",
                            **_ask_step(
                                "tasks",
                                "Add these tasks to the Garage project:\n- buy the steel rack "
                                "on monday\n- assemble the shelves on wednesday\n- sweep the "
                                "garage floor on friday",
                            ),
                        }
                    ],
                }
            )
        ),
        _plan({"do": "confirm", "answer": "yes"}),
        text_turn(
            "Added buy the steel rack (Mon), assemble the shelves (Wed) and sweep the garage "
            "floor (Fri)."
        ),
    ],
    # Memory (#80). S5 on a fresh world: nothing on record, said plainly.
    S5.id: lambda w: [
        _plan({"do": "recall", "query": "garage shelves decision"}),
        text_turn("I have no record of a decision about the garage shelves."),
    ],
    # S6: the new conversation's State matches "that Caribou song" to Odessa.
    S6.id: lambda w: [
        _plan(_ask_step("music", "grab the song Odessa by Caribou")),
        text_turn("Saved “Odessa” by Caribou."),
        _plan(_ask_step("music", "put Odessa by Caribou in the house folder")),
        text_turn("Filed “Odessa” under house."),
    ],
    S15.id: lambda w: [
        _S3_PLAN,
        text_turn("Queued “Honey” by Caribou and added “sort Honey” for the weekend."),
        _plan(
            _ask_step("music", "download the other album by Caribou"),
            _ask_step("tasks", "add a task to sort {s1} this weekend"),
        ),
        text_turn("Queued “Suddenly” and added “sort Suddenly” for the weekend."),
    ],
    S11.id: lambda w: [
        _S3_PLAN,
        text_turn("Queued “Honey” by Caribou and added “sort Honey” for the weekend."),
        _plan({"do": "undo", "target": "last"}),
        _plan({"do": "confirm", "answer": "yes"}),
        text_turn("Trashed “sort Honey”. The Honey download can't be undone, though."),
    ],
}


@pytest.mark.parametrize("scenario", [S4, S5, S6, S10, S11, S14, S15], ids=lambda s: s.id)
def test_compiled_golden_script_passes(scenario: Scenario, compiled: _ScriptedRun) -> None:
    sample = compiled.run(scenario, COMPILED_SCRIPTS[scenario.id](World()))
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert compiled.provider._turns == [], "the golden script wasn't fully consumed"


def _is_planner(request: dict[str, Any]) -> bool:
    form = request.get("response_format") or {}
    return bool(form.get("json_schema", {}).get("name") == "plan")


def test_compiled_s6_finds_the_song_in_state_across_conversations(
    compiled: _ScriptedRun,
) -> None:
    # #80: the entity registry, not the planner, carries "Odessa" across days.
    sample = compiled.run(S6, COMPILED_SCRIPTS[S6.id](World()))
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in compiled.provider.requests if _is_planner(r)]
    opening = planner[-1]["messages"][1]["content"]
    assert "Matches for words in the message, from earlier conversations" in opening
    assert '"Odessa" (outside text)' in opening


def test_compiled_s15_sees_no_matches_from_its_own_conversation(compiled: _ScriptedRun) -> None:
    # Matches are leads from *other* conversations: S15's second turn points
    # back ("their other album") within one conversation, so its planner input
    # is what it was before #80.
    sample = compiled.run(S15, COMPILED_SCRIPTS[S15.id](World()))
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in compiled.provider.requests if _is_planner(r)]
    assert "Matches for words" not in planner[-1]["messages"][1]["content"]


def test_compiled_s5_a_guessed_decision_still_gets_codes_no_record(
    compiled: _ScriptedRun,
) -> None:
    script = COMPILED_SCRIPTS[S5.id](World())
    script[-1] = text_turn("You picked the wooden shelves.")
    sample = compiled.run(S5, script)
    assert "no record" in (sample.session.last.reply or "")


def test_compiled_s11_says_what_cant_be_undone_when_the_narrator_doesnt(
    compiled: _ScriptedRun,
) -> None:
    # Spike C1b: the narrator never offered what it couldn't undo. Code appends it.
    script = COMPILED_SCRIPTS[S11.id](World())
    script[-1] = text_turn("Trashed “sort Honey”.")
    sample = compiled.run(S11, script)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert "can't be undone" in (sample.session.last.reply or "")


def _supported_today(scenario: Scenario) -> bool:
    # Today's loop has no way to read chess.
    return scenario.id != S12.id


@pytest.mark.parametrize(
    "scenario",
    [s for s in SCENARIOS if s.script is not None and _supported_today(s)],
    ids=lambda s: s.id,
)
def test_golden_script_passes(scenario: Scenario, scripted: _ScriptedRun) -> None:
    sample = scripted.run(scenario)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert scripted.provider._turns == [], "the golden script wasn't fully consumed"


@pytest.mark.parametrize("scenario", [S12], ids=lambda s: s.id)
def test_missing_capability_fails_unsupported(scenario: Scenario, scripted: _ScriptedRun) -> None:
    sample = scripted.run(scenario)
    assert sample.taken.outcome is Outcome.FAIL
    assert (sample.taken.failure or "").startswith("unsupported:"), sample.taken.failure


def _fails(sample: Sample, reason: str) -> None:
    assert sample.taken.outcome is Outcome.FAIL, "a wrong session passed"
    assert reason in (sample.taken.failure or ""), sample.taken.failure


def test_confirming_its_own_question_fails(scripted: _ScriptedRun) -> None:
    # PCC's gate (#68) means asking can no longer write; the way to act before
    # the yes is to answer the app's question in the same turn. confirm takes
    # only a question from an earlier turn, so nothing is trashed, and the
    # claim that it was is caught.
    sample = scripted.run(
        S9,
        [
            tool_calls_turn(("ask_tasks", {"message": "delete the Kitchen project"})),
            tool_calls_turn(("confirm", {"app": "tasks"})),
            text_turn("Done: the Kitchen project went to the trash."),
            text_turn("Already done."),
        ],
    )
    _fails(sample, "didn't ask before wiping")


def test_a_dropped_title_fails_s3(scripted: _ScriptedRun) -> None:
    # #55: the relay drops the album's real name, so the task can't carry it.
    sample = scripted.run(
        S3,
        [
            tool_calls_turn(("ask_music", {"message": "download the new Caribou album"})),
            tool_calls_turn(
                ("ask_tasks", {"message": "add a task to sort the new album this weekend"})
            ),
            text_turn("Queued Honey and added “sort the new album” for Saturday."),
        ],
    )
    _fails(sample, "no task titled with 'honey'")


def test_claiming_a_write_that_did_not_happen_fails(scripted: _ScriptedRun) -> None:
    sample = scripted.run(S2, [text_turn("Added “water the plants” for tomorrow.")])
    _fails(sample, "unbacked claim")


def test_an_unnamed_write_fails(scripted: _ScriptedRun) -> None:
    sample = scripted.run(
        S2,
        [
            tool_calls_turn(("ask_tasks", {"message": "add a task to water the plants tomorrow"})),
            text_turn("Done, it's on your list."),
        ],
    )
    _fails(sample, "doesn't name what it touched")


def test_a_recap_that_invents_fails(scripted: _ScriptedRun) -> None:
    sample = scripted.run(
        S16,
        [
            tool_calls_turn(("ask_tasks", {"message": "add a task to water the plants tomorrow"})),
            text_turn("Added “water the plants” for tomorrow."),
            text_turn("I added “water the plants” and queued an album for you."),
        ],
    )
    _fails(sample, "unbacked claim")


def test_a_provider_death_is_infra_not_a_miss(scripted: _ScriptedRun) -> None:
    sample = scripted.run(S2, [ProviderRequestError("llama-server request failed: timed out")])
    assert sample.taken.outcome is Outcome.INFRA, sample.taken


def test_samples_never_share_a_world(scripted: _ScriptedRun) -> None:
    first = scripted.run(S2)
    second = scripted.run(S2)
    assert first.session.world is not second.session.world
    assert len(second.session.world.receipts) == 1


def test_samples_never_share_a_database(scripted: _ScriptedRun) -> None:
    def conversations() -> list[int]:
        listed = scripted.client.get("/api/agent/conversations")
        assert listed.status_code == 200, listed.text
        return sorted(c["id"] for c in listed.json())

    scripted.run(S9)
    first = conversations()
    assert first
    scripted.run(S9)
    # The second sample sees only its own conversations, never the first's.
    assert len(conversations()) == len(first)


# --- the live run (opt-in) -------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _report_suite() -> Generator[None, None, None]:
    started = time.monotonic()
    yield
    if _LIVE:
        print(f"\n[session] suite {time.monotonic() - started:.1f}s {_INFRA.summary()}")
        print(f"[session] {_CLAIMS.summary()}")


@pytest.fixture
def live(
    db_session: DbSession,
    session_factory: sessionmaker[DbSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[LoopDriver, None, None]:
    for client, holder in _started(WORKSPACE_ROOT, db_session, session_factory, monkeypatch):
        if _SEED_BASE is not None:
            app.dependency_overrides[routes_agent.get_agent_loop] = _seeded_loop
        if not {"ask_tasks", "ask_music", "open_chess"} <= _tool_names():
            pytest.skip(
                f"fleet discovery registered {sorted(_tool_names())}; is "
                f"{WORKSPACE_ROOT} the workspace root?"
            )
        provider = eval_seed.seeded_provider if _SEED_BASE is not None else provider_from_settings
        yield LoopDriver(
            client,
            holder,
            tool_names=_tool_names,
            reset=rate_limit._reset,
            fresh_db=database_wiper(db_session),
            # A think (#78) runs on the real 26B, unseeded, when S14 waits.
            settle=_watching(session_factory, holder, provider, think_provider_from_settings),
        )


def _seeded_loop() -> Generator[AgentLoop, None, None]:
    provider = eval_seed.seeded_provider()
    try:
        yield loop_from_settings(
            provider,
            fleet_section=getattr(app.state, "fleet_section", None),
            fleet=getattr(app.state, "fleet", None),
        )
    finally:
        provider.close()


def _take(
    scenario: Scenario, driver: LoopDriver, samples: list[Sample]
) -> Callable[[int | None], Taken]:
    def take(seed: int | None) -> Taken:
        eval_seed.current.seed = seed
        try:
            sample = run_sample(scenario, driver)
        finally:
            eval_seed.current.seed = None
        sample.seed = seed
        print(sample_line(scenario, sample))
        if sample.taken.outcome in (Outcome.PASS, Outcome.FAIL):
            samples.append(sample)
        return sample.taken

    return take


def _record(scenario: Scenario, result: Any, samples: list[Sample], tier: str) -> None:
    seconds = sorted(t.seconds for s in samples for t in s.session.turns)
    p50 = seconds[len(seconds) // 2] if seconds else 0.0
    _CLAIMS.add(claim for s in samples for _, claim in s.session.claims)
    print(
        f"[session] scenario={scenario.id} tier={tier} PASS_RATE={result.summary()} "
        f"turn_p50={p50:.1f}s"
    )
    for mode, count in sorted(result.failure_modes.items(), key=lambda item: -item[1]):
        print(f"[session]   ×{count} {mode}")
    if _REPORT:
        line = scenario_record(
            scenario.id,
            result,
            0.8,
            kind="session",
            tier=tier,
            turn_p50=p50,
            samples=[{"seed": s.seed, "pass": s.taken.outcome == "PASS"} for s in samples],
        )
        with open(_REPORT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")


def _live_params(tier: str) -> list[Any]:
    return [
        pytest.param(
            s,
            id=s.id,
            marks=[pytest.mark.xfail(reason=s.known_red, strict=False)] if s.known_red else [],
        )
        for s in SCENARIOS
        if s.tier == tier
    ]


@pytest.mark.skipif(not _LIVE, reason="session evals run the real model: CONDUCTOR_AGENT_EVALS=1")
@pytest.mark.parametrize("scenario", _live_params("gate"))
def test_live_gate(scenario: Scenario, live: LoopDriver) -> None:
    samples: list[Sample] = []
    try:
        result = run_gate(
            scenario.id,
            _take(scenario, live, samples),
            budget=_INFRA,
            block_runs=_BLOCK_RUNS,
            max_runs=_MAX_RUNS,
            seed_base=_SEED_BASE,
        )
    except GateAborted as aborted:
        pytest.fail(str(aborted))
    _record(scenario, result, samples, "gate")
    assert result.decision is not Decision.BELOW_FLOOR, (
        f"{scenario.id} is below its floor with evidence, not noise: {result.summary()}"
    )


@pytest.mark.skipif(
    not (_LIVE and _FRONTIER),
    reason="the frontier tier is opt-in: CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1",
)
@pytest.mark.parametrize("scenario", _live_params("frontier"))
def test_live_frontier(scenario: Scenario, live: LoopDriver) -> None:
    samples: list[Sample] = []
    try:
        result = run_gate(
            scenario.id,
            _take(scenario, live, samples),
            budget=_INFRA,
            block_runs=_FRONTIER_RUNS,
            max_runs=_FRONTIER_RUNS,
            seed_base=_SEED_BASE,
        )
    except GateAborted as aborted:
        pytest.skip(f"frontier sample aborted on infra: {aborted}")
    _record(scenario, result, samples, "frontier")
    # Never gated: the frontier is measured over time, not held to a floor.
