"""Session evals (#63): scripted multi-turn scenarios S1–S16 over a fake world.

Two runs share every scenario, check and driver (``session_harness.py``,
``session_scenarios.py``, ``fake_world.py``, ``receipt_claims.py``):

- **Scripted (CI, no GPU).** Each scenario's golden ``script`` plays the
  model (the planner's plans, then the narrator's or speaker's words) through
  the compiled runtime (#75); the run must PASS — proof the checks are satisfiable, so a live 0/5
  is the model's. Deliberately wrong scripts must FAIL with the right reason —
  proof the checks bite. A scenario whose runtime capability is missing must
  FAIL ``unsupported:``.
- **Live (opt-in, the real model).** ``CONDUCTOR_AGENT_EVALS=1`` runs the
  ``gate`` tier under the vendored Wilson gate (``evalstats.run_gate``, floor
  0.8, known reds xfail); ``CONDUCTOR_FRONTIER_EVALS=1`` also measures the
  ``frontier`` tier — fixed samples, printed, never asserted. Each sample is a
  whole session on a fresh world through the real route, planner, interpreter,
  narrator and workspace manifests; only the apps are fake. ``CONDUCTOR_EVAL_REPORT=path``
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
from app.ai.interpreter import Interpreter, conductor_from_settings
from app.ai.provider import ChatProvider, ChatResult, ProviderRequestError
from app.ai.providers.llamacpp import provider_from_settings
from app.api import rate_limit, routes_agent, routes_threads
from app.config import get_settings
from app.db.session import get_db
from app.fleet.thread_store import DbThreadStore
from app.main import app
from app.services.job_watcher import JobWatcher
from app.services.thinker import Thinker, think_provider_from_settings
from app.tools import registry
from tests import eval_seed
from tests.eval_db import database_ager, database_wiper
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
from tests.scripted_provider import ScriptedProvider, text_turn
from tests.session_harness import (
    LoopDriver,
    Sample,
    Say,
    Scenario,
    WorldHolder,
    run_sample,
    sample_line,
)
from tests.session_scenarios import (
    FORGET_GONE,
    INCOGNITO,
    REMEMBER_USED,
    S2,
    S3,
    S5,
    S6,
    S9,
    S11,
    S15,
    S16,
    SCENARIOS,
    THREAD_FLOWS_BACK,
    THREAD_KEEPS_INCOGNITO,
    TIMELINE_RECALL,
    YES,
    answer,
    ask,
    plan,
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
    """A started app whose conductor plays a fresh script per sample: the
    compiler, interpreter and narrator, as live runs them."""

    def __init__(
        self,
        client: TestClient,
        holder: WorldHolder,
        db_session: DbSession,
        session_factory: sessionmaker[DbSession],
    ) -> None:
        self.client = client
        self.provider = ScriptedProvider([])
        # A job's note is narrated by the same scripted model, after the turns,
        # and a think's proposal (#78) comes from it too, when the person waits.
        settle = _watching(session_factory, holder, lambda: self.provider, lambda: self.provider)
        app.dependency_overrides[routes_agent.get_conductor] = lambda: Interpreter(
            self.provider, app.state.fleet, narrate=True
        )

        # A side thread's result (#139) is narrated by the same scripted model.
        def thread_chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            return self.provider.chat(messages, max_tokens=max_tokens)

        app.dependency_overrides[routes_threads.get_thread_chat] = lambda: thread_chat
        self.driver = LoopDriver(
            client,
            holder,
            tool_names=_tool_names,
            reset=rate_limit._reset,
            fresh_db=database_wiper(db_session),
            pass_time=database_ager(db_session),
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


@pytest.mark.parametrize(
    "scenario", [s for s in SCENARIOS if s.script is not None], ids=lambda s: s.id
)
def test_golden_script_passes(scenario: Scenario, scripted: _ScriptedRun) -> None:
    sample = scripted.run(scenario)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert scripted.provider._turns == [], "the golden script wasn't fully consumed"


def test_every_scenario_has_a_golden_script() -> None:
    assert [s.id for s in SCENARIOS if s.script is None] == []


def _is_planner(request: dict[str, Any]) -> bool:
    form = request.get("response_format") or {}
    return bool(form.get("json_schema", {}).get("name") == "plan")


def test_compiled_s6_finds_the_song_in_state_across_conversations(
    scripted: _ScriptedRun,
) -> None:
    # #80: the entity registry, not the planner, carries "Odessa" across days.
    sample = scripted.run(S6)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    opening = planner[-1]["messages"][1]["content"]
    assert "Matches for words in the message, from earlier conversations" in opening
    assert '"Odessa" (outside text)' in opening


def test_compiled_m1_puts_the_saved_fact_in_state_and_m2_takes_it_out(
    scripted: _ScriptedRun,
) -> None:
    # #81: remember → the fact is in State another day; forget → it is gone
    # from State and from recall.
    sample = scripted.run(REMEMBER_USED)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    assert '- "I sort music on Sundays" (saved ' in planner[-1]["messages"][1]["content"]
    sample = scripted.run(FORGET_GONE)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    assert "blue pot" not in planner[-1]["messages"][1]["content"]
    recalled = sample.session.turn("after").tools[0]
    assert recalled["tool"] == "recall" and recalled["app_record"] == []


def test_a_memory_reply_claiming_an_app_write_fails(scripted: _ScriptedRun) -> None:
    assert REMEMBER_USED.script is not None
    script = REMEMBER_USED.script(World())
    script[1] = text_turn("Got “I sort music on Sundays”, and queued your Sunday playlist.")
    sample = scripted.run(REMEMBER_USED, script)
    _fails(sample, "unbacked claim")


def test_compiled_s15_sees_no_matches_from_its_own_conversation(scripted: _ScriptedRun) -> None:
    # Matches are leads from *other* conversations: S15's second turn points
    # back ("their other album") within one conversation, so its planner input
    # is what it was before #80.
    sample = scripted.run(S15)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    assert "Matches for words" not in planner[-1]["messages"][1]["content"]


def test_compiled_s5_a_guessed_decision_still_gets_codes_no_record(
    scripted: _ScriptedRun,
) -> None:
    assert S5.script is not None
    script = S5.script(World())
    script[-1] = text_turn("You picked the wooden shelves.")
    sample = scripted.run(S5, script)
    assert "no record" in (sample.session.last.reply or "")


def test_compiled_t1_sees_only_recall_of_the_day_before(scripted: _ScriptedRun) -> None:
    # #82: past the quiet gap the decision isn't "this conversation" any more;
    # the planner's opening has no record of it, and the recall cites it by day.
    sample = scripted.run(TIMELINE_RECALL)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    assert "steel" not in planner[-1]["messages"][1]["content"].split("The person said")[0]
    (recalled,) = sample.session.turn("later").tools
    assert "steel rack" in recalled["app_record"][0]["fact"]


def test_a_t1_answer_without_the_recall_fails(scripted: _ScriptedRun) -> None:
    assert TIMELINE_RECALL.script is not None
    script = TIMELINE_RECALL.script(World())
    script[2:] = [answer("chat"), text_turn('{"about_you": false, "reply": "Wood, I think."}')]
    _fails(scripted.run(TIMELINE_RECALL, script), "wasn't recalled")


def test_compiled_t2_keeps_the_incognito_words_out_of_the_next_plan(
    scripted: _ScriptedRun,
) -> None:
    sample = scripted.run(INCOGNITO)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    assert "4417" not in planner[-1]["messages"][1]["content"].split("The person said")[0]


def test_a_t2_plan_carrying_the_incognito_words_fails(scripted: _ScriptedRun) -> None:
    assert INCOGNITO.script is not None
    script = INCOGNITO.script(World())
    script[2:] = [
        plan({"do": "recall", "query": "gate code 4417"}),
        text_turn("I have no record of the gate code."),
    ]
    _fails(scripted.run(INCOGNITO, script), "carried the incognito words")


def test_compiled_th1_carries_the_task_in_and_the_thread_sees_it(scripted: _ScriptedRun) -> None:
    sample = scripted.run(THREAD_FLOWS_BACK)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    planner = [r for r in scripted.provider.requests if _is_planner(r)]
    inside = str(planner[-1]["messages"][1]["content"])
    # In the thread: the carried task in State, none of the timeline's turns.
    assert "measure the garage wall" in inside.split("[End of state]")[0]
    assert "The last exchange" not in inside


def test_a_th1_result_without_the_thread_write_is_completed_by_code(
    scripted: _ScriptedRun,
) -> None:
    assert THREAD_FLOWS_BACK.script is not None
    script = THREAD_FLOWS_BACK.script(World())
    script[-1] = text_turn("All done in there.")
    sample = scripted.run(THREAD_FLOWS_BACK, script)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    note = sample.session.threads["garage"].note["content"]
    assert "Garage shelves" in note and "order the shelf brackets" in note


def test_a_th2_result_carrying_the_incognito_write_fails(scripted: _ScriptedRun) -> None:
    assert THREAD_KEEPS_INCOGNITO.script is not None
    script = THREAD_KEEPS_INCOGNITO.script(World())
    script[-1] = text_turn("Side thread “Party”: added “buy candles” and wrapped the gift.")
    _fails(scripted.run(THREAD_KEEPS_INCOGNITO, script), "incognito turn's")


def test_compiled_s11_says_what_cant_be_undone_when_the_narrator_doesnt(
    scripted: _ScriptedRun,
) -> None:
    # Spike C1b: the narrator never offered what it couldn't undo. Code appends it.
    assert S11.script is not None
    script = S11.script(World())
    script[-1] = text_turn("Trashed “sort Honey”.")
    sample = scripted.run(S11, script)
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert "can't be undone" in (sample.session.last.reply or "")


def test_a_missing_capability_fails_unsupported(scripted: _ScriptedRun) -> None:
    # Every scenario runs on the compiled runtime (#75); the harness still
    # fails a sample whose runtime lacks what it needs, never skips it.
    lacking = Scenario(
        "Sx",
        "needs a tool no fleet app offers",
        (Say("anything"),),
        (),
        needs=frozenset({"tool:ask_nothing"}),
    )
    sample = scripted.run(lacking, [])
    assert sample.taken.outcome is Outcome.FAIL
    assert (sample.taken.failure or "").startswith("unsupported:"), sample.taken.failure


def _fails(sample: Sample, reason: str) -> None:
    assert sample.taken.outcome is Outcome.FAIL, "a wrong session passed"
    assert reason in (sample.taken.failure or ""), sample.taken.failure


def test_claiming_the_wipe_without_asking_fails(scripted: _ScriptedRun) -> None:
    # The loop could ask and confirm in one turn; a plan can't (confirm comes
    # first, and only for a question from an earlier turn), and tasks asks its
    # own question either way. The way left to skip the question is a planner
    # that answers instead of asking, and a speaker that claims the wipe: no
    # question went out and nothing was trashed.
    sample = scripted.run(
        S9,
        [
            answer("chat"),
            text_turn(json.dumps({"about_you": True, "reply": "Done: Kitchen is in the trash."})),
            answer("chat"),
            text_turn(json.dumps({"about_you": True, "reply": "Already done."})),
        ],
    )
    _fails(sample, "didn't ask before wiping")


def test_a_dropped_title_fails_s3(scripted: _ScriptedRun) -> None:
    # #55: the album's real name never reaches the task. The validator sends
    # "the new album" back for {s1} (#72), so the plan that loses it here says
    # "it", which no check can tell from a name.
    sample = scripted.run(
        S3,
        [
            plan(
                ask("music", "download the new Caribou album"),
                ask("tasks", "add a task to sort it this weekend"),
            ),
            text_turn("Queued “Honey” and added “sort it” for Saturday."),
        ],
    )
    _fails(sample, "no task titled with 'honey'")


def test_claiming_a_write_that_did_not_happen_fails(scripted: _ScriptedRun) -> None:
    sample = scripted.run(
        S2,
        [
            answer("chat"),
            text_turn(
                json.dumps({"about_you": True, "reply": "Added “water the plants” for tomorrow."})
            ),
        ],
    )
    _fails(sample, "unbacked claim")


def test_an_unnamed_write_is_named_by_code(scripted: _ScriptedRun) -> None:
    # Was a wrong script ("Done, it's on your list." fails S2 as unnamed). The
    # narrator is asked again for a missing title, then code appends it (#65),
    # so the reply can't leave the write unnamed any more; the check itself
    # stays covered in test_receipt_claims.py.
    sample = scripted.run(
        S2,
        [
            plan(ask("tasks", "Add a task to water the plants, due tomorrow")),
            text_turn("Done, it's on your list."),
            text_turn("Done, it's on your list."),
        ],
    )
    assert sample.taken.outcome is Outcome.PASS, sample.taken.failure
    assert "water the plants" in (sample.session.last.reply or "")


def test_a_recap_that_invents_fails(scripted: _ScriptedRun) -> None:
    sample = scripted.run(
        S16,
        [
            plan(ask("tasks", "Add a task to water the plants, due tomorrow")),
            text_turn("Added “water the plants” for tomorrow."),
            answer("self"),
            text_turn("I added “water the plants” and queued an album for you."),
        ],
    )
    _fails(sample, "unbacked claim")


def test_a_yes_to_nothing_waiting_never_writes(scripted: _ScriptedRun) -> None:
    # S9's second turn alone: a confirm with no question waiting is refused
    # by the validator (twice), and code says so rather than guess.
    sample = scripted.run(S9, [YES, YES, YES, YES, YES, YES])
    assert sample.session.world.receipts == []
    assert sample.taken.outcome is Outcome.FAIL


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
            app.dependency_overrides[routes_agent.get_conductor] = _seeded_conductor
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
            pass_time=database_ager(db_session),
            # A think (#78) runs on the real 26B, unseeded, when S14 waits.
            settle=_watching(session_factory, holder, provider, think_provider_from_settings),
        )


def _seeded_conductor() -> Generator[Interpreter, None, None]:
    provider = eval_seed.seeded_provider()
    try:
        yield conductor_from_settings(provider, app.state.fleet)
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
