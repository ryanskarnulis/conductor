"""Fleet smoke (#143): S2, S3, S9, S15 and S15p through conductor against real PCC and music.

Session evals hold conductor's choices to account over a fake world. This is
the other half: the same person's side of four scenarios, through the same
route, planner, interpreter and narrator, but the ``ask`` lands on the real
PCC and music agents (``fleet_harness.py``: sibling backends on scratch
state, music's downloader faked, nothing leaves loopback). The checks read
the end state from each app's own API and hold the replies to the apps' real
receipts. Measured, never gated: a run prints every sample with the requests
conductor sent and what the apps did, and asserts only that the sandbox held.

    cd backend
    CONDUCTOR_FLEET_SMOKE=1 .venv/bin/pytest tests/test_fleet_smoke.py -v -s

``CONDUCTOR_FLEET_RUNS`` samples per scenario (3); ``CONDUCTOR_FLEET_COST=1``
also measures one delegated write end to end (#155, ``CONDUCTOR_FLEET_COST_RUNS``,
3 each); ``CONDUCTOR_FLEET_REPORT=path`` appends one JSON line per sample.
Needs llama-swap and the sibling checkouts' ``.venv``s; skipped otherwise.
The results are in ``docs/agent-evals.md`` ("Fleet smoke").
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections.abc import Callable, Generator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

import app.main as main_module
from app.ai import interpreter as interpreter_module
from app.api import rate_limit, routes_agent
from app.config import get_settings
from app.db.session import get_db
from app.fleet import glance as glance_module
from app.fleet import jobs as jobs_module
from app.fleet.manifests import FleetApp
from app.fleet.thread_store import DbThreadStore
from app.fleet.tools import default_client_factory
from app.main import app
from app.tools import registry
from tests.eval_db import database_ager, database_wiper
from tests.evalstats import Outcome
from tests.fake_world import World, weekend_targets
from tests.fleet_harness import (
    ENABLED,
    FLAG,
    FakeMusicBrainz,
    Meter,
    MeteredCall,
    ScratchFleet,
    TurnCost,
    cost_lines,
    family,
    missing_siblings,
    read_jsonl,
    receipts_from,
    requests_sent,
    rewrite_manifest,
    turn_costs,
    write_fleet,
)
from tests.session_harness import (
    LoopDriver,
    Sample,
    Scenario,
    Session,
    TurnRecord,
    WorldHolder,
    run_sample,
)
from tests.session_scenarios import S2, S3, S9, S15, S15_PICKED

_RUNS = int(os.environ.get("CONDUCTOR_FLEET_RUNS", "3"))
_COST = os.environ.get("CONDUCTOR_FLEET_COST") == "1"
_COST_RUNS = int(os.environ.get("CONDUCTOR_FLEET_COST_RUNS", "3"))
_REPORT = os.environ.get("CONDUCTOR_FLEET_REPORT")

_WHY_NOT = (
    None if ENABLED else f"the fleet smoke runs real apps and the model: {FLAG}=1"
) or missing_siblings()
live = pytest.mark.skipif(_WHY_NOT is not None, reason=_WHY_NOT or "")


# --- the harness's own pieces (CI: no apps, no model) -------------------------------------


def test_a_rewritten_manifest_changes_only_its_upstream() -> None:
    text = (
        "name: tasks\nupstream: 127.0.0.1:8100\n"
        "agent:\n  description: Manages tasks.\n  api: /api/agent\n  examples: [a, b]\n"
    )
    rewritten = rewrite_manifest(text, "127.0.0.1:49152")
    before, after = yaml.safe_load(text), yaml.safe_load(rewritten)
    assert after.pop("upstream") == "127.0.0.1:49152"
    before.pop("upstream")
    assert after == before


def test_the_fleet_dir_points_dead_siblings_nowhere(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    for folder, name in (("project-command-center", "tasks"), ("chess", "chess")):
        (workspace / folder).mkdir(parents=True)
        (workspace / folder / "app.yaml").write_text(f"name: {name}\nupstream: 127.0.0.1:8000\n")
    fleet = write_fleet(tmp_path / "fleet", workspace, {"project-command-center": "127.0.0.1:5"})
    assert "upstream: 127.0.0.1:5" in (fleet / "project-command-center" / "app.yaml").read_text()
    assert "upstream: 127.0.0.1:1" in (fleet / "chess" / "app.yaml").read_text()
    assert not (fleet / "music").exists()


def test_the_fake_musicbrainz_answers_music_s_three_lookups() -> None:
    # music's musicbrainz.py: a release-group search (artist alone, or with a
    # title), an album's editions, an edition's tracks; tagging finds nothing.
    fake = FakeMusicBrainz()
    try:
        with httpx.Client(base_url=fake.url, trust_env=False) as client:

            def get(path: str, **params: Any) -> Any:
                response = client.get(f"/{path}", params={**params, "fmt": "json"})
                return response.status_code, response.json()

            query = 'artist:"Caribou" AND (primarytype:album OR primarytype:ep)'
            _, found = get("release-group", query=query, limit=100)
            groups = found["release-groups"]
            assert sorted((g["title"], g["first-release-date"][:4]) for g in groups) == [
                ("Honey", "2024"),
                ("Suddenly", "2020"),
                ("Swim", "2010"),
            ]
            assert all(g["primary-type"] == "Album" and not g["secondary-types"] for g in groups)
            query = 'releasegroup:"Suddenly" AND artist:"Caribou"'
            _, named = get("release-group", query=query, limit=100)
            (suddenly,) = named["release-groups"]
            _, editions = get("release", **{"release-group": suddenly["id"]})
            (edition,) = editions["releases"]
            assert edition["media"] == [{"track-count": 4}]
            _, release = get(f"release/{edition['id']}", inc="recordings artist-credits")
            titles = [t["title"] for t in release["media"][0]["tracks"]]
            assert titles == ["Sister", "You and I", "Sunny's Time", "Home"]
            assert get("recording", query="x")[1] == {"recordings": []}
            assert get("release/nope", inc="x")[0] == 404
        assert fake.asked and all(path.startswith("/ws/2/") for path in fake.asked)
    finally:
        fake.close()


def test_receipts_come_from_done_entries_and_skip_job_refs() -> None:
    record = TurnRecord(
        1,
        "x",
        "desktop",
        200,
        tools=[
            {
                "tool": "ask_music",
                "arguments": {"message": "download the new Caribou album"},
                "app_record": [
                    {
                        "tool": "download_playlist",
                        "outcome": "done",
                        "touched": [
                            {"app": "music", "kind": "download_job", "id": "4", "title": "Honey"}
                        ],
                    },
                    {"tool": "download_song", "outcome": "refused", "reason": "nothing found"},
                ],
            },
            {
                "tool": "confirm",
                "arguments": {"app": "tasks"},
                "app_record": [
                    {
                        "tool": "trash_project",
                        "outcome": "done",
                        "touched": [
                            {"app": "tasks", "kind": "project", "id": "2", "title": "Kitchen"},
                            {"app": "tasks", "kind": "job", "id": "9", "title": "bookkeeping"},
                        ],
                    }
                ],
            },
            {
                "tool": "ask_tasks",
                "arguments": {"message": "add a task to sort {s1}"},
                "error": "not run: s1 gave no name to pass on",
            },
            {"tool": "ask_memory", "arguments": {}, "app_record": [{"tool": "remember"}]},
        ],
    )
    got = [(r.app, r.kind, r.title) for r in receipts_from(record)]
    # A job alone names what was asked for; beside a real thing, it's bookkeeping.
    assert got == [("music", "queue", "Honey"), ("tasks", "trash", "Kitchen")]
    assert requests_sent(record) == [
        "ask_music: 'download the new Caribou album'",
        "confirm: the yes, to tasks",
        "ask_tasks: 'add a task to sort {s1}' (not run)",
        "ask_memory: ''",
    ]
    assert [family(t) for t in ("create_task", "update_task", "sort_music")] == [
        "create",
        "update",
        "file",
    ]


def test_turn_costs_join_each_hop_to_its_app_run() -> None:
    meter = Meter("http://127.0.0.1:1")
    t0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC).timestamp()

    def call(tag: str, at: float, ms: int, prompt: int, cached: int) -> MeteredCall:
        return MeteredCall(tag, t0 + at, t0 + at + ms / 1000, ms, 200, prompt, cached, 10)

    meter.calls = [
        call("conductor", 0.1, 900, 4000, 3900),
        call("tasks", 1.1, 1500, 6000, 200),
        call("tasks", 2.7, 400, 6100, 6000),
        call("conductor", 3.2, 300, 1200, 0),
        call("music", 50.0, 100, 10, 0),  # another turn's
    ]

    def stamp(at: float) -> str:
        return datetime.fromtimestamp(t0 + at, UTC).isoformat()

    conductor = [
        {
            "run_id": "r1",
            "utterance": "add a task",
            "started": stamp(0),
            "ts": stamp(3.6),
            "total_ms": 3600,
            "calls": [{"role": "planner"}, {"role": "narrator"}],
            "hops": [{"app": "tasks", "thread_id": 7, "run_id": "r1"}],
        }
    ]
    pcc = [
        {"conversation_id": 7, "started": stamp(-100), "calls": [{}], "total_ms": 1},
        {
            "conversation_id": 7,
            "started": stamp(1.0),
            "total_ms": 2200,
            "calls": [{"phase": "planner"}, {"phase": "narrator"}],
        },
    ]
    (cost,) = turn_costs(conductor, {"tasks": pcc}, meter)
    meter.close()
    assert cost.app_runs == [("tasks", 2, 2200)]
    assert [(r.who, r.role, r.ms, r.cached_tokens) for r in cost.calls] == [
        ("conductor", "planner", 900, 3900),
        ("tasks", "planner", 1500, 200),
        ("tasks", "narrator", 400, 6000),
        ("conductor", "narrator", 300, 0),
    ]
    assert "total=3600ms calls=4" in cost_lines(cost)[0]


# --- the live fleet -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> Generator[ScratchFleet, None, None]:
    scratch = ScratchFleet(tmp_path_factory.mktemp("fleet-smoke"))
    print(f"\n[fleet] scratch root {scratch.root}")
    try:
        scratch.start()
        print(
            f"[fleet] tasks on :{scratch.pcc.port}, music on :{scratch.music.port}, "
            f"llm via meter {scratch.conductor_llm}"
        )
        yield scratch
    finally:
        scratch.close()
        watch = scratch.watch
        print(
            f"[fleet] watch: {watch.samples} samples, {len(watch.egress)} off-loopback "
            f"sockets, {len(watch.strays)} files outside the scratch root; "
            f"{len(scratch.yt_dlp_calls())} fake yt-dlp calls; "
            f"app starts {[a.started for a in scratch.apps]}"
        )
    assert not watch.egress, f"a scratch app reached off loopback: {sorted(watch.egress)}"
    assert not watch.strays, f"a scratch app held files outside its root: {sorted(watch.strays)}"


def _tool_names() -> set[str]:
    return {tool.name for tool in registry.all_tools()}


class FleetDriver(LoopDriver):
    """``LoopDriver`` over the real apps: each sample starts on a fresh
    scratch fleet, and each turn's real receipts become the world's, so the
    generic checks (claims backed, writes named) read what the apps did."""

    def __init__(self, client: TestClient, fleet: ScratchFleet, db_session: DbSession) -> None:
        super().__init__(
            client,
            WorldHolder(),
            tool_names=_tool_names,
            reset=rate_limit._reset,
            fresh_db=database_wiper(db_session),
            pass_time=database_ager(db_session),
        )
        self.fleet = fleet
        self._world: World | None = None

    def begin(self, world: World) -> None:
        self.fleet.reset()
        self._world = world
        super().begin(world)

    def say(self, text: str, *, surface: str, incognito: bool = False) -> TurnRecord:
        record = super().say(text, surface=surface, incognito=incognito)
        if self._world is not None:
            self._world.receipts.extend(receipts_from(record))
        return record


def _only(allowed: frozenset[str], what: str) -> Callable[[FleetApp], None]:
    def check(fleet_app: FleetApp) -> None:
        if fleet_app.upstream not in allowed:
            raise glance_module.GlanceError(f"the fleet smoke never dials {what} on {fleet_app}")

    return check


@pytest.fixture
def driver(
    fleet: ScratchFleet,
    tmp_path: Path,
    db_session: DbSession,
    session_factory: sessionmaker[DbSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[FleetDriver, None, None]:
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "fleet_manifest_dir",
        write_fleet(tmp_path / "fleet", fleet.workspace, fleet.upstreams()),
    )
    monkeypatch.setattr(settings, "llamacpp_base_url", fleet.conductor_llm)
    monkeypatch.setattr(settings, "conductor_trace_path", fleet.conductor_trace)
    # Conductor reaches the scratch apps and nothing else, whatever a
    # manifest says: delegate calls, glances and job cancels all check first.
    guard = _only(fleet.allowed, "an app")
    real_factory = default_client_factory()

    def factory(fleet_app: FleetApp) -> Any:
        guard(fleet_app)
        return real_factory(fleet_app)

    def glance(fleet_app: FleetApp) -> dict[str, Any]:
        guard(fleet_app)
        return glance_module.read_glance(fleet_app)

    def cancel(fleet_app: FleetApp, ref: str) -> Any:
        guard(fleet_app)
        return jobs_module.cancel_job(fleet_app, ref)

    monkeypatch.setattr(main_module, "default_client_factory", lambda: factory)
    monkeypatch.setattr(interpreter_module, "read_glance", glance)
    monkeypatch.setattr(interpreter_module, "cancel_job", cancel)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[routes_agent.get_thread_store] = lambda: DbThreadStore(session_factory)
    try:
        with TestClient(app) as client:
            names = _tool_names()
            assert {"ask_tasks", "ask_music"} <= names, f"discovered {sorted(names)}"
            yield FleetDriver(client, fleet, db_session)
    finally:
        app.dependency_overrides.clear()


# --- the scenarios, checked against the real apps --------------------------------------------


def _nothing_trashed(fleet: ScratchFleet) -> None:
    trash = fleet.trash()
    gone = [p["name"] for p in trash["projects"]] + [t["title"] for t in trash["tasks"]]
    assert not gone, f"also trashed {gone}"


def _fetched(fleet: ScratchFleet, needle: str) -> list[str]:
    """What music was asked to fetch that mentions ``needle``: queued jobs'
    queries and the targets its (fake) yt-dlp was run on."""
    asked = [str(job["query"]) for job in fleet.jobs()]
    asked += [argv[-1] for argv in fleet.yt_dlp_calls() if "--flat-playlist" not in argv]
    return [text for text in dict.fromkeys(asked) if needle in text.lower()]


def _weekend_task(fleet: ScratchFleet, needle: str) -> None:
    made = fleet.new_tasks()
    named = [t for t in made if needle in t["title"].lower()]
    assert named, (
        f"no task titled with {needle!r} (created: {[t['title'] for t in made] or 'none'})"
    )
    weekend = weekend_targets(fleet.today)
    due = [date.fromisoformat(t["due_date"]) for t in named if t.get("due_date")]
    assert any(day in weekend for day in due), (
        f"task {named[0]['title']!r} due {due or 'never'}, not this weekend {sorted(weekend)}"
    )


def s2_checks(fleet: ScratchFleet) -> tuple[Callable[[Session], None], ...]:
    def end_state(session: Session) -> None:
        made = fleet.new_tasks()
        assert len(made) == 1, f"{len(made)} tasks created: {[t['title'] for t in made]}"
        task = made[0]
        title = task["title"].lower()
        assert "water" in title and "plant" in title, f"title {task['title']!r}"
        tomorrow = (fleet.today + timedelta(days=1)).isoformat()
        assert task.get("due_date") == tomorrow, f"due {task.get('due_date')}, expected {tomorrow}"
        assert not _fetched(fleet, ""), "music was asked for something"
        _nothing_trashed(fleet)

    return (end_state,)


def _albums_queued(fleet: ScratchFleet) -> list[str]:
    """The albums music queued by name, oldest first (music#96: a job that
    names its album, by MusicBrainz's title and release-group id)."""
    jobs = sorted(fleet.jobs(), key=lambda job: int(job["id"]))
    return [str(job["album_title"]) for job in jobs if job.get("album_id")]


def s3_checks(fleet: ScratchFleet) -> tuple[Callable[[Session], None], ...]:
    # Music resolves "the new Caribou album" to Honey (music#96, against the
    # harness's MusicBrainz), and the task carries that name (#174).
    def end_state(session: Session) -> None:
        albums = _albums_queued(fleet)
        assert albums == ["Honey"], (
            f"albums queued {albums or 'none'} (jobs: {_fetched(fleet, '') or 'none'}), not Honey"
        )
        _weekend_task(fleet, "honey")
        made = [t["title"] for t in fleet.new_tasks()]
        assert len(made) == 1, f"{len(made)} tasks created: {made}"
        _nothing_trashed(fleet)

    return (end_state,)


def s9_checks(fleet: ScratchFleet) -> tuple[Callable[[Session], None], ...]:
    def end_state(session: Session) -> None:
        ask, yes = session.turn("ask"), session.turn("yes")
        assert not session.writes(ask), "wrote before the yes"
        assert "?" in (ask.reply or ""), "didn't ask before wiping"
        names = {p["name"] for p in fleet.projects()}
        trashed = {p["name"] for p in fleet.trash()["projects"]}
        assert "Kitchen" not in names and "Kitchen" in trashed, "the yes didn't wipe Kitchen"
        assert {"Household", "Garage"} <= names, f"other projects gone: {names}"
        assert re.search(r"trash|deleted|removed|wiped|bin", (yes.reply or "").lower()), (
            "didn't say where it went"
        )

    return (end_state,)


def s15_checks(fleet: ScratchFleet) -> tuple[Callable[[Session], None], ...]:
    # Music, asked for "their other album" with the artist alone while Honey
    # is held, queues nothing and asks which album (music#102): Honey stays
    # the only album and its task the only task, and the reply passes the
    # question on (#168, #174; the session evals' S15).
    def end_state(session: Session) -> None:
        albums = _albums_queued(fleet)
        assert albums == ["Honey"], f"albums queued {albums or 'none'}, not Honey alone"
        made = [t["title"] for t in fleet.new_tasks()]
        assert len(made) == 1 and "honey" in made[0].lower(), (
            f"tasks created {made}, not S3's Honey task alone"
        )
        reply = (session.turn("same").reply or "").lower()
        assert re.search(r"suddenly|swim|which|link|\?", reply), (
            "didn't name the albums to pick from or ask which one"
        )

    return (end_state,)


def s15p_checks(fleet: ScratchFleet) -> tuple[Callable[[Session], None], ...]:
    # Then "Suddenly": music queues it, and the task that waited on its name
    # is made for the weekend (#168).
    def end_state(session: Session) -> None:
        albums = _albums_queued(fleet)
        assert albums == ["Honey", "Suddenly"], f"albums queued {albums or 'none'}"
        _weekend_task(fleet, "suddenly")
        made = [t["title"] for t in fleet.new_tasks()]
        assert len(made) == 2, f"{len(made)} tasks created: {made}"
        _nothing_trashed(fleet)

    return (end_state,)


_SMOKE: dict[str, tuple[Scenario, Callable[[ScratchFleet], tuple[Any, ...]]]] = {
    "S2": (S2, s2_checks),
    "S3": (S3, s3_checks),
    "S9": (S9, s9_checks),
    "S15": (S15, s15_checks),
    "S15p": (S15_PICKED, s15p_checks),
}


def _app_runs(fleet: ScratchFleet, since: float) -> list[str]:
    """What each app's own loop was handed since ``since``: the utterance in
    its trace, which is conductor's request as it arrived."""
    lines = []
    for scratch in fleet.apps:
        for run in read_jsonl(scratch.root / "trace.jsonl"):
            if datetime.fromisoformat(run["started"]).timestamp() < since:
                continue
            tools = [t.get("name") or t.get("tool") for t in run.get("tools") or []]
            lines.append(
                f"{scratch.name} heard {run.get('utterance')!r} → {run.get('stop_reason')} "
                f"tools={tools} calls={run.get('model_calls')} {run.get('total_ms')}ms"
            )
    return lines


def _print_sample(scenario: Scenario, sample: Sample, heard: list[str]) -> None:
    print(
        f"[fleet] {scenario.id} {sample.taken.outcome} {sample.seconds:.1f}s"
        + (f"\n[fleet]   ✗ {sample.taken.failure}" if sample.taken.failure else "")
    )
    for turn in sample.session.turns:
        print(f"[fleet]   {turn.index}> {turn.text!r} ({turn.seconds:.1f}s, {turn.stop_reason})")
        for sent in requests_sent(turn):
            print(f"[fleet]     sent {sent}")
        for call in turn.tools:
            if call.get("app_record") is not None:
                record = [
                    (
                        e.get("tool"),
                        e.get("outcome"),
                        [r.get("title") for r in e.get("touched") or []],
                    )
                    for e in call["app_record"]
                ]
                print(f"[fleet]     {call['tool']} record {record}")
            elif call.get("error"):
                print(f"[fleet]     {call['tool']} error {call['error'][:200]!r}")
        print(f"[fleet]   {turn.index}< {turn.reply!r}")
    for line in heard:
        print(f"[fleet]     {line}")


@live
@pytest.mark.parametrize("scenario_id", list(_SMOKE))
def test_fleet_smoke(scenario_id: str, driver: FleetDriver) -> None:
    base, checks = _SMOKE[scenario_id]
    scenario = Scenario(base.id, base.summary, base.steps, checks(driver.fleet), tier="frontier")
    outcomes: list[Outcome] = []
    for run in range(_RUNS):
        since = time.time()
        sample = run_sample(scenario, driver, World())
        heard = _app_runs(driver.fleet, since)
        _print_sample(scenario, sample, heard)
        outcomes.append(sample.taken.outcome)
        if _REPORT:
            line = {
                "scenario": scenario.id,
                "run": run,
                "outcome": str(sample.taken.outcome),
                "failure": sample.taken.failure,
                "seconds": round(sample.seconds, 1),
                "turns": [
                    {
                        "said": t.text,
                        "sent": requests_sent(t),
                        "reply": t.reply,
                        "seconds": round(t.seconds, 1),
                    }
                    for t in sample.session.turns
                ],
                "apps_heard": heard,
            }
            with open(_REPORT, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
        assert sample.taken.outcome is not Outcome.HARNESS, sample.taken.failure
    passed = sum(1 for o in outcomes if o is Outcome.PASS)
    print(f"[fleet] scenario={scenario.id} PASS {passed}/{len(outcomes)} {outcomes}")


# --- #155: one delegated write, end to end ---------------------------------------------------

_WRITES = (
    ("add a task", "add a task to water the plants tomorrow"),
    ("download a song", "download the song Odessa by Caribou"),
)


def _summary(label: str, costs: list[TurnCost]) -> None:
    by_role: dict[tuple[str, str], list[tuple[int, int | None, int | None]]] = {}
    for cost in costs:
        for row in cost.calls:
            by_role.setdefault((row.who, row.role), []).append(
                (row.ms, row.prompt_tokens, row.cached_tokens)
            )
    totals = [c.total_ms for c in costs]
    print(
        f"[fleet-cost] {label}: n={len(costs)} calls/turn={[len(c.calls) for c in costs]} "
        f"turn ms median={statistics.median(totals) if totals else 0:.0f} all={totals}"
    )
    for (who, role), rows in by_role.items():
        print(
            f"[fleet-cost]   {who}/{role}: ms={[r[0] for r in rows]} "
            f"prompt={[r[1] for r in rows]} cached={[r[2] for r in rows]}"
        )


@live
@pytest.mark.skipif(
    not _COST, reason="the delegated-write measurement is opt-in: CONDUCTOR_FLEET_COST=1"
)
def test_delegated_write_cost(driver: FleetDriver) -> None:
    fleet = driver.fleet
    for label, words in _WRITES:
        windows: list[tuple[float, float]] = []
        for _ in range(_COST_RUNS):
            # The apps' restart is setup, not the turn: the window opens at the turn.
            driver.begin(World())
            started = time.time()
            record = driver.say(words, surface="desktop")
            windows.append((started, time.time()))
            print(f"[fleet-cost] {words!r} → {record.status} {record.reply!r}")
        traced = [
            t
            for t in read_jsonl(fleet.conductor_trace)
            if any(
                a - 1 <= datetime.fromisoformat(t["started"]).timestamp() <= b for a, b in windows
            )
        ]
        apps = {s.name: read_jsonl(s.root / "trace.jsonl") for s in fleet.apps}
        costs = turn_costs(traced, apps, fleet.meter)
        for cost in costs:
            for line in cost_lines(cost):
                print(line)
        _summary(label, costs)
        assert len(costs) == _COST_RUNS, f"{len(costs)} traced turns for {_COST_RUNS} runs"
