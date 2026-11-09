"""Foreground latency while a ``think`` runs on the 26B (#78), measured live.

The GPU is exclusive (``../llama-swap/config.yaml``): a 12B call can't run
while the 26B holds it. This measures what the person waits for: one
foreground planner call on the 12B (the first model call of a turn, with the
real planner prompt and schema), timed from the moment the turn starts.

Arms, each a fresh sample at a time:

- ``warm``: no think anywhere; the 12B is loaded.
- ``yield``: a real think call is generating on the 26B (its own prompt,
  thinking on, ``json_schema``) when the turn starts; the thinker's watch
  (``Thinker._call``) sees the turn and cuts the think off, as in
  production, and the 12B call goes through llama-swap's swap.
- ``no_yield``: the same, but the think is left running: what llama-swap
  alone does with a 12B request while the 26B is busy.

Opt-in, shared GPU: ``CONDUCTOR_THINK_EVALS=1``. Never starts or stops
llama-swap. Prints p50/p95/max per arm; ``docs/agent-evals.md`` records the
run.

    cd backend
    CONDUCTOR_THINK_EVALS=1 CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace \\
        .venv/bin/pytest tests/test_think_contention.py -s
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app import clock
from app.ai import compiler, foreground, think
from app.ai import plan as plans
from app.ai.providers.llamacpp import LlamaCppProvider
from app.config import get_settings
from app.fleet.manifests import discover_fleet
from app.services.thinker import Thinker, _Cut, think_provider_from_settings

_LIVE = os.environ.get("CONDUCTOR_THINK_EVALS") == "1"
_RUNS = int(os.environ.get("CONDUCTOR_THINK_EVAL_RUNS", "10"))
# A think left running holds the GPU for minutes: fewer runs of that arm.
_NO_YIELD_RUNS = int(os.environ.get("CONDUCTOR_THINK_EVAL_NO_YIELD_RUNS", "3"))
WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)
# How long the 26B generates before the turn starts (after it is loaded).
_GENERATING_FOR = 3.0

pytestmark = pytest.mark.skipif(
    not _LIVE, reason="contention runs the real models: CONDUCTOR_THINK_EVALS=1"
)


def _running(model: str) -> bool:
    base = get_settings().llamacpp_base_url.removesuffix("/v1")
    try:
        body = httpx.get(f"{base}/running", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(
        entry.get("model") == model and entry.get("state") == "ready"
        for entry in body.get("running", [])
    )


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def test_foreground_latency_during_a_think(session_factory: sessionmaker[Session]) -> None:
    fleet = discover_fleet(WORKSPACE_ROOT)
    if not fleet.agent_apps():
        pytest.skip(f"no fleet under {WORKSPACE_ROOT}")
    settings = get_settings()
    planner = compiler.planner_prompt(fleet)
    plan_format = {
        "type": "json_schema",
        "json_schema": {"name": "plan", "strict": True, "schema": plans.schema(fleet)},
    }
    turn = [
        {"role": "system", "content": planner},
        {"role": "user", "content": compiler.user_message(None, "what's due today?")},
    ]
    think_messages: list[dict[str, Any]] = [
        {"role": "system", "content": think.system_prompt(fleet)},
        {
            "role": "user",
            "content": think.opening(
                "plan my week around the garage project",
                "plan my week around the garage project",
                [],
                clock.local_today(),
            ),
        },
    ]
    thinker = Thinker(session_factory, fleet, provider=lambda: None)

    def foreground_call() -> float:
        with LlamaCppProvider(
            settings.llamacpp_base_url, settings.llamacpp_model, timeout_seconds=300
        ) as twelve:
            started = time.monotonic()
            twelve.chat(turn, max_tokens=compiler.MAX_TOKENS, response_format=plan_format)
            return time.monotonic() - started

    def during_think(*, yield_gpu: bool) -> tuple[float, str]:
        twenty_six = think_provider_from_settings()
        outcome: dict[str, str] = {}

        def run() -> None:
            try:
                thinker._call(twenty_six, think_messages, "contention")
                outcome["think"] = "finished"
            except _Cut:
                outcome["think"] = "cut off"
            except Exception as exc:
                outcome["think"] = f"ended: {type(exc).__name__}"

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        deadline = time.monotonic() + 120
        while not _running(settings.conductor_think_model) and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(_GENERATING_FOR)
        if yield_gpu:
            foreground.entered()
        try:
            seconds = foreground_call()
        finally:
            if yield_gpu:
                foreground.left()
        worker.join(timeout=300)
        twenty_six.close()
        return seconds, outcome.get("think", "still running")

    results: dict[str, list[float]] = {"warm": [], "yield": [], "no_yield": []}
    endings: dict[str, list[str]] = {"yield": [], "no_yield": []}
    foreground_call()  # load the 12B
    for _ in range(_RUNS):
        results["warm"].append(foreground_call())
    for arm in ("yield", "no_yield"):
        for _ in range(_RUNS if arm == "yield" else _NO_YIELD_RUNS):
            foreground._reset()
            seconds, ending = during_think(yield_gpu=arm == "yield")
            results[arm].append(seconds)
            endings[arm].append(ending)
    foreground._reset()
    for arm, values in results.items():
        if not values:
            continue
        print(
            f"[contention] {arm}: n={len(values)} p50={_pct(values, 0.5):.2f}s "
            f"p95={_pct(values, 0.95):.2f}s max={max(values):.2f}s "
            f"all={[round(v, 2) for v in values]}"
        )
    for arm, ends in endings.items():
        print(f"[contention] {arm} think endings: {ends}")
