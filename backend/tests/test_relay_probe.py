"""The relay probe (spike R1 → #65, from #55): does the reply keep the saved song's title?

"download this song <link>" through conductor's real loop, prompt and workspace
manifests on the live model, with music's hop canned to music's real handoff:
its note ("bet. it's in the Unsorted folder now.") and its v1.1 record, which
names the song as outside text. "Kept" means the final reply names the title.
Two titles, as the spike ran them: a real-looking one and a placeholder-looking
one the narrator alone dropped ~40% of the time.

    cd backend
    CONDUCTOR_AGENT_EVALS=1 .venv/bin/pytest tests/test_relay_probe.py -v -s

``CONDUCTOR_RELAY_RUNS`` sets runs per title (default 20); the bar is ≥95% kept
across both. ``CONDUCTOR_NARRATOR=0`` measures the planner's own relay (the
spike's "today" arm). Baseline in ``docs/agent-evals.md`` ("Relay probe").
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.ai.loop import loop_from_settings
from app.ai.providers.llamacpp import provider_from_settings
from app.config import get_settings
from app.fleet.context import DelegationContext, InMemoryThreadStore, use_delegation_context
from app.fleet.delegate import (
    EntityRef,
    MessageExchange,
    MessageRead,
    RecordEntry,
    ToolCallRead,
    Undo,
)
from app.fleet.manifests import fleet_from_settings
from app.fleet.tools import build_delegate_tools, render_fleet_section
from tests.evalstats import wilson_interval

_LIVE = os.environ.get("CONDUCTOR_AGENT_EVALS") == "1"
_RUNS = int(os.environ.get("CONDUCTOR_RELAY_RUNS", "20"))
WORKSPACE_ROOT = Path(
    os.environ.get("CONDUCTOR_EVAL_FLEET_DIR", str(Path(__file__).resolve().parents[3]))
)
SAID = "download this song please https://example.com/watch?v=abc123"
NOTE = "bet. it's in the Unsorted folder now."
TITLES = ("Odessa", "Real Title")
BAR = 0.95


class _Music:
    """Music's hop, canned: its note and a record that names the song."""

    def __init__(self, title: str) -> None:
        self.title = title

    def __enter__(self) -> _Music:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_conversation(self, *, title: str | None = None) -> int:
        return 1

    def send_message(
        self, conversation_id: int, message: str, *, confirm_token: str | None = None
    ) -> MessageExchange:
        now = datetime.now(UTC)
        song = EntityRef(app="music", kind="song", id="812", title=self.title, untrusted=True)
        record = [
            RecordEntry(
                tool="download_song",
                outcome="done",
                touched=[song],
                undo=Undo(none="a download can't be undone"),
            )
        ]
        return MessageExchange(
            user_message=MessageRead(
                id=1, conversation_id=1, role="user", content=message, created_at=now
            ),
            assistant_message=MessageRead(
                id=2,
                conversation_id=1,
                role="assistant",
                content=NOTE,
                tool_calls=[ToolCallRead(tool="download_song", arguments={}, result="ok")],
                stop_reason="completed",
                created_at=now,
                record=record,
            ),
        )


@pytest.fixture
def fleet_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setattr(get_settings(), "fleet_manifest_dir", WORKSPACE_ROOT)
    yield


@pytest.mark.skipif(
    not _LIVE, reason="the relay probe runs the real model: CONDUCTOR_AGENT_EVALS=1"
)
def test_relay_keeps_the_title(fleet_dir: None) -> None:
    fleet = fleet_from_settings()
    current: dict[str, str] = {}

    def factory(app: Any) -> Any:
        if app.name != "music":
            pytest.fail(f"the probe asked {app.name}, not music")
        return _Music(current["title"])

    tools = build_delegate_tools(fleet, factory)
    if "ask_music" not in tools:
        pytest.skip(f"no ask_music in {tools}; is {WORKSPACE_ROOT} the workspace root?")
    provider = provider_from_settings()
    loop = loop_from_settings(provider, fleet_section=render_fleet_section(fleet))
    kept = total = 0
    seconds: list[float] = []
    try:
        for title in TITLES:
            current["title"] = title
            for run in range(_RUNS):
                context = DelegationContext(
                    master_conversation_id=f"relay-{title}-{run}",
                    thread_store=InMemoryThreadStore(),
                    calls_per_turn_per_app=3,
                )
                started = time.monotonic()
                with use_delegation_context(context):
                    result = loop.run(SAID)
                seconds.append(time.monotonic() - started)
                asked = [call.tool for call in result.tool_calls]
                hit = "ask_music" in asked and title.lower() in (result.reply or "").lower()
                kept += hit
                total += 1
                mark = "KEPT" if hit else "drop"
                print(f"[relay] {title!r} {run} {mark} {asked} {result.reply!r}", flush=True)
    finally:
        provider.close()
    low, high = wilson_interval(kept, total)
    p50 = sorted(seconds)[len(seconds) // 2]
    print(f"[relay] kept {kept}/{total} ci [{low:.2f}, {high:.2f}] turn_p50={p50:.2f}s")
    assert kept / total >= BAR, f"kept {kept}/{total}, below {BAR:.0%}"
