"""Fleet discovery: scanning app.yaml manifests into a Fleet.

Everything runs against tmp-dir fixtures — no live workspace, no network.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from app.fleet.manifests import discover_fleet

_PCC_MANIFEST = """\
name: tasks
title: Project Command Center
upstream: 127.0.0.1:8100
agent:
  description: "Manages projects and tasks."
  api: /api/agent
  examples:
    - "what's due today"
    - "mark the groceries task done"
"""

_CHESS_MANIFEST = """\
name: chess
title: Chess
upstream: 127.0.0.1:8000
agent:
  description: "Plays chess."
  api: /api/agent
  examples:
    - "move my knight to f3"
"""

# An app with no agent block (like odysseus): a fleet member, but no tool.
_NO_AGENT_MANIFEST = """\
name: odysseus
title: Odysseus
upstream: 127.0.0.1:7000
"""


def _write(root: Path, app_dir: str, manifest: str) -> None:
    directory = root / app_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "app.yaml").write_text(textwrap.dedent(manifest), encoding="utf-8")


def test_discovers_agent_bearing_apps(tmp_path: Path) -> None:
    _write(tmp_path, "project-command-center", _PCC_MANIFEST)
    _write(tmp_path, "chess", _CHESS_MANIFEST)

    fleet = discover_fleet(tmp_path)

    # Sorted by directory name: chess dir, then project-command-center dir.
    assert [app.name for app in fleet.apps] == ["chess", "tasks"]
    tasks = fleet.get("tasks")
    assert tasks is not None
    assert tasks.agent is not None
    assert tasks.agent.description == "Manages projects and tasks."
    assert tasks.agent.api == "/api/agent"
    assert tasks.agent.examples == ("what's due today", "mark the groceries task done")
    assert tasks.agent_base_url == "http://127.0.0.1:8100/api/agent"


def test_app_without_agent_block_is_a_member_but_not_delegable(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)
    _write(tmp_path, "odysseus", _NO_AGENT_MANIFEST)

    fleet = discover_fleet(tmp_path)

    assert {app.name for app in fleet.apps} == {"chess", "odysseus"}
    assert [app.name for app in fleet.agent_apps()] == ["chess"]
    assert [app.name for app in fleet.non_agent_apps()] == ["odysseus"]
    odysseus = fleet.get("odysseus")
    assert odysseus is not None and odysseus.agent is None


def test_malformed_yaml_is_skipped_not_raised(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)
    _write(tmp_path, "broken", "name: [unclosed\n  : : :")

    fleet = discover_fleet(tmp_path)

    assert [app.name for app in fleet.apps] == ["chess"]


def test_missing_required_field_is_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)
    # No upstream — gen.py requires it, so conductor skips it too.
    _write(tmp_path, "half", "name: half\ntitle: Half\n")

    fleet = discover_fleet(tmp_path)

    assert [app.name for app in fleet.apps] == ["chess"]


def test_conductor_own_manifest_is_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)
    # Conductor's own manifest — even present in the scanned dir, never a member.
    _write(
        tmp_path,
        "conductor",
        "name: conductor\ntitle: Conductor\nupstream: 127.0.0.1:8300\n",
    )

    fleet = discover_fleet(tmp_path)

    assert [app.name for app in fleet.apps] == ["chess"]
    assert fleet.get("conductor") is None


def test_malformed_agent_block_degrades_to_non_agent_member(tmp_path: Path) -> None:
    # An agent block missing its `api` is unusable → recorded without an agent.
    _write(
        tmp_path,
        "wonky",
        "name: wonky\ntitle: Wonky\nupstream: 127.0.0.1:9000\n"
        'agent:\n  description: "does things"\n',
    )

    fleet = discover_fleet(tmp_path)

    wonky = fleet.get("wonky")
    assert wonky is not None
    assert wonky.agent is None  # degraded, not delegable
    assert wonky.name in {app.name for app in fleet.non_agent_apps()}


def test_upstream_host_is_rewritten_port_preserved(tmp_path: Path) -> None:
    _write(tmp_path, "project-command-center", _PCC_MANIFEST)
    _write(tmp_path, "chess", _CHESS_MANIFEST)

    fleet = discover_fleet(tmp_path, upstream_host="host.docker.internal")

    tasks = fleet.get("tasks")
    chess = fleet.get("chess")
    assert tasks is not None and chess is not None
    # Host rewritten, port kept.
    assert tasks.upstream == "host.docker.internal:8100"
    assert chess.upstream == "host.docker.internal:8000"
    assert tasks.agent_base_url == "http://host.docker.internal:8100/api/agent"


def test_empty_upstream_host_keeps_manifest_value(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)

    fleet = discover_fleet(tmp_path, upstream_host="")

    chess = fleet.get("chess")
    assert chess is not None and chess.upstream == "127.0.0.1:8000"


def test_missing_manifest_dir_returns_empty_fleet(tmp_path: Path) -> None:
    fleet = discover_fleet(tmp_path / "does-not-exist")
    assert fleet.apps == ()
