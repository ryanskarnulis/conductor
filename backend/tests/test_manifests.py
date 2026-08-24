"""Fleet discovery: scanning app.yaml manifests into a Fleet.

Everything runs against tmp-dir fixtures — no live workspace, no network.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

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

# Chess as it actually ships: openable, not delegable — the user is handed
# over to the board rather than proxied through conductor.
_OPEN_CHESS_MANIFEST = """\
name: chess
title: Chess
upstream: 127.0.0.1:8000
open:
  description: "Opens the chess board."
  path: /
  intent_param: intent
  examples:
    - "let's play chess"
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


def test_app_without_agent_or_open_block_is_a_member_but_inert(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)
    _write(tmp_path, "odysseus", _NO_AGENT_MANIFEST)

    fleet = discover_fleet(tmp_path)

    assert {app.name for app in fleet.apps} == {"chess", "odysseus"}
    assert [app.name for app in fleet.agent_apps()] == ["chess"]
    assert [app.name for app in fleet.inert_apps()] == ["odysseus"]
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


def test_malformed_agent_block_degrades_to_inert_member(tmp_path: Path) -> None:
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
    assert wonky.name in {app.name for app in fleet.inert_apps()}


# --- open: blocks (handoff apps) ----------------------------------------------


def test_discovers_an_openable_app(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _OPEN_CHESS_MANIFEST)

    fleet = discover_fleet(tmp_path)

    chess = fleet.get("chess")
    assert chess is not None
    assert chess.open is not None
    assert chess.open.description == "Opens the chess board."
    assert chess.open.path == "/"
    assert chess.open.intent_param == "intent"
    assert chess.open.examples == ("let's play chess",)
    # Openable but not delegable: it gets an open tool, not an ask tool, and
    # it is emphatically not inert.
    assert [app.name for app in fleet.open_apps()] == ["chess"]
    assert fleet.agent_apps() == ()
    assert fleet.inert_apps() == ()


def test_open_block_defaults_path_and_takes_no_intent_when_unset(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "arcade",
        'name: arcade\ntitle: Arcade\nupstream: 127.0.0.1:8500\nopen:\n  description: "Opens it."\n',
    )

    arcade = discover_fleet(tmp_path).get("arcade")

    assert arcade is not None and arcade.open is not None
    assert arcade.open.path == "/"
    # No intent_param → hand off to a bare URL, carrying nothing.
    assert arcade.open.intent_param is None


def test_an_app_can_be_both_delegable_and_openable(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "both",
        "name: both\ntitle: Both\nupstream: 127.0.0.1:9100\n"
        'agent:\n  description: "Does things."\n  api: /api/agent\n'
        'open:\n  description: "Opens things."\n',
    )

    fleet = discover_fleet(tmp_path)

    app = fleet.get("both")
    assert app is not None
    assert app.has_agent and app.has_open
    assert [a.name for a in fleet.agent_apps()] == ["both"]
    assert [a.name for a in fleet.open_apps()] == ["both"]


def test_malformed_open_block_degrades_without_touching_the_agent_block(tmp_path: Path) -> None:
    # An open block with no description is unusable → recorded without it. The
    # app keeps its (valid) agent block: one bad block never costs the other.
    _write(
        tmp_path,
        "wonky",
        "name: wonky\ntitle: Wonky\nupstream: 127.0.0.1:9000\n"
        'agent:\n  description: "does things"\n  api: /api/agent\n'
        "open:\n  path: /play\n",
    )

    wonky = discover_fleet(tmp_path).get("wonky")

    assert wonky is not None
    assert wonky.open is None  # degraded, not openable
    assert wonky.has_agent


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


# --- the actions prefix -------------------------------------------------------

_ACTIONS_MANIFEST = """\
name: music
title: Music
upstream: 127.0.0.1:8500
agent:
  description: "Downloads and sorts music."
  api: /api/agent
  actions: /api/sorting/
  examples:
    - "sort my music"
"""


def test_an_app_can_declare_a_prefix_a_page_may_act_on(tmp_path: Path) -> None:
    """One prefix, so a panel in conductor's UI can answer without a model turn."""
    _write(tmp_path, "music", _ACTIONS_MANIFEST)

    music = discover_fleet(tmp_path).get("music")

    assert music is not None and music.agent is not None
    # Trailing slash trimmed, so the proxy joins one way rather than two.
    assert music.agent.actions == "/api/sorting"
    assert music.actions_base_url == "http://127.0.0.1:8500/api/sorting"


def test_most_apps_declare_none_and_have_no_such_url(tmp_path: Path) -> None:
    _write(tmp_path, "chess", _CHESS_MANIFEST)

    chess = discover_fleet(tmp_path).get("chess")

    assert chess is not None and chess.agent is not None
    assert chess.agent.actions is None
    with pytest.raises(ValueError, match="no actions prefix"):
        _ = chess.actions_base_url


def test_a_malformed_actions_prefix_costs_the_prefix_and_nothing_else(tmp_path: Path) -> None:
    """Same rule as the block itself: a bad key never costs the app its tool.

    It must be a rooted path — it is the whole of what conductor will forward
    to, so "anything" is not an option.
    """
    _write(tmp_path, "music", _ACTIONS_MANIFEST.replace("actions: /api/sorting/", "actions: true"))

    music = discover_fleet(tmp_path).get("music")

    assert music is not None and music.agent is not None
    assert music.agent.actions is None
    assert music.has_agent, "the app still delegates"
