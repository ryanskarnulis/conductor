"""Conductor MCP server (stdio): the delegate tools over the MCP protocol.

Run with ``python -m app.mcp.server`` from ``backend/`` (the repo's
``.mcp.json`` does exactly that). Conductor discovers the fleet, builds one
``ask_<app>`` tool per agent (plus ``list_agents``) on the shared registry, and
exposes every one on a FastMCP instance — so an MCP host (Claude Code) can
route to the house's apps through conductor. FastMCP re-derives each tool's
schema from the same signatures/docstrings the registry uses.

Unlike the HTTP loop (Slice 4), the MCP server has no per-request scope, so it
registers **one process-global** :class:`DelegationContext`: subagent threads
persist across tool calls within a session (follow-ups carry context app-side),
the driver is tagged ``agent:mcp``, and the per-turn call budget is disabled
(the driving agent is the trusted MCP host, not conductor's shallow loop). The
delegate client still presents ``agent:conductor`` downstream, as the contract
requires.

All fleet discovery and tool registration happens in :func:`main`, *after*
logging is pointed at stderr — the import must stay side-effect-free and
silent, because stdout is the JSON-RPC transport and a stray log line there
corrupts it.
"""

from __future__ import annotations

import sys

import structlog
from mcp.server.fastmcp import FastMCP

from app.fleet.context import (
    MCP_DRIVER,
    DelegationContext,
    InMemoryThreadStore,
    set_process_delegation_context,
)
from app.fleet.manifests import Fleet, fleet_from_settings
from app.fleet.tools import build_delegate_tools, default_client_factory
from app.logging_config import configure_logging
from app.tools import registry

logger = structlog.get_logger(__name__)

mcp = FastMCP(
    "conductor",
    instructions=(
        "Conductor: the household fleet's master agent. Each tool delegates to one "
        "app's agent over the workspace delegate contract and returns what that agent "
        "said; call list_agents to see which apps you can reach and what each does. "
        "Conductor acts only through these tools — it has no abilities of its own."
    ),
)


def setup() -> Fleet:
    """Discover the fleet, build the delegate tools, and mount them on FastMCP.

    Idempotent-enough for one process: registers the tools on the shared
    registry, exposes them on :data:`mcp`, and binds the process-wide
    delegation context. Call once, after logging is configured.
    """
    fleet = fleet_from_settings()
    build_delegate_tools(fleet, default_client_factory())
    set_process_delegation_context(
        DelegationContext(
            master_conversation_id="mcp",
            thread_store=InMemoryThreadStore(),
            calls_per_turn_per_app=0,  # disabled: the driver is the trusted MCP host
            driver=MCP_DRIVER,
        )
    )
    for registered in registry.all_tools():
        mcp.add_tool(registered.fn)
    return fleet


def main() -> None:
    # stdout carries the JSON-RPC transport; all logging must go to stderr,
    # configured before any fleet discovery logs a line.
    configure_logging(stream=sys.stderr)
    fleet = setup()
    logger.info(
        "mcp_server_starting",
        server="conductor",
        tools=len(registry.all_tools()),
        agents=len(fleet.agent_apps()),
    )
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
