# Conductor

The workspace fleet's master agent. Conductor is where a user talks to "the
house" — it figures out which app's agent should handle a request and
delegates to it, holding the conversation across apps.

This is Phase 3 of the workspace's
[`agent-standard/AGENTS-MASTER-PLAN.md`](../agent-standard/AGENTS-MASTER-PLAN.md).
**Current state: conversations API (Slice 4 backend).** On top of the agent
engine (tool registry, llama.cpp provider, bounded loop, layered Glitch
personality) and fleet delegation (`ask_<app>` tools discovered from `app.yaml`
`agent:` blocks, also exposed over an MCP stdio server via `.mcp.json`),
conductor now persists master conversations in SQLite (SQLAlchemy + Alembic)
and fronts its loop with a REST chat API: conversation CRUD,
`POST /api/agent/conversations/{id}/messages` for the synchronous
message → loop round trip, and `GET …/activity` — the poll target that reports
"asking chess…"-style progress while a turn blocks (no SSE in v1). The
subagent-thread map is DB-backed, so follow-ups keep their app-side context
across restarts. Still to come: the chat web UI (Slice 4 frontend) and the
routing eval harness. See `CLAUDE.md` for what's live today versus what's
coming.

## Architecture sketch

```
Browser ──► conductor frontend (nginx :8300) ──► conductor backend (:8301)
                                                        │  /api/agent/* + SQLite
                                                        │  delegates to:
                          ┌─────────────────────────────┼─────────────────────┐
                          ▼                              ▼                     ▼
                  PCC   /api/agent/*            chess  /api/agent/*    future app agents
                  (project-command-center)      (chess)
```

Conductor discovers per-app agents from their `app.yaml` `agent:` blocks and
calls them over the standardized REST contract in
[`../agent-standard/delegate-api.md`](../agent-standard/delegate-api.md) — one
delegate tool per app, depth-1 only (app agents never call each other; only
conductor calls out). Every app agent (this one included, once it has one)
follows [`../agent-standard/STANDARD.md`](../agent-standard/STANDARD.md), with
[`project-command-center`](../project-command-center) as the reference
implementation.

## Stack

```
Frontend:  React + Vite + TypeScript
Backend:   FastAPI + SQLAlchemy 2.0 + Alembic (SQLite in data/)
Config:    pydantic-settings
Agent:     tool registry + llama.cpp provider + bounded loop (gemma-4-12b)
```

## Ports

Conductor owns the `8300`–`8399` block in the workspace gateway's port
registry ([`../gateway/README.md`](../gateway/README.md)):

| Service | Port |
|---|---|
| frontend (docker, published) | `127.0.0.1:8300` |
| backend (dev) | `127.0.0.1:8301` |
| frontend (vite dev server) | `127.0.0.1:5174` |

## Dev quickstart

```bash
./main.sh                 # bootstrap env/deps, start backend + frontend
./test.sh                 # backend pytest/ruff/mypy + frontend Vitest/lint/build
```

Frontend: http://127.0.0.1:5174 · Backend: http://127.0.0.1:8301/health

## Deploy with Docker

```bash
cp .env.example .env
docker compose up --build     # backend + frontend
```

The frontend is host-only by default (`http://127.0.0.1:8300`); the
workspace gateway proxies `conductor.$HOMELAB_DOMAIN` here once `app.yaml` is
picked up by `gateway/gen.py`. The backend publishes no host port — it's
reachable only via nginx and the compose network.
