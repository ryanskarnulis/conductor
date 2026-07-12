# Conductor

The workspace fleet's master agent. Conductor is where a user talks to "the
house" — it figures out which app's agent should handle a request and
delegates to it, holding the conversation across apps.

This is Phase 3 of the workspace's
[`agent-standard/AGENTS-MASTER-PLAN.md`](../agent-standard/AGENTS-MASTER-PLAN.md).
**Current state: Phase 3 complete (PRs #1–#6, routing evals GO 12/12); only
the hub-app "Later" item remains open.** On top of the agent
engine (tool registry, llama.cpp provider, bounded loop, layered Glitch
personality) and fleet delegation (`ask_<app>` tools discovered from `app.yaml`
`agent:` blocks, also exposed over an MCP stdio server via `.mcp.json`),
conductor persists master conversations in SQLite (SQLAlchemy + Alembic) and
fronts its loop with a REST chat API — conversation CRUD,
`POST /api/agent/conversations/{id}/messages` for the synchronous
message → loop round trip, and `GET …/activity`, the poll target for live
progress (no SSE in v1). The web UI is a PCC-style chat panel: conversation
sidebar, markdown replies with the delegate-call trajectory above each one,
and a live "Asking chess… · 12s" progress line polled from the activity
endpoint while a turn runs. The subagent-thread map is DB-backed, so
follow-ups keep their app-side context across restarts. The routing eval
harness (`docs/agent-evals.md`) closed Phase 3's go/no-go gate: gemma-4-12b
routes all goldens, refuses out-of-fleet asks, and holds destructive-op
confirmation. See `CLAUDE.md` for details.

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

## Voice (STT/TTS)

Voice rides the same agent pipeline: `app/ai/speech.py` speaks the OpenAI
audio wire format over `httpx` to the shared workspace `../speech/` service
per the fleet contract (`../agent-standard/voice.md`), and
`app/api/routes_voice.py` proxies it — `POST /api/voice/transcribe` (audio →
text, biased toward conductor's fleet-routing vocabulary and app names) and
`POST /api/voice/speak` (text → mp3, the fleet house voice). Both are
rate-limited per client IP (`VOICE_REQUESTS_PER_MIN`, default 30). The chat
panel's vendored MicButton (chess-canonical modules in `frontend/src/voice/`)
gives push-to-talk and hands-free conversation mode; voice-initiated turns
get the reply spoken, typed turns stay silent. Configure with
`SPEECH_BASE_URL` / `TTS_BASE_URL` / `STT_MODEL` / `TTS_MODEL` / `TTS_VOICE`
(defaults in `app/config.py`); leave `SPEECH_BASE_URL` unset to run
voiceless — the voice endpoints answer 503 and nothing else changes.

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
