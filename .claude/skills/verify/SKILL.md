---
name: verify
description: Launch and drive conductor's backend to verify a change end-to-end against the real fleet (llama-swap + sibling app agents).
---

# Verifying conductor

The surface is the HTTP API on `127.0.0.1:8301` (the chat UI proxies to it).
Real end-to-end needs two things running on the host:

- **llama-swap** on `127.0.0.1:8200` (`curl -s 127.0.0.1:8200/v1/models`) —
  gemma-4-12b cold-loads in ~10-100s on the first call; warm turns are ~1-5s.
- **At least one fleet app** — PCC's backend on `127.0.0.1:8101/health` is the
  usual read-only target (`"what tasks are due today?"` routes to `ask_tasks`
  and mutates nothing). Chess asks can mutate a live game; prefer analysis
  questions if you must route there.

## Launch

```bash
cd backend && .venv/bin/alembic upgrade head   # dev DB: ../data/conductor.db
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8301   # background it
```

Startup log must show `fleet_discovered` and `delegate_tools_built` with the
expected `ask_<app>` tools (fleet scans the workspace root by default).
Don't use `python -m app.main` for verification — it runs with `--reload`.

## Drive

```bash
BASE=http://127.0.0.1:8301/api/agent
curl -s -X POST $BASE/conversations -d '{}' -H 'Content-Type: application/json'
# the one model-calling endpoint (blocks; give it -m 400 for a cold model):
curl -s -m 400 -X POST $BASE/conversations/1/messages \
  -H 'Content-Type: application/json' -d '{"content":"what tasks are due today?"}'
```

While a POST blocks, poll the progress channel from another shell — it should
show `kind:"tool", tool:"ask_<app>"` mid-run and `active:false` after:

```bash
curl -s $BASE/conversations/1/activity
```

Worth checking after a delegate run:

- The reply's `tool_calls` names the expected `ask_<app>` with a real result.
- One `delegate_call` structlog line per delegate (app, subagent id, latency).
- `data/conductor.db` → `delegate_threads` maps `(master_id, app)` to the
  subagent conversation; a **restart** of uvicorn followed by a follow-up
  message must reuse the same `subagent_conversation_id` (the DB store's job).

## Gotchas

- Verification conversations land in the dev DB and in the target app's DB
  (conductor creates a real subagent thread there) — fine in dev, but keep
  asks read-only.
- The activity registry and rate limiter are in-process: they only make sense
  against a single uvicorn worker (which is all we ever run).
