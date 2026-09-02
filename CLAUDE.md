# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

Conductor is the workspace fleet's master agent: it routes a user's request to
the right per-app agent over the standardized REST delegate contract
(`../agent-standard/delegate-api.md`) and holds the conversation. Apps with an
`agent:` block in their `app.yaml` get an `ask_<app>` tool; apps with an
`open:` block get an `open_<app>` handoff (the user's words carried as
`?intent=`). The backend mirrors PCC's reference implementation
(`../project-command-center/backend/app/`) — read the code as "same as PCC,
minus deltas". `TODO.md` is the backlog; `DONE.md` the log;
`docs/agent-evals.md` the routing eval baseline.

## Commands

```bash
./main.sh                 # bootstrap env/deps, migrate, start backend + frontend
./test.sh                 # backend pytest/ruff/mypy + frontend Vitest/lint/build
```

Ports (gateway registry): frontend `127.0.0.1:8300` (nginx, proxies `/api`),
backend dev `8301`, vite dev `5174`.

## Git workflow

- Never commit to `main`. Branch → PR → squash-merge on green CI
  (`gh pr checks --watch`, then `gh pr merge --squash`). Run `./test.sh`
  before pushing. Deploys via `~/deploy/conductor` clean clones; an `app.yaml`
  change regenerates the gateway too.
- **Eval gate:** any change to the base prompt, vendored personality, the
  loop, the model, **or any fleet app's `agent:`/`open:` manifest block**
  runs the routing evals (`CONDUCTOR_AGENT_EVALS=1 pytest
  tests/test_agent_evals.py -v -s`, llama-swap up) and must not regress
  `docs/agent-evals.md`. The manifest half is the easy one to forget — the PR
  that breaks routing usually lands in another repo.

## Conventions

Mirrors PCC byte-for-byte where reasonable: Python 3.11+ / FastAPI /
SQLAlchemy 2.0 typed / Alembic for every schema change / soft deletes /
strict mypy / ruff (line 100). React + Vite + TS strict, Night-Silk tokens
(`frontend/src/styles/tokens.css` sourced from `../gateway/theme/silk.css` —
re-copy to fix drift, never edit). Docker: backend (no host port) + nginx
frontend on `8300`.

## Rules

- **Conductor is the delegation root** (depth-1): it is the only client of
  other apps' delegate APIs, and it never grows an `agent:` block of its own.
- **Vendored Glitch** (`app/ai/personality-global.md`) is a verbatim copy of
  the canonical `../agent-standard/` file plus one header line — never edit in
  place; re-copy and run `check-sync.sh`. Prompt composition: conductor base →
  Glitch → fleet section → date (no app flavor).
- **Routing hints live in the system prompt** (`render_fleet_section`), not
  tool docstrings. If routing degrades, tighten manifest `examples` first — a
  bigger model is the last resort.
- **Destructive requests to `ask_` apps get confirmed first** — operational
  voice with a worked example (descriptive safety rules don't bind a 12B).
  `open_` apps confirm their own resets; "resign the game" is a plain handoff.
- **Delegate calls are budgeted** (3 per app per turn) and every one emits a
  `delegate_call` structlog event; replies relay as text plus a compact
  activity note — never raw transcripts into conductor's history, and history
  replays text turns only.
- **`max_iterations` is 6** — each iteration can wrap a full subagent loop, so
  depth stacks.
- **The fleet action proxy** (`/api/fleet/{app}/actions/{path}`) forwards only
  to declared `agent.actions` prefixes: no path escapes, no forwarded browser
  headers, short timeouts (5s/30s), the app's status passes through. It is the
  one model-free path conductor offers a page.
- **Note turns** (`POST …/notes`) record what the person did in the UI as the
  person's own turn — never an invented assistant reply.
- **Single-worker constraint:** the in-memory turn-activity registry and rate
  limiter pin the deployment to one uvicorn worker until the SSE item ships.
