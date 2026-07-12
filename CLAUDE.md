# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What This Is

Conductor is the workspace fleet's master agent: a new sibling app (like
`project-command-center` and `chess`) whose job is to route a user's request
to the right per-app agent over a standardized REST delegate contract, and
hold the conversation. It is Phase 3 of
`../agent-standard/AGENTS-MASTER-PLAN.md` (Phases 0–2 — the standard itself,
PCC alignment, and the chess migration — are done; PCC is the reference
implementation every app, including this one, follows).

**This repo is currently Slice 1 of Phase 3: scaffold only.** Backend and
frontend exist, CI/CD is wired, and the gateway manifest is in place, but
there is no agent logic yet — no tool registry, no provider, no loop, no
delegate calls to other apps. `/health` and a themed placeholder page are all
that's live. Subsequent slices add subagent discovery from `app.yaml agent:`
blocks, the standard agent stack, and the chat UI, per the master plan.

See `../agent-standard/STANDARD.md` for the contract every agent (once this
one has one) must satisfy, and `../agent-standard/delegate-api.md` for the
REST shape conductor will speak to PCC, chess, and future app agents.

## Commands

```bash
./main.sh                 # bootstrap env/deps, start backend + frontend
./test.sh                 # backend pytest/ruff/mypy + frontend Vitest/lint/build
```

## Ports

Conductor owns the `8300`–`8399` block in the workspace gateway's port
registry (`../gateway/README.md`):

| Service | Port | Notes |
|---|---|---|
| frontend (docker, published) | `8300` | nginx serving the built SPA, proxies `/api` to the backend |
| backend (dev) | `8301` | `python -m app.main`; no host port published in docker |
| frontend (vite dev server) | `5174` | `5173` is PCC's dev server |

All ports bind `127.0.0.1` — the workspace gateway (Caddy) is the LAN front
door, same as every other app.

## Git workflow

- **Never commit or push directly to `main`.** For every change: create a
  branch (`feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`), commit,
  push, and open a PR with `gh pr create`.
- PRs are **squash-merged once CI is green**: after opening a PR run
  `gh pr checks --watch` and, when green, `gh pr merge --squash`. Never merge
  with failing or pending checks.
- CI (`.github/workflows/ci.yml`) mirrors `./test.sh`: backend pytest, ruff
  (check + format), and mypy; frontend Vitest, lint, and build. Run
  `./test.sh` before pushing.
- Deploys via the `~/deploy/conductor` clean-clone pattern
  (`.github/workflows/deploy.yml`) — same as PCC and chess. If `app.yaml`
  changes, the deploy workflow regenerates the gateway too.

## Conventions

This app mirrors `project-command-center`'s conventions byte-for-byte where
reasonable — it's the fleet's reference implementation:

- Python 3.11+, FastAPI, pydantic-settings `Settings` (env-overridable),
  ruff (line-length 100), strict mypy, pytest.
- React + Vite + TypeScript (strict), plain global CSS with Night-Silk design
  tokens (`frontend/src/styles/tokens.css`, sourced from
  `../gateway/theme/silk.css` — the canonical copy; re-copy `silk.css` to fix
  drift, never edit it in place), ESLint flat config, Vitest + jsdom +
  Testing Library.
- Docker compose: backend (uvicorn, no published host port) + frontend
  (nginx, SPA + `/api` proxy, published on `127.0.0.1:8300`).

## Later

Once conductor grows an agent (Phase 3 slices beyond scaffold), it must also
follow `../agent-standard/NEW-APP-CHECKLIST.md` and the depth-1 delegation
rule in `../agent-standard/STANDARD.md` — conductor is the delegation root,
so it is the only client of other apps' delegate APIs, and its own delegate
surface (if it ever gets one) must never be exposed to another agent.
