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

**This repo is currently Slice 2 of Phase 3: the standard agent stack.** The
engine is in place — tool registry, llama.cpp provider, bounded loop, and the
layered Glitch personality (adapted from the PCC reference implementation,
`../project-command-center/backend/app/`) — plus exhaustive tests. Not yet
built (Slices 3–4): the per-app delegate tools (the registry ships **empty**),
the MCP stdio server, and the REST conversations API that fronts the loop.
`/health` and a themed placeholder page are still all that's exposed over
HTTP. Subsequent slices add subagent discovery from `app.yaml agent:` blocks,
wire the delegate tools into the registry, and add the chat UI, per the master
plan.

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

## Agent stack

The standard four layers (`../agent-standard/STANDARD.md`), adapted from PCC so
the modules diff cleanly against the reference implementation — read a change
as "same as PCC, minus the deltas below":

```
backend/app/
├── ai/
│   ├── provider.py            # ChatProvider Protocol + shared wire-neutral
│   │                          #   types (ToolSpec/ToolCall/ChatResult, typed
│   │                          #   provider errors). In PCC these live inside
│   │                          #   loop.py + llamacpp.py; split out here so
│   │                          #   nothing above the seam imports a backend.
│   ├── providers/llamacpp.py  # LlamaCppProvider: OpenAI wire format over
│   │                          #   plain httpx, Pydantic-validated at the
│   │                          #   boundary. reasoning_content dropped; thinking
│   │                          #   OFF by default (routing turns stay fast).
│   ├── loop.py                # AgentLoop (bounded), build_system_prompt, the
│   │                          #   conductor base prompt, loop_from_settings.
│   └── personality-global.md  # vendored Glitch (see re-vendor rule below)
└── tools/
    ├── registry.py            # @tool decorator, call_tool(..., actor=…). Ships
    │                          #   EMPTY — delegate tools arrive in Slice 3.
    └── runtime.py             # the actor contextvar only (PCC's tool_session /
                               #   DB plumbing is adapted out — conductor has no
                               #   local DB; writes are attributed downstream
                               #   via the X-Agent-Actor header)
```

Intentional deltas from PCC: provider protocol/types split into `provider.py`;
no `chat_structured` (conductor has no structured-output need); no `runtime`
DB session; no `resolve_actor` (conductor is the delegation root — it never
receives an inbound `X-Agent-Actor`); `max_iterations` defaults to 6, not 10.

**Re-vendor rule.** `ai/personality-global.md` is a **verbatim** copy of
`../agent-standard/personality-global.md` with exactly one added first line
(`<!-- vendored … re-copy to fix drift -->`). It is the canonical Glitch text
and must never be edited in place — fix any drift by re-copying the canonical
file and re-adding that header, then confirm with
`bash ../agent-standard/check-sync.sh`. `loop.py` strips the header at
prompt-build time. Prompt composition is `conductor base → global Glitch →
date` (the app-flavor layer is deliberately empty).

**Why `max_iterations = 6`** (`CONDUCTOR_MAX_ITERATIONS`, default in
`config.py`). Conductor's loop is deliberately shallower than the app loops'
~10: each conductor iteration may wrap a full subagent loop — a delegate call
fans out into that app's own bounded tool-calling loop — so latency stacks.
Keeping conductor's loop shallow bounds the worst-case depth-1 fan-out.

## Later

Beyond the agent stack (Phase 3 Slices 3–4 and Phase 4), conductor must also
follow `../agent-standard/NEW-APP-CHECKLIST.md` and the depth-1 delegation
rule in `../agent-standard/STANDARD.md` — conductor is the delegation root,
so it is the only client of other apps' delegate APIs, and its own delegate
surface (if it ever gets one) must never be exposed to another agent.
