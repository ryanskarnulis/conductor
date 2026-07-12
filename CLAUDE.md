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

**Phase 3 is complete (PRs #1–#6, routing evals GO 12/12); only the hub-app
"Later" item remains open — see `TODO.md`.**
On top of the Slice 2 engine (tool registry, llama.cpp provider, bounded loop,
layered Glitch personality — adapted from the PCC reference implementation,
`../project-command-center/backend/app/`) and Slice 3's `fleet/` package
(manifest discovery, the typed delegate REST client, the per-app `ask_<app>`
tools with guardrails, the MCP stdio server), Slice 4 added persistence and
the full chat front: SQLite via SQLAlchemy 2.0 + Alembic (`app/db/`), the
conversations service and REST API fronting conductor's own loop
(`app/services/`, `app/api/routes_agent.py`), the DB-backed subagent-thread
store (`app/fleet/thread_store.py`), per-IP rate limiting, request-ID logging
middleware, the turn-activity poll endpoint (`app/api/turn_activity.py`), and
the web UI — a PCC-style chat panel (`frontend/src/features/agent/`) with a
conversation sidebar, markdown replies with the delegate-call trajectory, and
a live "Asking chess… · 12s" progress line polled from the activity endpoint
while a synchronous run blocks (no SSE in v1). The routing eval harness
(`backend/tests/test_agent_evals.py`, baseline in `docs/agent-evals.md`)
closed Phase 3's go/no-go gate: gemma-4-12b routes 12/12 goldens. **The
baseline gates every prompt / manifest-examples / model / loop change** — run
the evals before merging one (`CONDUCTOR_AGENT_EVALS=1 pytest
tests/test_agent_evals.py -v -s` from `backend/`; needs llama-swap up).

See `../agent-standard/STANDARD.md` for the contract every agent (once this
one has one) must satisfy, and `../agent-standard/delegate-api.md` for the
REST shape conductor will speak to PCC, chess, and future app agents.

## Commands

```bash
./main.sh                 # bootstrap env/deps, migrate, start backend + frontend
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
- SQLAlchemy 2.0 typed syntax (`Mapped[...]`/`mapped_column`); **Alembic for
  every schema change** (`alembic revision --autogenerate` from `backend/`,
  review the file before applying). Soft deletes for user-facing rows via the
  service-layer helper.
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
├── api/
│   ├── routes_agent.py        # conversations CRUD, the message → loop round
│   │                          #   trip, and the turn-activity poll endpoint
│   ├── turn_activity.py       # in-memory registry of in-flight turns — the
│   │                          #   UI's progress poll target (single worker)
│   ├── rate_limit.py          # per-IP sliding-window limiter (PCC copy)
│   └── request_ip.py          # spoof-resistant client key (trimmed from PCC)
├── db/
│   ├── models.py              # Conversation, ConversationMessage (loop outcome
│   │                          #   denormalized), DelegateThread (thread map)
│   └── session.py             # engine + SQLite pragmas + get_db (PCC copy)
├── services/
│   ├── common.py              # active()/soft_delete() (trimmed from PCC)
│   └── conversations.py       # the only conversations write path (PCC minus
│   │                          #   activity_events — audit is the delegate_call
│   │                          #   structlog event)
├── schemas/
│   ├── common.py              # NonBlankStr, UTCDateTime
│   └── conversations.py       # wire models, incl. TurnActivityRead
├── alembic/                   # migrations (alembic.ini sits in backend/)
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
│   ├── loop.py                # AgentLoop (bounded), build_system_prompt (now
│   │                          #   takes the fleet layer), conductor base
│   │                          #   prompt, loop_from_settings.
│   └── personality-global.md  # vendored Glitch (see re-vendor rule below)
├── fleet/                     # Slice 3 — conductor's job (see below)
│   ├── manifests.py           # discover_fleet: scan {fleet_manifest_dir}/*/
│   │                          #   app.yaml → Fleet of FleetApp/AgentSpec
│   ├── delegate.py            # DelegateClient (httpx) + typed errors + the
│   │                          #   wire models mirroring PCC/chess
│   ├── context.py             # DelegationContext: thread map + per-turn call
│   │                          #   budget + audit hook; ThreadStore seam
│   ├── thread_store.py        # DbThreadStore: the ThreadStore seam over the
│   │                          #   delegate_threads table (per-op sessions)
│   └── tools.py               # build_delegate_tools (ask_<app> + list_agents),
│                              #   render_fleet_section (the prompt layer)
├── mcp/server.py              # FastMCP stdio server: the registry (delegate
│                              #   tools + list_agents) over MCP; .mcp.json
├── logging_config.py          # configure_logging(stream) + RequestIDMiddleware
│                              #   (the MCP server points logging at stderr —
│                              #   stdout is the transport)
└── tools/
    ├── registry.py            # @tool decorator, call_tool(..., actor=…). Ships
    │                          #   empty; build_delegate_tools populates it at
    │                          #   startup — the one surface loop + MCP consume.
    └── runtime.py             # the actor contextvar only (PCC's tool_session /
                               #   DB plumbing is adapted out — tool bodies never
                               #   touch conductor's DB; writes are attributed
                               #   downstream via the X-Agent-Actor header)
```

Intentional deltas from PCC: provider protocol/types split into `provider.py`;
no `chat_structured` (conductor has no structured-output need); no `runtime`
DB session (tool bodies make HTTP calls, not local writes); no `resolve_actor`
(conductor is the delegation root — it never receives an inbound
`X-Agent-Actor`, so the conversations API takes no actor header);
`max_iterations` defaults to 6, not 10; no `activity_events` table (the
delegation audit is the `delegate_call` structlog event); the loop grew an
`on_activity` seam PCC doesn't have (progress beats for the activity poll).

## Fleet delegation (`fleet/`)

Conductor's actual job. Discovery is declarative: `discover_fleet` scans
`{settings.fleet_manifest_dir}/*/app.yaml` (dev default: the workspace root,
computed from the package location; docker sets `FLEET_MANIFEST_DIR=/fleet`, a
read-only mount of `..`, with `FLEET_UPSTREAM_HOST=host.docker.internal` to
rewrite each manifest's upstream host — port preserved). Every well-formed app
becomes a `FleetApp`; the ones with an `agent:` block get an `AgentSpec`.
Conductor's own manifest is skipped (depth-1: it is never its own delegate);
malformed/incomplete manifests are skipped with a `structlog` warning, never a
crash; a malformed `agent:` block degrades an app to a non-agent member.

`build_delegate_tools(fleet, client_factory)` registers one `ask_<app>` tool
per agent-bearing app (`ask_tasks`, `ask_chess`, …) — docstring from the
manifest's `agent.description` — plus a local `list_agents`. An `ask_<app>`
call, over `DelegateClient` (which always sends `X-Agent-Actor: agent:conductor`
and uses a 300s read / 5s connect timeout for cold model loads):

- resolves or creates the app's subagent thread for the current master
  conversation, and on a `404` (pruned thread) recreates it and retries
  **exactly once** (a second 404 fails the call);
- maps every typed delegate fault (`DelegateThreadGone` / `DelegateRateLimited`
  with `Retry-After` / `DelegateUnavailable` / `DelegateRequestRejected`, a 4xx
  other than 404/429 — the request itself is invalid, never retried as-is /
  `DelegateProtocolError`) to an informative `ToolError` the model reads and
  adapts to — 429 is surfaced, never auto-retried;
- relays the assistant reply plus a compact `[app did: …]` activity note —
  never raw transcripts into conductor's history.

Routing hints live in the **system prompt**, not tool docstrings:
`render_fleet_section(fleet)` builds a dynamic layer (each agent's name /
description / examples) that `build_system_prompt` slots in as
`conductor base → Glitch → fleet → date`.

**Guardrails + seams (`context.py`).** A `DelegationContext` is bound around
each run (a contextvar for the HTTP loop — `routes_agent.post_message` binds
one per request; a process-global fallback for the MCP server). It holds: (a)
the **thread map** via a `ThreadStore` keyed by
`(master_conversation_id, app_name)` — the HTTP loop uses the DB-backed
`DbThreadStore` (`fleet/thread_store.py`, `delegate_threads` table) so
subagent threads survive restarts, while the MCP server keeps a
process-lifetime `InMemoryThreadStore` (its driving host manages its own
session); (b) a **per-turn per-app call budget**
(`conductor_delegate_calls_per_turn_per_app`, default 3; `<= 0` disables it, as
the MCP server does since its driver is the trusted MCP host) — exceeding it
raises `ToolError` so the model stops hammering one app; (c) an **audit hook** —
every delegate call emits one `delegate_call` structlog event (app, subagent
thread id, latency_ms, and stop_reason or error class), tagged with the driver
(`agent:loop` for the loop, `agent:mcp` for the MCP server).

The **MCP server** (`app/mcp/server.py`, wired by `.mcp.json`) exposes the same
registry over stdio; it discovers the fleet and registers the tools in `main()`
*after* pointing logging at stderr — the import stays silent because stdout is
the JSON-RPC transport.

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

**Delegation-root rule.** Conductor is the delegation root and nothing else:
per the depth-1 rule (`../agent-standard/STANDARD.md`) it is the only client
of other apps' delegate APIs, and its own delegate surface (if it ever gets
one) must never be exposed to another agent — which is why `app.yaml`
deliberately ships no `agent:` block. Like every fleet app it also stays on
`../agent-standard/NEW-APP-CHECKLIST.md`.

## Conversations API (`api/`, `services/`, `db/`)

The HTTP front for conductor's own loop, PCC's conversations shape with the
delegation seams wired in. All under `/api/agent`:

- `GET|POST /conversations`, `GET|DELETE /conversations/{id}` — CRUD; soft
  delete only; an untitled conversation is titled from its first user message.
- `POST /conversations/{id}/messages` — the one model-calling endpoint,
  rate-limited per client IP (`AGENT_MESSAGES_PER_MIN`). Synchronous and
  non-streaming (v1): it commits the user turn *first* (a provider failure →
  502 must not swallow it), binds the `DelegationContext` (DB thread store,
  call budget, `agent:loop` driver) around exactly the run, then persists the
  assistant turn with its tool trajectory (`tool_calls` JSON) and stop reason.
  History replays **text turns only** — stale delegate transcripts never
  re-enter the model's window.
- `GET /conversations/{id}/activity` — the progress poll target while a POST
  blocks: the loop's `on_activity` beats land in the in-memory
  `turn_activity` registry (`kind: model|tool`, tool name, iteration, elapsed
  seconds), which the UI renders as "asking chess…". This is v1's no-SSE
  progress channel; it assumes the single-worker uvicorn the Dockerfile runs.

Persistence is SQLite (`data/conductor.db` in dev, `/data` volume in docker)
via SQLAlchemy 2.0; `./main.sh` and the docker CMD run `alembic upgrade head`
before serving. The `delegate_threads` table backs the fleet thread map — the
"Guardrails + seams" section above covers how the stores split between the
HTTP loop and the MCP server.

## Task tracking

`TODO.md` is the living backlog — the next sprint plus the deferred and
blocked items. One task = one branch = one PR (see Git workflow above); when
an item ships, record the outcome on its line (bold note with the date and PR
number) rather than deleting it silently.
