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
"Later" item remains open — see `TODO.md`. Voice shipped 2026-07-12 (#8–#10,
`../agent-standard/VOICE-PLAN.md` Phase 4): `/api/voice` STT/TTS proxies to
the shared `../speech/` service, vendored chess-canonical frontend modules
(push-to-talk + hands-free in the chat panel), and the spoken "Asking
chess…" slow-turn acknowledgment on voice-initiated turns
(`frontend/src/features/agent/voiceAck.ts`) — voice fronts the same
`/api/agent` pipeline, so the eval baseline was untouched (12/12 re-run).**
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
│   ├── routes_voice.py        # STT/TTS proxies to the shared ../speech/
│   │                          #   service (PCC copy; fleet voice standard)
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
│   ├── speech.py              # SpeechClient (PCC copy modulo the STT prompt,
│   │                          #   biased to fleet routing vocabulary) — see
│   │                          #   ../agent-standard/voice.md and README
│   └── personality-global.md  # vendored Glitch (see re-vendor rule below)
├── fleet/                     # Slice 3 — conductor's job (see below)
│   ├── manifests.py           # discover_fleet: scan {fleet_manifest_dir}/*/
│   │                          #   app.yaml → Fleet of FleetApp/AgentSpec/
│   │                          #   OpenSpec (agent: and open: are independent)
│   ├── delegate.py            # DelegateClient (httpx) + typed errors + the
│   │                          #   wire models mirroring PCC/chess
│   ├── context.py             # DelegationContext: thread map + per-turn call
│   │                          #   budget + audit hook; ThreadStore seam
│   ├── thread_store.py        # DbThreadStore: the ThreadStore seam over the
│   │                          #   delegate_threads table (per-op sessions)
│   └── tools.py               # build_delegate_tools (ask_<app> + open_<app> +
│                              #   list_agents), render_fleet_section (prompt)
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
becomes a `FleetApp`; the ones with an `agent:` block get an `AgentSpec`, the
ones with an `open:` block an `OpenSpec` (independent — an app may have either,
both, or neither). Conductor's own manifest is skipped (depth-1: it is never
its own delegate); malformed/incomplete manifests are skipped with a `structlog`
warning, never a crash; a malformed block degrades that one capability (a bad
`agent:` costs the app its ask tool, not its open tool), and an app with neither
block is *inert*: a fleet member conductor can only name.

`build_delegate_tools(fleet, client_factory)` registers one `ask_<app>` tool
per agent-bearing app (`ask_tasks`, …) — docstring from the manifest's
`agent.description` — one `open_<app>` per **openable** app (see below), plus a
local `list_agents`. An `ask_<app>` call, over `DelegateClient` (which always
sends `X-Agent-Actor: agent:conductor` and uses a 300s read / 5s connect
timeout for cold model loads):

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

**Handoff apps (`open:`).** Not every app should be delegated to. Chess is a
board you play *in* — voice-first, stateful, full-screen — so relaying moves
through conductor's chat panel is a strictly worse game than the app itself.
Such an app declares an `open:` block instead of (or alongside) `agent:`
(`../agent-standard/app-yaml-open-block.md`) and gets an `open_<app>` tool that
hands the **user** over: it returns a handoff payload (target app, path, and the
user's request as an `intent`), conductor's frontend reads that off the
persisted trajectory and navigates the tab to `<app>.<same host the page came
from>?intent=…`, and the app runs the intent through its own agent on arrival —
so nothing the user asked for is lost in the trip. The tool is purely local: no
network, no delegate thread, no call budget (nothing is *asked* of the app).
Chess ships `open:` only — its `/api/agent` endpoints still exist, it just
stops advertising them, which is what retired `ask_chess`. Note the
destructive-op confirmation rule is scoped to `ask_` apps: conductor isn't
acting on an open app, and chess confirms its own resets, so "resign the game"
is a plain handoff.

Routing hints live in the **system prompt**, not tool docstrings:
`render_fleet_section(fleet)` builds a dynamic layer (each app's tool /
description / examples, split into the ones conductor delegates to and the ones
it hands the user over to) that `build_system_prompt` slots in as
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

### Fleet action proxy (`api/routes_fleet.py`)

`GET|POST /api/fleet/{app}/actions/{path}` forwards one call to the prefix an
app declares as `agent.actions` (`../agent-standard/app-yaml-agent-block.md`)
— the **only** thing conductor does without a model in the path.

It exists for one shape of work: a person answering the same question dozens of
times. Music's sorting pass is ~150 questions, and a click that waits for a
local 12B to read a sentence and re-emit it as a tool call is a slow way of
typing, not a button (`../future-plans/music-agent.md`, Phase 2.6).

Proxying rather than letting the page dial the app is what keeps every app in
the fleet as headless as it was: the page stays same-origin with conductor, so
no app grows a CORS allow-list for another app's page or becomes writable by a
foreign origin. The boundary:

- **Only declared prefixes.** No `agent.actions`, no proxy — which is every app
  but music today. Unknown app and no-actions app answer the same 404; telling
  them apart only maps the fleet for whoever is asking.
- **No climbing out of it.** A `..` segment (percent-encoded is the form that
  survives a client's normalization) is a 400 before a socket opens.
- **The browser's headers are not forwarded** — a proxy that passes them on
  hands one origin's cookies and auth to another service.
- **Short timeouts** (5s connect / 30s read), not the delegate client's 300s:
  nothing here waits on a model, so a button that hangs a page for five minutes
  is a bug, not patience. Per-IP capped at `FLEET_ACTIONS_PER_MIN` (120 —
  higher than the agent surface, because one answer per click is the point).
- **The app's own status and body pass through untouched**: a 422 saying "that
  artist is already sorted" is the app's answer to the person.

### The sort panel (`frontend/src/features/sort/`)

The one piece of app-specific UI in conductor, and the client half of the proxy
above. It opens under the thread when a turn's trajectory shows music actually
ran `sort_music` (`sortTurn.ts` — read off `app_tools`, never grepped out of the
reply text: a panel that looks for a tool name inside a 12B's paraphrase breaks
the first time the model rephrases) and **stays** open for the rest of the pass,
keyed to the newest sorting turn anywhere in the thread rather than to the last
message. A pass outlives the turn that starts it: saying "one at a time" out
loud is a turn of its own and need not run the tool again, and keying to the
last message made the panel vanish mid-pass and stay gone. Dismissing leaves a
"Back to sorting" chip, so closing it is never a one-way door. It renders the
**live** worklist fetched from music, not a snapshot of what the reply said, and
re-reads it whenever a turn lands so a spoken answer moves the buttons. So an answer given by
voice, by `python -m app.sort` in a terminal, or by dragging a file in a file
manager moves the buttons too.

- Folder buttons file the whole group; `One at a time` opens it up and files
  only the ticked songs — an artist orders the questions, it never answers them.
  An opened group **stays open until it is empty**: going song by song means
  several answers about one artist, so filing part of a group returns to the
  rest of it. `All of them` collapses back, as a choice rather than something
  that happens to you mid-answer.
- Skipping is client-side: not answering is not an act and writes nothing.
- The panel never decides a genre. It offers the folders that exist and shows
  the `tags say:` hint exactly as the terminal pass does — the only signal the
  file carries, and half of them are junk.
- Every filing posts a note turn, so a click reads in the thread as the answer
  it was. Typing and speaking keep working while it is open; it is an input, not
  a mode.

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
- `POST /conversations/{id}/notes` — record something the person did in the UI
  (filing a song from the sort panel, which reaches the app through the fleet
  action proxy). **No model runs**: it appends the turn and commits. Stored as
  the person's own turn, because the person is who acted — inventing an
  assistant reply conductor never made is the one thing a truthful transcript
  cannot do. It replays into the next turn's context like any other user turn,
  which is the point: otherwise the model's next reply contradicts the panel.
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
