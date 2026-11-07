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
minus deltas". `docs/agent-evals.md` is the routing eval baseline.

## Commands

```bash
./main.sh                 # bootstrap env/deps, migrate, start backend + frontend
./test.sh                 # backend pytest/ruff/mypy + frontend Vitest/lint/build
```

Ports (gateway registry): frontend `127.0.0.1:8300` (nginx, proxies `/api`),
backend dev `8301`, vite dev `5174`.

- After bumping a backend dep: `cd backend && .venv/bin/python -m pip freeze
  --exclude-editable > requirements.lock`. CI, the Docker image and fresh
  venvs install against it (`-c requirements.lock`).

## Git workflow

- The backlog is GitHub issues (`gh issue list`). File follow-up work as an
  issue, never in a markdown file; PRs close issues with `Closes #N`.
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
- **Destructive requests: the app that owns the write asks** (#68). An `ask_`
  app with `agent.confirms: true` gets the request straight through; its
  `confirm_required` question goes to the person and the token stays on the
  turn's record (`app_confirm`, never sent to the model). Only a question from
  the last turn can take a yes (`confirm(app)`, which sends `Confirm-Token` on
  the asking thread); anything else lapses it. For an `ask_` app without the
  flag, conductor asks first — operational voice with a worked example
  (descriptive safety rules don't bind a 12B). `open_` apps confirm their own
  resets; "resign the game" is a plain handoff.
- **Delegate calls are budgeted** (3 per app per turn) and every one emits a
  `delegate_call` structlog event; replies relay as text plus a compact
  activity note — never raw transcripts into conductor's history.
- **No chat turns are replayed** (#64). Each turn's one user message opens
  with the state block (`app/ai/state_block.py`: calendar, surface, panel,
  running jobs, a pending question, recent entities from the apps' records)
  and recall (`app/ai/recall.py`: this thread's record, requests and last
  exchange as data, plus word-matched calls from earlier conversations), then
  the person's words. System prompt and tool schemas stay byte-identical; the
  opening is capped (`turn_context.MAX_CHARS`).
- **Delegate v1.1** (#67): each `ask` mints one `Idempotency-Key`; only a
  connection dropped after send is resent, under that key. The app's typed
  record is stored per call (`app_record`, never sent to the model), and a 5xx
  whose record lists `done` work is reported as partly happened, by name
  (untrusted titles by kind and id). No record means unknown: never retry blind.
- **The planner decides, the narrator speaks** (#65). A completed run's reply
  is a tool-less call on a typed receipt of what ran (`app/ai/narrator.py`);
  the persona lives only there. Code guarantees the facts (titles, counts,
  undo, quoted outside text, two sentences on voice); handoffs and an app's
  confirm question pass through verbatim.
- **The ledger** (#69, `app/services/ledger.py`) is the record of what
  happened: a turn opens with the person's words before the loop runs and
  closes with its plan, one step per call and the apps' receipts. Every app
  write has a receipt row; an app call with no readable record gets
  conductor's `unknown` receipt, never "nothing changed". Journaling runs in
  savepoints and never costs the turn. The state block's recent entities
  come from it.
- **Compiled plans** (#70, `CONDUCTOR_COMPILER`): the 12B compiles each turn
  to a typed plan under `json_schema` (`app/ai/plan.py`, `compiler.py`), code
  validates it (one repair, else the loop takes the turn), and the
  interpreter (`app/ai/interpreter.py`) runs single-app `ask`/`open`/
  `ask_user`/`confirm` plans and the direct answers (`app/ai/speaker.py`).
  Plans have no designed length; 12 steps is a runaway ceiling. Every other
  kind, cross-app plans and `{sN}` data flow fall back to the loop until
  their issue lands; the fallen-back plan stays on the ledger. A destructive
  ask to an app without `confirms` pauses on code's question; `ask_user` and
  an app's `confirm_required` pause too, and the next turn resumes the same
  plan. The planner prompt and the ten step kinds stay byte-stable: plan-level
  evals (`tests/test_plan_evals.py`, `CONDUCTOR_PLAN_EVALS=1`) gate any change
  to them, to the compiler, or to a manifest.
- **`max_iterations` is 6** — each iteration can wrap a full subagent loop, so
  depth stacks.
- **The loop keeps STANDARD.md §3's safety nets**: `max_tokens` on every
  call, the stall rule, a 12-call per-run cap, and the full stop vocabulary. A
  stop short of `completed` replies from the code-kept run record, and a
  failed run is stored before the route answers 502/500, so a write never
  vanishes from the thread.
- **Stop** (#66): `POST …/cancel` ends the turn `cancelled` at its next
  provider call or dispatch, replying from the record (never narrated); a
  delegate call in flight gets the app's `…/cancel`, best-effort.
- **The fleet action proxy** (`/api/fleet/{app}/actions/{path}`) forwards only
  to declared `agent.actions` prefixes: no path escapes, no forwarded browser
  headers, short timeouts (5s/30s), the app's status passes through. It is the
  one model-free path conductor offers a page.
- **Note turns** (`POST …/notes`) record what the person did in the UI as the
  person's own turn — never an invented assistant reply.
- **Single-worker constraint:** the in-memory turn-activity registry and rate
  limiter pin the deployment to one uvicorn worker until #38 (SSE) ships.
- **The delegate contract never assumes a text-only client** — voice
  convergence is a master-plan placeholder.
