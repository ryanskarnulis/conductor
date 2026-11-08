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
  loop, the model, **or any fleet app's `agent:`/`open:`/`glance:` manifest block**
  runs the routing evals (`CONDUCTOR_AGENT_EVALS=1 pytest
  tests/test_agent_evals.py -v -s`, llama-swap up) and must not regress
  `docs/agent-evals.md`. The manifest half is the easy one to forget — the PR
  that breaks routing usually lands in another repo.

## Conventions

Mirrors PCC byte-for-byte where reasonable: Python 3.11+ / FastAPI /
SQLAlchemy 2.0 typed / Alembic for every schema change / soft deletes /
strict mypy / ruff (line 100). React + Vite + TS strict. Docker: backend (no
host port) + nginx frontend on `8300`.

**The look grows from Night-Silk** (#85) into conductor's design system
([artifact](https://claude.ai/artifact/QWR5kpuitBY2rJYz1eFxN7), #86):
`styles/silk.css` is a verbatim copy of `../gateway/theme/silk.css` (re-copy
to fix drift, never edit); `tokens.css` maps conductor's semantic names onto
it; `conductor.css` holds the design system's additions (motion, scene,
spacing, radii), generated from its `tokens.json` — change a value there and
regenerate. Components live in `src/design/` (`cd-` classes in
`styles/components.css`), the shell in `features/agent/`.

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
  undo, a plan's unrun steps and why, quoted outside text, two sentences on
  voice); handoffs and an app's confirm question pass through verbatim.
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
  interpreter (`app/ai/interpreter.py`) runs `ask`/`look`/`open`/`ask_user`/
  `confirm` plans across apps, a last `think`, the control steps, and the
  direct answers (`app/ai/speaker.py`).
  Plans have no designed length; 12 steps is a runaway ceiling. Every other
  kind falls back to the loop until its issue lands; the fallen-back plan
  stays on the ledger. A destructive
  ask to an app without `confirms` pauses on code's question; `ask_user` and
  an app's `confirm_required` pause too, and the next turn resumes the same
  plan. The planner prompt and the ten step kinds stay byte-stable: plan-level
  evals (`tests/test_plan_evals.py`, `CONDUCTOR_PLAN_EVALS=1`) gate any change
  to them, to the compiler, or to a manifest.
- **`look(app)` reads a glance, never asks an agent** (#73). An app's
  `glance:` block (`../agent-standard/app-yaml-glance-block.md`) puts its
  `holds` line in the planner prompt and `look` in the schema for that app
  only; no block, no look (the validator sends a stray one back for an `ask`).
  The interpreter makes one plain GET (`app/fleet/glance.py`: declared path,
  no forwarded headers, 2s/5s timeouts, no redirects, no model, no delegate
  budget) and files `look_<app>` with a `looked_up` receipt built by code
  (`app/ai/glance.py`; outside titles cleaned and marked) and no writes. It is
  not a registry tool, so the loop's schemas don't change. A glance that can't
  be read asks the same app the person's words, but only for a plain question,
  a plan whose words are the request, an app with an agent and no ask to it
  already in the plan; otherwise the step fails and the plan ends. The failed
  read stays on the ledger and out of the narrator's story. A look gives
  nothing to `{sN}`. Adjacent looks run in parallel with other reads.
  Session evals read a world-backed glance (`FakeGlances`/`World.glance`, set in
  the harness): no eval ever reaches a real app, and a look counts as a read.
  The speaker's `chat` answer sees the ledger's facts and the about-you check,
  since the planner files "what did you just do?" there when glance lines shift.
- **Data flows as quoted data, never as control** (#72, `app/ai/dataflow.py`).
  Code fills `{sN}` from step N's receipt: its main touched title (job refs
  skipped, up to 3 joined), cleaned and quoted. No title, or more than 3,
  means the step doesn't run and the plan ends, with the reason on the
  record. Never guess a name or splice in an app's reply text. The validator
  repairs a lost ref: a bare `s1`, a step re-describing what an earlier ask
  will find, or an ask pointing at State ("their other album"). The plan is
  fixed before any step runs, so a title can't steer control flow. An
  `untrusted` title goes only into a non-destructive ask to an app with
  `confirms`, never into a destructive ask, an app without `confirms`, or an
  `open` intent. Only trusted values survive a pause. Adjacent reads to
  different apps (`plan.reads_only`) run at once in threads with the turn's
  context. Everything else runs in order, one call in flight. The budget and
  thread store are locked, and Stop cancels every call in flight.
- **Control steps act over the ledger** (#74, `app/ai/control.py`), each as
  a plan of its own. `cancel` ends the waiting plan (its row says
  `cancelled`), lets an app's token lapse unsent, and says what won't
  happen. `amend` puts the change after the waiting step's words and resumes
  the same plan; a gate asks again, and an app asks its own question again.
  `undo` takes the last plan's write receipts, or older ones only when the
  target names them, and asks each owning app to redeem its handles through
  the usual confirmation (the app's, else code's gate). `undo_none`,
  `unknown` and unreachable receipts are never tried: they ride the record as
  steps not run, so the narrator says them or code appends them. A receipt is
  undone once (`undone_by_step_id`); "undo that" twice finds the same plan,
  and an undo's inverse is never offered back. An undo streams like any
  plan: its notes and asks are the plan's steps on the live channel.
  `cancel`/`amend` aimed at a running job (one runs and nothing waits, or
  the target names it) are the jobs runtime's (`app/ai/job_control.py`).
- **Jobs** (#76): a step whose receipt carries a `job` detaches: the reply
  says it's queued, and the ledger's `ledger_jobs` row tracks it. The watcher
  (`app/services/job_watcher.py`, one thread, started with the app, resumes
  from the rows) polls `…/jobs/{ref}` (`app/fleet/jobs.py`: no model, no
  forwarded headers, short timeouts) every 5 s, 30 s after 2 minutes. When a
  job ends it posts **one** note to the conversation that started it,
  narrated from the app's final report with code checks on done/total and the
  failed items' names (`narrator.job_note`); `noted_at`, the message and the
  job's receipts land in one commit, so restarts and races never post twice.
  No end after 6 hours, or a 404, is code's "lost track of it": unknown,
  never failed. A note isn't an exchange: it never lapses a waiting question.
  `cancel` calls the app's `…/cancel` and says the `done` count it answered;
  `amend` can't change a running job and says so (the contract has none).
- **`think` is a background job on the local 26B** (#78, `app/ai/think.py`,
  `app/services/thinker.py`; `CONDUCTOR_THINK` off falls back to the loop).
  Always the last step: it files a `ledger_jobs` row conductor owns
  (`app = "conductor"`) with what its inputs found, and code replies at once;
  the conversation never waits. The watcher hands it to the thinker:
  gemma-4-26b-a4b, thinking on, `json_schema` on the answer, its own prompt
  and sampling. The answer is a **proposal**, validated like any plan (one
  repair) and tainted: it posts as the job's one note (`think_proposal`, the
  confirm card) and nothing in it runs without the person's yes. Yes runs it
  as a normal plan (gates and apps' questions still ask); no, an amend or
  anything else sets it aside. **It yields the GPU**: no try starts during a
  turn, and one in flight is cut off when a turn starts
  (`app/ai/foreground.py`). The call streams so the cut-off closes it between
  chunks: llama-swap finishes an in-flight request before it swaps, so
  without that a turn waits out the whole thought (~2 min measured). Another
  app's model call still waits behind a think (no cross-app priority yet). A
  cut-off or failed call is `interrupted` (a `not_run` receipt)
  and retried after 60 s with no turn, 3 tries, then code's honest "couldn't
  finish". Never claim a proposal that wasn't made.
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
- **Single-worker by design:** conductor runs SQLite on one uvicorn worker,
  and the in-memory turn-activity registry (the live channel), the rate
  limiter, the job watcher's one thread and the thinker's one try at a time
  (with its turn tracker and cancel flags) rely on it; multi-worker safety is
  not a goal.
- **The live channel** (#38, `app/api/turn_activity.py`): `GET …/events` is
  SSE for the turn in flight: a snapshot on connect (so a reconnect loses
  nothing), then beats, `plan`, `step_started`, `receipt`, `paused`,
  `resumed` (a turn picking up a paused plan: why, the step, an `ask_user`
  answer as `{s1}` reads it) and `turn_ended`, plus `job` events (progress,
  then the end with its note's id) that keep the stream open past
  `turn_ended` while a job runs. Step events go through one seam
  (`app/ai/turn_events.py`); a receipt is the step as the ledger journals it (`ledger.live_step`) and
  lands only once its record exists, and `turn_ended` goes out after the
  ledger commit. Queues and turn state are bounded; a lagging follower is
  resynced from a snapshot. `GET …/activity` stays as the poll fallback.
- **The delegate contract never assumes a text-only client** — voice
  convergence is a master-plan placeholder.
- **The panel is the accessible truth; the 3D web is decoration** (#88). The
  Living Web (`src/scene/livingWeb.js`, three.js) is `aria-hidden`, loads as
  a lazy chunk after first paint, and shows nothing the panel doesn't say in
  words. Both read one derivation (`turnView.ts` → `sceneState.ts`), and
  nothing looks done before its receipt says so.
- **The live plan view** (#89) draws from the same stream: the plan appears
  queued beside the code-built ack, each step lights on its own event, a
  `{sN}` stays a dashed ref until its receipt names a title (never filled
  from an outside title early), and a step that didn't run is `not run`,
  never `failed`. A pause shows as a card in the plan; the turn that ended on
  one keeps its stream (`held`) until the next turn, so the answer visibly
  resumes the same plan. A strand lights only on a real step event.
