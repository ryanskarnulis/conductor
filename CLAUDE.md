# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

Conductor is the workspace fleet's master agent: it routes a user's request to
the right per-app agent over the standardized REST delegate contract
(`../agent-standard/delegate-api.md`) and holds the conversation. Every turn
is planner → validator → interpreter → narrator (#75; no tool-calling loop).
A plan is typed steps, and an app is a step's parameter, never a tool of its
own: `ask` an app whose
`app.yaml` has an `agent:` block, `look` at one with a `glance:` block,
`open` one with an `open:` block (the user's words carried as `?intent=`);
see "Compiled plans are the only path" below. The backend mirrors
PCC's reference implementation (`../project-command-center/backend/app/`) —
read the code as "same as PCC, minus deltas". `docs/agent-evals.md` is the
eval baseline.

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
- **Eval gate** (#59, since #75): any change to the planner prompt, the step
  kinds, compiler/validator, interpreter, narrator, the model, **or any fleet
  app's `agent:`/`open:`/`glance:` manifest block** runs the plan-level evals
  (`CONDUCTOR_PLAN_EVALS=1 pytest tests/test_plan_evals.py -v -s`, n=20 per
  case, #142) and the session evals (`CONDUCTOR_AGENT_EVALS=1
  CONDUCTOR_FRONTIER_EVALS=1 pytest tests/test_session_evals.py -v -s`),
  llama-swap up, both under the Wilson gate, and must not regress
  `docs/agent-evals.md`, which records each plan group's first-try rate
  beside its score. The held-out set (`tests/held_out_cases.py`,
  `CONDUCTOR_HELDOUT_EVALS=1`) runs only at milestones and is never tuned
  against: its misses are reported, never fixed on it. The routing goldens
  retired at #75. The manifest half is the easy one to forget — the PR that
  breaks a plan usually lands in another repo.

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

- **One household timezone** (#130, `CONDUCTOR_TIMEZONE`, env only, default
  `America/Denver`): storage is UTC; every "today", calendar line and day a
  timestamp is labelled with goes through `app/clock.py`, never
  `date.today()` or a UTC `.date()`.
- **Conductor is the delegation root** (depth-1): it is the only client of
  other apps' delegate APIs, and it never grows an `agent:` block of its own.
- **Vendored Glitch** (`app/ai/personality-global.md`) is a verbatim copy of
  the canonical `../agent-standard/` file plus one header line — never edit in
  place; re-copy and run `check-sync.sh`. It is the voice (#256,
  `narrator.PERSONA`) of every reply the model writes: narrated replies, job
  and thread notes, the speaker's direct answers. The planner has no persona;
  code's own lines stay plain. `speaker.ABOUT_YOU` must agree with it.
- **Model knowledge lives in profiles** (agent-standard#32): one TOML per
  model in `app/ai/profiles/`, named for its llama-swap id: sampling, the
  thinking kwarg, each role's cap and temperature (`planner`, `narrator`,
  `speaker`), and `think`'s cap and thinking. The strict loader is vendored
  (`app/ai/agentkit/model_profile.py`: re-copy and run `check-sync.sh`, never
  edit); `app/ai/model_profile.py` holds conductor's phases. A model with no
  file runs on the server's sampling, with a `model_profile_missing` warning.
  `CONDUCTOR_PLANNER_TEMPERATURE` overrides the planner's for an A/B. A
  profile edit moves request bytes (`tests/test_request_bytes.py`): eval gate.
- **The provider's response half is vendored** (agent-standard#32,
  `app/ai/agentkit/chat.py`: re-copy and run `check-sync.sh`, never edit):
  the typed errors, `ChatResult`/`Usage`/`ServerStamp` and the body and
  stream readers, re-exported by `app/ai/provider.py`. The request side
  (payload, roles, capture, logging) is conductor's, in `providers/llamacpp.py`.
- **Routing hints live in the planner prompt** (`compiler.planner_prompt`,
  from manifest descriptions and `examples`), not tool docstrings. If plans
  degrade, tighten manifest `examples` first — a bigger model is the last
  resort.
- **Destructive requests: the app that owns the write asks** (#68). An `ask_`
  app with `agent.confirms: true` gets the request straight through; its
  `confirm_required` question goes to the person and the token stays on the
  turn's record (`app_confirm`, never sent to the model). Several gated calls
  are several questions (`confirms_required` → `app_confirms`, one token
  each). For an `ask_` app without the flag, code's gate asks first (a
  `destructive` ask pauses on it). `open_` apps confirm their own resets;
  "resign the game" is a plain handoff.
- **Questions never block the chat** (#251, `app/services/cards.py`). Every
  open question is a card: an app's confirm, code's gate, an undo's
  question, a think's proposal, a pick-one (`GET …/cards`, the page's
  "Waiting on you", each with its age). **Only a tap approves**
  (`POST …/cards`, no model): there is no `confirm` step (the schema
  doesn't offer it and the validator refuses it; the prompt's bytes still
  describe it, since rewording them cost the declines), and a bare "yes"
  with a card waiting gets code's "tap the card" (`plan.bare_yes`), never a
  plan. A tap sends that card's token (the app re-checks: a thing changed or
  finished since is said in its words, nothing runs) or picks the waiting
  plan up; the answer goes on the asking turn's record (`answers` names the
  token). One request's cards are a group with "Approve all". Other turns
  leave cards alone; a card lapses with the window (quiet gap) or its
  token. A new question about the same entity and action replaces the old
  card. **Words reach only the question asked in the turn right before**,
  read by code, never the planner: an option's name picks it
  (`options.answers`), "never mind" dismisses that group
  (`plan.dismisses`), a correction amends it in their words (#77,
  `plan.changes_what_waits`) and the question is asked again on a new card. Anything else is a new request and the card
  stays; the planner's own `cancel` never closes a card (one repair,
  `plan.cancels_a_card`) and reaches a running job only by name. An open question
  with nothing to tap (an `ask_user` with no options) waits on words for one
  turn, as before (`resume`). **Only dependents wait**: steps that use the
  asking step's `{sN}`, or would stop for a question of their own, wait with
  the card (`plan.waiting_on`, the paused body's `waiting`); the rest of the
  plan runs at once. **A tap resumes them with no chat turn**
  (`Interpreter.resume`): a turn of its own, the tap recorded as the
  person's note, streamed and narrated like any turn; an undo resumes the
  receipts it chose when it asked, never "the last plan" as it is by then.
- **Asks to one app have no cap of their own** (#218): the plan says what
  will run, and its 12-step ceiling is the runaway stop. A part of an ask
  that is another app's manifest example ("sort my music" asked of tasks)
  goes back to that app (#186). Every delegate call emits a `delegate_call`
  structlog event; never raw transcripts into a model's
  context.
- **No chat turns are replayed** (#64). Each turn's one user message opens
  with the state block (`app/ai/state_block.py`: calendar, surface, panel,
  running jobs, the question the turn right before asked (older cards are
  not shown), recent entities from the apps' records,
  and matches from earlier conversations, #80) and recall (`app/ai/recall.py`:
  this thread's record, requests and last exchange as data, plus word-matched
  calls from earlier conversations when the registry has no match), then the
  person's words. The planner prompt stays byte-identical; the opening is
  capped (`turn_context.MAX_CHARS`).
- **One timeline** (#82, `app/services/conversations.py`): every surface
  writes to one conversation (`kind = timeline`, created clean on first use;
  `POST /agent/conversations` answers it and it is never deleted: "forget
  that" is how). Older conversations are `legacy`: out of the API, still
  read by recall and matches. "This conversation" for the planner is the
  timeline since its last quiet gap (`QUIET_GAP`, 3 h): stored turns, recent
  entities, a waiting plan, undo, the self-report, "forget that" and the
  apps' subagent threads (`thread_key`) all follow that window. Older turns
  arrive only as cross-day hits labelled with their day. Session evals'
  "new conversation" is the record moved back a day (`tests/eval_db.age`).
  The page opens on that same window (#226, `timeline.windowStart`, from
  `quiet_gap_seconds`): older turns and notes sit behind "Show earlier",
  a batch a press with the place kept (#233). A phone's sheet fits the
  latest turn so the web keeps its room; the window's earlier turns are
  scrolled just above it, and reading back gives the sheet its full height.
- **Incognito** (#82, `app/ai/incognito.py`): a per-request switch. The
  person's words and the reply are stored as "(incognito)", and calls,
  steps and finished plans keep no words; receipts and entities stay. Memory
  indexing, recall, matches, recent entities, the self-report and timeline
  search skip it; no fact is saved or attached; the trace and the capture
  write nothing, and log lines blank wordy fields. Only the live response
  carries the reply. The page turns it off after the same quiet gap.
- **Side threads** (#139, `app/services/threads.py`, `app/ai/thread_note.py`):
  a page control starts one (no step kind, no planner change): a
  conversation with `kind = thread`, read whole, with its own apps' threads
  and live channel; several may be open, one on the page. It `carries` the
  timeline window's entities and saved facts as quoted data, never turns;
  incognito and forgotten turns lend nothing, and every read re-checks. Its
  State lists its own entities, then what it carries. Closing posts one
  `thread_note` to the timeline, narrated from its receipts with code checks
  (titles, counts), never from an incognito or forgotten turn, an undone
  write or the memory app; like a job's note it lapses no question. A
  thread where nothing was changed or found posts no note (#226,
  `threads.leaves_note`); the page's Side thread panel lists the few closed
  most recently (`GET /agent/threads?closed=true`) to read again. Its
  `thread` ledger turn copies the receipts without undo handles, so the
  timeline's State sees what it touched. A closed thread takes no turns; its
  jobs note into the timeline.
- **Delegate v1.1** (#67): each `ask` mints one `Idempotency-Key`; only a
  connection dropped after send is resent, under that key. The app's typed
  record is stored per call (`app_record`, never sent to the model), and a 5xx
  whose record lists `done` work is reported as partly happened, by name
  (untrusted titles by kind and id). No record means unknown: never retry blind.
- **The planner decides, the narrator speaks** (#65). A completed run's reply
  is a tool-less call on a typed receipt of what ran (`app/ai/narrator.py`),
  except a read's answer, which passes through (#241, next rule);
  the persona (vendored Glitch, #256) lives only in the speaking prompts, and
  never overrides a fact. Code guarantees the facts (titles, counts,
  undo, a plan's unrun steps and why, quoted outside text, two sentences on
  voice, and a job's state: queued or started, never done, its query no
  second thing, #188; and a write's dates named with their weekday in the
  app's note, `narrator.with_weekdays`, #215); handoffs and an app's confirm question pass through
  verbatim, and so does an app's own question when it did nothing (quoted as
  its words, #168). A v1.2 app's question is its typed `needs_input` alone
  (#202); for a v1.1 app, one that ran nothing may ask without a "?" ("Please
  tell me the artist…", #189), and a few request phrasings read it.
- **A read's answer passes through in the app's own words** (#241,
  `app/ai/relay.py`, delegate v1.3's `answer {text, spoken, untrusted}`).
  It passes only when the step was a read (`plans.marked_read`: sent
  `mode: "read"` to a v1.2 app, `plan.reads_only` for a v1.1 one), the call
  came back, its record is all `looked_up` (at least one; no `done`,
  `refused` or `not_run`) and the app isn't asking. So no write is ever
  claimed from an app's words; asked for a change, the narrator speaks from
  the receipt whatever the app sent. Per step: the narrator speaks for the
  plan's other steps and sees a passed read only as "this app answered"
  (`relay.stand_in`: no record, note, titles or count); a plan that is all
  passed reads calls no narrator. The answer lives on the call's record
  (`ToolCallRecord.app_answer`, `passed`), **never in a prompt**: the stored
  reply is conductor's own words plus code's "Tasks answered.", which is
  what recall, the state block and the self-report quote. Nothing in it
  fills a `{sN}` or is followed. The page shows it under the app's name,
  links and images off, marked "includes outside text" when the app says
  `untrusted`, its record has an untrusted ref, or it names a title this
  window knows as outside text. Voice says `spoken` (`MessageRead.spoken`):
  "Tasks says: …", two sentences when one app alone answered, one each
  beside conductor's own words or another app's, at most two apps; the
  whole answer is on the page. A lone answer about three things or fewer
  that names none gets code's "That's “A” and “B”." from the record. An incognito turn stores none of it. A plan
  that ends on a question, a handoff or a stop passes nothing. Tasks sends
  `answer`; music doesn't yet, so its reads are still narrated.
- **Delegate v1.2** (#202, decided in #145): only an app whose `agent:`
  block says `context: true` gets the v1.2 fields; a v1.1 app keeps every
  guess. Its ask sends the words as written, plus `context`: the `{sN}`
  refs with their ids (beside the spliced titles, which stay until that
  app's evals pass without them; for an app in `dataflow.TITLES_OUT`, today
  tasks, `{sN}` is a pointer, "that album", and the title rides the ref
  alone, #204), its own entities this window touched
  (newest `STATE_REFS`), and the saved facts `facts.related` chose (they no
  longer ride the words). An ask the planner marks `read` (#204,
  `plans.marked_read`) goes `mode: "read"` and may run beside other reads,
  so a wrong mark costs a refused write; a v1.1 app's reads stay
  `plan.reads_only`'s guess. The step's `refs`/`read` arguments stay out of
  every model's view, like `facts`. Its question is `needs_input` alone
  (`ToolCallRecord.app_asks`), never its words; an answer that picks one of
  its candidates goes back as that ref, `picked` (#196). The `_NOUNS`
  repairs (`_UNRESOLVED`, `_UNNAMED`, `_described`) are what's left of #204.
- **The ledger** (#69, `app/services/ledger.py`) is the record of what
  happened: a turn opens with the person's words before it runs and closes
  with its plan (a control step split off first is its own plan), one step
  per call and the apps' receipts. Old rows may hold a `loop` plan or a
  `fell_back` one; they still read. Every app
  write has a receipt row; an app call with no readable record gets
  conductor's `unknown` receipt, never "nothing changed". Journaling runs in
  savepoints and never costs the turn. The state block's recent entities
  come from it.
- **Compiled plans are the only path** (#70, #75): the 12B compiles each turn
  to a typed plan under `json_schema` (`app/ai/plan.py`, `compiler.py`), code
  validates it, and the interpreter (`app/ai/interpreter.py`, on
  `app/ai/runtime.py`) runs every step kind and the direct answers
  (`app/ai/speaker.py`). Each reply opens with `about` (#151: what the
  message is about; the running-job and waiting values only when State has
  one), and a plan its `about` contradicts gets one repair
  (`plan.disagrees`); it isn't stored with the plan. Words that change
  something State doesn't show get one repair to an amend, never a new ask
  (`plan.points_at_nothing`, #176). **No guessed action:** a plan still invalid after
  the compiler's repair, or with a `{sN}` no step can name
  (`plan.unrunnable`), gets one more round (`compiler.repair_turn`), then
  code asks which app was meant when the tries disagreed, else files a
  `think` on the person's own words (#146: nothing waiting, no try read the
  words as a confirm, resume or control, `CONDUCTOR_THINK` on), else says it
  couldn't put that together; a model outage is a 502, never a question or
  a think. A control step inside a longer plan runs first as its own plan,
  then the rest, unless it left a question waiting (then not run, and said). Plans have no designed length; 12 steps is a runaway
  ceiling. A destructive
  ask to an app without `confirms` pauses on code's question; `ask_user`, an
  app's `confirm_required` and an app's own question (`app_question`, #168:
  the answer goes back to the step that asked) pause too, each holding up
  only the steps that need its answer (#251). **A named group goes to the app in the person's
  words** (#248): State's recent entities resolve pointers ("it", "those"),
  never a group ("the test tasks"), since they hold only what was touched
  lately. A one-ask plan that lists State's titles or numbers the person
  didn't name (two or more, no pointer in the words) gets its request put
  back to their words by code (`plan.in_their_words`); a longer plan gets
  one repair (`plan.names_for_a_group`). The planner prompt and the nine step kinds (no `confirm` since #251) stay byte-stable: plan-level
  evals (`tests/test_plan_evals.py`, `CONDUCTOR_PLAN_EVALS=1`) gate any change
  to them, to the compiler, or to a manifest.
- **`look(app)` reads a glance, never asks an agent** (#73). An app's
  `glance:` block (`../agent-standard/app-yaml-glance-block.md`) puts its
  `holds` line in the planner prompt and `look` in the schema for that app
  only; no block, no look (the validator sends a stray one back for an `ask`).
  The interpreter makes one plain GET (`app/fleet/glance.py`: declared path,
  no forwarded headers, 2s/5s timeouts, no redirects, no model, no delegate
  budget) and files `look_<app>` with a `looked_up` receipt built by code
  (`app/ai/glance.py`; outside titles cleaned and marked; a ref keeps its
  other scalar fields, bounded and cleaned the same way, and a last-changed
  time such as chess's `last_move_at` becomes a stated age on the household
  clock, "last move 19 days ago", #161) and no writes. It is
  not a registry tool. A glance that can't
  be read asks the same app the person's words, but only for a plain question,
  a plan whose words are the request, an app with an agent and no ask to it
  already in the plan; otherwise the step fails and the plan ends. The failed
  read stays on the ledger and out of the narrator's story. A look gives
  nothing to `{sN}`. Adjacent looks run in parallel with other reads.
  Session evals read a world-backed glance (`FakeGlances`/`World.glance`, set in
  the harness): no eval ever reaches a real app, and a look counts as a read.
  The speaker's `chat` answer sees the ledger's facts and the about-you check,
  since the planner files "what did you just do?" there when glance lines shift.
- **Memory over the ledger** (#80, `app/services/memory.py`, `app/ai/memory.py`).
  Code only: no model call writes memory, and it reads the person's words and
  the apps' receipts, never conductor's replies or plan text. Each ended turn
  is indexed (words, the entities it touched, a quiet topic: shared words or
  entities, else the turn before), oldest first, as turns close. Topics are
  never shown or managed. Recall and matches also rank by meaning (#147,
  `app/services/turn_vectors.py`): each indexed turn is embedded once as it
  closes by the CPU embeddings service (`CONDUCTOR_EMBEDDINGS`,
  `EMBEDDINGS_BASE_URL`), both searches are fused, and a hit by meaning alone
  must lift `memory.LIFT_ALONE` over the median cosine (set from
  `tests/recall_cases.py`'s paraphrases and controls). Meaning only ranks;
  `{sN}` stays on words, and so does `facts.related` (meaning only adds a
  fact when it finds none, below). Incognito and forgotten turns are
  never sent (forgetting deletes the vector); a service down or slow means
  words alone, never a failed or waiting turn; `scripts/embed_turns.py`
  backfills. `recall` cites turns from every live conversation by
  day and conversation, scoped by matching topics, as a `conductor` receipt
  (`fact`, `cites_turn_id`); `{sN}` on it is the one entity that clearly
  answers (a tie names nothing), with its trust. Nothing found is said, never
  guessed (code appends it). The state block's matches are registry leads
  from other conversations, only when the words point back; outside titles
  stay quoted labels. Everything recalled is labelled "as of then" (#127),
  and a `knowledge`/`chat` answer to a question an app answered before gets
  the validator's repair (`plan.recalled_answer`: ask or look again; plan
  cases `stale-*`). The prompt's worked examples for this were measured and
  dropped: they pulled unrelated goldens into asks.
- **Memory you can see** (#81, `app/ai/facts.py`, `app/services/facts.py`).
  "Remember …", "what do you know about me?" and "forget that" are asks to
  `memory`, conductor's own app: one `ask` enum value and one planner-prompt
  line, run by code (no agent, model, budget or gate). A fact is cut from the
  person's own words, never a model's, and so is what to forget (#134);
  outside text never becomes one. Its
  receipts have no undo handle and no registry entity. Saved facts ride the
  state block (capped) and the speaker's facts. Forgetting soft-deletes the
  fact and `unindex`es the turn it was said in and the one that forgot it
  (`forgotten_at`: off topics, entities, recall, matches and receipt quotes,
  blanked in every replay); the visible transcript keeps the words. The
  memory list (`/api/memory/facts`, a lazy UI chunk) shows each fact's
  source; its edits and forgets are the person's note turns. An app ask
  whose request shares two terms with a saved fact carries it (#134,
  `facts.related`, at most two), and so does one whose request writes
  something of the fact the person didn't say ("Sunday") when their own
  words share two terms with it (#179). When words find none, an ask that
  isn't a read may carry the one fact it means (#204,
  `app/services/fact_meaning.py`, `CONDUCTOR_FACTS_BY_MEANING`): its lift over
  `facts.BACKGROUND` and its gap over the next fact must clear the bar set
  from `tests/fact_cases.py`; reads never do (topic isn't relevance). Code
  sends it after the request, quoted and
  marked as context, never rewriting the words; the step's `facts` argument
  records it, and the record lines models read leave it out. Four
  validator nudges (one repair, then the plan runs as compiled) keep plans
  on that path: "what do you remember about X?" with no saved fact about X
  is a `recall` (`plan.remembered_topic`), a lookup of a fact State lists
  before another app's ask is dropped (`plan.looked_up_fact`), "what
  can you do?" or "what do you know about me?" is never `knowledge`
  (`plan.about_you`), and "forget that" with nothing waiting or running is
  never a control step (`plan.forgets_by_control`, #183).
- **Data flows as quoted data, never as control** (#72, `app/ai/dataflow.py`).
  Code fills `{sN}` from step N's receipt: its main touched title (job refs
  skipped, up to 3 joined), cleaned and quoted. No title, or more than 3,
  means the step doesn't run and the plan ends, with the reason on the
  record; so does an ask its app turned down with nothing done (#168). Never
  guess a name or splice in an app's reply text. The validator repairs a lost
  ref: a bare `s1`, a step re-describing what an earlier ask will find (a
  compound ask for one thing too, #188; not an ask for several), an ask
  pointing at State ("their other album"), or a recall or memory lookup for
  one (`plan.recalls_unnamed`, a nudge, #168); and it
  splits an ask carrying a second app's work ("… and add a task", #127) into
  two steps. The plan is
  fixed before any step runs, so a title can't steer control flow. An
  `untrusted` title goes only into a non-destructive ask to an app with
  `confirms`, never into a destructive ask, an app without `confirms`, or an
  `open` intent. Only trusted values survive a pause. An `ask_user`'s options
  resolve in code to the one entity State named for each (#120,
  `app/ai/options.py`, kept on the paused body; the schema is unchanged): the
  person's pick (by name, distinct words or place) fills `{s1}` with that
  entity's title and trust. An option no entity names stays a title; an
  answer that picks none goes in as said. Adjacent reads to
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
  With nothing waiting, the person's words pick between them (#185): "undo"
  is an undo even compiled as a cancel, and "stop"/"cancel" with a job
  running is a cancel even compiled as an undo. A running job takes an undo
  only when it is all the last plan wrote; beside other writes it is left
  running and the reply says so (code's reason, guaranteed like #163's).
  **Spoken corrections** (#77, `plan.is_correction`: a no, yes or
  "actually", then what instead: "no, Saturday") change what waits or
  runs: a cancel, chat or self for one gets one repair to an amend
  (`plan.corrects`); words that pick an option and say more
  ("no, the bill one on Saturday", `plan.changes_too`) fill `{s1}` and
  go after the resumed step (#198). With nothing waiting or
  running, an amend goes to the app that made the last plan's finished
  write, named from its receipt by kind and id (one app, at most three
  things, else nothing changes and code says so), the change in the
  person's own words; the validator asks for that amend, also in place of
  one new ask (which names the thing again and loses it), only when the
  ledger has such a write (`Checks.changeable`) and the words take it back
  (a no, "actually" or "wait", never a yes or an okay).
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
  A new ask to an app with a running job, when the words narrow it ("just
  the first ten"), gets one repair to an amend (`plan.narrows_job`, #160).
- **`think` is a background job on the local 26B** (#78, `app/ai/think.py`,
  `app/services/thinker.py`; `CONDUCTOR_THINK` off: the think isn't run and
  code says so). Code files one too when no plan could be assembled at all
  (#146; the reply says it is thinking it over, never that anything ran;
  off, it is the old "couldn't put that together"). The plan ends at a
  think (later steps not run, and said): it files a `ledger_jobs` row conductor owns
  (`app = "conductor"`) with what its inputs found, and code replies at once;
  the conversation never waits. The watcher hands it to the thinker:
  gemma-4-26b-a4b, thinking on, `json_schema` on the answer, its own prompt
  and profile. The answer is a **proposal**, validated like any plan (one
  repair; a plan asked for and answered with no steps gets it too, #255,
  `think.wants_steps`) and tainted: it posts as the job's one note (`think_proposal`, a
  card) and nothing in it runs without the person's tap. The tap runs it
  as a normal plan (gates and apps' questions still ask); "Not now", a
  spoken no or an amend sets it aside. **It yields the GPU**: no try starts during a
  turn, and one in flight is cut off when a turn starts
  (`app/ai/foreground.py`). The call streams so the cut-off closes it between
  chunks: llama-swap finishes an in-flight request before it swaps, so
  without that a turn waits out the whole thought (~2 min measured). Another
  app's model call still waits behind a think (no cross-app priority yet). A
  cut-off or failed call is `interrupted` (a `not_run` receipt)
  and retried after 60 s with no turn, 3 tries, then code's honest "couldn't
  finish". Never claim a proposal that wasn't made.
- **STANDARD.md §3's safety nets, without the loop** (#75): `max_tokens` on
  every model call, the plan's 12-step ceiling and the stop vocabulary; depth stays 1 (a step is one app call). The loop's
  iteration, stall and 12-call caps went with it (#111). A stop short of
  `completed` replies from the code-kept record, and a failed turn is stored
  with its plan before the route answers 502/500, so a write never vanishes
  from the thread.
- **Stop** (#66): `POST …/cancel` ends the turn `cancelled` at its next model
  call or step, replying from the record (never narrated); a delegate call
  in flight gets the app's `…/cancel`, best-effort.
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
  answer as `{s1}` reads it, with its trust and picked entity) and `turn_ended`, plus `job` events (progress,
  then the end with its note's id) that keep the stream open past
  `turn_ended` while a job runs. Step events go through one seam
  (`app/ai/turn_events.py`); a receipt is the step as the ledger journals it (`ledger.live_step`) and
  lands only once its record exists, and `turn_ended` goes out after the
  ledger commit. Queues and turn state are bounded; a lagging follower is
  resynced from a snapshot. `GET …/activity` stays as the poll fallback.
- **Barge-in** (#77): the page passes `bargeIn` to the vendored MicButton
  ("Talk over replies", on unless turned off, `localStorage`), and a turn
  said over a reply carries the voice client's `Heard` (`MessageCreate.heard`,
  snake case, bounded). Code keeps it on the newest answered turn in the
  window (`ledger_turns.heard`) only when its `total_chars` is that reply's
  UTF-16 length; the state block then says the reply was cut and quotes
  what wasn't heard (capped, `HEARD_MAX_CHARS`), and the self-report says
  what they heard. Incognito turns send, store and show none of it; a
  forgotten turn lends nothing, re-checked on every read. The mic calls the
  page through a latest-callback ref (it keeps the one it started with).
  Real-audio behaviour (echo tripping the mic) is unverified by tests.
- **The delegate contract never assumes a text-only client** — voice
  convergence is a master-plan placeholder.
- **The panel is the accessible truth; the 3D web is decoration** (#88). The
  Living Web (`src/scene/livingWeb.js`, three.js) is `aria-hidden`, loads as
  a lazy chunk after first paint, and shows nothing the panel doesn't say in
  words. Both read one derivation (`turnView.ts` → `sceneState.ts`), and
  nothing looks done before its receipt says so. A finished turn's or job's
  strand lets go on the derivation's clock (#232, `HOLD_MS`, `RELEASE_MS`); a
  running or waiting one keeps it.
- **The live plan view** (#89) draws from the same stream: the plan appears
  queued beside the code-built ack, each step lights on its own event, a
  `{sN}` stays a dashed ref until its receipt names a title (never filled
  from an outside title early), and a step that didn't run is `not run`,
  never `failed`. The turn that ended on a question keeps its stream
  (`held`) until the next turn; its card is under "Waiting on you". A strand lights only on a real step event.
