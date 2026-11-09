# Agent eval harness

Conductor's live evals run the **real** model (gemma-4-12b behind llama-swap)
against the **real** workspace manifests. Since #75 (2026-11-08) every turn
goes planner → validator → interpreter → narrator, and the eval gate is two
suites under the vendored Wilson gate (#59):

- **Plan-level evals** (`tests/test_plan_evals.py`, below): does the 12B
  compile the right plan? Every routing golden re-labelled as a plan, the
  S1–S16 scenarios, a held-out set and long plans, per group.
- **Session evals** (`tests/test_session_evals.py`, below): whole multi-turn
  sessions over a fake fleet, checked on the end state and the replies.

The single-utterance routing goldens (`tests/test_agent_evals.py`) **retired
at #75** with the loop they measured. Their runs stay below as history; the
cases live on as plans (`refuse-playback`, `music-save` and `chess-resign`
included). Full measurement narratives are in this file's git history.

## The gating rule

Any change to the planner prompt, the step kinds, the compiler or validator,
the interpreter or narrator, the model, **or any fleet app's
`agent:`/`open:`/`glance:` manifest block — wherever that PR lands** — runs
both suites and must not regress them:

```bash
cd backend
export CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace   # from a worktree
CONDUCTOR_PLAN_EVALS=1 .venv/bin/pytest tests/test_plan_evals.py -v -s
CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1 \
    .venv/bin/pytest tests/test_session_evals.py -v -s
```

A sibling app widening its manifest changes these suites' inputs (that is
how `refuse-playback` went red without anyone noticing). If plans degrade,
tighten manifest `examples` first; a bigger model is the last resort.

**Nothing may go statistically below its floor** (`STANDARD.md` §6), judged by
the vendored `backend/tests/evalstats.py` (spec `test_evalstats.py`,
conductor's drift guards `test_evalstats_app.py`; re-copy from
`../agent-standard/`, never edit): a one-sided 95% Wilson upper bound below
the floor is red, a rate at the floor green. Plan groups have their own
floors; session scenarios in the `gate` tier are held to 0.8, sampled in
blocks of 5 up to 20; the `frontier` tier is measured, never asserted. A
known red is `xfail`, not gated. `scripts/eval_campaign.sh --suite plan|session`
runs a paired A/B.

## Infrastructure is retried, not scored

A provider death (llama-swap evicting or reloading the model mid-suite: a 502
`llama-server request failed: timed out`, a refused connection, or a 5xx while
it swaps) never decided anything, so it is not scored. The harness throws the
sample away and re-takes it on a fresh conversation, up to
`CONDUCTOR_EVAL_INFRA_RETRIES` per scenario and `CONDUCTOR_EVAL_INFRA_BUDGET`
across the suite (`InfraBudget` in the vendored `evalstats.py`). A timed-out
sample still costs the full 300 s, so the defaults stop a dead server inside
~an hour.

Reading a run:

- `[eval] scenario=… PASS_RATE=p/n (…) ci=[lo,hi] DECISION … infra=N`: the
  verdict over real routing decisions, after N thrown-away samples.
- `[eval] suite …s infra_spent=S/B (scenario=n, …)` closes the run — a green
  suite that limped says so, and says which scenarios ate the budget.
- `INFRA_ABORTED` — the scenario ran out of retries. Routing was not tested:
  fix the server, not the manifests.
- `llama-server refused the request` — a 4xx (context overrun) or an
  unparseable body. Deterministic, so it fails on the first one without
  retrying; look at the request.

Any other red is a routing answer.

## History: the routing goldens (retired at #75, 2026-11-08)

What follows ran `tests/test_agent_evals.py` against the old loop, which
#75 deleted. Kept as history; none of it gates anything now.

### Known reds (xfail, not gated)

Under the sampled gate (2026-10-28) two goldens read below the floor on main;
both are `xfail` in `KNOWN_REDS` until fixed:

- `refuse-playback` — 0/5, deterministic (below).
- `music-save` — "save this track to the music folder": 2/5, then 13/20 (65%,
  ci [0.47, 0.80]); every miss answers locally with no app call. The
  one-sample gate passed it by luck in every recorded baseline.

The rest of the suite: 19/21 ABOVE_FLOOR at 5/5, infra 2/12 (two llama-swap
timeouts on `chess-reset`, re-taken, never scored), 684 s.

### Routing run, 2026-11-08 — `chess-status` reads `look(chess)` (#118)

Since #73 a glance answers "how is our chess game going?" (chess#536's
point), so `chess-status` now expects `look_chess`, the record the
interpreter files for a `look` step. It pins no `intent` and no delegate call.
A `look_` call counts as an action only for a golden that expects one; ahead
of an ask it stays free, so `tasks-due` and `music-sort-ask` (which now look
first) keep passing as they did. Chess's manifest is unchanged.

Run on gemma-4-12b, 5 samples per golden, infra 0/12, 105 s:

| golden | result |
|---|---|
| tasks-due, tasks-create | 5/5, 5/5 |
| **tasks-week** | **1/5 red** (earlier run the same day 2/5): every miss answers locally with no app call |
| chess-play, chess-move, chess-reset | 5/5 each |
| **chess-status** | **5/5** (`look_chess`; was 0/5 on main as `open_chess`) |
| **chess-analysis** | **0/5 red**, known miss: `look_chess` every time, expects `open_chess` |
| **chess-resign** | **1/5 red** (1/5 in both runs): the misses route to `ask_music` |
| music-download-link, -download-named | 5/5, 5/5 |
| music-save | 4/5 (xpass; 5/5 in the earlier run) |
| music-sort, -sort-ask, -organize | 5/5 each |
| refuse-lights, -weather, -playback | 5/5 each (`refuse-playback` xpass; 12/15 in the earlier run) |
| confirm-delete-task, -wipe-project | 5/5, 5/5 |
| local-capabilities | 5/5 |

The reds besides `chess-analysis` are not from this change: `tasks-week` and
`chess-resign` read below the floor on this run. Both are explained, and
`tasks-week` fixed, in the next section (#122).

### The `tasks-week` and `chess-resign` reds (#122, 2026-11-08)

Two causes, bisected with the live goldens (gemma-4-12b, 5 samples) over the
merges since the last recorded baseline:

| commit | `tasks-week` | `chess-resign` |
|---|---|---|
| `11068a8` (#100, loop) | 5/5 | 5/5 |
| `19c9874` (#102, recall, loop) | 5/5 | 5/5 |
| `186bcd0` (#70, compiler) | **1/5** | **2/5** |
| `757c42d` (#73, look) | 1/5 | 12/20 |
| main, 2026-11-08 | 1/5 | 4/5; 11/20 at n=20 |

Both turned red with the compiler, which is the first thing that reads #64's
recall. The bisect held the manifests fixed at today's, so neither is a manifest
regression.

- **`tasks-week` was the harness, not routing.** Samples share one database,
  and #64's recall adds "earlier conversations' calls that share words" to the
  turn. Sample 1 routed; samples 2–5 found sample 1's answered
  `ask_tasks "what am I working on this week?"` in the context and the
  planner replied `{"answer": "knowledge"}`: no app call, the speaker
  declined. Session evals got a wiped database per sample in #103; routing
  evals didn't. Now `tests/eval_db.py` is shared and both wipe. `tasks-week`
  5/5. The same leak is the likeliest cause of `music-save`'s "answers locally"
  misses (5/5 now, in two runs; left `xfail` until a 20-sample read).
- **`chess-resign` is a real planner miss**, no leak: `{"do":"ask","app":"music",
  "request":"resign the game"}`, 9 of 20, though "resign the game" is
  one of chess's own examples. Listing more examples made it worse (add
  "I resign" and "reset the game": 3/20). The `open.description` was the lever:
  naming resigning and what "the game" means took it to **20/20** (the same
  sentence without the clause: 17/20). Chess owns `app.yaml`, so the fix is
  a draft for that repo:

```diff
-  description: "Opens the chess board in the browser and starts a session with the chess agent. Use this for anything to do with playing, watching, or changing a chess game — you do not play chess yourself, you hand the user to the board."
+  description: "Opens the chess board in the browser and starts a session with the chess agent. Use this for anything to do with playing, watching, resetting, resigning or otherwise changing a chess game ('the game' always means this one) — you do not play chess yourself, you hand the user to the board."
```

`chess-resign` is `xfail` in `KNOWN_REDS` until that lands.

Same run, the draft manifest read from a scratch fleet dir ("after"; before is
the real manifests, both with the wiped database):

| golden | before | after |
|---|---|---|
| tasks-week | 5/5 (was 1/5) | 5/5 |
| chess-resign | 4/5 (11/20 at n=20) | **5/5** (20/20 at n=20) |
| chess-analysis | 0/5 known red | 0/5 |
| refuse-playback | 9/15 (xfail) | 5/5 (xpass) |
| every other golden | 5/5 | 5/5 |

Plan evals, shipped prompt (unchanged), 5 samples per case:

| group | floor | before | after |
|---|---|---|---|
| golden | 0.8 | 95/105 (`chess-resign` 1/5) | 98/105 (`chess-resign` 5/5) |
| cross | 0.8 | 35/35 | 35/35 |
| ref | 0.85 | 16/20 | 14/20 |
| control | 0.8 | 20/25 | 20/25 |
| memory | 0.8 | 7/10 | 5/10 |
| knowledge | 0.65 | 15/15 | 14/15 |
| hard | 0.5 | 5/5 | 5/5 |
| self | 0.55 | 6/15 | 10/15 |
| holdout | 0.85 | 46/50 | 46/50 |
| long | 0.7 | 36/40 | 38/40 |

At n=5 `ref` and `memory` read red after (and `ref` nominally under its floor
before). They're two cases, `ref-ambiguous` and `memory-caribou`, that don't
involve chess. At n=20 on each side: ref 61/80 before, 67/80 after; memory
27/40 before, 26/40 after. So the change doesn't move them, and they are
red on main at n=20 independent of this work (`ref-ambiguous` plans an `ask` to
tasks instead of `ask_user` about 3 in 4 times; `memory-caribou` plans a
`look` of music instead of recalling or asking).

Session gate tier, compiler on, 5 samples: every scenario 5/5 before and
after except S7 4/5 after (ABOVE_FLOOR; S7's miss is already recorded in
this file as happening on main). Claims 40/40 backed in both runs.

Left open: the leak is also a product weakness. A person asking the same
question in a new conversation gets the planner trusting a recalled
answered call over asking the app again. The planner prompt is unchanged
here; it wants its own change, gated by the plan evals.

### Open regression — `refuse-playback` red since music #9

"play some jazz music" routes to `ask_music` instead of refusing,
deterministic 0/3, bisected to music #9's manifest line *"queues a whole album
or playlist"* (2026-08-23). An attempted fix (a no-playback negative sentence
in music's description) fixed it 3/3 but broke `music-save` in 2 of 3 full
runs — the negation suppresses music routing generally. Options, none taken
(tracked in #36): narrow the negative until both hold; retire the golden
(music's own agent refuses truthfully one hop later); or leave red until music
Phase 6 makes the route correct.

### Baseline — 2026-08-22, gemma-4-12b, 18/18 (3 consecutive runs)

| scenario | kind | expected |
|---|---|---|
| tasks-due / tasks-create / tasks-week | route | ask_tasks |
| chess-play / -move / -status / -analysis / -reset / -resign | route | open_chess (+intent) |
| music-download-link / -download-named / -save | route | ask_music |
| refuse-lights / refuse-weather / refuse-playback | refuse | no app call |
| confirm-delete-task / confirm-wipe-project | route (#68; was confirm) | ask_tasks: PCC asks itself |
| local-capabilities | local | list_agents only |

Warm routing turns are sub-second to ~2 s; real latency is the subagent's own
loop. (`refuse-playback` passed here; see the regression above for its state
since music #9.)

Standing observations:

- Adding a third agent didn't blur existing routes — manifest `examples` do
  the work; re-baseline when music gains playback vocabulary.
- Descriptive safety rules don't bind a 12B: the confirm rule holds only in
  operational voice with a worked example (the harness caught unconfirmed
  delegation on its first run).
- Same-app repeats within a turn are allowed by the goldens; the per-turn
  budget (3) caps the worst case.

## Session evals (#63)

`backend/tests/test_session_evals.py`: scenarios S1–S16 from #56 (the research
doc's §1.1) as whole multi-turn sessions over a fake fleet whose apps really
change state (`fake_world.py`: a tasks app and a music app that read
conductor's free text the way a cooperative app agent would, and leave a
receipt for every write). Checks read the **end state** per app plus the
replies held to the receipts (`receipt_claims.py`): nothing claimed that
didn't happen, and every write named in its turn's reply (by title, or by
count for a batch). Paths are never pinned. The scenarios are in
`session_scenarios.py`; the driver seam in `session_harness.py` is what a
later runtime plugs into.

- **Scripted (CI).** Each scenario's golden script plays the model (plans, then
  the narrator's words; since #75 the compiled runtime only) and must
  pass, so every check is shown to be satisfiable. Deliberately wrong scripts
  (acting before the yes, a dropped album title, a claimed write that didn't
  happen, an invented recap) must fail for the right reason; an unnamed write
  is now named by code, so its script passes.
- **Live (opt-in).** `CONDUCTOR_AGENT_EVALS=1` runs the `gate` tier under the
  vendored Wilson gate. `CONDUCTOR_FRONTIER_EVALS=1` also measures the
  `frontier` tier: 5 samples per scenario, never asserted. Every sample is a
  fresh world and a fresh conversation through the real route, interpreter, prompt
  and workspace manifests.

```bash
cd backend
CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1 \
    .venv/bin/pytest tests/test_session_evals.py -k live -v -s
```

From a worktree, set `CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace`.
`CONDUCTOR_SESSION_VERBOSE=1` prints every turn's words and reply;
`CONDUCTOR_EVAL_REPORT=<path>` appends a JSON line per scenario. The run
closes with `[session] claims backed=b/n`, the claim tally #65 is judged on.

A scenario that needs a capability the runtime lacks fails `unsupported:`
instead of skipping, so it reads 0 for a named reason and
starts measuring on its own once the capability lands.

### Baseline: today's loop, 2026-11-05, gemma-4-12b

Two full runs on a shared GPU (a PCC eval ran alongside the second). Gate
scenarios ran under the Wilson gate; frontier ones ran 5 samples each.
Run 1 used the first cut of the checks; between the runs the naming check
learned stems ("a call to your mom" names "call mom") and three fake-app
misreadings were fixed (a "last week" search read as a date filter, "plan" read
as "create", no cancel path in music). Run 2 is the baseline.

| scenario | what it checks | tier | run 1 | run 2 | main failure mode |
|---|---|---|---|---|---|
| S1 | what's due today, from PCC's data | gate | 1/5 | **5/5** | — |
| S7 | voice: "dennis" → dentist moved to Friday, one spoken line | gate | 12/20 | **8/10** | the reply doesn't name what moved |
| S8 | honest "no app does weather"; egg answered or declined | gate | 5/5 | **5/5** | — |
| S2 | add a task: title, date, reply names it | known red | 1/5 | 0/5 | the reply doesn't name the task |
| S2r | a second turn reuses the delegate thread (#37) | known red | 0/5 | 0/5 | the reply doesn't name the task (reuse holds) |
| S5 | what did we decide: cite it, or "I don't have that" | known red | 0/5 | 1/5 | finds "steel rack", doesn't say where from |
| S9 | asks once, acts on "yes", says what went to the trash (#37) | known red | 4/5 | 1/5 | "project's in the trash" without naming it |
| S16 | "what did you just do?" names what happened | known red | 0/5 | 1/3, then INFRA_ABORTED | turn 1 doesn't name the task |
| S3 | queue the album, then a task titled with its real name | frontier | 0/5 | 0/5 | the task says "the new Caribou album", not "Honey" (#55) |
| S4 | a long job reports back on its own | frontier | 0/5 | 0/5 | nothing unprompted arrives |
| S6 | "that Caribou song", days later | frontier | 0/5 | 0/5 | no memory across conversations |
| S10 | Stop partway | frontier | 0/5 | 0/5 | `unsupported: stop` (#66) |
| S11 | "undo that" after S3 | frontier | 0/5 | 0/5 | doesn't say the download can't be undone |
| S12 | read the last chess game without opening the board | frontier | 0/5 | 0/5 | `unsupported: tool:ask_chess` |
| S13 | "anything I should know?" | frontier | 0/5 | 0/5 | reads today's tasks only: no overdue, no music |
| S14 | plan my week around the garage project | frontier | 0/5 | 0/5 | no concrete plan to confirm |
| S15 | "do the same for their other album" | frontier | 0/5 | 0/5 | the task never carries "Suddenly" |

**#68 (2026-11-06), PCC asks and conductor answers with its token:** routing
19/19 plus both relabelled destructive goldens at 5/5. S9 2/5, and every
miss is narration ("didn't say where it went", #65): it asked once and
trashed on the yes in all 5. S7 read 2/5 in one block, then 17/20 on a
20-sample rerun, so it stays green.

Claims, run 2: **29/29 backed** (create 1/1, queue 16/16, trash 2/2, update
10/10; run 1 had 45/46, one "got the task sorted" with no sort). So today's
relay doesn't invent writes; it leaves them unnamed. That is #55, and #65's
narrator is the fix, so the known reds point there. Floors stay at 0.8. Warm
turn p50 was 1–3 s; the GPU was shared with a PCC eval, so S16 hit four 300 s
provider timeouts and aborted on infra (retried, never scored).

**#66 (2026-11-06), Stop:** S10 0/5 → **5/5** (the reply is the run record:
what got filed, and the app's undo note). Gate tier unchanged at 5/5 each;
routing 19 pass, `refuse-playback` still the known red.

### #65, the narrator (2026-11-06, gemma-4-12b)

Paired on one harness: `CONDUCTOR_NARRATOR=0` is the planner's own relay
("before"), the default is the narrator ("after"). The fake apps now send
v1.1 records shaped like PCC's and music's, so both arms see the same receipts.
Gate tier, 5-sample blocks; S5 also at 20.

| scenario | before | after | turn p50 before → after |
|---|---|---|---|
| S1 | 5/5 | 5/5 | 0.8 → 1.3 s |
| S2 | 0/5 | **5/5** | 0.9 → 1.6 s |
| S2r | 1/5 | **5/5** | 0.9 → 1.7 s |
| S5 | 2/5 | **5/5**, then 18/20 | 1.0 → 2.0 s |
| S7 | 5/5 | 5/5 | 0.7 → 1.3 s |
| S8 | 5/5 | 5/5 | 0.5 → 0.5 s |
| S9 | 15/20 | **5/5** | 0.8 → 1.4 s |
| S16 | 0/5 | **5/5** | 0.8 → 1.6 s |

- Claims backed: 24/24 before, **30/30** after (create 20/20, trash 5/5,
  update 5/5). The known reds on S2, S2r, S5, S9 and S16 are dropped.
- **Relay probe** (`tests/test_relay_probe.py`, spike R1 as a standing probe:
  "download this song", music's real note and record, "Odessa" and "Real
  Title"): **40/40** kept after, twice (Wilson lower 0.94); 0/20 before.
  Turn p50 0.75 → 1.24 s.
- **Latency:** the median of the gate scenarios' p50s goes 0.8 → 1.45 s
  (+0.65 s); the relay probe +0.49 s. The worst is S5 (+1.0 s): a long receipt
  of notes to answer from.
- **Routing** with the narrator on: 19 pass, `music-save` 5/5 (xpass),
  `refuse-playback` 0/5 (known red, #36). The planner prompt and tool schemas
  are byte-identical, so routing can't move.
- The first cut required every title a read touched; S5 fell to 9/20 (lists
  crowded out the answer, and every search re-spoke). A read now owes one
  thing it found; a write owes every title. The narrator re-speaks in about 1
  of 20 S5 turns.

## Plan-level evals (#70)

`backend/tests/test_plan_evals.py` measures the compiler alone. Each sample is
one compile as the interpreter does it: the shipped prompt, schema, validator,
its repair and (since #75) one more round when the plan still can't run, against
the real llama-server and the workspace manifests. Nothing runs. A sample
passes when the plan is valid and matches one of the case's shapes
(`plan_cases.py`).

The cases are:
- spike D1's bench: every routing golden re-labelled as a plan, the S1–S16
  scenarios and the held-out set;
- eight long-plan cases (6–10 steps when compiled one thing per step), added
  when the 5-step limit went. They check coverage, not exact shape.

The gate is per group. Red takes evidence: the Wilson upper bound below the
floor. Floors sit a little under the spike's 12B v2 rates.

```bash
cd backend
CONDUCTOR_PLAN_EVALS=1 CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace \
    .venv/bin/pytest tests/test_plan_evals.py -v -s
```

Since #75 this suite and the session evals are the eval gate (above): any
change to the planner prompt, the step kinds, the validator or a fleet app's
manifest runs both.

### Baseline: 2026-11-06, gemma-4-12b, 5 samples per case

| group | floor | first run | shipped prompt |
|---|---|---|---|
| golden | 0.8 | 82/105 | 88/105 |
| cross | 0.8 | 30/35 | 30/35 |
| ref | 0.85 | 20/20 | 19/20 |
| control | 0.8 | **11/25** | 20/25 |
| memory | 0.8 | 8/10 | 7/10 |
| knowledge | 0.65 | 15/15 | 15/15 |
| hard | 0.5 | **0/5** | 5/5 |
| self | 0.55 | 14/15 | 6/15 |
| holdout | 0.85 | 45/50 | 45/50 |
| long | 0.7 | 34/40 | 37/40 |

Every plan was valid in both runs. Long plans ran to 7 steps at a 2.2 s
median compile; single-app compiles take 0.5–0.6 s.

The first run's reds were fixed in the prompt, each change A/B'd live first:
- **"yes" to a waiting question came back as an answer, not `confirm`.** A
  rule line took it from 1/10 to 10/10.
- **A plan added steps after `think`.** `think` is now always last.
- **"reset the chess game" became a destructive ask to music.** The
  "destructive" wording is now scoped to what an ask app keeps, and "the game"
  always means chess.

Two misses were fixed in code instead of the prompt:
- **Invented `ask_user` options.** The validator now refuses options that
  State doesn't name.
- **Self questions filed as `knowledge`.** A knowledge answer sees the ledger
  facts and flags `about_you` under a schema. This is why `self` reads 6/15
  here while S16 is 5/5 end to end.

Known misses:
- `chess-resign` (1–2/5) and `chess-analysis` (0/5, the glance) are fixes for
  chess's manifest examples.
- `refuse-playback` is #36.
- `amend` compiles to an ask. It falls back to the loop until #74.

### Session evals with the compiler on (#70, 2026-11-06)

`CONDUCTOR_COMPILER=1` covers the gate tier and the frontier tier. The planner
prompt is final.

| scenario | narrator (#65) | compiled |
|---|---|---|
| S1 | 5/5 | 5/5 |
| S2 | 5/5 | 5/5 |
| S2r | 5/5 | 5/5 |
| S5 | 5/5 | 5/5 |
| S7 | 5/5 | 5/5 |
| S8 | 5/5 | 5/5 |
| S9 | 5/5 | 5/5 |
| S16 | 5/5 | 5/5 |
| S6 (frontier) | 0/5 | 2/5 |
| S10 (frontier) | 5/5 | 5/5 |

- Claims were 93/94 backed. The one miss is S4's "playlist download", which
  is cross-app and runs on the loop.
- Turn p50s were 1.3–3.2 s.
- Before the final prompt, S7 read 4/10. The planner offered invented
  options, then rewrote the request into a search. The options rule and the
  keep-their-words line fixed it.
- S3, S4, S11, S13, S14 and S15 stay 0/5. They need cross-app data flow (#72),
  jobs (#76), undo (#74) or `think` (#78), which fall back to the loop today.
- S12 is still `unsupported: tool:ask_chess`.

### Cross-app plans and data flow (#72, 2026-11-07)

Cross-app single-writer plans and `{sN}` data flow run on the interpreter
now. `look` still falls back, until #73. The planner prompt is unchanged.
What changed for the 12B is the validator. It now catches a ref the planner
meant but didn't write:
- a bare `s1`;
- a later step describing what an earlier ask will find ("the album I just
  downloaded");
- an ask left pointing at State ("their other album", "the other album" with
  no artist).

The one repair turns each of these into `{s1}` or the name State gives.

Session evals, frontier tier with the compiler on. "Before" is #70's run;
the other columns are A/B arms of this change.

| scenario | before (5) | data flow only (5) | + validator v1 (10) | + prompt line (10) | validator, no prompt line (10) | shipped (10) |
|---|---|---|---|---|---|---|
| S3 | 0/5 | 2/5 | 8/10 | 10/10 | 10/10 | **10/10** |
| S15 | 0/5 | 1/5 | 1/10 | 8/10 | 3/10 | **9/10** |

- With data flow alone, S3 was 2/5. The planner wrote `{s1}` about half
  the time. Otherwise it wrote "step s1" or re-described the album ("the new
  Caribou album"), and the real title was lost.
- S15 lost the artist: "download their other album" reached music, which
  can't see State. The repair then asked music which album, so code had no
  title for `{s1}` and refused the task step instead of guessing.
- A prompt line about pronouns lifted S15 to 8/10, but memory went red
  (5/10, twice). It was cut. The shipped arm gets S15 to 9/10 without it, by
  giving that check's repair a worked shape ("the other album by <the artist
  in State>").

Plan-level evals, every group, shipped validator vs #70's baseline:

| group | floor | before | this change |
|---|---|---|---|
| golden | 0.8 | 89/105 | 89/105 |
| cross | 0.8 | 32/35 | **35/35** |
| ref | 0.85 | 18/20 | 18/20 |
| control | 0.8 | 20/25 | 20/25 |
| memory | 0.8 | 8/10 | 9/10 |
| knowledge | 0.65 | 15/15 | 15/15 |
| hard | 0.5 | 4/5 | 5/5 |
| self | 0.55 | 9/15 | 14/15 |
| holdout | 0.85 | 47/50 | 46/50 |
| long | 0.7 | 39/40 | 37/40 |

All groups are green. The baseline column is a same-day run with the prompt,
schema and validator identical to main, not #70's table.

Session gate tier with the compiler on, in two runs. The 5-sample run is the
usual gate; the 10-sample run is a paired A/B against origin/main on the same
day:

| scenario | shipped (5) | origin/main (10) | shipped (10) |
|---|---|---|---|
| S1 | 5/5 | 10/10 | 10/10 |
| S2 | 5/5 | 10/10 | 10/10 |
| S2r | 5/5 | 10/10 | 10/10 |
| S5 | 5/5 | 10/10 | 9/10 |
| S7 | 4/5 | 9/10 | 9/10 |
| S8 | 5/5 | 10/10 | 10/10 |
| S9 | 5/5 | 10/10 | 10/10 |
| S16 | 5/5 | 10/10 | 8/10 |

- Every gate scenario is above its floor in every run.
- S7's miss (the dentist's due date) happens on main too.
- S5 is a `recall` turn, which still falls back to the loop, so this change
  doesn't run in it.
- S16's two misses are the speaker's recap leaving out the task's name. All
  ten runs compiled and ran the same one-step plan, and the speaker and the
  ledger's self-report are unchanged. Two misses against none isn't evidence
  at n=10, so it's recorded here and not treated as a regression.

- The other frontier scenarios: S4 (jobs, #76), S11 (undo, #74), S12 (no
  `ask_chess`) and S13/S14 (`think`, #78) stay 0. S10 is 10/10, and S6 is
  7/10 (#70: 2/5).
- Turn p50s were 1.4–5.1 s. The two-turn cross-app scenarios run 2.5–4.5 s.

### Jobs (#76, 2026-11-08)

S4 moves to the gate tier. A queued playlist detaches as a job; the harness's
`Advance` finishes it in the fake world, and reading what arrived unprompted
runs one watcher pass (`settle`: time has passed). S4 now checks exactly one
note, "38 of 40", and both misses ("Lost Signal", "Paper Moons") by name.

Gate tier, 5 samples each, compiler on, planner prompt and schema unchanged:

| scenario | result |
|---|---|
| S1 | 5/5 |
| S2 | 5/5 |
| S2r | 5/5 |
| **S4** | **5/5** (was frontier, 0/5: nothing unprompted arrived) |
| S5 | 4/5 (the miss: answered without saying where it came from) |
| S7 | 5/5 |
| S8 | 5/5 |
| S9 | 5/5 |
| S16 | 5/5 |

- All five live S4 notes were the narrator's own words, every fact present
  (no code append needed).
- Claims were 40/40 backed.
- Plan-level evals on the same day: every group green (golden 94/105, cross
  35/35, ref 19/20, control 22/25, memory 8/10, knowledge 15/15, hard 2/5,
  self 9/15, holdout 47/50, long 40/40). The planner prompt and step kinds
  didn't change.

### Escalation: `think` on the local 26B (#78, 2026-11-08)

A `think` step runs as a background job on gemma-4-26b-a4b (thinking on,
`json_schema` on the answer, its own prompt and sampling). Its proposal posts
as the job's note and runs only on a yes. The session harness's `Settle` step
lets the think finish before the person says yes (in evals the watcher's
`sweep(force=True)` runs it to its end). The 12B planner prompt and step kinds
didn't change.

**S14 (frontier), "plan my week around the garage project", then "yes".** 10
samples per arm. The check counts garage tasks put on days this week by the
"yes" turn, new or rescheduled. Before this change it counted only new
tasks, and the 26B mostly reschedules the three garage tasks already there.

| arm | S14 | when the 12B planned a `think` |
|---|---|---|
| 12B only (`CONDUCTOR_THINK=0`: `think` falls back to the loop) | 0/10 | – |
| 26B think, first prompt, new-tasks-only check (the full suite run) | 0/5 | 0/5 |
| 26B think, first prompt | 7/10 | 7/9 |
| 26B think, shipped prompt | 9/10 | 9/9 |
| 26B think, shipped prompt, streamed call (shipped) | 6/10 | 6/6 |

- The 12B-only arm never writes a plan: it lists the garage tasks and asks,
  and the "yes" reaches nothing concrete.
- The first prompt let the 26B put the whole schedule in its summary with no
  steps (2/10). The shipped prompt says a schedule always has steps, one
  change per line.
- Every miss in the shipped arms is the 12B not escalating: it plans a plain
  ask (1/10 and 4/10 across two runs), and once its own ask wrote a junk task
  before the yes. Whenever it planned a think, S14 passed (15/15).
- Two of the 26B's 15 answers needed the one repair. All 15 were proposals
  of asks to tasks. None was claimed without a plan.
- Thinks took 39–116 s on the 26B (p50 ~55 s), including the swap. Turn 1
  answers at once ("Thinking it over…"), p50 2.7–9 s with the ask before it.
- The fake tasks app now takes a list of moves, one per line, as PCC's agent
  does. Two first-prompt passes came from it reading "schedule X to Monday"
  as new tasks.

**Foreground latency during an escalation.** `tests/test_think_contention.py`
(`CONDUCTOR_THINK_EVALS=1`) times one foreground planner call on the 12B (the
shipped planner prompt and schema, "what's due today?"). It starts the call
3 s after the 26B is loaded and generating a real S14 think. Arms run back to
back on the shared GPU:

| arm | n | p50 | p95 | max |
|---|---|---|---|---|
| warm, no think | 10 | 0.34 s | 0.34 s | 0.34 s |
| think yields (shipped: the turn cuts the streamed call off) | 10 | 5.57 s | 5.64 s | 5.64 s |
| think left running | 3 | 125 s | 128 s | 128 s |
| think yields, non-streamed call (first cut) | 10 | 124 s | 133 s | 133 s |

- llama-swap doesn't stop the 26B for a 12B request. It finishes the
  in-flight request first, so a foreground turn would wait out the whole
  thought.
- Closing a non-streamed call from another thread didn't unblock its read,
  so the first cut waited too. The think's call now streams and stops
  reading at the next chunk. The ~5.6 s that's left is the 26B → 12B swap
  plus a cold prefill.
- Another app's model call (PCC's agent, chess) isn't seen by conductor's
  turn tracker, so it still waits behind a running think. That needs the GPU
  priority gate (#56, later).

Same day, same branch:
- Plan-level evals: every group passed its gate (golden 91/105, cross 35/35,
  ref 15/20, control 20/25, memory 7/10, knowledge 15/15, hard 4/5, self
  9/15, holdout 48/50, long 36/40). The planner prompt, schema and validator
  are unchanged.
- Session gate tier: S1, S2, S2r, S4, S7, S8, S9 and S16 were 5/5, and S5
  was 4/5.
- Frontier: S3 5/5, S6 4/5, S10 5/5, S11 3/5, S12 0/5 (`unsupported`), S13
  0/5 (unchanged; in one sample the 12B planned a think for "anything I
  should know?", and the 26B answered with a summary) and S15 4/5.
- This full run went out with the first prompt and the new-tasks-only S14
  check. The S14 rows above are reruns of S14 alone.
- Claims were 111/111 backed.

### Memory over the ledger (#80, 2026-11-08)

`recall` runs on the interpreter over the ledger, and the entity registry puts
matches from earlier conversations in State. The planner prompt, the schema
and the validator are unchanged; the narrator prompt gained one line on
recalled results. Compiler on, gemma-4-12b.

Session evals, gate tier at 5 samples, frontier at 5 then 15:

| scenario | before | this change |
|---|---|---|
| S1, S2, S2r, S4, S7, S8, S9, S16 | 5/5 | 5/5 each |
| S5 | 4/5 (#76) | **5/5** |
| **S6** | 7/10 (#72), frontier | **19/20** (4/5, then 15/15): moves to the gate tier |
| S15 | 9/10 (#72), frontier | 14/20 (3/5, then 11/15): stays frontier |
| S3 | 10/10 | 5/5 |
| S10 | 10/10 | 5/5 |
| S11 | — | 4/5 |
| S12, S13, S14 | 0 | 0 (no `ask_chess`; `think` is #78) |

- S5 is now a `recall` plan, not a loop fallback. On the fresh world there's
  nothing on record, so all five said so ("I have no record of a decision
  regarding the garage shelves"). Code appends that line if the narrator
  guesses.
- S6 reads the registry's match in State (`"Odessa" (outside text), Tuesday …,
  you said "grab Odessa by Caribou"`). 19 of 20 asks to music named Odessa.
  The miss asked for "the Caribou song", and music asked which one.
- S15's planner input is unchanged by this PR: matches only come from other
  conversations, and a scripted test asserts it. Its misses are the planner's:
  one ask to music carried both the download and the task (4 times), and
  once it asked the person which album instead. Recorded, not treated as a
  regression, since the prompt and State it sees are byte-identical.
- Claims were 107/107 backed.

Plan-level evals, same prompt and validator, n=5 for every group:

| group | floor | this run |
|---|---|---|
| golden | 0.8 | 94/105 |
| cross | 0.8 | 35/35 |
| ref | 0.85 | 15/20 |
| control | 0.8 | 22/25 |
| memory | 0.8 | 7/10 |
| knowledge | 0.65 | 15/15 |
| hard | 0.5 | 5/5 |
| self | 0.55 | 10/15 |
| holdout | 0.85 | 47/50 |
| long | 0.7 | 38/40 |

Then `ref` and `memory` at n=20, against main's n=20 (#122's investigation):

| group | main | this change |
|---|---|---|
| ref | 61/80 | 64/80 (`ref-ambiguous` 5/20) |
| memory | 27/40 | 33/40 (`memory-caribou` 13/20, `s5-garage` 20/20) |

- Every group passed its gate.
- This PR doesn't move `memory-caribou` or `ref-ambiguous`. Plan evals feed
  a fixed State and nothing here changes what the planner sees, so those
  differences are sampling. `memory-caribou`'s misses are `look:music`
  (7/20), and `ref-ambiguous` asks tasks instead of asking the person
  (15/20). Both are planner-side follow-ups.

### The loop retired (#75, 2026-11-08)

Every turn is a plan now: no loop fallback. What used to fall back gets one
more repair round, then code's question or "I couldn't put that together";
a control step in a longer plan is split off and runs first; a think ends
the plan. The planner prompt, schema and validator are unchanged. Plan-level
evals now compile as the interpreter does (the extra round included) and
print `third-call` and `right-within-two` beside the old columns.
gemma-4-12b, shared GPU, n=5.

| group | floor | #80 | this change |
|---|---|---|---|
| golden | 0.8 | 94/105 | 90/105 |
| cross | 0.8 | 35/35 | 35/35 |
| ref | 0.85 | 15/20 | 17/20 |
| control | 0.8 | 22/25 | 20/25 |
| memory | 0.8 | 7/10 | 8/10 |
| knowledge | 0.65 | 15/15 | 14/15 |
| hard | 0.5 | 5/5 | 4/5 |
| self | 0.55 | 10/15 | 8/15 |
| holdout | 0.85 | 47/50 | 49/50 |
| long | 0.7 | 38/40 | 38/40 |

- Every group passed its gate; every plan was valid, and no sample needed the
  third call (`right-within-two` equals the score). The moves are sampling:
  the planner's input is byte-identical.
- `chess-resign` 1/5 (4 asks to music; chess#540 pending), `chess-analysis`
  0/5 (the glance), `refuse-playback` 2/5, `music-save` 5/5.

Session evals, gate tier at 5 samples, frontier at 5:

| scenario | #80 | this change |
|---|---|---|
| S1, S2, S2r, S5, S7, S8, S9, S16 | 5/5 | 5/5 each |
| S4 | 5/5 | 4/5 (the narrator called the playlist a created "job") |
| S6 (gate) | 19/20 | 4/5 (asked music without the title once) |
| S3, S10 | 5/5 | 5/5 |
| S11 | 4/5 | 5/5 |
| **S12** | 0 (`unsupported`: needed `ask_chess`) | **5/5**: it needs no `ask_chess` now; open and look run it |
| S13 | 0 | 0/5 (the brief leaves out the songs to sort) |
| S14 | 6/10 (#78) | 2/5: the 12B planned a think 2 of 5 times; both passed |
| S15 | 14/20 | 2/5, then 5/10: every plan was `ask:music \| ask:tasks`, and the misses dropped `{s1}` (#127) |

- None of the new paths fired live (no extra round, no split, no think cut);
  the misses are the planner's and the narrator's, as before.
- Claims were 108/109 backed (S4's "job").

### Memory you can see (#81, 2026-11-08)

"Remember …", "what do you know about me?" and "forget that" are asks to
`memory`, conductor's own app: one more `ask` enum value and one planner-prompt
line after the fleet's apps. Saved facts are in State. Compiler on,
gemma-4-12b, n=5.

Session evals, gate tier. M1 and M2 are new and gated:

| scenario | this change |
|---|---|
| S1, S2, S2r, S4, S5, S6, S7, S8, S9 | 5/5 each |
| S16 | 4/5 (the recap left out the task's name once, as in #72) |
| **M1** remember, then another day "what day do I sort music?" | **5/5** |
| **M2** remember, "forget that", then another day "where is the spare key?" | **5/5** |

- M1's later turn compiled to `recall` 5/5 and found the remember turn; the
  fact was also in State. M2's later turn declined or said it didn't know 5/5:
  nothing in recall or State still held it.
- Claims were 52/52 backed (memory's saves count as creates, its forgets as
  trashes). No infra misses.

Plan-level evals, the shipped line, against main the same day:

| group | floor | main | this change |
|---|---|---|---|
| golden | 0.8 | 96/105 | 88/105 |
| cross | 0.8 | — | 35/35 |
| ref | 0.85 | 17/20 | 17/20 |
| control | 0.8 | — | 20/25 |
| memory | 0.8 | 7/10 | 25/30 |
| knowledge | 0.65 | — | 15/15 |
| hard | 0.5 | — | 5/5 |
| self | 0.55 | 10/15 | 7/15 |
| holdout | 0.85 | 48/50 | 44/50 |
| long | 0.7 | — | 38/40 |

- Every group passed its gate. The new memory cases (remember, about-me,
  forget-that, forget-named) were 20/20.
- Regressions to watch: `ho-recall` ("what do you remember about the fence
  project?") went from 5/5 `recall` to 0/5, asking tasks; `local-capabilities`
  went from 4/5 to 0/5, filed as `knowledge` (the speaker's about-you check
  still answers it from the facts). Two other wordings of the memory line
  didn't fix `ho-recall`, and one lost "what do you know about me?".
- A fact used in another app's request ("a task for my music sorting day")
  was 0/5 on all three wordings: the planner passes the person's words
  through, as the prompt tells it to. That case was dropped; it's a planner
  follow-up.
