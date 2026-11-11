# Agent eval harness

Conductor's live evals run the **real** model (gemma-4-12b behind llama-swap)
against the **real** workspace manifests. Since #75 (2026-11-08) every turn
goes planner → validator → interpreter → narrator, and the eval gate is two
suites under the vendored Wilson gate (#59):

- **Plan-level evals** (`tests/test_plan_evals.py`, below): does the 12B
  compile the right plan? Every routing golden re-labelled as a plan, the
  S1–S16 scenarios and long plans, per group, at n=20 with each group's
  first-try rate beside its score; and a held-out set run only at milestones.
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

Landed as chess#540 (see the last section).

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
Phase 6 makes the route correct. Retired in the end (last section).

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
  scenarios and D1's held-out set, which every PR since 2026-11-06 has run
  and decided on, so it is the ordinary `mixed` group since #142;
- eight long-plan cases (6–10 steps when compiled one thing per step), added
  when the 5-step limit went. They check coverage, not exact shape.

The gate is per group, at 20 samples per case (`CONDUCTOR_PLAN_SAMPLES`,
default 20 since #142; n=5 read `chess-resign` anywhere from 5/20 to 15/20
on one day). Red takes evidence: the Wilson upper bound below the floor.
Floors sit a little under the spike's 12B v2 rates. Each group line also
prints its **first-try** rate: samples right on the planner's first reply,
before any repair or nudge. Record it beside the score: a validator nudge can
lift a score without the planner changing (`self` 29/60 to 60/60 in #134),
and only the first-try rate shows that.

```bash
cd backend
CONDUCTOR_PLAN_EVALS=1 CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace \
    .venv/bin/pytest tests/test_plan_evals.py -v -s
```

**The held-out set** (`tests/held_out_cases.py`, #142): about twenty cases
over every kind of request, new phrasings and situations. Run it only at
milestones, never in a PR's tuning loop, and never tune the prompt, the
validator, a manifest or its expectations against it. A miss is reported and
fixed on the ordinary groups; a case that gets tuned against moves to
`plan_cases.py` and a new one replaces it. It is gated as one pool at 0.8,
a floor fixed before its first run.

```bash
CONDUCTOR_HELDOUT_EVALS=1 CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace \
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
- `chess-resign` 1/5 (4 asks to music; fixed by chess#540), `chess-analysis`
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

### Recall trust, mixed asks, option entities (#127, #120, 2026-11-08)

- **Recall trust (#127.1).** Recall's record and cross-conversation hits say
  "as of then". A `knowledge`/`chat` answer to a question an app already
  answered gets a validator repair (`plan.recalled_answer`), which tells the
  planner to ask or look again. The new `stale` group (4 cases, floor 0.8)
  reproduces #122's leak. Same-thread "what's due today?" over its earlier
  answer read 0/20 on main. With this change it reads 20/20, and every pass
  was the repair: the first compile still answered `knowledge` all 20 times.
  The worked examples for this rule were A/B'd and dropped. One paragraph
  took `stale` to 10/10, but `chess-resign` and `refuse-playback` fell to
  0/10. A narrower example didn't bind (2/10).
- **Plan reds (#127.2).** The planner prompt got two clauses. In `ask_user`:
  "When State names two or more that fit their words, ask_user first, never
  an ask that picks one." In `recall`: "("when did I last...", "what did we
  decide..."). Never a look for these: a glance only shows now." An
  `ask_user` worked example also fixed `ref-ambiguous`, but it made
  `confirm-delete-task` search and then ask (18/20 invalid), so it was
  dropped. The validator now splits an ask that carries a second app's work
  ("… and add a task", "remind me") into one step per app.
- **#120.** `ask_user` options resolve in code (`app/ai/options.py`). Neither
  the prompt nor the schema changed for this.

Plan evals, n=20, gemma-4-12b, shared GPU. Main is `75ea878` with the same
cases:

| group | floor | main | this change |
|---|---|---|---|
| golden | 0.8 | 384/420 | 375/420 |
| cross | 0.8 | 140/140 | 140/140 |
| ref | 0.85 | 64/80 (`ref-ambiguous` 4/20) | **77/80** (20/20) |
| control | 0.8 | 81/100 | 84/100 |
| memory | 0.8 | 29/40 (`memory-caribou` 9/20) | **40/40** (20/20) |
| knowledge | 0.65 | 57/60 | 56/60 |
| hard | 0.5 | 13/20 | 14/20 |
| self | 0.55 | 34/60 | 41/60 |
| holdout | 0.85 | 184/200 | 186/200 |
| long | 0.7 | 152/160 | 151/160 |
| stale | 0.8 | 60/80 (same-thread 0/20) | **80/80** |

- Every group passed its gate. Every plan was valid, and nothing needed the
  third call.
- Golden's drop is `chess-resign`, which read 11/20 on main and 2/20 here.
  In a paired re-run in the same window it read 5/20 on main's prompt and
  4/20 on this one. Main read it anywhere from 5/20 to 15/20 today, so this
  is sampling; chess#540 fixed it.
- `knowledge-egg` read 16/20 in both runs here (4 `answer:self`), against
  20/20 twice for main's prompt. That's small but looks real.
- `s6-that-song` read 17/20 (main 20/20). In 3 asks music was told "that
  Caribou song" instead of Odessa.

Sessions run on the same code. S15 is paired at n=20:

| scenario | main | this change |
|---|---|---|
| S15 (frontier) | 12/20 | 15/20 |

- Main's S15 misses: turn 2 ran nothing (4), music only (2), and Honey was
  queued again (1).
- This change's S15 misses: Honey queued again (3), turn 2 ran nothing (1),
  and a task named "the other album" (1).
- Neither side showed the mixed music ask in these 40 samples.

Gate tier at 5: S1, S2, S2r, S5, S6, S7, S8, S9 and S16 were 5/5 each. S4
was 4/5 (playlist never queued, as in #75's run). Frontier at 5: S3, S10,
S11, S12 and S15 were 5/5, S14 4/5, and S13 0/5 (the brief, as before).
Claims were 110/110 backed.

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

### chess#540 lands; `refuse-playback` retired as a decline (#36, 2026-11-08)

**chess#540.** Chess's `open.description` now says "resetting, resigning or
otherwise changing a chess game ('the game' always means this one)", the draft
from #122 as written. `chess-resign` at n=20, three paired runs, real
manifests against a scratch fleet dir with the changed chess:

| run | before | after |
|---|---|---|
| 1 | 1/20 | 20/20 |
| 2 | 2/20 | 15/20 |
| 3 | 3/20 | 19/20 |

6/60 before, 54/60 after. The misses on both sides are asks to music. So the
change takes the golden from about 10% to about 90% but not to 100%; #122's
20/20 was the top of that range.

**#36.** No fleet app plays music, and music's own agent answers truthfully
one hop later, so routing "play some jazz music" to `ask music` is right.
`refuse-playback` no longer requires a decline: it accepts `ask music` or a
plain decline. #56's spike D1 saw the 12B ask music 5/5, but the shipped
prompt doesn't: at n=20 it asked music 1/20 in both arms, declined most of
the time and answered `knowledge` (which neither shape accepts) the rest.
Under the new shapes it reads 13, 18, 17/20 before and 17, 20, 18/20 after.
Requiring `ask music` alone would have made the case a standing 1/20 red.

Plan evals, n=5, gemma-4-12b, shared GPU, before against after (the first
run; `refuse-playback` counted under the new shapes in both):

| group | floor | before | after |
|---|---|---|---|
| golden | 0.8 | 93/105 | 93/105 |
| cross | 0.8 | 35/35 | 35/35 |
| ref | 0.85 | 19/20 | 20/20 |
| control | 0.8 | 20/25 | 20/25 |
| memory | 0.8 | 28/30 | 30/30 |
| knowledge | 0.65 | 15/15 | 15/15 |
| hard | 0.5 | 5/5 | 5/5 |
| self | 0.55 | 7/15 | 11/15 |
| holdout | 0.85 | 41/50 | 42/50 |
| long | 0.7 | 39/40 | 40/40 |
| stale | 0.8 | 20/20 | 20/20 |

- Every group passed its gate. `holdout` reads under its nominal floor on both
  sides (82%, 84%) but not with evidence (Wilson upper bounds 0.89, 0.91).
  `self` moved 7/15 to 11/15, which is sampling.
- Golden at n=20, three pairs: 369, 373, 373/420 before and 387, 378, 382/420
  after. The gain is `chess-resign` and `refuse-playback`.
- **Watch `local-capabilities`.** It answers `knowledge` instead of listing the
  apps, and read 15, 12, 10/20 before against 10, 7, 11/20 after (37/60 vs
  28/60, not significant: the spread inside one arm is as wide as the gap).
  Nothing in the new clause concerns it, and #81 already moved it from 4/5 to
  0/5 on main. It was not gated or fixed here.

Session evals, gate tier at 5, before and after (S1, S2, S2r, S4, S5, S6, S7,
S8, S9 and S16 were 5/5 in both):

| scenario | before | after |
|---|---|---|
| M1 | 8/10 (a first block of 3/5) | 5/5 |
| M2 | 5/5 | 4/5 |

- Every scenario ABOVE_FLOOR and STABLE; infra 0/12 both. Claims backed 53/55
  before (two queued) and 55/55 after. M1 and M2 touch no chess manifest, so
  those are sampling.

### Memory line regressions and saved facts in asks (#134, 2026-11-09)

- **Saved facts reach other apps in code.** When an ask's request shares two
  terms with a saved fact, the interpreter sends that fact after the
  request, quoted and marked as context (`facts.related`/`facts.attach`).
  `memory-used` is back. It passes when tasks gets the words that carry the
  fact, or when the plan writes Sunday itself. Session scenario M3 checks
  that the tasks app received the fact.
- **The memory line is unchanged.** Three rewordings were A/B'd, and none
  held up across the groups:
  - v1 says facts are in State, code sends them, and the rest is recall.
  - v2 is a shorter version of that.
  - v3 says a project or thing is recall.

  Each fixed one case and cost another. At n=10 with no repair, v3 took
  `ho-recall` to 10/10 but `memory-about-me` fell to 1/10. v2 took
  `memory-about-me` to 10/10 but `ho-recall` stayed at 0/10. With the
  repairs on at n=5, v1 and v2 lowered golden (90–93/105 against 96 for
  the shipped line). v2 also took `refuse-playback` to 1/5.
- **Three validator nudges.** Each gets one repair. If the repaired plan
  keeps its choice, it still runs.
  - `plan.remembered_topic`: "what do you remember about X?" goes to recall
    when no saved fact is about X.
  - `plan.looked_up_fact`: drops a recall or memory lookup of a fact State
    already lists before another app's ask.
  - `plan.about_you`: sends back a `knowledge` answer to "what can you
    do?", "what did you just do?" or "what do you know about me?".

Plan evals at n=20 on gemma-4-12b, after #130 and with chess#540's manifest.
"main" is this branch's compile path with main's behaviour: the shipped line
and none of the three nudges. It ran in the same window.

| group | floor | main | this change |
|---|---|---|---|
| golden | 0.8 | 372/420 | **395/420** |
| cross | 0.8 | 140/140 | 139/140 |
| ref | 0.85 | 78/80 | 78/80 |
| control | 0.8 | 80/100 | 80/100 |
| memory | 0.8 | 113/140 | **139/140** |
| knowledge | 0.65 | 60/60 | 58/60 |
| hard | 0.5 | 20/20 | 20/20 |
| self | 0.55 | 29/60 | **60/60** |
| stale | 0.8 | 80/80 | 80/80 |
| holdout | 0.85 | 165/200 | **190/200** |
| long | 0.7 | 151/160 | 150/160 |

| case | main | this change |
|---|---|---|
| `ho-recall` | 0/20 | 20/20 |
| `local-capabilities` | 8/20 | 19/20 |
| `memory-used` | 0/20 by the old criterion, 11/20 by the new one | 20/20 |
| `memory-about-me` | 2/20 | 19/20 |
| `status-last` / `status-now` | 0/20 / 9/20 | 20/20 / 20/20 |

- Every group passed its gate, and every plan was valid. Nothing needed the
  third call.
- `decline-news` read 18/20 (main 20/20). Its two misses were `knowledge`,
  and no nudge touches that case.
- `chess-resign` read 19/20 and `refuse-playback` 19/20 (main 17/20 and
  16/20). These runs came before #136 widened `refuse-playback` to accept
  ask music. Rescored on #136's criterion, both counts stay the same,
  because every miss was `knowledge`.

**M2 and "forget that".** The first session run had M2 at 2/5, a gate
red. The memory line was unchanged, and these plans touch no nudge or
attachment. Traces showed what happened: told "forget that", the planner
wrote the memory line's own example, "forget that I sort music on
Sundays". It did that 12 times in 20, and memory refused each time. The
target to forget now comes from the person's own words when the planner's
target isn't in them, as a fact to remember already does. M2 then read
20/20, and the planner still wrote the example 12 times in those 20.

Session evals, final code, gate tier at 5. M3 is new and gated:

| scenario | this change |
|---|---|
| S1, S2, S2r, S4, S5, S6, S8, S9, S16, M1, M2 | 5/5 each |
| S7 | 8/10 (first block 3/5, both misses "dentist due"; 5/5 in the first run) |
| **M3** remember, then another day "add a task for my music sorting day …" | **5/5** (5/5 in the first run too) |

- Frontier: S3, S11, S12, S14 and S15 were 5/5, S10 4/5 (one Stop reply
  quoted a call line), and S13 0/5 (the brief, as before).
- Claims were 128/129 backed (the S10 Stop line). No infra misses.

### One timeline and incognito (#82, 2026-11-09)

The planner prompt, step kinds, compiler and validator are unchanged. What
changed is the opening: "this conversation" is now the timeline since its
last quiet gap (3 h), and older turns come only as cross-day hits.

- **The harness.** `NewConversation` is now another day on the same
  timeline: `tests/eval_db.age` moves the record back a day, past the gap.
  `Say(..., incognito=True)` sends the switch, and a `Search` step runs the
  person's timeline search.
- **New scenarios**, in the gate after 10/10 across two live runs:
  - **T1**: a decision said in words, then a day later "what did we
    decide?". It can only come back through recall.
  - **T2**: an incognito "remember the gate code is 4417" saves no fact.
    The next turn's plan and recall never carry the code, and search
    doesn't find it.
  - **S1v**: S1 with the voice surface tag.

Plan evals at n=20 on gemma-4-12b, main (`826af54`) and this branch, run back
to back:

| group | floor | main | this change |
|---|---|---|---|
| golden | 0.8 | 393/420 | 390/420 |
| cross | 0.8 | 139/140 | 138/140 |
| ref | 0.85 | 78/80 | 77/80 |
| control | 0.8 | 80/100 | 80/100 |
| memory | 0.8 | 139/140 | 138/140 |
| knowledge | 0.65 | 60/60 | 59/60 |
| hard | 0.5 | 19/20 | 20/20 |
| self | 0.55 | 59/60 | 60/60 |
| holdout | 0.85 | 185/200 | 189/200 |
| long | 0.7 | 152/160 | 153/160 |
| stale | 0.8 | 80/80 | 80/80 |

Every group passed its gate on both, and every plan was valid. The compile
path is identical, so the differences are sampling noise.

Session evals, gate tier at 5 and frontier at 5:

| scenario | run 1 | run 2 |
|---|---|---|
| S1, S2, S2r, S4, S5, S6, S7, S8, S9, S16, M1, M2, M3 | 5/5 each | — |
| **S1v** by voice | **5/5** | **5/5** |
| **T1** recall across the gap | **5/5** | **5/5** |
| **T2** incognito | **5/5** | **5/5** |

- Frontier: S3, S10, S12 and S14 were 5/5. S11 was 4/5 (one reply didn't
  say the download can't be undone), then 5/5. S13 was 0/5 (the brief, as
  before).
- S15 was 2/5, then 3/5. Every miss was "Suddenly never queued": music
  filed the library instead. Main read 1/5 in the same window, so the
  timeline didn't cause it. Both turns sit in one window, where the opening
  is unchanged.
- Claims were 126/126 backed. No infra misses.

### A fresh held-out set, the n=20 gate and first-try rates (#142, 2026-11-09)

The planner prompt, step kinds, compiler, validator and manifests are
unchanged. D1's held-out set is now the `mixed` group, plan evals default to
n=20, and each group records its first-try rate (right on the planner's first
reply) beside the final score. Floors are unchanged; `control`'s waits for
the state-narrowed schema (#140).

Plan evals at n=20 on gemma-4-12b, this branch on `2af6371`:

| group | floor | score | first-try |
|---|---|---|---|
| golden | 0.8 | 394/420 | 377/420 |
| cross | 0.8 | 140/140 | 123/140 |
| ref | 0.85 | 76/80 | 76/80 |
| control | 0.8 | **80/100** | 80/100 |
| memory | 0.8 | 139/140 | 116/140 |
| knowledge | 0.65 | 60/60 | 60/60 |
| hard | 0.5 | 20/20 | 20/20 |
| self | 0.55 | 59/60 | **31/60** |
| mixed (was `holdout`) | 0.85 | 190/200 | 170/200 |
| long | 0.7 | 154/160 | 143/160 |
| stale | 0.8 | 80/80 | 60/80 |

- Every group passed its gate, and every plan was valid. Nothing needed the
  third call.
- `control` sits exactly on its floor, as before: `amend-job` read 0/20,
  an ask to music every time. `ho-amend` (now in `mixed`) read 10/20, its
  misses the same ask.
- The gap between score and first-try is the validator's repairs and #134's
  nudges: `self` is 31/60 on the planner alone, `stale` 60/80, `memory`
  116/140.
- Other misses: `chess-analysis` 0/20 (`look_chess`, as before) and
  `long-cleanup` 14/20 (each miss drops "sort my music").

**Held-out baseline** (`CONDUCTOR_HELDOUT_EVALS=1`, 24 cases, n=20, the same
code, run once): **438/480 (91%, Wilson 0.89–0.93), first-try 437/480**,
above its 0.8 pool floor.

| kind | score | first-try |
|---|---|---|
| routing | 74/80 | 74/80 |
| cross | 60/60 | 60/60 |
| ref | 59/60 | 59/60 |
| control | 79/80 | 79/80 |
| memory | 57/60 | 57/60 |
| knowledge | 47/60 | 47/60 |
| self | 22/40 | 22/40 |
| long | 40/40 | 39/40 |

Misses, reported and not tuned against:
- `h-self-apps` "which apps are you connected to?" 2/20: `knowledge` 18 times.
  `plan.about_you`'s nudge doesn't cover the phrasing.
- `h-know-al-dente` "what does al dente mean?" 7/20: `decline` 6, `self` 6,
  `chat` 1.
- `h-route-rematch` "set up a chess rematch" 14/20: the misses add an ask to
  tasks before the open.
- `h-memory-keep-in-mind` 17/20: all three misses wrote the memory line's
  own example, "remember I sort music on Sundays" (M2's pattern, #134).
- `h-ref-album-playlist` and `h-control-drop-album` 19/20 each.

### Planner temperature A/B (#150, 2026-11-09)

Sampling is now per role (`app/ai/providers/llamacpp.py`: planner, narrator,
speaker), every role on the model profile (temperature 1.0, top_p 0.95,
top_k 64), so request bytes are unchanged. The planner's temperature is the
one knob: `CONDUCTOR_PLANNER_TEMPERATURE` (unset = 1.0), so this A/B can be
rerun. The plan-level harness now sends `role="planner"`; its request was
checked per arm and carried exactly 1.0, 0, 0.3 and 0.6.

Plan evals, n=20 (`CONDUCTOR_PLAN_SAMPLES=20`), every group, gemma-4-12b,
`233fc83`, four runs on one server (one arm per GPU hold, not interleaved).
Each cell is passed/runs, then first-try (a valid plan with no repair, the
harness's `first-try`).

Read with the #142 section above: this A/B ran before #142 merged, so its
"first-try" is what the harness now prints as `first-valid` (a valid plan with
no repair, right or not), not #142's first-try (right on the first reply), and
`holdout` is the group now called `mixed`.

| group | 1.0 | 0 | 0.3 | 0.6 |
|---|---|---|---|---|
| golden | 395/420 (94%), 416/420 (99%) | 400/420 (95%), 400/420 (95%) | 399/420 (95%), 410/420 (98%) | 397/420 (95%), 417/420 (99%) |
| cross | 138/140 (99%), 128/140 (91%) | 140/140 (100%), 120/140 (86%) | 140/140 (100%), 124/140 (89%) | 140/140 (100%), 123/140 (88%) |
| ref | 73/80 (91%), 80/80 (100%) | 80/80 (100%), 80/80 (100%) | 80/80 (100%), 80/80 (100%) | 78/80 (98%), 80/80 (100%) |
| control | 80/100 (80%), 100/100 (100%) | 80/100 (80%), 100/100 (100%) | 80/100 (80%), 100/100 (100%) | 80/100 (80%), 100/100 (100%) |
| memory | 138/140 (99%), 115/140 (82%) | 140/140 (100%), 120/140 (86%) | 140/140 (100%), 120/140 (86%) | 140/140 (100%), 118/140 (84%) |
| knowledge | 60/60 (100%), 60/60 (100%) | 60/60 (100%), 60/60 (100%) | 60/60 (100%), 60/60 (100%) | 60/60 (100%), 60/60 (100%) |
| hard | 18/20 (90%), 20/20 (100%) | 20/20 (100%), 20/20 (100%) | 20/20 (100%), 20/20 (100%) | 20/20 (100%), 20/20 (100%) |
| self | 59/60 (98%), 37/60 (62%) | 60/60 (100%), 30/60 (50%) | 60/60 (100%), 34/60 (57%) | 60/60 (100%), 37/60 (62%) |
| holdout | 187/200 (94%), 180/200 (90%) | 180/200 (90%), 180/200 (90%) | 185/200 (92%), 180/200 (90%) | 181/200 (90%), 180/200 (90%) |
| long | 155/160 (97%), 147/160 (92%) | 160/160 (100%), 160/160 (100%) | 156/160 (98%), 146/160 (91%) | 155/160 (97%), 151/160 (94%) |
| stale | 80/80 (100%), 60/80 (75%) | 80/80 (100%), 60/80 (75%) | 80/80 (100%), 60/80 (75%) | 80/80 (100%), 60/80 (75%) |
| **all** | 1383/1460 (95%), 1343/1460 (92%) | 1400/1460 (96%), 1330/1460 (91%) | 1400/1460 (96%), 1334/1460 (91%) | 1391/1460 (95%), 1346/1460 (92%) |

**The winner rule did not fire; the planner stays at 1.0.** The rule: an arm
wins only if no group is Wilson-significantly below the 1.0 arm (read as the
arm's upper bound below the 1.0 arm's lower bound, `evalstats.wilson_interval`
at 1.645, on both pass and first-try) and it improves both `control` and the
overall first-try rate.

- 0: `golden` first-try is significantly below (400/420 against 416/420), and
  overall first-try is lower (91.1% against 92.0%).
- 0.3: no group is significantly below, but overall first-try is lower
  (91.4%).
- 0.6: no group is significantly below and overall first-try is up by three
  plans in 1460 (92.2%), but `control` did not move.
- `control` cannot improve on first-try (100/100 at 1.0), and its pass count
  was 80/100 in all four arms, so no reading of the rule lets it improve.

What the table shows besides:

- `control` is `amend-job`, 0/20 in every arm (the planner files `ask:music`
  where the case expects an amend), so the group sits at its 0.8 floor at any
  temperature. It is a case or prompt question, not sampling.
- Lower temperature raised passes a little (0 and 0.3: 1400/1460 against
  1383/1460, mostly `ref`, `golden`, `cross` and `hard`; `holdout` fell at 0),
  and moved no first-try figure for the better. `ho-amend` is the other
  steady miss: 13/20 wrong at 1.0, 20/20 at 0.
- At 0 the 20 samples of a case are near-identical, so its counts are close
  to one sample per case: they overstate the evidence.
- Noise at 1.0 is real: `golden`'s 395/420 here against 393/420 and 390/420
  in the last two runs, the same code.

Session evals were not run: no default changed.

### A remembered fact comes from the person's words; one memory-line A/B (#159, #138, 2026-11-10)

The planner prompt, step kinds, compiler and validator are unchanged.
`facts.own_words` changed: the fact is cut from the person's own
"remember", "keep in mind" or "don't forget" anywhere in what they said,
up to the end of that sentence or the next ask. The planner's version
stands only when it sits word for word inside that cut. Before this, code
only read a "remember" at the start of the person's words. If the plan
copied the prompt's example, "by the way, keep in mind that …", "…, keep
that in mind" and "add a task … and keep in mind that …" saved nothing,
and "keep in mind that X and add a task …" saved the second ask's words
along with the fact. A plain "keep in mind that X" was already right.

New `memory` cases, fresh wordings with facts that aren't the prompt's
example: `memory-keep-in-mind`, `memory-remember-place`,
`memory-dont-forget`, `memory-remember-and-ask` (a second ask in the same
message) and `memory-forget-just-saved` ("forget that" right after saving,
as M2 says it). The remember cases check the fact code would save
(`M.saves`), not the request text.

Plan evals at n=20 on gemma-4-12b, this branch on `8bf298a`. The main
column is the #142 run above (same prompt and compile path):

| group | floor | main | first-try | this change | first-try |
|---|---|---|---|---|---|
| golden | 0.8 | 394/420 | 377/420 | 385/420 | 378/420 |
| cross | 0.8 | 140/140 | 123/140 | 139/140 | 120/140 |
| ref | 0.85 | 76/80 | 76/80 | 75/80 | 75/80 |
| control | 0.8 | 80/100 | 80/100 | 80/100 | 80/100 |
| memory | 0.8 | 139/140 | 116/140 | 238/240 (12 cases) | 212/240 |
| knowledge | 0.65 | 60/60 | 60/60 | 60/60 | 60/60 |
| hard | 0.5 | 20/20 | 20/20 | 18/20 | 18/20 |
| self | 0.55 | 59/60 | 31/60 | 58/60 | 37/60 |
| mixed | 0.85 | 190/200 | 170/200 | 185/200 | 165/200 |
| long | 0.7 | 154/160 | 143/160 | 156/160 | 150/160 |
| stale | 0.8 | 80/80 | 60/80 | 80/80 | 60/80 |

- Every group passed its gate, and every plan was valid. The steady misses
  are as before: `amend-job` 0/20, `ho-amend` 5/20 and `chess-analysis`
  0/20 (`look:chess`).
- The new cases were 20/20 each. None of the 100 remember samples copied the
  example, so main's `own_words` saves the same facts on these plans. The
  fix shows in the unit tests, on phrasings the plan cases didn't produce.
- The copying #138 describes shows in `memory-forget-just-saved`: 17/20
  plans wrote "forget that I sort music on Sundays", and `memory-forget-that`
  3/20. Both still pass, because what to forget comes from the person's
  words (#134).

Session evals, gate tier at 5 and frontier at 5: every gate scenario passed.
S6 and M3 were 4/5, the rest 5/5 (M1 and M2 5/5). The S6 miss never filed
Odessa. In the M3 miss, the planner wrote "on Sunday" into the ask itself,
so no fact went along; the fact saved was right. Frontier: S3, S10, S11 and
S12 were 5/5, S14 4/5, S13 0/5 (the brief, as before) and S15 1/5. Claims
were 117/117 backed. No infra misses.

**One memory-line A/B, dropped.** The examples became placeholders:
`"remember <the fact, as they said it>"` and `"forget that <the fact>"`.
This was measured at n=20 against the shipped line, same code, one arm per
GPU hold:

| group | shipped | first-try | placeholders | first-try |
|---|---|---|---|---|
| golden | 385/420 | 378/420 | 396/420 | 389/420 |
| memory | 238/240 | 212/240 | 240/240 | 227/240 |
| self | **58/60** | **37/60** | **51/60** | **31/60** |
| plans copying the example | 20/240 | | 0/240 | |

- The rewording took copying to zero, with no literal placeholder in any
  plan. It also lifted `memory-about-me` first-try from 2/20 to 20/20.
- `self` fell: `status-last` gave `answer:chat` 9 times in 20 against 2.
  The rule was at least as good in every group, so the line stays as
  shipped. #134 also found the line fragile.

### Questions about conductor itself (#158, #140's guard; 2026-11-10)

The planner prompt, step kinds and compiler are unchanged. `plan.about_you`'s
nudge (a `knowledge` answer sent back once) now also catches "what did you
(just) say" and "which/what apps are you connected to / can you reach / do
you have (access to)". The speaker's facts carry conductor's last reply
(`ledger.self_report`: the newest turn's only, clipped, never an incognito or
forgotten one), and its about-you wording says "just did or said".

New plan cases: `self-said-just-now` ("what did you say just now?", after an
exchange) and `self-apps-reach` ("what apps can you reach from here?") in
`self`; and four empty-state `control` cases kept from #140's narrowed-schema
attempt, which was not shipped (`yes-nothing-waiting`,
`cancel-nothing-running`, `amend-nothing-waiting`, `undo-nothing-done`; a
control step, chat or self all count, since none acts).

Plan evals at n=20 on gemma-4-12b, back to back with the same cases: main's
code (`8bf298a`) and this change.

| group | floor | main | first-try | this change | first-try |
|---|---|---|---|---|---|
| golden | 0.8 | 392/420 | 384/420 | 395/420 | 386/420 |
| cross | 0.8 | 139/140 | 122/140 | 139/140 | 126/140 |
| ref | 0.85 | 74/80 | 74/80 | 79/80 | 79/80 |
| control (9 cases) | 0.8 | 155/180 | 133/180 | 150/180 | 130/180 |
| memory | 0.8 | 139/140 | 120/140 | 140/140 | 118/140 |
| knowledge | 0.65 | 59/60 | 59/60 | 60/60 | 60/60 |
| hard | 0.5 | 19/20 | 19/20 | 19/20 | 19/20 |
| self (5 cases) | 0.55 | 71/100 | 43/100 | **99/100** | 43/100 |
| mixed | 0.85 | 191/200 | 171/200 | 182/200 | 162/200 |
| long | 0.7 | 152/160 | 143/160 | 150/160 | 144/160 |
| stale | 0.8 | 80/80 | 60/80 | 80/80 | 60/80 |

- `self-said-just-now` went from 8/20 to 20/20 and `self-apps-reach` from
  4/20 to 20/20; on main every miss but one was `knowledge`. The first-try
  rate is unchanged (7/20 and 2/20): the planner still files them as
  knowledge, and the nudge's one repair turns them into `self`.
- `control`: the existing five cases were 80/100 on both (`amend-job` 0/20,
  as before). The empty-state cases were 75/80 on main and 70/80 here.
  `amend-nothing-waiting` read 17/20 and then 13/20; its misses are asks to
  tasks, knowledge, one decline and one cancel. The nudge can't match those
  words, so its compile path didn't change. `yes-nothing-waiting` is never
  right on the first try: the validator refuses a confirm with nothing
  waiting, and the repair answers chat or self.
- `mixed` fell by 9, all of it `ho-amend` ("make it the first twenty
  instead", 11/20 and then 3/20, its misses the usual ask to music). It
  doesn't match the nudge either, and it read 7/20 and 10/20 in earlier
  runs. No group is Wilson-significantly below main (#150's rule).

Session evals, gate tier at 5 and frontier at 5, this change:

| scenario | result |
|---|---|
| S1, S1v, S2, S2r, S4, S5, S8, S9, S16, M2, M3, T1, T2 | 5/5 each |
| S7, M1 | 4/5 each |
| **S6** | **2/5**, then 19/20; main's code 17/20 in the same window |

- S6's misses are turn 2's ask to music without the song ("Odessa is in
  None"), compiled with no repair. Nothing this change touches runs on that
  path, and at 20 runs each it reads 19/20 here against 17/20 on main.
- Frontier: S3, S10, S11 and S12 5/5, S14 4/5, S15 3/5 (the known "Suddenly
  never queued" misses), S13 0/5 (the brief, as before).
- Claims were 116/116 backed. One infra miss (a llama-swap timeout) was
  re-taken and not scored.

### A repeated question, without its earlier request (#141, 2026-11-10)

Asked "what's due today?" again, the planner's first compile answered
`knowledge` every time, and every pass was `plan.recalled_answer`'s repair (a
second planner call). The trigger is the earlier request in the same words,
not the app's answer: stripping the answers (tried 2026-11-09, not shipped)
moved nothing, and a record of a different question read the app 20/20. Now
the planner's message (`compiler.user_message`, through
`turn_context.for_planner`) leaves out an earlier request whose terms equal
the person's words, the terms `recalled_answer` matches on: its record lines,
an earlier conversation's hit, the earlier requests, a registry match it
touched, and the last exchange when it was that request. The validator still
reads the whole opening, so the repair stays the backstop. The planner prompt
is unchanged. Of the 98 plan and held-out cases, only four `stale` ones get a
different planner message; every other case's input is byte-identical. New
case: `stale-due-earlier-in-thread` (the same question two exchanges back).

Plan evals at n=20 on gemma-4-12b, this branch on `8bf298a`. `stale` was
paired with main on the same cases; the other groups are read against the
#142 table above:

| group | floor | score | first-try |
|---|---|---|---|
| golden | 0.8 | 397/420 | 389/420 |
| cross | 0.8 | 139/140 | 126/140 |
| ref | 0.85 | 76/80 | 76/80 |
| control | 0.8 | **80/100** | 80/100 |
| memory | 0.8 | 140/140 | 120/140 |
| knowledge | 0.65 | 58/60 | 58/60 |
| hard | 0.5 | 19/20 | 19/20 |
| self | 0.55 | 57/60 | 35/60 |
| mixed | 0.85 | 188/200 | 168/200 |
| long | 0.7 | 156/160 | 147/160 |
| stale | 0.8 | 100/100 | **100/100** (main: 60/100) |

- `stale-due-same-thread` and `stale-due-earlier-in-thread`: first-try 0/20
  each on main (`knowledge` 40 of 40, repaired), 20/20 each here, one planner
  call per sample. The other three `stale` cases read 20/20 first-try on both.
- No group moved significantly; the small dips (`knowledge` 58/60, `self`
  57/60) are on cases whose input is unchanged. `control` sits on its floor
  as before (`amend-job` 0/20).

Session evals, gate tier at 5 and the frontier at 5, this branch: every gate
scenario 5/5 but **M3, 5/10, below its floor**. M3's planner input is
unchanged by this branch (neither turn's words share the other's terms), and
its misses are main's: the plan writes "Sunday" itself, so `facts.related`
attaches nothing, or the narrator claims a save. Paired at n=20: M3 **15/20
here, 16/20 on main**, the same two failure modes. S15 paired at n=20: **14/20
here, 11/20 on main** (its input is unchanged too). Frontier: S3, S10, S11,
S12 5/5, S14 4/5, S13 0/5 (the brief misses the counts, as before).

**Combined run (2026-11-10), this branch on top of #159, #158 and #140's cases**,
the three changes gated separately before: every plan group above its floor
(golden 391/420, cross 139/140, ref 78/80, control 156/180, memory 239/240,
knowledge 60/60, hard 20/20, self 99/100 first-try 47, mixed 185/200, long
153/160, stale 100/100 first-try 100). Session gate all 5/5 except S16 4/5,
M3 and S6 both 5/5; frontier S3, S10, S11, S12 5/5, S14 and S15 4/5, S13 0/5
(as before); claims 124/125 backed.

### A leading `about` field (#151, 2026-11-11)

Each planner reply now opens with `about`, what the message is about, before
the plan's shape: `an app`, `conductor itself`, `general knowledge`, `the
outside world`, `the waiting question`, `a running job` or `small talk`.
`a running job` and `the waiting question` are in the schema's enum only when
State lists a running job or something waits on the person
(`plan.offered_about`). The prompt gains one block describing the
values, and its JSON examples carry the field. `plan.disagrees` is one more
nudge (one repair, then the plan runs): it sends back a plan its own `about`
contradicts, such as `general knowledge` with an ask, `an app` with a direct
answer, `a running job` with a new ask and no cancel or amend, or `conductor
itself` with only a recall or a think. The field isn't stored with the plan;
the eval report records it per reply.

Plan evals at n=20 on gemma-4-12b, main (`70476ea`) and three variants, one
arm per GPU hold in this order, each cell passed then first-try:

| group | floor | main | v1 | v2 | **v3 (shipped)** |
|---|---|---|---|---|---|
| golden | 0.8 | 392, 385 /420 | 401, 400 | 400, 400 | 400, 400 |
| cross | 0.8 | 138, 127 /140 | 136, **112** | 139, 122 | 138, 114 |
| ref | 0.85 | 76, 76 /80 | **64, 64** | 74, 74 | 76, 76 |
| control (9 cases) | 0.8 | 154, 133 /180 | 168, 159 | 152, 152 | 143, 141 |
| memory | 0.8 | 239, 220 /240 | 236, 228 | 237, 237 | 239, 238 |
| knowledge | 0.65 | 60, 60 /60 | 60, 60 | 60, 60 | 60, 60 |
| hard | 0.5 | 20, 20 /20 | **7, 7** | **7, 7** | 19, 18 |
| self | 0.55 | 97, 46 /100 | **80**, 80 | **80**, 80 | 90, 90 |
| mixed | 0.85 | 183, 163 /200 | 199, 180 | 200, 180 | 198, 178 |
| long | 0.7 | 153, 146 /160 | 155, **129** | 153, 137 | 155, 140 |
| stale | 0.8 | 100, 100 /100 | 100, 99 | 100, 100 | 100, 100 |
| **all** | | 1612, 1476 /1700 | 1606, 1518 | 1602, 1549 | 1618, 1555 |

Bold is Wilson-significantly below main by #150's rule (the arm's upper bound
under main's lower bound, 1.645), on the score or the first-try rate.

- **v1**: all seven values always offered. `a running job` drew "undo that",
  "cancel that" and "push the dennis thing to friday" (an amend of a task, so
  `s7-dennis` 6/20), "yes" with nothing waiting anchored on `the waiting
  question` (12 of 20 invalid), "plan my week" became `an app` and an ask to
  tasks (`s14-plan-week` 7/20), and "what did you say just now?" became a
  recall (`self-said-just-now` 0/20).
- **v2**: the job and waiting values only when State has one, and `an app`
  no longer says "plan its steps". `ref`, `cross` and `long` came back;
  `s14-plan-week` (7/20) and `self-said-just-now` (0/20, recall 14, think 6)
  did not.
- **v3**: `an app` says planning around it ends in a think, `conductor
  itself` says what you did or said is a self answer and what you know about
  them is memory's, `small talk` covers a yes or no with nothing waiting, and
  `conductor itself` with only a recall or think goes back. No group is below
  main by the rule, on either rate, so it ships.

What moved, v3 against main:

| case | main | v3 (first-try) |
|---|---|---|
| `amend-job` | 0/20 | 16/20 (16) |
| `ho-amend` | 3/20 | 20/20 (20) |
| `status-now`, `status-last`, `self-apps-reach`, `local-capabilities` first-try | 7, 7, 5, 12 /20 | 20 each |
| `memory-about-me`, `memory-used` first-try | 9, 11 /20 | 20 each |
| `chess-resign`, `refuse-playback`, `s6-that-song` | 17 each | 20 each |
| `self-said-just-now` | 18/20 (7) | **10/20** (10): recall 9 times, kept through the repair |
| `amend-nothing-waiting` | 16/20 (13) | **5/20**: ask tasks "make it the first ten instead" 12 times |
| `yes-nothing-waiting` | 18/20 (0) | **2/20**: confirm, then decline after the repair, 18 times |
| `s3-album-task`, `cross-playlist-task` first-try | 14, 15 /20 | 7, 9 (the lost-ref repair; 20/20 each) |

- Read these with care. The `self` score (97 to 90) and `cross` first-try
  (127 to 114) would be significant on a pooled two-proportion test (z −2.0
  and −2.2), just not by #150's rule; the empty-state guards (#140's) lost
  most: with no waiting or job value to pick, "yes" and "make it the first
  ten" land on `the outside world` or `an app`.
- The other nudges barely fire. On main's first replies `plan.about_you`
  (#134, #158) fired 67 times in 1700 and `plan.looked_up_fact` 9; on v3's,
  neither fired once. `plan.disagrees` fired 11 times (9 of them
  `self-said-just-now`, kept as recall). `remembered_topic` still catches
  `ho-recall` 20 times in both.

Session evals were not run with this change; they are still owed before it
merges.

### A running job the words narrow is amended (#160, 2026-11-11)

`amend-job` ("actually just grab the first ten songs", a music download
running) was 0/20 on main, an ask to music every time. #151's `about` took
it to 16/20 on its own; its four misses were `an app` and an ask to music (3)
and a cancel then an ask (1). Now `plan.narrows_job` is one more nudge: when
State lists a running job for the app an ask goes to, the person's words
narrow something under way ("just", "only", "instead", "the first ten",
"stop after …") and no step amends, the plan goes back once to amend that job
(target `app:job:ref`). The jobs contract has no amend for a running job, so
the amend then says it can't change it; that is expected. Of the plan cases,
only `amend-job` and `ho-amend` meet the rule; every other case compiles as
on #151.

Plan evals at n=20 on gemma-4-12b, this branch on #151's commit, against
#151's v3 run and main's above:

| group | floor | main | #151 | this change | first-try |
|---|---|---|---|---|---|
| golden | 0.8 | 392/420 | 400/420 | 401/420 | 401/420 |
| cross | 0.8 | 138/140 | 138/140 | 136/140 | 117/140 |
| ref | 0.85 | 76/80 | 76/80 | 75/80 | 75/80 |
| control | 0.8 | 154/180 | 143/180 | 148/180 | 142/180 |
| memory | 0.8 | 239/240 | 239/240 | 239/240 | 234/240 |
| knowledge | 0.65 | 60/60 | 60/60 | 60/60 | 60/60 |
| hard | 0.5 | 20/20 | 19/20 | 19/20 | 19/20 |
| self | 0.55 | 97/100 | 90/100 | 91/100 | 89/100 |
| mixed | 0.85 | 183/200 | 198/200 | 199/200 | 179/200 |
| long | 0.7 | 153/160 | 155/160 | 159/160 | 141/160 |
| stale | 0.8 | 100/100 | 100/100 | 100/100 | 100/100 |

- `amend-job` **20/20** (18 first-try; 2 were the repair, both from `an app`
  and an ask to music). `ho-amend` 20/20 as on #151.
- No group is Wilson-significantly below main or #151 (#150's rule).
- **The `control` floor stays 0.8.** Its first five cases are now 100/100
  (Wilson 0.97–1.00), but the group is 148/180 (0.82, Wilson 0.77–0.86):
  #140's empty-state guards fell with #151 (`yes-nothing-waiting` 4/20,
  `amend-nothing-waiting` 4/20, against 18 and 16 on main), so the group's
  rate sits just above 0.8 and leaves nothing to raise it into.

## Fleet smoke (#143)

`backend/tests/test_fleet_smoke.py`: S2, S3, S9 and S15 through conductor
against the **real** PCC and music agents, the seam session evals fake. Opt-in,
never a CI or PR gate; measured first, gated later if it holds steady.

```bash
cd backend
CONDUCTOR_FLEET_SMOKE=1 .venv/bin/pytest tests/test_fleet_smoke.py -v -s
```

The harness (`tests/fleet_harness.py`) starts each app's backend from its
sibling checkout with that checkout's own `.venv` (code used read-only), on a
free loopback port (never 8000/8100/8200/8300/8500 or a dev port), a scratch
SQLite migrated with the app's `alembic upgrade head`, and scratch library and
trace files, all in one pytest temp folder. Each sample restarts both apps on
fresh databases; PCC is seeded through its REST API (fake_world's projects and
tasks). Conductor discovers them the normal way: the siblings' `app.yaml`
copied into `fleet_manifest_dir` with `upstream` rewritten (chess pointed at a
dead port, so the planner prompt is production's). Its delegate client, glance
and job cancel are wrapped to refuse any other upstream.

Nothing leaves loopback: music's yt-dlp is a fake (`YT_DLP_BIN`, music's own
test seam) that writes a silent mp3 and logs its argv; MusicBrainz and the
likes poller are off; Genius, Spotify and SoundCloud are unset; every proxy
variable points at a dead port. The subprocesses run from the temp folder with
a minimal environment, so no checkout's `.env` is read. A watcher samples the
apps' sockets (`ss`) and open files (`/proc`) throughout and fails the module
on a peer off loopback or a database, audio or trace file outside the temp
folder (checked against a positive control). Checks read each app's end state
through its own API, and the replies are held to the apps' real receipts.

`CONDUCTOR_FLEET_RUNS` (3) samples per scenario; `CONDUCTOR_FLEET_REPORT=path`
appends a JSON line per sample; `CONDUCTOR_FLEET_COST=1` adds #155's
measurement below.

### First measurement, 2026-11-10, gemma-4-12b, `8bf298a`

A measurement, not a floor. 3 samples each; watch: 492 samples, 0 sockets off
loopback, 0 files outside the temp folder, 8 fake yt-dlp calls.

| scenario | pass | turn time |
|---|---|---|
| S2 add a task | 3/3 | 3.6–7.3 s |
| S3 album, then a task named for it | 0/3 | 4.1–5.9 s |
| S9 wipe Kitchen, yes | 3/3 | 1.8–2.5 s per turn |
| S15 S3, then "do the same for their other album" | 0/3 | 3.8–6.2 s per turn |

What the apps received (conductor's `ask` words, verbatim):

- S2: tasks heard "add a task to water the plants tomorrow" and called
  `create_task {title: "water the plants", due_date: tomorrow}` every time.
- S9: tasks heard "wipe the kitchen project" (`destructive: true`), asked
  "Trash project "Kitchen" and its 3 tasks?" itself, and the `confirm` step
  trashed it. Conductor's gate stayed out of the way, as `confirms: true` says.
- S3: music heard "download the new Caribou album" and queued
  `download_playlist {query: "Caribou album"}` (2/3; once it answered that it
  needed a link and did nothing). The tasks step, "add a task to sort {s1}
  this weekend" (once "… sort the album {s1} …"), never ran: "s1 gave no name
  to pass on".
- S15 turn 2: twice `recall` ("what was the other album the person
  mentioned?", "what other album did they mention?") then "download the new
  {s1} album" / "download the album {s1}", not run; once the plan dropped the
  music step and tasks heard "add a task to sort the other Caribou album this
  weekend" (created, due Sunday, but not "Suddenly").

Why S3 and S15 fail against the real music, where the fake passes:

- **Music's receipt names no album.** `download_playlist`'s record touches
  only a `download_job` ref, titled with the query the 12B wrote ("Caribou
  album"). `dataflow.from_record` skips job refs by design, so `{s1}` is
  empty. The fake world adds an `album` ref titled "Honey"; the real app
  never resolves an album name (no catalog, MusicBrainz is tagging-only).
- **The fake knows Caribou's discography; music doesn't.** "Their other
  album" has nothing to resolve against in any real app.
- **Music's job runs `ytsearch1:Caribou album` with `--yes-playlist`**: real
  yt-dlp would fetch one search hit, not an album (a music finding).
- The narrator blames the person: "I could not add the task … because a
  specific name for the task was not provided."

S2 and S9 behave as the fake predicts; the writes, receipts, confirm token
and replies all line up.

### A delegated write end to end (#155, 2026-11-10)

Same run, all three traced (`CONDUCTOR_TRACE_PATH`, `PCC_TRACE_PATH`,
`MUSIC_TRACE_PATH`) and every model call through the harness's meter, a
pass-through to llama-swap recording llama-server's own `usage`/`timings`
(music's trace keeps no cache numbers). Neither app records `X-Agent-Turn`,
so a hop is joined to its app run by thread (`conversation_id` = the hop's
`thread_id`) inside the turn's window. 3 runs each, back to back, fresh
databases and a new delegate thread per run; the 12B stayed loaded, so no
run includes a model load. "Cold" below is the first run after a different
utterance (the planner's prefix differs past the state block).

"add a task to water the plants tomorrow" (4 calls every run; turn 3694, 4300,
4217 ms, median 4217):

| call | ms | prompt | cached | prefill ms |
|---|---|---|---|---|
| conductor planner | 689 / 1314 / 1282 | 1860 | 1833 / 1652 / 1652 | 109 / 782 / 760 |
| PCC tool turn | 1484 / 1482 / 1480 | 3977 | 3614 | ~910 |
| PCC closing turn | 559 / 550 / 553 | 4067 | 3973 | ~168 |
| conductor narrator | 885 / 879 / 820 | 772 | 471 | ~472 |

"download the song Odessa by Caribou" (yt-dlp faked, so this is the model
calls; 4 calls every run; turn 3490, 2443, 2315 ms, median 2443):

| call | ms | prompt | cached | prefill ms |
|---|---|---|---|---|
| conductor planner | 1358 / 627 / 611 | 1859 | 1652 / 1858 / 1858 | 778 / 49 / 32 |
| music tool turn | 877 / 290 / 290 | 3147 | 3052 / 3143 / 3143 | 632 / 45 / 46 |
| music closing turn | 461 / 683 / 571 | 3223 | 3143 | ~156 |
| conductor narrator | 708 / 762 / 762 | 769 | 471 | ~472 |

Read:

- Four sequential model calls per delegated write, as the issue counted, and
  they are nearly the whole turn: 3.6–4.2 s of 3.7–4.3 s for a task, 2.4–3.4 s
  of 2.4–3.5 s for a song.
- The narrator's prefix stops matching at token 471 on every call, so ~300
  tokens are re-read each turn (~470 ms, half its time).
- PCC's tool turn re-reads ~360 tokens every run (~910 ms prefill) even when
  warm; its closing turn rides the tool turn's cache.
- The planner's prefix is cached to ~1650 of 1860 after a different
  utterance (~770 ms prefill) and fully when the words repeat (~40 ms).

## S3 and S15 after the fake matched real music (#163, 2026-11-11)

**Every S3 and S15 score above measured the fake.** The baseline table, #72's
10/10 and 9/10, the 5/5 and n=20 pairs and the "Suddenly" misses are
conductor against `fake_world.py`'s music, which knew more than real music:
an `album` ref with an id of its own, "Their other album" resolved against a
discography (`year >= 2015`, minus what was already queued). The fleet smoke
above is the same two scenarios against the real app: 0/3 each.

The fake now returns what real music will once ryanskarnulis/music#96 lands:

- One album resolved ("download the new Caribou album", a title, or an
  artist with one album): the record touches an `album` entity ref
  (`kind: "album"`, `id`: a MusicBrainz release-group id, `title`) beside the
  `download_job` ref.
- Not resolved to one ("the Caribou album", "the other album by Caribou"):
  nothing written, nothing touched, and the reply names the candidates and
  asks for one, or a link. Music keeps no idea which album is "the other".

So S15's turn 2 can no longer end in "Suddenly": music names the candidates,
the tasks step has no `{s1}` and doesn't run, and the check is that nothing
was written and the reply asks which one. S3's check also requires the queued
download to name its album. The old scores are not comparable with the first
run after this: both rows restart from it.

**Not run.** The model-backed evals (plan, session, fleet smoke) were left
unrun here; the GPU was reserved. The eval gate runs once after the batch
that changes the narrator and the glance view merges (#161, #163):

- the glance view keeps a ref's other scalar fields and states its age, so
  every `look` receipt changes;
- the narrator's user message is reordered (the surface's length line, the
  receipt, then the person's words; was words first) and a reply that blames
  the person for a missing name is re-spoken or appended to.

The narrator's cache prefix (471 tokens above) is exactly its system prompt,
which is the same on every call, so nothing varied inside it: the words that
followed it did, and so did the receipt. What was fixed text after them
(the length line, the label) now leads, so the cacheable prefix grows by
tens of tokens, not the ~300 that remain per turn. Re-measure it with
`CONDUCTOR_FLEET_COST=1` after the batch.
