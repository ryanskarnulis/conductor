# Routing eval harness

`backend/tests/test_agent_evals.py` — golden utterance→route scenarios against
the **real** model (gemma-4-12b behind llama-swap) and the **real** workspace
manifests. This was Phase 3's go/no-go gate (verdict: GO). Full measurement
narratives are in this file's git history.

## What it is

Each golden drives one utterance through the same seam the web UI uses, with
the real provider, the real layered prompt (fleet section rendered from the
real sibling `app.yaml`s), and the real loop. Only the network hop is stubbed
(canned delegate replies), so the routing decision is entirely the model's.
Assertions are behavioral (temp 1.0): **route** goldens pin "exactly this
app's tool, no other" (handoffs also pin the `intent` carried the user's
words); **refuse** goldens pin "no app acted on"; **confirm** goldens pin a
question with no tool call on destructive `ask_` requests; **local** goldens
allow `list_agents` only.

## Running it

```bash
cd backend
CONDUCTOR_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s
```

Opt-in; needs llama-swap up (cold load ~100 s). Overrides: `LLAMACPP_BASE_URL`,
`LLAMACPP_MODEL`, `CONDUCTOR_EVAL_FLEET_DIR`, `CONDUCTOR_EVAL_RUNS` (block
size, default 5), `CONDUCTOR_EVAL_MAX_RUNS` (default 20),
`CONDUCTOR_EVAL_INFRA_RETRIES` (default 3), `CONDUCTOR_EVAL_INFRA_BUDGET`
(default 12). The suite skips itself unless
discovery finds both `ask_tasks` and `open_chess`.

## The gating rule

This suite gates every change to the base prompt, the vendored personality,
the model, the loop, **and any fleet app's `agent:`/`open:` descriptions and
examples — wherever that PR lands.** A sibling app widening its manifest is a
change to this suite's inputs (that is exactly how `refuse-playback` went red
without anyone noticing). If routing degrades, tighten manifest `examples`
first; a bigger model is the last resort.

**No golden may go statistically below its floor** (0.8, the agent standard's
sampled gate, `STANDARD.md` §6). Each golden is sampled in blocks of 5, a fresh
conversation per sample, and judged by the vendored `backend/tests/evalstats.py`
(spec `test_evalstats.py`, conductor's drift guards `test_evalstats_app.py`; re-copy
from `../agent-standard/`, never edit): a one-sided 95% Wilson upper bound below
the floor is red, a rate at the floor green, anything between buys another
block up to 20. At n=5: 0–2 red, 3 escalates, 4–5 green. An `UNDECIDED` count at
the cap passes, flagged. A known red is `xfail`, not gated.

## Infrastructure is retried, not scored

A provider death (llama-swap evicting or reloading the model mid-suite: a 502
`llama-server request failed: timed out`, a refused connection, or a 5xx while
it swaps) never decided a route, so it is not scored. The harness throws the
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

## Known reds (xfail, not gated)

Under the sampled gate (2026-10-28) two goldens read below the floor on main;
both are `xfail` in `KNOWN_REDS` until fixed:

- `refuse-playback` — 0/5, deterministic (below).
- `music-save` — "save this track to the music folder": 2/5, then 13/20 (65%,
  ci [0.47, 0.80]); every miss answers locally with no app call. The
  one-sample gate passed it by luck in every recorded baseline.

The rest of the suite: 19/21 ABOVE_FLOOR at 5/5, infra 2/12 (two llama-swap
timeouts on `chess-reset`, re-taken, never scored), 684 s.

## Open regression — `refuse-playback` red since music #9

"play some jazz music" routes to `ask_music` instead of refusing,
deterministic 0/3, bisected to music #9's manifest line *"queues a whole album
or playlist"* (2026-08-23). An attempted fix (a no-playback negative sentence
in music's description) fixed it 3/3 but broke `music-save` in 2 of 3 full
runs — the negation suppresses music routing generally. Options, none taken
(tracked in #36): narrow the negative until both hold; retire the golden
(music's own agent refuses truthfully one hop later); or leave red until music
Phase 6 makes the route correct.

## Baseline — 2026-08-22, gemma-4-12b, 18/18 (3 consecutive runs)

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

- **Scripted (CI).** Each scenario's golden script plays the model and must
  pass, so every check is shown to be satisfiable. Deliberately wrong scripts
  (acting before the yes, a dropped album title, a claimed write that didn't
  happen, an unnamed write, an invented recap) must fail for the right reason.
- **Live (opt-in).** `CONDUCTOR_AGENT_EVALS=1` runs the `gate` tier under the
  same Wilson gate as routing. `CONDUCTOR_FRONTIER_EVALS=1` also measures the
  `frontier` tier: 5 samples per scenario, never asserted. Every sample is a
  fresh world and a fresh conversation through the real route, loop, prompt
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

A scenario that needs a capability the runtime lacks (`stop`, `tool:ask_chess`)
fails `unsupported:` instead of skipping, so it reads 0 for a named reason and
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
