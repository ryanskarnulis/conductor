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
`LLAMACPP_MODEL`, `CONDUCTOR_EVAL_FLEET_DIR`, `CONDUCTOR_EVAL_INFRA_RETRIES`
(default 2), `CONDUCTOR_EVAL_INFRA_BUDGET` (default 6). The suite skips itself unless
discovery finds both `ask_tasks` and `open_chess`.

## The gating rule

This suite gates every change to the base prompt, the vendored personality,
the model, the loop, **and any fleet app's `agent:`/`open:` descriptions and
examples — wherever that PR lands.** A sibling app widening its manifest is a
change to this suite's inputs (that is exactly how `refuse-playback` went red
without anyone noticing). If routing degrades, tighten manifest `examples`
first; a bigger model is the last resort.

## Infrastructure is retried, not scored

A provider death (llama-swap evicting or reloading the model mid-suite: a 502
`llama-server request failed: timed out`, a refused connection, or a 5xx while
it swaps) never decided a route, so it is not scored. The harness throws the
attempt away and re-takes it on a fresh conversation, up to
`CONDUCTOR_EVAL_INFRA_RETRIES` per scenario and `CONDUCTOR_EVAL_INFRA_BUDGET`
across the suite (`backend/tests/evalstats.py`, conductor's cut of chess's).
A timed-out attempt still costs the full 300 s, so the defaults stop a dead
server inside ~30 min.

Reading a run:

- `[eval] … infra=N` on a scenario line: it passed or failed on a real
  routing decision after N thrown-away attempts.
- `[eval] suite …s infra_spent=S/B (scenario=n, …)` closes the run — a green
  suite that limped says so, and says which scenarios ate the budget.
- `INFRA_ABORTED` — the scenario ran out of retries. Routing was not tested:
  fix the server, not the manifests.
- `llama-server refused the request` — a 4xx (context overrun) or an
  unparseable body. Deterministic, so it fails on the first one without
  retrying; look at the request.

Any other red is a routing answer.

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
| confirm-delete-task / confirm-wipe-project | confirm | no call, asks first |
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
