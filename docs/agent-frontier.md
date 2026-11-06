# Frontier tier

Asks conductor isn't expected to pass yet, measured over time and never gated
(STANDARD.md §6; PCC's `docs/agent-frontier.md` is the reference). The gate
answers "did this change break something that worked"; the frontier answers
"how far is the thing we're building from working". A frontier scenario that
reaches its floor moves to the gate in the PR that got it there.

## What's in it

The `frontier` tier of the session evals (`backend/tests/session_scenarios.py`,
#63): whole multi-turn scenarios from #56's S1–S16 that need a capability
today's loop doesn't have, or that it has but fails as a matter of course.

| scenario | why it's frontier | moves with |
|---|---|---|
| S3, S15 | cross-app: the album's real name must reach the task (#55) | #65, #72 (≥80% end state) |
| S4 | a job reporting back on its own: no unprompted messages yet | #76, #38 |
| S6 | reference resolution across conversations: no memory across them | #64, #80 |
| S10 | Stop partway: no cancel route (`unsupported: stop`) | #66 |
| S11 | undo across apps: no record of what to invert | #69, #74 |
| S12 | reading chess without opening the board (`unsupported: tool:ask_chess`) | #73 |
| S13 | a brief from two apps: asks one and stops | #64 |
| S14 | hard planning: the 12B's ceiling | #78 / #79 |

A scenario that needs a capability the runtime lacks fails `unsupported:` and
reads 0 for a reason the report names. It becomes a real measurement the day
the capability lands, with no harness change: the driver looks for a cancel
route and for tools by name.

The routing gate's known reds (`refuse-playback`, `music-save`) stay in the gate
as `xfail`, not here: they are regressions to fix, not capabilities to build.

## Running it

```bash
cd backend
CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1 \
    CONDUCTOR_EVAL_REPORT=/tmp/frontier.jsonl \
    .venv/bin/pytest tests/test_session_evals.py -k live_frontier -v -s
```

`CONDUCTOR_FRONTIER_RUNS` (default 5) is the fixed sample count; there is no
escalation and no assertion. From a worktree, also set
`CONDUCTOR_EVAL_FLEET_DIR=~/Documents/workspace` (the default guesses the
workspace from the repo's location).

## History

Append each measured run's records to `docs/frontier-history.jsonl` (one
`evalstats.scenario_record` line per scenario, with `tier`, the per-sample
seeds and verdicts, and the date in the commit that adds them). Compare runs
only within one campaign (`backend/scripts/eval_campaign.sh --suite session`):
the GPU is shared, and two batches run hours apart are not paired.
