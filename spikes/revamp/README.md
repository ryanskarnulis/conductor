# Conductor revamp spikes (throwaway, never merged)

Spikes for ryanskarnulis/conductor#56, run 2026-11-05 on the real llama-swap
(gemma-4-12b, gemma-4-26b-a4b; RTX 3060 12 GB, 31 GB RAM). Results are in
`results/` (JSONL, one row per call) and summarized in #56.

| Spike | File | Question |
|---|---|---|
| S0 | `s0_swap.py` | How long is a 12B ↔ 26B swap with today's RAM? |
| D1 | `d1_bench.py`, `d1_cases.py`, `plan_schema.py`, `plan_v2.py` | Can the 12B compile typed plans under `json_schema`? |
| R1 | `r1_relay.py` | Does a narrator reading a typed receipt keep what the app touched? |
| C1b | `c1b_async.py` | Does the narrator stay honest about jobs in mixed states? |

Run from this directory with conductor's backend venv (the fleet section
comes from the real `app.yaml`s):

```bash
PY=../../../conductor/backend/.venv/bin/python   # or your checkout's venv
$PY s0_swap.py 3
$PY d1_bench.py 12b-v2 5 && $PY d1_bench.py --report
$PY r1_relay.py receipt 40 && $PY r1_relay.py --report
$PY c1b_async.py 10
```

Caveats: D1 v2's prompt was revised after reading v1's misses on the same
cases, so the held-out group (written after v1, never tuned against) is the
honest number. R1's "today" arm rebuilds music's handoff from
`run_record.handoff`; #55 used real music replies. C1b was graded by reading
every reply.
