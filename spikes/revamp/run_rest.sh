#!/usr/bin/env bash
# The rest of the spike runs, 12B first (no swaps), then the 26B arms.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/rskarnulis/Documents/workspace/conductor/backend/.venv/bin/python
run() { "$PY" "$@" 2>&1 | grep -v "fleet_" ; }
run d1_bench.py 12b 5 holdout            # v1 on the held-out cases, for comparison
run d1_bench.py 12b-v2 5
run d1_bench.py 12b-t03-v2 5
run r1_relay.py today 40
run r1_relay.py receipt 40
run c1b_async.py 10
run d1_bench.py 26b-v2 5
run d1_bench.py 26b-think-v2 3 cross,ref,control,memory,hard,self,holdout
echo ALL-DONE
