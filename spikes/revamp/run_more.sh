#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/rskarnulis/Documents/workspace/conductor/backend/.venv/bin/python
run() { "$PY" "$@" 2>&1 | grep -v "fleet_" ; }
run r1_relay.py receipt2 40
run d1_bench.py 12b-v2c 5 golden,holdout
echo MORE-DONE
