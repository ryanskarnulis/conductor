#!/usr/bin/env bash
# Paired, interleaved A/B for an agent change: alternating blocks between two
# trees on ONE llama-server, so server drift lands on both arms alike
# (STANDARD.md §6, #53; after PCC's backend/scripts/eval_campaign.sh).
#
#   backend/scripts/eval_campaign.sh --a /path/to/main --b /path/to/branch \
#       --k 'music or refuse' --blocks 4
#
# An arm may also be an environment setting on the same tree:
#
#   backend/scripts/eval_campaign.sh --a . --b . --env-b LLAMACPP_MODEL=other --k ...
#
# --suite plan (default) runs tests/test_plan_evals.py (unseeded: it samples
# CONDUCTOR_PLAN_SAMPLES=<runs> per case; the routing goldens retired at #75);
# --suite session runs
# the session evals' live tiers (tests/test_session_evals.py -k live, frontier
# included). Each block runs the suite once per arm from that arm's own
# backend/ (its own app and its own harness are what is measured) with
# CONDUCTOR_EVAL_RUNS=CONDUCTOR_EVAL_MAX_RUNS=<runs> (and
# CONDUCTOR_FRONTIER_RUNS=<runs>) so a block is exactly <runs> scored samples:
# escalation would give the arms different counts. Arm order flips every block.
#
# Arms are paired: block <n> of both arms runs with
# CONDUCTOR_AGENT_EVALS_SEED=<seed> + 1000*<n>, so sample j of each arm's block
# n draws the same sampling stream (tests/eval_seed.py). --seed <base> picks the
# base (default 53); --no-seed runs unseeded. Pairing is approximate and does
# not replace the flipped block order.
#
# Before each arm-block the log records the tree's HEAD, a sha of its harness
# files, `app.__file__` as Python resolves it (a run that can't prove which
# tree it measured measured nothing), and llama-swap /running. The GPU is
# shared: other sessions' load lands on both arms alike only because the
# blocks alternate — never compare arms run in separate batches.
#
# Output lands in --out (default $TMPDIR/conductor-campaign-<utc>/, never the
# repo) as block-<n>-<arm>.jsonl plus .out and campaign.log; the last thing
# printed is the joined table (scripts/campaign_report.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-$BACKEND/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

A=""; B=""; K=""; BLOCKS=4; OUT=""; RUNS=5; SEED=53; SUITE=plan
ENV_A=(); ENV_B=()
BASE_URL="${LLAMACPP_BASE_URL:-http://127.0.0.1:8200/v1}"
FLEET_DIR="${CONDUCTOR_EVAL_FLEET_DIR:-$(cd "$BACKEND/../.." && pwd)}"
while [ $# -gt 0 ]; do
  case "$1" in
    --a) A="$2"; shift 2 ;;
    --b) B="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --env-a) ENV_A+=("$2"); shift 2 ;;
    --env-b) ENV_B+=("$2"); shift 2 ;;
    --blocks) BLOCKS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no-seed) SEED=""; shift ;;
    -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$A" ] && [ -n "$B" ] || { echo "need --a and --b" >&2; exit 2; }
case "$SUITE" in
  plan) TEST=tests/test_plan_evals.py; HARNESS_FILES=(tests/test_plan_evals.py tests/plan_cases.py) ;;
  session) TEST=tests/test_session_evals.py
    HARNESS_FILES=(tests/test_session_evals.py tests/session_harness.py tests/session_scenarios.py
                   tests/fake_world.py tests/receipt_claims.py tests/eval_seed.py) ;;
  *) echo "--suite is plan or session" >&2; exit 2 ;;
esac
A="$(cd "$A" && pwd)"; B="$(cd "$B" && pwd)"
for tree in "$A" "$B"; do
  for f in "${HARNESS_FILES[@]}"; do
    [ -f "$tree/backend/$f" ] || { echo "$tree is not a conductor tree with $f" >&2; exit 2; }
  done
done
OUT="${OUT:-${TMPDIR:-/tmp}/conductor-campaign-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
LOG="$OUT/campaign.log"
SWAP_ROOT="${BASE_URL%/v1}"

log() { echo "$*" | tee -a "$LOG"; }

arm_env() {  # $1 = arm name → that arm's KEY=VALUE settings, one per line
  if [ "$1" = a ]; then printf '%s\n' "${ENV_A[@]+"${ENV_A[@]}"}"; else printf '%s\n' "${ENV_B[@]+"${ENV_B[@]}"}"; fi
}

declare -A HARNESSES

identity() {  # $1 = arm name, $2 = tree
  local head file running
  head="$(git -C "$2" rev-parse --short HEAD)"
  HARNESSES["$1"]="$(cd "$2/backend" && cat "${HARNESS_FILES[@]}" | sha256sum | cut -c1-12)"
  file="$(cd "$2/backend" && "$PYTHON" -c 'import app; print(app.__file__)')"
  running="$(curl -s -m 5 "$SWAP_ROOT/running" || echo '{}')"
  log "  arm $1: tree $2 HEAD $head harness ${HARNESSES[$1]} app $file env [$(arm_env "$1" | tr '\n' ' ')] running $running"
  case "$file" in "$2"/*) ;; *) log "  !! app resolved outside $2; aborting"; exit 1 ;; esac
}

run_block() {  # $1 = block number, $2 = arm name, $3 = tree
  local report="$OUT/block-$1-$2.jsonl" seed=""
  [ -n "$SEED" ] && seed=$((SEED + 1000 * $1))
  log "block $1 arm $2 start $(date -u +%H:%M:%SZ) seed ${seed:-none}"
  identity "$2" "$3"
  local -a extra=() select=()
  while IFS= read -r kv; do [ -n "$kv" ] && extra+=("$kv"); done < <(arm_env "$2")
  if [ "$SUITE" = session ]; then
    select=(-k "live${K:+ and ($K)}")
  elif [ -n "$K" ]; then
    select=(-k "$K")
  fi
  (cd "$3/backend" && env CONDUCTOR_AGENT_EVALS=1 CONDUCTOR_FRONTIER_EVALS=1 \
      CONDUCTOR_EVAL_RUNS="$RUNS" CONDUCTOR_EVAL_MAX_RUNS="$RUNS" CONDUCTOR_FRONTIER_RUNS="$RUNS" \
      CONDUCTOR_EVAL_REPORT="$report" CONDUCTOR_AGENT_EVALS_SEED="$seed" \
      CONDUCTOR_PLAN_EVALS=1 CONDUCTOR_PLAN_SAMPLES="$RUNS" \
      CONDUCTOR_EVAL_FLEET_DIR="$FLEET_DIR" "${extra[@]}" \
      "$PYTHON" -m pytest "$TEST" "${select[@]}" -q -s -p no:logging >"$OUT/block-$1-$2.out" 2>&1) || true
  log "block $1 arm $2 end $(date -u +%H:%M:%SZ) ($(wc -l <"$report" 2>/dev/null || echo 0) records)"
}

log "campaign $SUITE: a=$A b=$B k='${K}' blocks=$BLOCKS runs=$RUNS seed=${SEED:-none} out=$OUT"
for n in $(seq 1 "$BLOCKS"); do
  if [ $((n % 2)) = 1 ]; then
    run_block "$n" a "$A"; run_block "$n" b "$B"
  else
    run_block "$n" b "$B"; run_block "$n" a "$A"
  fi
done
if [ "${HARNESSES[a]}" != "${HARNESSES[b]}" ]; then
  log "!! the arms ran different harness files (${HARNESSES[a]} vs ${HARNESSES[b]}): not a fair"
  log "!! A/B on any scenario the difference touches"
fi
"$PYTHON" "$HERE/campaign_report.py" "$OUT" | tee -a "$LOG"
