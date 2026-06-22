#!/usr/bin/env bash
set -euo pipefail

CASE_ID="${CASE_ID:-OPM7_HEM_LAYERED_OPEN_01_07}"

cd "$(dirname "$0")/.."

python -m co2_wellbore.coupling.opm_restart \
  --run-flow \
  --deck Base.DATA \
  --case-id "$CASE_ID" \
  --pressure-control-mode opening_choke \
  --choke-model hem \
  --thermal-model layered \
  --valve-diameter-m 0.067 \
  --opening-schedule-percent 1 2 3 4 5 6 7 \
  --n-steps 7 \
  --step-days 30 \
  --q-min-sm3-day 0 \
  --q-max-sm3-day 300000 \
  --rate-schedule-is-cap \
  --max-coupling-iter 28 \
  --hem-flow-tolerance-kg-s 0.05 \
  --thp-min-bar 20 \
  --thp-max-bar 350 \
  --no-accept-capacity-surplus \
  --no-accept-last-feasible \
  --no-allow-force-accept \
  --figures-dir "$PWD/figures_${CASE_ID}" \
  2>&1 | tee "run_${CASE_ID}.log"
