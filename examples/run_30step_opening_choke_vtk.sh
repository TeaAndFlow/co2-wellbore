#!/usr/bin/env bash
set -euo pipefail

CASE_ID="${1:-LOCAL_OPM_HEM_LAYERED_D067_OPEN_01_08_HOLD30_VTK_TEMPVD}"

cd "$(dirname "$0")/.."

rm -rf "$PWD/data/runs_restart_coupled_coolprop_manywells/$CASE_ID"

python examples/07_opm_opening_choke_coupling.py \
  --run-flow \
  --base-dir "$PWD/data" \
  --deck Base.DATA \
  --case-id "$CASE_ID" \
  --pressure-control-mode opening_choke \
  --choke-model hem \
  --thermal-model layered \
  --valve-diameter-m 0.067 \
  --opening-schedule-percent \
    1 2 3 4 5 6 7 8 \
    8 8 8 8 8 8 8 8 8 8 \
    8 8 8 8 8 8 8 8 8 8 \
    8 8 \
  --n-steps 30 \
  --q-min-sm3-day 0 \
  --q-max-sm3-day 1200000 \
  --max-coupling-iter 28 \
  --hem-flow-tolerance-kg-s 0.05 \
  --thp-min-bar 20 \
  --no-accept-capacity-surplus \
  --no-accept-last-feasible \
  --no-allow-force-accept \
  --enable-vtk-output \
  --vtk-write-temperature \
  --vtk-write-pressures \
  --vtk-write-saturations \
  --flow-exe flow \
  --mpi-nproc 1 \
  2>&1 | tee "${CASE_ID}.log"
