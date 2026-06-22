#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
pytest -q tests/validation
python scripts/validation/run_hem_grid_convergence.py
