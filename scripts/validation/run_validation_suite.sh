#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
pytest -q tests/validation
python scripts/validation/run_hem_grid_convergence.py
python scripts/validation/generate_validation_figures.py
python scripts/validation/generate_hem_ideal_gas_limit.py
python scripts/validation/generate_property_reference_comparison.py
