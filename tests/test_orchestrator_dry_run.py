from __future__ import annotations

from pathlib import Path

import pandas as pd

from co2_wellbore.coupling.cli import build_parser
from co2_wellbore.coupling.orchestrator import run_orchestrator


def test_orchestrator_fixed_rate_dry_run_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "data"

    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--base-dir",
            str(base_dir),
            "--deck",
            "Base.DATA",
            "--output-root",
            str(tmp_path / "runs"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--case-id",
            "PYTEST_FIXED_RATE_DRY_RUN",
            "--pressure-control-mode",
            "fixed_rate",
            "--n-steps",
            "1",
            "--max-coupling-iter",
            "6",
        ]
    )

    case_dir = run_orchestrator(args)

    accepted_csv = case_dir / "coupled_exchange_accepted.csv"
    iterations_csv = case_dir / "coupled_exchange_iterations.csv"
    profiles_csv = case_dir / "wellbore_profiles_accepted.csv"

    assert accepted_csv.exists()
    assert iterations_csv.exists()
    assert profiles_csv.exists()

    accepted = pd.read_csv(accepted_csv)
    iterations = pd.read_csv(iterations_csv)

    assert len(accepted) == 1
    assert bool(accepted["accepted"].iloc[0]) is True
    assert "thermal_update_method" in iterations.columns
    assert "thermal_wtemp_next_C" in iterations.columns
