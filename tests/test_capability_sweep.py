from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_capability_sweep_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "stress" / "run_capability_sweep.py"),
            "--mode",
            "smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    csv_path = output_dir / "capability_envelope.csv"
    summary_path = output_dir / "capability_summary.md"

    assert csv_path.exists()
    assert summary_path.exists()

    df = pd.read_csv(csv_path)

    assert len(df) > 0
    assert {"component", "case_id", "status", "failure_reason"}.issubset(df.columns)
    assert set(df["component"]).issuperset({"properties", "hem_choke", "wellbore"})

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "stress" / "plot_capability_results.py"),
            "--csv",
            str(csv_path),
            "--figures-dir",
            str(figures_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    assert (figures_dir / "capability_status_by_component.png").exists()
