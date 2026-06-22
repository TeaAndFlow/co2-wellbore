from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def _fmt_float(value: object, fmt: str = ".3g") -> str:
    try:
        x = float(value)
        if not math.isfinite(x):
            return ""
        return format(x, fmt)
    except Exception:
        return ""


def write_run_summary(
    *,
    case_dir: Path,
    case_id: str,
    accepted_df: pd.DataFrame,
    iterations_df: pd.DataFrame,
    figure_paths: Iterable[Path],
) -> Path:
    """Write compact human-readable Markdown summary for one coupled run."""

    case_dir = Path(case_dir)
    summary_path = case_dir / "RUN_SUMMARY.md"

    n_steps = int(len(accepted_df))
    n_iterations = int(len(iterations_df))

    max_iter_per_step = ""
    if "iteration" in accepted_df.columns and len(accepted_df) > 0:
        max_iter_per_step = _fmt_float(accepted_df["iteration"].max(), ".0f")

    max_abs_dT = ""
    if "residual_temperature_bht_minus_wtemp_C" in accepted_df.columns and len(accepted_df) > 0:
        max_abs_dT = _fmt_float(
            accepted_df["residual_temperature_bht_minus_wtemp_C"].abs().max(),
            ".3f",
        )

    max_abs_choke = ""
    if "choke_capacity_residual_kg_s" in accepted_df.columns and len(accepted_df) > 0:
        numeric = pd.to_numeric(accepted_df["choke_capacity_residual_kg_s"], errors="coerce")
        if numeric.notna().any():
            max_abs_choke = _fmt_float(numeric.abs().max(), ".3f")

    lines: list[str] = []
    lines.append(f"# Run summary: `{case_id}`")
    lines.append("")
    lines.append("## Convergence overview")
    lines.append("")
    lines.append(f"- Accepted steps: **{n_steps}**")
    lines.append(f"- Total coupling iterations: **{n_iterations}**")
    lines.append(f"- Max accepted iteration: **{max_iter_per_step}**")
    lines.append(f"- Max accepted |BHT - WTEMP|: **{max_abs_dT} C**")
    lines.append(f"- Max accepted |choke residual|: **{max_abs_choke} kg/s**")
    lines.append("")

    lines.append("## Accepted steps")
    lines.append("")
    lines.append(
        "| step | it | reason | open % | q sm3/d | WBHP bar | THP bar | WTEMP C | BHT C | dT C | choke kg/s |"
    )
    lines.append(
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in accepted_df.iterrows():
        values = [
            _fmt_float(row.get("step"), ".0f"),
            _fmt_float(row.get("iteration"), ".0f"),
            str(row.get("accepted_reason", "")),
            _fmt_float(row.get("opening_percent"), ".1f"),
            _fmt_float(row.get("reservoir_rate_sm3_day"), ".0f"),
            _fmt_float(row.get("reservoir_bhp_bar"), ".2f"),
            _fmt_float(row.get("wellbore_required_thp_bar"), ".2f"),
            _fmt_float(row.get("control_wtemp_used_C"), ".2f"),
            _fmt_float(row.get("wellbore_bht_calc_C"), ".2f"),
            _fmt_float(row.get("residual_temperature_bht_minus_wtemp_C"), "+.3f"),
            _fmt_float(row.get("choke_capacity_residual_kg_s"), "+.3f"),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `coupled_exchange_accepted.csv`")
    lines.append("- `coupled_exchange_iterations.csv`")
    lines.append("- `wellbore_profiles_accepted.csv`")
    lines.append("- `coupling_metadata.json`")
    lines.append("")

    fig_list = [Path(p) for p in figure_paths]
    if fig_list:
        lines.append("## Figures")
        lines.append("")
        for path in fig_list:
            lines.append(f"- `{path.name}`")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
