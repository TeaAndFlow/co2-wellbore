#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def pick_file(case_dir: Path, names):
    for name in names:
        p = case_dir / name
        if p.exists():
            return p
    return None


def pick_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def add_step_colorbar(fig, ax, steps, cmap):
    norm = Normalize(vmin=float(np.min(steps)), vmax=float(np.max(steps)))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Coupling step")
    return norm


def plot_residual(case_dir: Path, out_dir: Path):
    csv = pick_file(case_dir, [
        "coupled_exchange_iterations.csv",
        "coupled_exchange_iterations_LIVE.csv",
        "coupled_exchange_all.csv",
    ])
    if csv is None:
        print("[skip] no iterations CSV")
        return

    df = pd.read_csv(csv)

    step_col = pick_col(df, ["step"])
    it_col = pick_col(df, ["iteration", "control_iteration"])
    res_col = pick_col(df, [
        "choke_capacity_residual_kg_s",
        "choke_kv_residual_m3_h",
        "pressure_residual_for_convergence",
    ])

    if not all([step_col, it_col, res_col]):
        print("[skip] residual plot missing columns")
        print(df.columns.tolist())
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    steps = sorted(df[step_col].dropna().unique())
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=min(steps), vmax=max(steps))

    for step in steps:
        sub = df[df[step_col] == step].sort_values(it_col)
        ax.plot(
            sub[it_col],
            sub[res_col],
            marker="o",
            linewidth=1.5,
            markersize=4,
            color=cmap(norm(step)),
            alpha=0.9,
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Choke Flow Residual")
    ax.set_xlabel("Coupling iteration")
    ax.set_ylabel("HEM residual, kg/s")
    ax.grid(True, alpha=0.35)
    add_step_colorbar(fig, ax, steps, cmap)

    out = out_dir / "clean_choke_flow_residual.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("[saved]", out)


def plot_summary_timeseries(case_dir: Path, out_dir: Path):
    csv = pick_file(case_dir, [
        "coupled_exchange_accepted.csv",
        "coupled_exchange_accepted_LIVE.csv",
    ])
    if csv is None:
        print("[skip] no accepted CSV")
        return

    df = pd.read_csv(csv)
    step_col = pick_col(df, ["step"])

    candidates = [
        ("opening_percent", "Opening, %"),
        ("reservoir_rate_sm3_day", "Rate, sm3/day"),
        ("reservoir_bhp_bar", "BHP, bar"),
        ("wellbore_required_thp_bar", "Required THP, bar"),
        ("control_wtemp_used_C", "WTEMP used, C"),
        ("wellbore_bht_calc_C", "BHT calculated, C"),
    ]

    for col, ylabel in candidates:
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
        ax.plot(df[step_col], df[col], marker="o", linewidth=2.0)
        ax.set_title(ylabel)
        ax.set_xlabel("Coupling step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)

        safe = col.replace("/", "_")
        out = out_dir / f"clean_{safe}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[saved]", out)


def plot_wellbore_profiles(case_dir: Path, out_dir: Path):
    csv = pick_file(case_dir, [
        "wellbore_profiles_accepted.csv",
        "wellbore_profiles.csv",
        "wellbore_profiles_LIVE.csv",
    ])
    if csv is None:
        print("[skip] no wellbore profile CSV")
        return

    df = pd.read_csv(csv)

    step_col = pick_col(df, ["step", "control_step_index"])
    depth_col = pick_col(df, ["depth_m", "depth", "z_m", "measured_depth_m"])
    p_col = pick_col(df, ["pressure_bar", "p_bar", "pressure"])
    t_col = pick_col(df, ["temperature_C", "temp_C", "temperature"])

    if not step_col or not depth_col:
        print("[skip] profile plot missing step/depth columns")
        print(df.columns.tolist())
        return

    steps = sorted(df[step_col].dropna().unique())
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=min(steps), vmax=max(steps))

    for value_col, title, xlabel, fname in [
        (p_col, "Wellbore Pressure", "Pressure, bar", "clean_wellbore_pressure.png"),
        (t_col, "Wellbore Temperature", "Temperature, C", "clean_wellbore_temperature.png"),
    ]:
        if value_col is None:
            continue

        fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)

        for step in steps:
            sub = df[df[step_col] == step].sort_values(depth_col)
            ax.plot(
                sub[value_col],
                sub[depth_col],
                linewidth=1.4,
                color=cmap(norm(step)),
                alpha=0.9,
            )

        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth, m")
        ax.grid(True, alpha=0.35)
        add_step_colorbar(fig, ax, steps, cmap)

        out = out_dir / fname
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[saved]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else case_dir / "clean_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_residual(case_dir, out_dir)
    plot_summary_timeseries(case_dir, out_dir)
    plot_wellbore_profiles(case_dir, out_dir)


if __name__ == "__main__":
    main()
