#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyVista viewer for coupled CO2 wellbore profiles.

Shows a vertical pipe colored by pressure, temperature, quality, density, etc.,
with a slider over accepted coupling steps.

Examples
--------
python scripts/view_coupled_pipe_pyvista.py \
  --case-dir data/runs_restart_coupled_coolprop_manywells/LOCAL_OPM_HEM_LAYERED_D067_OPEN_02_10 \
  --variable pressure

python scripts/view_coupled_pipe_pyvista.py \
  --case-dir data/runs_restart_coupled_coolprop_manywells/LOCAL_OPM_HEM_LAYERED_D067_OPEN_02_10 \
  --variable temp

python scripts/view_coupled_pipe_pyvista.py \
  --case-dir data/runs_restart_coupled_coolprop_manywells/LOCAL_OPM_HEM_LAYERED_D067_OPEN_02_10 \
  --variable quality
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "depth": ["depth_m", "z_m", "measured_depth_m", "tvd_m"],
    "pressure": ["pressure_bar", "p_bar", "P_bar", "pressure"],
    "temp": ["temperature_C", "temp_C", "T_C", "temperature", "T"],
    "quality": ["quality_mass", "vapor_quality", "x_vapor", "x_co2", "gas_quality"],
    "vapor": ["vapor_fraction", "gas_fraction", "alpha_g", "void_fraction"],
    "density": ["density_kg_m3", "rho_kg_m3", "rho"],
    "velocity": ["velocity_m_s", "mixture_velocity_m_s", "u_m_s"],
    "enthalpy": ["enthalpy_J_kg", "h_J_kg", "h"],
}


def find_col(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for cand in candidates:
        cc = cand.lower()
        for col in columns:
            lc = str(col).lower()
            if cc in lc or lc in cc:
                return str(col)

    if required:
        raise KeyError(f"Cannot find any of {candidates}. Available columns: {list(columns)}")
    return None


def discover_profile_csv(case_dir: Path) -> Path:
    candidates = [
        case_dir / "wellbore_profiles_accepted.csv",
        case_dir / "wellbore_profiles.csv",
        case_dir / "wellbore_profiles_LIVE.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = sorted(case_dir.glob("*profile*.csv"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"No wellbore profile CSV found in {case_dir}")


def discover_accepted_csv(case_dir: Path) -> Optional[Path]:
    for name in ["coupled_exchange_accepted.csv", "coupled_exchange_accepted_LIVE.csv"]:
        p = case_dir / name
        if p.exists():
            return p
    return None


def finite_clim(values: np.ndarray, percentile: float) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0

    if percentile < 100.0:
        lo = float(np.percentile(v, 100.0 - percentile))
        hi = float(np.percentile(v, percentile))
    else:
        lo = float(v.min())
        hi = float(v.max())

    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def make_pipe_mesh(depth_m: np.ndarray, values: np.ndarray, radius: float):
    import pyvista as pv

    depth = np.asarray(depth_m, dtype=float)
    val = np.asarray(values, dtype=float)

    order = np.argsort(depth)
    depth = depth[order]
    val = val[order]

    # Vertical pipe: z = -depth, slight x-offset zero.
    points = np.column_stack([
        np.zeros_like(depth),
        np.zeros_like(depth),
        -depth,
    ])

    # One polyline through all depth points.
    n = len(points)
    if n < 2:
        raise ValueError("Need at least two depth points for pipe visualization.")

    cells = np.concatenate([[n], np.arange(n)]).astype(np.int64)
    line = pv.PolyData(points)
    line.lines = cells
    line.point_data["value"] = val.astype(np.float32)

    tube = line.tube(radius=float(radius), n_sides=24)
    return tube


def step_info_text(step: int, accepted_df: Optional[pd.DataFrame]) -> str:
    if accepted_df is None or "step" not in accepted_df.columns:
        return ""

    row = accepted_df[accepted_df["step"].astype(int) == int(step)]
    if row.empty:
        return ""

    r = row.iloc[0]
    fields = [
        ("opening", "opening_percent", "%.3g %%"),
        ("q", "reservoir_rate_sm3_day", "%.6g sm3/day"),
        ("BHP", "reservoir_bhp_bar", "%.4g bar"),
        ("THP", "wellbore_required_thp_bar", "%.4g bar"),
        ("reason", "accepted_reason", "%s"),
        ("constraint", "active_constraint", "%s"),
    ]

    parts = []
    for label, col, fmt in fields:
        if col in r.index:
            val = r[col]
            try:
                text = fmt % float(val)
            except Exception:
                text = fmt % str(val)
            parts.append(f"{label}: {text}")
    return "\\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="View coupled CO2 wellbore profiles as a PyVista pipe.")
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--variable", choices=["pressure", "temp", "quality", "vapor", "density", "velocity", "enthalpy"], default="pressure")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--radius", type=float, default=20.0, help="Visual tube radius in plotting units, not physical tubing radius.")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--window-width", type=int, default=850)
    ap.add_argument("--window-height", type=int, default=900)
    args = ap.parse_args()

    import pyvista as pv

    case_dir = args.case_dir.expanduser().resolve()
    profile_csv = args.csv.expanduser().resolve() if args.csv else discover_profile_csv(case_dir)

    df = pd.read_csv(profile_csv)
    if "step" not in df.columns:
        raise KeyError(f"Profile CSV must contain 'step'. Columns: {list(df.columns)}")

    depth_col = find_col(df.columns, COLUMN_CANDIDATES["depth"])
    value_col = find_col(df.columns, COLUMN_CANDIDATES[args.variable])

    accepted_csv = discover_accepted_csv(case_dir)
    accepted_df = pd.read_csv(accepted_csv) if accepted_csv is not None else None

    steps = sorted(int(s) for s in pd.unique(df["step"].dropna().astype(int)))
    if not steps:
        raise RuntimeError("No steps found in profile CSV.")

    clim = finite_clim(df[value_col].to_numpy(dtype=float), float(args.percentile))

    print(f"[case] {case_dir}")
    print(f"[profiles] {profile_csv}")
    print(f"[accepted] {accepted_csv}")
    print(f"[variable] {args.variable} -> column={value_col}")
    print(f"[depth] {depth_col}")
    print(f"[steps] {steps}")

    plotter = pv.Plotter(window_size=(args.window_width, args.window_height))
    plotter.set_background("#d8e6f3")

    actor = None
    text_actor = None
    first = True

    def draw(pos_float):
        nonlocal actor, text_actor, first

        pos = int(round(float(pos_float)))
        pos = int(np.clip(pos, 0, len(steps) - 1))
        step = steps[pos]

        dstep = df[df["step"].astype(int) == int(step)].copy()
        dstep = dstep[np.isfinite(dstep[depth_col]) & np.isfinite(dstep[value_col])]
        if dstep.empty:
            return

        mesh = make_pipe_mesh(
            depth_m=dstep[depth_col].to_numpy(dtype=float),
            values=dstep[value_col].to_numpy(dtype=float),
            radius=float(args.radius),
        )

        try:
            if actor is not None:
                plotter.remove_actor(actor, reset_camera=False)
            if text_actor is not None:
                plotter.remove_actor(text_actor, reset_camera=False)
        except Exception:
            pass

        actor = plotter.add_mesh(
            mesh,
            scalars="value",
            cmap=args.cmap,
            clim=clim,
            show_edges=False,
            scalar_bar_args={
                "title": value_col,
                "vertical": True,
                "position_x": 0.05,
                "position_y": 0.18,
                "width": 0.10,
                "height": 0.55,
                "n_labels": 7,
                "fmt": "%.4g",
            },
            reset_camera=False,
        )

        info = step_info_text(step, accepted_df)
        text = (
            f"Wellbore pipe viewer\\n"
            f"step {step} ({pos + 1}/{len(steps)})\\n"
            f"variable={args.variable}\\n"
            f"column={value_col}"
        )
        if info:
            text += "\\n\\n" + info

        text_actor = plotter.add_text(text, position="upper_left", font_size=10, color="black")

        plotter.show_bounds(
            grid="front",
            location="outer",
            xtitle="X",
            ytitle="Y",
            ztitle="-depth, m",
            color="black",
            font_size=9,
        )

        if first:
            plotter.reset_camera()
            first = False

        plotter.render()

    draw(0)

    if len(steps) > 1:
        plotter.add_slider_widget(
            callback=draw,
            rng=[0, len(steps) - 1],
            value=0,
            title="accepted coupling step",
            pointa=(0.22, 0.04),
            pointb=(0.78, 0.04),
            style="modern",
            fmt="%.0f",
        )

    plotter.add_axes()
    print("[viewer] mouse: left rotate, wheel zoom, right pan, bottom slider = accepted step.")
    plotter.show()


if __name__ == "__main__":
    main()
