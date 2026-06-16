from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from co2_wellbore.constants import BAR_TO_PA

from .opening_choke import co2_phase_split_from_ph
from .reporting import _safe_float

def co2_saturation_curve_PT(
    *,
    T_min_C: float = -56.4,
    T_max_C: float = 30.8,
    n: int = 250,
) -> pd.DataFrame:
    """Return pure-CO2 vapor-liquid saturation / dew curve in P-T coordinates.

    For pure CO2, dew and bubble pressure are the same saturation pressure
    at a given temperature. The curve is valid from near the triple point
    to below the critical point.
    """
    from CoolProp.CoolProp import PropsSI

    fluid = "CO2"
    Tcrit_C = float(PropsSI("Tcrit", fluid)) - 273.15
    Ttriple_C = float(PropsSI("Ttriple", fluid)) - 273.15

    Tmin = max(float(T_min_C), Ttriple_C + 0.05)
    Tmax = min(float(T_max_C), Tcrit_C - 0.05)

    rows: List[Dict[str, float]] = []
    for T_C in np.linspace(Tmin, Tmax, int(n)):
        try:
            P_bar = float(PropsSI("P", "T", T_C + 273.15, "Q", 1.0, fluid)) / 1.0e5
            if math.isfinite(P_bar):
                rows.append({"temperature_C": float(T_C), "pressure_bar": float(P_bar)})
        except Exception:
            continue

    return pd.DataFrame(rows)


def _finite_series(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def plot_pt_diagram_for_step(
    *,
    row: pd.Series,
    profile_step: pd.DataFrame,
    figures_dir: Path,
    case_id: str,
) -> Optional[Path]:
    """Create a clean P-T diagram for one accepted coupling step."""
    import matplotlib.pyplot as plt

    if profile_step.empty:
        return None
    if "temperature_C" not in profile_step or "pressure_bar" not in profile_step:
        return None

    step = int(row.get("step", 0))
    opening = float(row.get("opening_percent", float("nan")))

    upstream_T = _safe_float(row.get("choke_upstream_temperature_C", float("nan")))
    upstream_P = _safe_float(row.get("choke_upstream_pressure_bar", float("nan")))

    downstream_T = _safe_float(row.get("choke_downstream_temperature_C", float("nan")))
    downstream_P = _safe_float(row.get("choke_downstream_pressure_bar", float("nan")))

    bh_T = _safe_float(row.get("wellbore_bht_calc_C", profile_step["temperature_C"].iloc[-1]))
    bh_P = _safe_float(row.get("wellbore_bhp_calc_bar", profile_step["pressure_bar"].iloc[-1]))

    quality = _safe_float(row.get("choke_downstream_quality_mass", row.get("choke_quality_mass", float("nan"))))
    phase = str(row.get("choke_downstream_phase_label", row.get("choke_phase_label", "")))
    q_sm3_day = _safe_float(row.get("reservoir_rate_sm3_day", row.get("control_rate_target_sm3_day", float("nan"))))

    dew = co2_saturation_curve_PT()

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    if not dew.empty:
        ax.plot(
            dew["temperature_C"],
            dew["pressure_bar"],
            linewidth=2.2,
            label="CO2 saturation line",
        )

    # Choke expansion line: upstream -> after choke.
    if all(math.isfinite(x) for x in [upstream_T, upstream_P, downstream_T, downstream_P]):
        ax.plot(
            [upstream_T, downstream_T],
            [upstream_P, downstream_P],
            marker="o",
            linewidth=2.5,
            label="Choke expansion",
        )
        ax.annotate(
            "Upstream",
            xy=(upstream_T, upstream_P),
            xytext=(upstream_T - 24.0, upstream_P + 13.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black"},
        )
        ax.annotate(
            "After choke",
            xy=(downstream_T, downstream_P),
            xytext=(downstream_T - 30.0, downstream_P + 16.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black"},
        )

    # Wellbore trajectory: after choke -> bottomhole.
    prof = profile_step.copy()
    prof = prof.sort_values("depth_m") if "depth_m" in prof else prof

    ax.plot(
        prof["temperature_C"],
        prof["pressure_bar"],
        linewidth=2.5,
        label="Wellbore path",
    )

    if all(math.isfinite(x) for x in [bh_T, bh_P]):
        ax.scatter([bh_T], [bh_P], s=60, zorder=5)
        ax.annotate(
            "Bottomhole",
            xy=(bh_T, bh_P),
            xytext=(bh_T + 4.0, bh_P - 23.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black"},
        )

    # Move saturation label away from the curve.
    if not dew.empty:
        idx = min(len(dew) - 1, max(0, int(0.48 * len(dew))))
        dew_T = float(dew["temperature_C"].iloc[idx])
        dew_P = float(dew["pressure_bar"].iloc[idx])
        ax.annotate(
            "Saturation line",
            xy=(dew_T, dew_P),
            xytext=(dew_T - 36.0, dew_P + 32.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black"},
        )

    vapor_pct = quality * 100.0 if math.isfinite(quality) else float("nan")
    info = (
        f"Rate: {q_sm3_day:.0f} sm3/day\n"
        f"Post-choke vapor: {vapor_pct:.1f}%\n"
        f"Phase: {phase}"
    )
    ax.text(
        0.02,
        0.03,
        info,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "black", "alpha": 0.9},
    )

    ax.set_title(f"P-T Path - Step {step} ({opening:.1f}% Opening)")
    ax.set_xlabel("Temperature, C")
    ax.set_ylabel("Pressure, bar")
    ax.grid(True)
    ax.legend(loc="best")

    T_values = _finite_series(
        list(prof["temperature_C"])
        + [upstream_T, downstream_T, bh_T]
        + ([] if dew.empty else list(dew["temperature_C"]))
    )
    P_values = _finite_series(
        list(prof["pressure_bar"])
        + [upstream_P, downstream_P, bh_P]
        + ([] if dew.empty else list(dew["pressure_bar"]))
    )

    if T_values:
        ax.set_xlim(min(T_values) - 8.0, max(T_values) + 8.0)
    if P_values:
        ax.set_ylim(max(0.0, min(P_values) - 8.0), max(P_values) + 15.0)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"07_PT_step_{step:04d}_opening_{opening:.1f}pct.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return out_path


def plot_pt_diagram_all_steps(
    *,
    accepted_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    figures_dir: Path,
    case_id: str,
) -> Optional[Path]:
    """Create one combined P-T diagram for all accepted coupling steps."""
    import matplotlib.pyplot as plt

    if accepted_df.empty or profiles_df.empty:
        return None
    if "temperature_C" not in profiles_df or "pressure_bar" not in profiles_df or "step" not in profiles_df:
        return None

    dew = co2_saturation_curve_PT()

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    if not dew.empty:
        ax.plot(
            dew["temperature_C"],
            dew["pressure_bar"],
            linewidth=2.5,
            label="CO2 saturation line",
        )

    for _, row in accepted_df.iterrows():
        step = int(row.get("step", 0))
        opening = float(row.get("opening_percent", float("nan")))
        g = profiles_df[profiles_df["step"] == step].copy()
        if g.empty:
            continue
        g = g.sort_values("depth_m") if "depth_m" in g else g

        ax.plot(
            g["temperature_C"],
            g["pressure_bar"],
            linewidth=1.9,
            label=f"Step {step}: {opening:.1f}%",
        )

        upstream_T = _safe_float(row.get("choke_upstream_temperature_C", float("nan")))
        upstream_P = _safe_float(row.get("choke_upstream_pressure_bar", float("nan")))
        downstream_T = _safe_float(row.get("choke_downstream_temperature_C", float("nan")))
        downstream_P = _safe_float(row.get("choke_downstream_pressure_bar", float("nan")))

        if all(math.isfinite(x) for x in [upstream_T, upstream_P, downstream_T, downstream_P]):
            ax.plot(
                [upstream_T, downstream_T],
                [upstream_P, downstream_P],
                linestyle="--",
                linewidth=1.2,
            )

    ax.set_title("P-T Paths - Accepted Steps")
    ax.set_xlabel("Temperature, C")
    ax.set_ylabel("Pressure, bar")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "07_PT_all_accepted_steps.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return out_path


def profile_phase_fraction_arrays(profile_step: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return vapor and liquid/dense fractions along the wellbore profile."""
    vapor: List[float] = []

    q_series = profile_step["quality_mass"] if "quality_mass" in profile_step else pd.Series([float("nan")] * len(profile_step))
    phase_series = profile_step["phase_label"] if "phase_label" in profile_step else pd.Series([""] * len(profile_step))

    for q_raw, phase_raw in zip(q_series, phase_series):
        q = _safe_float(q_raw)
        phase = str(phase_raw).lower()

        if math.isfinite(q) and 0.0 <= q <= 1.0:
            vf = q
        elif ("vapor" in phase or "gas" in phase) and "dense" not in phase:
            vf = 1.0
        else:
            vf = 0.0

        vapor.append(float(vf))

    vapor_arr = np.asarray(vapor, dtype=float)
    liquid_arr = 1.0 - vapor_arr
    return vapor_arr, liquid_arr


def plot_wellbore_phase_distribution_for_step(
    *,
    row: pd.Series,
    profile_step: pd.DataFrame,
    figures_dir: Path,
) -> Optional[Path]:
    """Plot vapor/liquid-dense mass fraction distribution along wellbore for one step."""
    import matplotlib.pyplot as plt

    if profile_step.empty or "depth_m" not in profile_step:
        return None

    step = int(row.get("step", 0))
    opening = float(row.get("opening_percent", float("nan")))

    g = profile_step.copy().sort_values("depth_m")
    vapor, liquid = profile_phase_fraction_arrays(g)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    ax.plot(vapor * 100.0, g["depth_m"], linewidth=2.2, label="Vapor")
    ax.plot(liquid * 100.0, g["depth_m"], linewidth=2.2, label="Liquid/dense")

    ax.invert_yaxis()
    ax.set_xlim(-2.0, 102.0)
    ax.set_xlabel("Mass fraction, %")
    ax.set_ylabel("Depth, m")
    ax.set_title(f"Phase Distribution - Step {step} ({opening:.1f}% Opening)")
    ax.grid(True)
    ax.legend(loc="best")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"07_wellbore_phase_distribution_step_{step:04d}_opening_{opening:.1f}pct.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return out_path


def plot_wellbore_vapor_fraction_all_steps(
    *,
    accepted_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    figures_dir: Path,
) -> Optional[Path]:
    """Plot vapor fraction profiles for all accepted steps."""
    import matplotlib.pyplot as plt

    if accepted_df.empty or profiles_df.empty or "step" not in profiles_df or "depth_m" not in profiles_df:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for _, row in accepted_df.iterrows():
        step = int(row.get("step", 0))
        opening = float(row.get("opening_percent", float("nan")))
        g = profiles_df[profiles_df["step"] == step].copy()
        if g.empty:
            continue
        g = g.sort_values("depth_m")
        vapor, _ = profile_phase_fraction_arrays(g)

        ax.plot(vapor * 100.0, g["depth_m"], linewidth=1.9, label=f"Step {step}: {opening:.1f}%")

    ax.invert_yaxis()
    ax.set_xlim(-2.0, 102.0)
    ax.set_xlabel("Vapor mass fraction, %")
    ax.set_ylabel("Depth, m")
    ax.set_title("Vapor Fraction Profiles")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "07_wellbore_vapor_fraction_profiles.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return out_path


def generate_coupling_figures(
    accepted_df: pd.DataFrame,
    iterations_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    figures_dir: Path,
    case_id: str,
) -> List[Path]:
    """Generate informative coupling figures for the 07 OPM/choke/wellbore run."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    saved: List[Path] = []

    def save_current(name: str) -> None:
        path = figures_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        saved.append(path)

    if not accepted_df.empty:
        x = accepted_df["opening_percent"] if "opening_percent" in accepted_df else accepted_df["step"]

        if "reservoir_rate_sm3_day" in accepted_df:
            plt.figure()
            plt.plot(x, accepted_df["reservoir_rate_sm3_day"], marker="o")
            plt.xlabel("Choke opening, %")
            plt.ylabel("Accepted CO2 injection rate, sm3/day")
            plt.title("Accepted Rate vs Opening")
            plt.grid(True)
            save_current("07_accepted_rate_vs_opening.png")

        pressure_cols = [
            ("reservoir_bhp_bar", "OPM WBHP"),
            ("wellbore_required_thp_bar", "Required THP after choke"),
        ]
        available = [(c, label) for c, label in pressure_cols if c in accepted_df]
        if available:
            plt.figure()
            for c, label in available:
                plt.plot(x, accepted_df[c], marker="o", label=label)
            plt.xlabel("Choke opening, %")
            plt.ylabel("Pressure, bar")
            plt.title("Pressure Response")
            plt.grid(True)
            plt.legend()
            save_current("07_pressure_response_vs_opening.png")

        temp_cols = [
            ("choke_downstream_temperature_C", "Post-choke T"),
            ("wellbore_bht_calc_C", "BHT calculated"),
            ("control_wtemp_used_C", "WTEMP used in OPM"),
        ]
        available = [(c, label) for c, label in temp_cols if c in accepted_df]
        if available:
            plt.figure()
            for c, label in available:
                plt.plot(x, accepted_df[c], marker="o", label=label)
            plt.xlabel("Choke opening, %")
            plt.ylabel("Temperature, C")
            plt.title("Temperature Response")
            plt.grid(True)
            plt.legend()
            save_current("07_temperature_response_vs_opening.png")

        if "choke_downstream_vapor_mass_fraction" in accepted_df:
            plt.figure()
            plt.plot(
                x,
                accepted_df["choke_downstream_vapor_mass_fraction"] * 100.0,
                marker="o",
                label="Vapor",
            )
            if "choke_downstream_liquid_or_dense_mass_fraction" in accepted_df:
                plt.plot(
                    x,
                    accepted_df["choke_downstream_liquid_or_dense_mass_fraction"] * 100.0,
                    marker="o",
                    label="Liquid/dense",
                )
            plt.xlabel("Choke opening, %")
            plt.ylabel("Mass fraction, %")
            plt.title("Post-Choke Phase Fractions")
            plt.ylim(-2.0, 102.0)
            plt.grid(True)
            plt.legend()
            save_current("07_post_choke_phase_fractions_vs_opening_pct.png")

        if "choke_downstream_vapor_mass_rate_kg_s" in accepted_df:
            plt.figure()
            plt.plot(x, accepted_df["choke_downstream_vapor_mass_rate_kg_s"], marker="o", label="Vapor CO2")
            if "choke_downstream_liquid_or_dense_mass_rate_kg_s" in accepted_df:
                plt.plot(x, accepted_df["choke_downstream_liquid_or_dense_mass_rate_kg_s"], marker="o", label="Liquid/dense CO2")
            plt.xlabel("Choke opening, %")
            plt.ylabel("Mass rate, kg/s")
            plt.title("Post-Choke Phase Mass Rates")
            plt.grid(True)
            plt.legend()
            save_current("07_post_choke_phase_mass_rates_vs_opening.png")

    if not iterations_df.empty:
        if "choke_capacity_residual_kg_s" in iterations_df:
            plt.figure()
            for step, g in iterations_df.groupby("step"):
                plt.plot(g["iteration"], g["choke_capacity_residual_kg_s"], marker="o", label=f"step {step}")
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Coupling iteration")
            plt.ylabel("HEM residual, kg/s")
            plt.title("Choke Flow Residual")
            plt.grid(True)
            plt.legend()
            save_current("07_convergence_choke_flow_residual_kg_s.png")

        if "choke_kv_residual_m3_h" in iterations_df:
            plt.figure()
            for step, g in iterations_df.groupby("step"):
                plt.plot(g["iteration"], g["choke_kv_residual_m3_h"], marker="o", label=f"step {step}")
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Coupling iteration")
            plt.ylabel("Kv residual, m3/h")
            plt.title("Choke Residual Convergence")
            plt.grid(True)
            plt.legend()
            save_current("07_convergence_choke_kv_residual.png")

        if "residual_temperature_bht_minus_wtemp_C" in iterations_df:
            plt.figure()
            for step, g in iterations_df.groupby("step"):
                plt.plot(g["iteration"], g["residual_temperature_bht_minus_wtemp_C"], marker="o", label=f"step {step}")
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Coupling iteration")
            plt.ylabel("BHT - WTEMP, C")
            plt.title("Temperature Residual Convergence")
            plt.grid(True)
            plt.legend()
            save_current("07_convergence_temperature_residual.png")

    if not profiles_df.empty:
        # Accepted wellbore profiles. Use all accepted steps; for one-step run it is one curve.
        for ycol, ylabel, fname in [
            ("pressure_bar", "Pressure, bar", "07_wellbore_profile_pressure.png"),
            ("temperature_C", "Temperature, C", "07_wellbore_profile_temperature.png"),
            ("quality_mass", "CO2 quality / vapor mass fraction", "07_wellbore_profile_quality.png"),
        ]:
            if ycol in profiles_df and "depth_m" in profiles_df:
                plt.figure()
                for step, g in profiles_df.groupby("step"):
                    plt.plot(g[ycol], g["depth_m"], marker=None, label=f"step {step}")
                plt.gca().invert_yaxis()
                plt.xlabel(ylabel)
                plt.ylabel("Depth, m")
                plt.title(f"Wellbore {ylabel}")
                plt.grid(True)
                plt.legend()
                save_current(fname)

    # P-T diagrams per accepted coupling step + combined overview.
    if not accepted_df.empty and not profiles_df.empty:
        if {"step", "temperature_C", "pressure_bar"}.issubset(set(profiles_df.columns)):
            for _, row in accepted_df.iterrows():
                step = int(row.get("step", 0))
                profile_step = profiles_df[profiles_df["step"] == step].copy()

                pt_path = plot_pt_diagram_for_step(
                    row=row,
                    profile_step=profile_step,
                    figures_dir=figures_dir,
                    case_id=case_id,
                )

                if pt_path is not None:
                    saved.append(pt_path)

            all_pt_path = plot_pt_diagram_all_steps(
                accepted_df=accepted_df,
                profiles_df=profiles_df,
                figures_dir=figures_dir,
                case_id=case_id,
            )

            if all_pt_path is not None:
                saved.append(all_pt_path)

    # Wellbore phase distribution figures.
    if not accepted_df.empty and not profiles_df.empty:
        if {"step", "depth_m"}.issubset(set(profiles_df.columns)):
            for _, row in accepted_df.iterrows():
                step = int(row.get("step", 0))
                profile_step = profiles_df[profiles_df["step"] == step].copy()

                phase_path = plot_wellbore_phase_distribution_for_step(
                    row=row,
                    profile_step=profile_step,
                    figures_dir=figures_dir,
                )

                if phase_path is not None:
                    saved.append(phase_path)

            vapor_profile_path = plot_wellbore_vapor_fraction_all_steps(
                accepted_df=accepted_df,
                profiles_df=profiles_df,
                figures_dir=figures_dir,
            )

            if vapor_profile_path is not None:
                saved.append(vapor_profile_path)

    return saved
