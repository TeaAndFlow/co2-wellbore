#!/usr/bin/env python3
"""Generate detailed capability/stress-test figures from capability_envelope.csv.

This script is designed for the v0.2.9 capability-envelope release.

It makes both:
1) simple global figures;
2) scientifically useful sliced figures:
   - HEM maps and pressure/temperature/opening responses;
   - wellbore warning maps;
   - wellbore BHP/velocity response under fixed setups;
   - n_segments convergence diagnostics;
   - runtime scaling diagnostics;
   - property density and phase maps.

No seaborn is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[figure] {path}")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_label(x: object) -> str:
    try:
        xf = float(x)
        if np.isfinite(xf):
            return f"{xf:g}"
    except Exception:
        pass
    return str(x)


def nearest_value(values: Iterable[float], target: float) -> float | None:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return None
    return min(vals, key=lambda v: abs(v - target))


def heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if pivot.empty:
        return

    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=cbar_label)
    plt.xticks(range(len(pivot.columns)), [clean_label(x) for x in pivot.columns], rotation=0)
    plt.yticks(range(len(pivot.index)), [clean_label(x) for x in pivot.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[table] {path}")


def prepare_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "pressure_bar",
        "temperature_C",
        "upstream_pressure_bar",
        "upstream_temperature_C",
        "downstream_pressure_bar",
        "opening_fraction",
        "thp_bar",
        "q_sm3_day",
        "mass_rate_kg_s",
        "tvd_m",
        "diameter_m",
        "n_segments",
        "quality_mass",
        "density_kg_m3",
        "enthalpy_J_kg",
        "viscosity_Pa_s",
        "cp_J_kgK",
        "critical_pressure_bar",
        "selected_throat_pressure_bar",
        "selected_throat_quality_mass",
        "mass_flow_capacity_kg_s",
        "bhp_bar",
        "bht_C",
        "min_pressure_bar",
        "max_pressure_bar",
        "max_velocity_m_s",
        "max_reynolds",
        "two_phase_fraction",
        "runtime_s",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def write_diagnostic_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    status = (
        df.groupby(["component", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["component", "status"])
    )
    write_table(status, tables_dir / "capability_status_counts.csv")

    bad = df[df["status"].isin(["fail", "warning"])].copy()
    if not bad.empty:
        reasons = (
            bad.groupby(["component", "status", "failure_reason"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        write_table(reasons, tables_dir / "capability_failure_reason_counts.csv")
        write_table(
            bad.sort_values(["component", "failure_reason", "case_id"]),
            tables_dir / "capability_bad_cases.csv",
        )

    runtime = (
        df.groupby("component")["runtime_s"]
        .agg(["count", "mean", "median", "max", "sum"])
        .reset_index()
        .sort_values("sum", ascending=False)
    )
    write_table(runtime, tables_dir / "capability_runtime_by_component.csv")

    slow = df.sort_values("runtime_s", ascending=False).head(200)
    write_table(slow, tables_dir / "capability_slowest_200_cases.csv")

    hem = df[df["component"] == "hem_choke"].copy()
    if not hem.empty:
        write_table(
            hem["flow_regime"].fillna("").value_counts().reset_index(name="count"),
            tables_dir / "capability_hem_flow_regime_counts.csv",
        )
        write_table(
            hem["critical_search_method"].fillna("").value_counts().reset_index(name="count"),
            tables_dir / "capability_hem_search_method_counts.csv",
        )

    wb = df[df["component"] == "wellbore"].copy()
    if not wb.empty:
        write_table(
            wb["failure_reason"].fillna("").replace("", "OK").value_counts().reset_index(name="count"),
            tables_dir / "capability_wellbore_reason_counts.csv",
        )
        write_table(
            wb["phase_label"].fillna("").value_counts().reset_index(name="count"),
            tables_dir / "capability_wellbore_phase_counts.csv",
        )


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------

def plot_status_counts(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df.groupby(["component", "status"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Capability sweep status by component")
    plt.xlabel("Component")
    plt.ylabel("Number of cases")
    plt.xticks(rotation=0)
    savefig(figures_dir / "capability_status_by_component.png")


def plot_status_fraction(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df.groupby(["component", "status"]).size().unstack(fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Capability sweep status fraction by component")
    plt.xlabel("Component")
    plt.ylabel("Fraction of cases")
    plt.xticks(rotation=0)
    savefig(figures_dir / "capability_status_fraction_by_component.png")


def plot_failure_reasons(df: pd.DataFrame, figures_dir: Path) -> None:
    bad = df[df["status"].isin(["fail", "warning"])].copy()
    if bad.empty:
        return

    counts = bad["failure_reason"].replace("", "UNKNOWN").value_counts().head(20)
    counts.sort_values().plot(kind="barh", figsize=(11, 7))
    plt.title("Top capability sweep failure/warning reasons")
    plt.xlabel("Number of cases")
    plt.ylabel("Failure/warning reason")
    savefig(figures_dir / "capability_failure_reasons.png")


def plot_failure_reasons_log(df: pd.DataFrame, figures_dir: Path) -> None:
    bad = df[df["status"].isin(["fail", "warning"])].copy()
    if bad.empty:
        return

    counts = bad["failure_reason"].replace("", "UNKNOWN").value_counts().head(20)
    counts.sort_values().plot(kind="barh", figsize=(11, 7))
    plt.xscale("log")
    plt.title("Top capability sweep failure/warning reasons, log scale")
    plt.xlabel("Number of cases, log scale")
    plt.ylabel("Failure/warning reason")
    savefig(figures_dir / "capability_failure_reasons_log.png")


# ---------------------------------------------------------------------------
# Property plots
# ---------------------------------------------------------------------------

def plot_property_ok_map(df: pd.DataFrame, figures_dir: Path) -> None:
    prop = df[df["component"] == "properties"].copy()
    if prop.empty:
        return

    prop["ok_flag"] = (prop["status"] == "ok").astype(float)

    pivot = prop.pivot_table(
        index="temperature_C",
        columns="pressure_bar",
        values="ok_flag",
        aggfunc="mean",
    )

    heatmap(
        pivot,
        title="CO2 property backend P/T capability map",
        xlabel="Pressure, bar",
        ylabel="Temperature, C",
        cbar_label="OK fraction",
        path=figures_dir / "capability_property_pt_ok_map.png",
        vmin=0.0,
        vmax=1.0,
    )


def plot_property_density_map(df: pd.DataFrame, figures_dir: Path) -> None:
    prop = df[(df["component"] == "properties") & (df["status"] == "ok")].copy()
    if prop.empty:
        return

    pivot = prop.pivot_table(
        index="temperature_C",
        columns="pressure_bar",
        values="density_kg_m3",
        aggfunc="median",
    )

    heatmap(
        pivot,
        title="CO2 density map over tested P/T grid",
        xlabel="Pressure, bar",
        ylabel="Temperature, C",
        cbar_label="Median density, kg/m3",
        path=figures_dir / "capability_property_density_map.png",
    )


def plot_property_phase_counts(df: pd.DataFrame, figures_dir: Path) -> None:
    prop = df[df["component"] == "properties"].copy()
    if prop.empty:
        return

    counts = prop["phase_label"].fillna("").replace("", "UNKNOWN").value_counts()
    counts.sort_values().plot(kind="barh", figsize=(9, 5))
    plt.title("CO2 property phase labels in tested P/T grid")
    plt.xlabel("Number of cases")
    plt.ylabel("Phase label")
    savefig(figures_dir / "capability_property_phase_counts.png")


# ---------------------------------------------------------------------------
# HEM plots
# ---------------------------------------------------------------------------

def hem_ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["component"] == "hem_choke") & (df["status"] == "ok")].copy()


def plot_hem_capacity_vs_opening(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = hem_ok(df)
    if hem.empty:
        return

    hem["opening_percent"] = 100.0 * hem["opening_fraction"]
    curve = hem.groupby("opening_percent")["mass_flow_capacity_kg_s"].median().dropna().sort_index()
    if curve.empty:
        return

    curve.plot(marker="o", figsize=(8, 5))
    plt.title("HEM choke median capacity vs opening")
    plt.xlabel("Opening, %")
    plt.ylabel("Median mass-flow capacity, kg/s")
    plt.grid(True)
    savefig(figures_dir / "capability_hem_capacity_vs_opening.png")


def plot_hem_capacity_map(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = hem_ok(df)
    if hem.empty:
        return

    opening = nearest_value(hem["opening_fraction"].dropna().unique(), 0.20)
    p2 = nearest_value(hem["downstream_pressure_bar"].dropna().unique(), 20.0)

    subset = hem[
        hem["opening_fraction"].sub(opening).abs().lt(1.0e-12)
        & hem["downstream_pressure_bar"].sub(p2).abs().lt(1.0e-12)
    ].copy()

    pivot = subset.pivot_table(
        index="upstream_temperature_C",
        columns="upstream_pressure_bar",
        values="mass_flow_capacity_kg_s",
        aggfunc="median",
    )

    heatmap(
        pivot,
        title=f"HEM capacity map, opening={100*opening:g}% / P2={p2:g} bar",
        xlabel="Upstream pressure, bar",
        ylabel="Upstream temperature, C",
        cbar_label="Median capacity, kg/s",
        path=figures_dir / "capability_hem_capacity_map_p1_t1.png",
    )


def plot_hem_capacity_vs_p1_by_temperature(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = hem_ok(df)
    if hem.empty:
        return

    opening = nearest_value(hem["opening_fraction"].dropna().unique(), 0.20)
    p2 = nearest_value(hem["downstream_pressure_bar"].dropna().unique(), 20.0)

    subset = hem[
        hem["opening_fraction"].sub(opening).abs().lt(1.0e-12)
        & hem["downstream_pressure_bar"].sub(p2).abs().lt(1.0e-12)
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    for t1, grp in subset.groupby("upstream_temperature_C"):
        grp = grp.sort_values("upstream_pressure_bar")
        plt.plot(
            grp["upstream_pressure_bar"],
            grp["mass_flow_capacity_kg_s"],
            marker="o",
            label=f"T1={t1:g} C",
        )

    plt.title(f"HEM capacity vs upstream pressure, opening={100*opening:g}% / P2={p2:g} bar")
    plt.xlabel("Upstream pressure, bar")
    plt.ylabel("Mass-flow capacity, kg/s")
    plt.grid(True)
    plt.legend(fontsize=8)
    savefig(figures_dir / "capability_hem_capacity_vs_p1_by_temperature.png")


def plot_hem_capacity_vs_p2_by_pressure(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = hem_ok(df)
    if hem.empty:
        return

    opening = nearest_value(hem["opening_fraction"].dropna().unique(), 0.20)
    t1 = nearest_value(hem["upstream_temperature_C"].dropna().unique(), 20.0)

    subset = hem[
        hem["opening_fraction"].sub(opening).abs().lt(1.0e-12)
        & hem["upstream_temperature_C"].sub(t1).abs().lt(1.0e-12)
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    for p1, grp in subset.groupby("upstream_pressure_bar"):
        grp = grp.sort_values("downstream_pressure_bar")
        plt.plot(
            grp["downstream_pressure_bar"],
            grp["mass_flow_capacity_kg_s"],
            marker="o",
            label=f"P1={p1:g} bar",
        )

    plt.title(f"HEM capacity vs downstream pressure, opening={100*opening:g}% / T1={t1:g} C")
    plt.xlabel("Downstream pressure, bar")
    plt.ylabel("Mass-flow capacity, kg/s")
    plt.grid(True)
    plt.legend(fontsize=8)
    savefig(figures_dir / "capability_hem_capacity_vs_p2_by_pressure.png")


def plot_hem_critical_pressure_ratio_map(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = hem_ok(df)
    if hem.empty:
        return

    opening = nearest_value(hem["opening_fraction"].dropna().unique(), 0.20)
    p2 = nearest_value(hem["downstream_pressure_bar"].dropna().unique(), 20.0)

    subset = hem[
        hem["opening_fraction"].sub(opening).abs().lt(1.0e-12)
        & hem["downstream_pressure_bar"].sub(p2).abs().lt(1.0e-12)
    ].copy()

    if subset.empty:
        return

    subset["critical_pressure_ratio"] = subset["critical_pressure_bar"] / subset["upstream_pressure_bar"]

    pivot = subset.pivot_table(
        index="upstream_temperature_C",
        columns="upstream_pressure_bar",
        values="critical_pressure_ratio",
        aggfunc="median",
    )

    heatmap(
        pivot,
        title=f"HEM critical pressure ratio Pc/P1, opening={100*opening:g}% / P2={p2:g} bar",
        xlabel="Upstream pressure, bar",
        ylabel="Upstream temperature, C",
        cbar_label="Critical pressure ratio Pc/P1",
        path=figures_dir / "capability_hem_critical_pressure_ratio_map.png",
    )


def plot_hem_runtime_map(df: pd.DataFrame, figures_dir: Path) -> None:
    hem = df[df["component"] == "hem_choke"].copy()
    if hem.empty:
        return

    pivot = hem.pivot_table(
        index="upstream_temperature_C",
        columns="upstream_pressure_bar",
        values="runtime_s",
        aggfunc="median",
    )

    heatmap(
        pivot,
        title="HEM median runtime over P1/T1 grid",
        xlabel="Upstream pressure, bar",
        ylabel="Upstream temperature, C",
        cbar_label="Runtime, s",
        path=figures_dir / "capability_hem_runtime_map.png",
    )


# ---------------------------------------------------------------------------
# Wellbore plots
# ---------------------------------------------------------------------------

def wellbore_all(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["component"] == "wellbore"].copy()


def wellbore_ok_warning(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["component"] == "wellbore")
        & (df["status"].isin(["ok", "warning"]))
    ].copy()


def representative_wellbore_slice(wb: pd.DataFrame) -> pd.DataFrame:
    """A clean fixed setup for line plots.

    Prefer:
    THP=100 bar, TVD=1600 m, D=0.10 m, N=160,
    thermal=overall_U, T_head=20 C.
    """

    out = wb.copy()

    targets = {
        "thp_bar": 100.0,
        "tvd_m": 1600.0,
        "diameter_m": 0.10,
        "n_segments": 160.0,
        "temperature_C": 20.0,
    }

    mask = pd.Series(True, index=out.index)

    for col, target in targets.items():
        val = nearest_value(out[col].dropna().unique(), target)
        if val is not None:
            mask &= out[col].sub(val).abs().lt(1.0e-12)

    if "thermal_model" in out.columns and (out["thermal_model"] == "overall_U").any():
        mask &= out["thermal_model"].eq("overall_U")

    sliced = out[mask].copy()

    if not sliced.empty:
        return sliced

    return out.copy()


def plot_wellbore_bhp_vs_rate(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    wb = representative_wellbore_slice(wb)

    curve = wb.groupby("q_sm3_day")["bhp_bar"].median().dropna().sort_index()
    if curve.empty:
        return

    curve.plot(marker="o", figsize=(9, 5))
    plt.title("Wellbore BHP vs injection rate, representative slice")
    plt.xlabel("Injection rate, sm3/day")
    plt.ylabel("BHP, bar")
    plt.grid(True)
    savefig(figures_dir / "capability_wellbore_bhp_vs_rate.png")


def plot_wellbore_velocity_vs_rate(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    wb = representative_wellbore_slice(wb)

    curve = wb.groupby("q_sm3_day")["max_velocity_m_s"].median().dropna().sort_index()
    if curve.empty:
        return

    curve.plot(marker="o", figsize=(9, 5))
    plt.title("Wellbore max velocity vs injection rate, representative slice")
    plt.xlabel("Injection rate, sm3/day")
    plt.ylabel("Max velocity, m/s")
    plt.grid(True)
    savefig(figures_dir / "capability_wellbore_velocity_vs_rate.png")


def plot_wellbore_bhp_vs_rate_by_tvd(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    # Fixed setup except TVD and q.
    d = nearest_value(wb["diameter_m"].dropna().unique(), 0.10)
    n = nearest_value(wb["n_segments"].dropna().unique(), 160.0)
    thp = nearest_value(wb["thp_bar"].dropna().unique(), 100.0)
    temp = nearest_value(wb["temperature_C"].dropna().unique(), 20.0)

    subset = wb[
        wb["diameter_m"].sub(d).abs().lt(1.0e-12)
        & wb["n_segments"].sub(n).abs().lt(1.0e-12)
        & wb["thp_bar"].sub(thp).abs().lt(1.0e-12)
        & wb["temperature_C"].sub(temp).abs().lt(1.0e-12)
        & wb["thermal_model"].eq("overall_U")
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    for tvd, grp in subset.groupby("tvd_m"):
        curve = grp.groupby("q_sm3_day")["bhp_bar"].median().dropna().sort_index()
        plt.plot(curve.index, curve.values, marker="o", label=f"TVD={tvd:g} m")

    plt.title(f"Wellbore BHP vs rate by TVD, THP={thp:g} bar, D={d:g} m, N={n:g}")
    plt.xlabel("Injection rate, sm3/day")
    plt.ylabel("BHP, bar")
    plt.grid(True)
    plt.legend()
    savefig(figures_dir / "capability_wellbore_bhp_vs_rate_by_tvd.png")


def plot_wellbore_bhp_vs_rate_by_thp(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    d = nearest_value(wb["diameter_m"].dropna().unique(), 0.10)
    n = nearest_value(wb["n_segments"].dropna().unique(), 160.0)
    tvd = nearest_value(wb["tvd_m"].dropna().unique(), 1600.0)
    temp = nearest_value(wb["temperature_C"].dropna().unique(), 20.0)

    subset = wb[
        wb["diameter_m"].sub(d).abs().lt(1.0e-12)
        & wb["n_segments"].sub(n).abs().lt(1.0e-12)
        & wb["tvd_m"].sub(tvd).abs().lt(1.0e-12)
        & wb["temperature_C"].sub(temp).abs().lt(1.0e-12)
        & wb["thermal_model"].eq("overall_U")
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    for thp, grp in subset.groupby("thp_bar"):
        curve = grp.groupby("q_sm3_day")["bhp_bar"].median().dropna().sort_index()
        plt.plot(curve.index, curve.values, marker="o", label=f"THP={thp:g} bar")

    plt.title(f"Wellbore BHP vs rate by THP, TVD={tvd:g} m, D={d:g} m, N={n:g}")
    plt.xlabel("Injection rate, sm3/day")
    plt.ylabel("BHP, bar")
    plt.grid(True)
    plt.legend()
    savefig(figures_dir / "capability_wellbore_bhp_vs_rate_by_thp.png")


def plot_wellbore_velocity_vs_rate_by_diameter(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    n = nearest_value(wb["n_segments"].dropna().unique(), 160.0)
    tvd = nearest_value(wb["tvd_m"].dropna().unique(), 1600.0)
    thp = nearest_value(wb["thp_bar"].dropna().unique(), 100.0)
    temp = nearest_value(wb["temperature_C"].dropna().unique(), 20.0)

    subset = wb[
        wb["n_segments"].sub(n).abs().lt(1.0e-12)
        & wb["tvd_m"].sub(tvd).abs().lt(1.0e-12)
        & wb["thp_bar"].sub(thp).abs().lt(1.0e-12)
        & wb["temperature_C"].sub(temp).abs().lt(1.0e-12)
        & wb["thermal_model"].eq("overall_U")
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    for d, grp in subset.groupby("diameter_m"):
        curve = grp.groupby("q_sm3_day")["max_velocity_m_s"].median().dropna().sort_index()
        plt.plot(curve.index, curve.values, marker="o", label=f"D={d:g} m")

    plt.title(f"Wellbore velocity vs rate by diameter, THP={thp:g} bar, TVD={tvd:g} m, N={n:g}")
    plt.xlabel("Injection rate, sm3/day")
    plt.ylabel("Max velocity, m/s")
    plt.grid(True)
    plt.legend()
    savefig(figures_dir / "capability_wellbore_velocity_vs_rate_by_diameter.png")


def plot_wellbore_warning_fraction_map(
    df: pd.DataFrame,
    figures_dir: Path,
    *,
    reason: str,
    x_col: str,
    y_col: str,
    filename: str,
    title: str,
) -> None:
    wb = wellbore_all(df)
    if wb.empty:
        return

    wb["flag"] = wb["failure_reason"].fillna("").eq(reason).astype(float)

    pivot = wb.pivot_table(
        index=y_col,
        columns=x_col,
        values="flag",
        aggfunc="mean",
    )

    heatmap(
        pivot,
        title=title,
        xlabel=x_col,
        ylabel=y_col,
        cbar_label=f"Fraction with {reason}",
        path=figures_dir / filename,
        vmin=0.0,
        vmax=1.0,
    )


def plot_wellbore_min_pressure_map(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    # Representative physical setup except q and THP.
    d = nearest_value(wb["diameter_m"].dropna().unique(), 0.10)
    n = nearest_value(wb["n_segments"].dropna().unique(), 160.0)
    tvd = nearest_value(wb["tvd_m"].dropna().unique(), 1600.0)
    temp = nearest_value(wb["temperature_C"].dropna().unique(), 20.0)

    subset = wb[
        wb["diameter_m"].sub(d).abs().lt(1.0e-12)
        & wb["n_segments"].sub(n).abs().lt(1.0e-12)
        & wb["tvd_m"].sub(tvd).abs().lt(1.0e-12)
        & wb["temperature_C"].sub(temp).abs().lt(1.0e-12)
        & wb["thermal_model"].eq("overall_U")
    ].copy()

    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="thp_bar",
        columns="q_sm3_day",
        values="min_pressure_bar",
        aggfunc="median",
    )

    heatmap(
        pivot,
        title=f"Wellbore minimum pressure map, TVD={tvd:g} m, D={d:g} m, N={n:g}",
        xlabel="q_sm3_day",
        ylabel="thp_bar",
        cbar_label="Median minimum pressure, bar",
        path=figures_dir / "capability_wellbore_min_pressure_q_thp_map.png",
    )


def plot_wellbore_runtime_vs_nsegments(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_all(df)
    if wb.empty:
        return

    curve = wb.groupby("n_segments")["runtime_s"].median().dropna().sort_index()
    if curve.empty:
        return

    curve.plot(marker="o", figsize=(8, 5))
    plt.title("Wellbore median runtime vs number of segments")
    plt.xlabel("n_segments")
    plt.ylabel("Median runtime, s")
    plt.grid(True)
    savefig(figures_dir / "capability_wellbore_runtime_vs_nsegments.png")


def grid_convergence_table(wb: pd.DataFrame, value_col: str) -> pd.DataFrame:
    group_cols = [
        "thp_bar",
        "q_sm3_day",
        "tvd_m",
        "diameter_m",
        "thermal_model",
        "temperature_C",
    ]

    records = []

    for _, grp in wb.groupby(group_cols):
        grp = grp.dropna(subset=["n_segments", value_col]).copy()
        if grp["n_segments"].nunique() < 2:
            continue

        ref_n = grp["n_segments"].max()
        ref_values = grp[grp["n_segments"] == ref_n][value_col]

        if ref_values.empty:
            continue

        ref = float(ref_values.median())

        for _, row in grp.iterrows():
            n = float(row["n_segments"])
            if n == ref_n:
                continue
            val = float(row[value_col])
            records.append(
                {
                    "n_segments": n,
                    "reference_n_segments": ref_n,
                    "abs_error": abs(val - ref),
                    "signed_error": val - ref,
                    "value_col": value_col,
                }
            )

    return pd.DataFrame(records)


def plot_wellbore_grid_convergence(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_ok_warning(df)
    if wb.empty:
        return

    for value_col, ylabel, filename in [
        ("bhp_bar", "Median |BHP(N) - BHP(Nmax)|, bar", "capability_wellbore_bhp_grid_convergence.png"),
        ("bht_C", "Median |BHT(N) - BHT(Nmax)|, C", "capability_wellbore_bht_grid_convergence.png"),
    ]:
        conv = grid_convergence_table(wb, value_col)
        if conv.empty:
            continue

        curve = conv.groupby("n_segments")["abs_error"].median().dropna().sort_index()
        if curve.empty:
            continue

        curve.plot(marker="o", figsize=(8, 5))
        plt.title(f"Wellbore grid convergence: {value_col}")
        plt.xlabel("n_segments")
        plt.ylabel(ylabel)
        plt.grid(True)
        savefig(figures_dir / filename)


def plot_wellbore_phase_counts(df: pd.DataFrame, figures_dir: Path) -> None:
    wb = wellbore_all(df)
    if wb.empty:
        return

    counts = wb["phase_label"].fillna("").replace("", "UNKNOWN").value_counts()
    counts.sort_values().plot(kind="barh", figsize=(9, 5))
    plt.title("Wellbore bottom phase labels")
    plt.xlabel("Number of cases")
    plt.ylabel("Phase label")
    savefig(figures_dir / "capability_wellbore_phase_counts.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="validation/results/capability_envelope.csv")
    parser.add_argument("--figures-dir", default="validation/figures")
    parser.add_argument(
        "--tables-dir",
        default=None,
        help="Directory for diagnostic CSV tables. Default: same directory as input CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    tables_dir = Path(args.tables_dir).expanduser().resolve() if args.tables_dir else csv_path.parent

    df = pd.read_csv(csv_path)
    df = prepare_numeric_columns(df)

    print(f"[input] {csv_path}")
    print(f"[rows] {len(df)}")
    print(f"[figures] {figures_dir}")
    print(f"[tables] {tables_dir}")

    write_diagnostic_tables(df, tables_dir)

    # Global
    plot_status_counts(df, figures_dir)
    plot_status_fraction(df, figures_dir)
    plot_failure_reasons(df, figures_dir)
    plot_failure_reasons_log(df, figures_dir)

    # Properties
    plot_property_ok_map(df, figures_dir)
    plot_property_density_map(df, figures_dir)
    plot_property_phase_counts(df, figures_dir)

    # HEM
    plot_hem_capacity_vs_opening(df, figures_dir)
    plot_hem_capacity_map(df, figures_dir)
    plot_hem_capacity_vs_p1_by_temperature(df, figures_dir)
    plot_hem_capacity_vs_p2_by_pressure(df, figures_dir)
    plot_hem_critical_pressure_ratio_map(df, figures_dir)
    plot_hem_runtime_map(df, figures_dir)

    # Wellbore
    plot_wellbore_bhp_vs_rate(df, figures_dir)
    plot_wellbore_velocity_vs_rate(df, figures_dir)
    plot_wellbore_bhp_vs_rate_by_tvd(df, figures_dir)
    plot_wellbore_bhp_vs_rate_by_thp(df, figures_dir)
    plot_wellbore_velocity_vs_rate_by_diameter(df, figures_dir)
    plot_wellbore_min_pressure_map(df, figures_dir)
    plot_wellbore_runtime_vs_nsegments(df, figures_dir)
    plot_wellbore_grid_convergence(df, figures_dir)
    plot_wellbore_phase_counts(df, figures_dir)

    plot_wellbore_warning_fraction_map(
        df,
        figures_dir,
        reason="VERY_LOW_WELLBORE_PRESSURE",
        x_col="q_sm3_day",
        y_col="thp_bar",
        filename="capability_wellbore_very_low_pressure_fraction_q_thp.png",
        title="VERY_LOW_WELLBORE_PRESSURE fraction over q/THP grid",
    )

    plot_wellbore_warning_fraction_map(
        df,
        figures_dir,
        reason="VERY_LOW_WELLBORE_PRESSURE",
        x_col="tvd_m",
        y_col="thp_bar",
        filename="capability_wellbore_very_low_pressure_fraction_tvd_thp.png",
        title="VERY_LOW_WELLBORE_PRESSURE fraction over TVD/THP grid",
    )

    plot_wellbore_warning_fraction_map(
        df,
        figures_dir,
        reason="HIGH_WELLBORE_VELOCITY",
        x_col="q_sm3_day",
        y_col="diameter_m",
        filename="capability_wellbore_high_velocity_fraction_q_diameter.png",
        title="HIGH_WELLBORE_VELOCITY fraction over q/diameter grid",
    )


if __name__ == "__main__":
    main()