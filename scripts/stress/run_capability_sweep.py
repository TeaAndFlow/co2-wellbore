#!/usr/bin/env python3
"""Capability/stress sweep for CO2 wellbore/choke/property components.

This script is intentionally component-level.
It does not run OPM Flow. The goal is to map where the external
CoolProp/HEM/wellbore model works, fails, or enters suspicious regions
before adding salt precipitation or hydrate prediction.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from co2_wellbore import (
    CO2HEMChokeValve,
    CO2WellboreCalculator,
    CoolPropCO2,
    DriftFluxConfig,
    HEMChokeConfig,
    ThermalConfig,
    WellboreGeometry,
)
from co2_wellbore.constants import BAR_TO_PA


def classify_exception(exc: BaseException) -> str:
    msg = str(exc).lower()

    if "saturation" in msg or "quality" in msg:
        return "SATURATION_AMBIGUOUS_BOUNDARY"
    if "coolprop" in msg or "propssi" in msg or "flash" in msg:
        return "COOLPROP_INVALID_STATE"
    if "no valid throat" in msg:
        return "HEM_NO_VALID_THROAT"
    if "downstream_pressure_bar" in msg:
        return "HEM_INVALID_DOWNSTREAM_PRESSURE"
    if "bracket" in msg and "thp" in msg:
        return "WELLBORE_THP_NOT_BRACKETED"
    if "pressure must be positive" in msg:
        return "INVALID_PRESSURE"
    if "temperature must be positive" in msg:
        return "INVALID_TEMPERATURE"

    return exc.__class__.__name__


def f(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan

def hydrate_risk_proxy_label(
    *,
    post_choke_temperature_C: float,
    post_choke_pressure_bar: float,
    quality_mass: float,
) -> tuple[str, float]:
    """Very rough pre-hydrate proxy for diagnostic plots.

    This is NOT a hydrate model.

    It only marks cold/high-pressure/two-phase-ish post-choke states that deserve
    future hydrate-model attention.
    """

    T = float(post_choke_temperature_C)
    P = float(post_choke_pressure_bar)
    x = float(quality_mass) if math.isfinite(float(quality_mass)) else math.nan

    score = 0.0

    # Cold CO2 after throttling.
    if math.isfinite(T):
        if T <= 0.0:
            score += 2.0
        elif T <= 10.0:
            score += 1.0

    # High pressure helps hydrate stability, but this is only a proxy.
    if math.isfinite(P):
        if P >= 40.0:
            score += 2.0
        elif P >= 20.0:
            score += 1.0

    # Two-phase CO2 is important for future coupled physics.
    if math.isfinite(x) and 0.0 < x < 1.0:
        score += 1.0

    if score >= 4.0:
        return "high_proxy", score
    if score >= 2.0:
        return "medium_proxy", score
    if score > 0.0:
        return "low_proxy", score
    return "none", score

def base_row(component: str, case_id: str, **kwargs: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "component": component,
        "case_id": case_id,
        "status": "ok",
        "failure_reason": "",
        "message": "",

        "pressure_bar": math.nan,
        "temperature_C": math.nan,

        "upstream_pressure_bar": math.nan,
        "upstream_temperature_C": math.nan,
        "downstream_pressure_bar": math.nan,
        "opening_fraction": math.nan,

        "thp_bar": math.nan,
        "q_sm3_day": math.nan,
        "mass_rate_kg_s": math.nan,
        "tvd_m": math.nan,
        "diameter_m": math.nan,
        "thermal_model": "",
        "n_segments": math.nan,

        "phase_label": "",
        "quality_mass": math.nan,
        "density_kg_m3": math.nan,
        "enthalpy_J_kg": math.nan,
        "viscosity_Pa_s": math.nan,
        "cp_J_kgK": math.nan,

        "flow_regime": "",
        "critical_pressure_bar": math.nan,
        "selected_throat_pressure_bar": math.nan,
        "selected_throat_quality_mass": math.nan,
        "mass_flow_capacity_kg_s": math.nan,
        "critical_search_method": "",
        "critical_optimizer_success": "",

        "bhp_bar": math.nan,
        "bht_C": math.nan,
        "min_pressure_bar": math.nan,
        "max_pressure_bar": math.nan,
        "max_velocity_m_s": math.nan,
        "max_reynolds": math.nan,
        "bottom_phase_label": "",
        "two_phase_fraction": math.nan,

        # Post-choke / P-H stress diagnostics.
        "post_choke_pressure_bar": math.nan,
        "post_choke_temperature_C": math.nan,
        "post_choke_density_kg_m3": math.nan,
        "post_choke_enthalpy_J_kg": math.nan,
        "post_choke_quality_mass": math.nan,
        "post_choke_phase_label": "",
        "post_choke_deltaT_C": math.nan,
        "hydrate_risk_proxy": "",
        "hydrate_risk_score": math.nan,

        "runtime_s": math.nan,
    }
    row.update(kwargs)
    return row


def grids(mode: str) -> Dict[str, List[float]]:
    if mode == "smoke":
        return {
            "p_bar": [60.0, 100.0],
            "t_C": [40.0, 60.0],
            "p1_bar": [60.0, 100.0],
            "t1_C": [20.0, 60.0],
            "p2_bar": [10.0, 20.0, 40.0],
            "opening": [0.05, 0.20],
            "q_sm3_day": [1.0, 50000.0, 100000.0],
            "thp_bar": [60.0, 100.0],
            "tvd_m": [800.0, 1600.0],
            "diameter_m": [0.10],
            "n_segments": [40, 80],
        }

    if mode == "quick":
        return {
            # Property backend: enough to touch gas-like, dense, near-critical-ish states.
            "p_bar": [10.0, 20.0, 40.0, 60.0, 73.8, 80.0, 100.0, 150.0],
            "t_C": [0.0, 5.0, 20.0, 31.0, 40.0, 60.0, 100.0],

            # HEM choke: representative but not insane.
            "p1_bar": [60.0, 80.0, 100.0, 150.0],
            "t1_C": [0.0, 20.0, 40.0, 60.0, 100.0],
            "p2_bar": [10.0, 20.0, 40.0, 80.0],
            "opening": [0.02, 0.05, 0.10, 0.20, 0.50, 1.00],

            # Wellbore: keep this compact. Wellbore sweep is expensive.
            "q_sm3_day": [1.0, 10000.0, 50000.0, 100000.0, 200000.0],
            "thp_bar": [60.0, 100.0, 150.0],
            "tvd_m": [1600.0, 3000.0],
            "diameter_m": [0.10, 0.16],
            "n_segments": [80, 160],
        }

    return {
        "p_bar": [6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 70.0, 73.8, 80.0, 100.0, 120.0, 150.0, 200.0],
        "t_C": [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 31.0, 35.0, 40.0, 60.0, 80.0, 100.0],
        "p1_bar": [30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0],
        "t1_C": [-20.0, -10.0, 0.0, 5.0, 20.0, 31.0, 40.0, 60.0, 80.0, 100.0],
        "p2_bar": [6.0, 8.0, 10.0, 20.0, 40.0, 80.0, 120.0, 180.0],
        "opening": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00],
        "q_sm3_day": [1.0, 100.0, 1000.0, 10000.0, 50000.0, 100000.0, 200000.0, 300000.0, 500000.0],
        "thp_bar": [10.0, 20.0, 60.0, 100.0, 150.0, 250.0],
        "tvd_m": [500.0, 1600.0, 3000.0, 4000.0],
        "diameter_m": [0.05, 0.075, 0.10, 0.16],
        "n_segments": [40, 80, 160, 320],
    }


def run_property_sweep(mode: str) -> List[Dict[str, Any]]:
    props = CoolPropCO2()
    g = grids(mode)
    rows: List[Dict[str, Any]] = []

    for p_bar, t_C in itertools.product(g["p_bar"], g["t_C"]):
        case_id = f"PROP_P{p_bar:g}_T{t_C:g}"
        t0 = time.time()
        row = base_row(
            "properties",
            case_id,
            pressure_bar=p_bar,
            temperature_C=t_C,
        )

        try:
            out = props.props(p_bar * BAR_TO_PA, t_C + 273.15)
            row.update(
                phase_label=str(out.get("phase_label", "")),
                quality_mass=f(out.get("quality_mass")),
                density_kg_m3=f(out.get("rho_kg_m3")),
                enthalpy_J_kg=f(out.get("enthalpy_J_kg")),
                viscosity_Pa_s=f(out.get("viscosity_Pa_s")),
                cp_J_kgK=f(out.get("cp_J_kgK")),
            )

            if not math.isfinite(row["density_kg_m3"]) or row["density_kg_m3"] <= 0:
                row["status"] = "warning"
                row["failure_reason"] = "NONPOSITIVE_DENSITY"

        except Exception as exc:
            row["status"] = "fail"
            row["failure_reason"] = classify_exception(exc)
            row["message"] = str(exc)[:500]

        row["runtime_s"] = time.time() - t0
        rows.append(row)

    return rows


def run_hem_sweep(mode: str) -> List[Dict[str, Any]]:
    g = grids(mode)
    rows: List[Dict[str, Any]] = []

    n_samples = 40 if mode == "smoke" else 60 if mode == "quick" else 100

    cases = [
        item
        for item in itertools.product(
            g["p1_bar"],
            g["t1_C"],
            g["p2_bar"],
            g["opening"],
        )
        if item[2] < 0.995 * item[0]
    ]

    n_cases = len(cases)
    print(f"[capability] HEM cases: {n_cases}, pressure samples per case: {n_samples}", flush=True)

    for case_no, (p1_bar, t1_C, p2_bar, opening) in enumerate(cases, start=1):
        if case_no == 1 or case_no % 50 == 0 or case_no == n_cases:
            print(
                f"[capability] HEM {case_no}/{n_cases}: "
                f"P1={p1_bar:g} bar, T1={t1_C:g} C, P2={p2_bar:g} bar, open={100*opening:g}%",
                flush=True,
            )

        case_id = f"HEM_P1{p1_bar:g}_T1{t1_C:g}_P2{p2_bar:g}_OPEN{opening:g}"
        t0 = time.time()
        row = base_row(
            "hem_choke",
            case_id,
            upstream_pressure_bar=p1_bar,
            upstream_temperature_C=t1_C,
            downstream_pressure_bar=p2_bar,
            opening_fraction=opening,
        )

        try:
            valve = CO2HEMChokeValve(
                config=HEMChokeConfig(
                    full_diameter_m=0.067,
                    opening_fraction=opening,
                    discharge_coefficient=0.84,
                    area_exponent=1.0,
                    min_pressure_bar=6.0,
                    n_pressure_samples=n_samples,
                )
            )
            out = valve.capacity_result(
                upstream_pressure_bar=p1_bar,
                upstream_temperature_C=t1_C,
                downstream_pressure_bar=p2_bar,
            )
            row.update(
                flow_regime=str(out.get("flow_regime", "")),
                phase_label=str(out.get("phase_label", "")),
                quality_mass=f(out.get("quality_mass")),
                critical_pressure_bar=f(out.get("critical_pressure_bar")),
                selected_throat_pressure_bar=f(out.get("selected_throat_pressure_bar")),
                selected_throat_quality_mass=f(out.get("selected_throat_quality_mass")),
                mass_flow_capacity_kg_s=f(out.get("mass_flow_capacity_kg_s")),
                critical_search_method=str(out.get("critical_search_method", "")),
                critical_optimizer_success=str(out.get("critical_optimizer_success", "")),
            )

            cap = row["mass_flow_capacity_kg_s"]
            if not math.isfinite(cap) or cap <= 0:
                row["status"] = "warning"
                row["failure_reason"] = "NONPOSITIVE_HEM_CAPACITY"

        except Exception as exc:
            row["status"] = "fail"
            row["failure_reason"] = classify_exception(exc)
            row["message"] = str(exc)[:500]

        row["runtime_s"] = time.time() - t0
        rows.append(row)

        if case_no == 1 or case_no % 50 == 0 or case_no == n_cases:
            print(
                f"[capability] HEM done {case_no}/{n_cases}: "
                f"status={row['status']} reason={row['failure_reason']} "
                f"runtime={row['runtime_s']:.3f}s",
                flush=True,
            )

    return rows

def run_post_choke_sweep(mode: str) -> List[Dict[str, Any]]:
    """Stress P-H post-choke states using HEM upstream enthalpy and downstream pressure.

    This component is intended for future salt/hydrate preparation.

    It does NOT predict hydrates. It only maps:
    - post-choke temperature,
    - density,
    - quality,
    - phase label,
    - a crude cold/high-pressure hydrate-risk proxy.
    """

    g = grids(mode)
    props = CoolPropCO2()
    rows: List[Dict[str, Any]] = []

    if mode == "smoke":
        p1_values = [60.0, 100.0]
        t1_values = [20.0, 60.0]
        p2_values = [10.0, 20.0, 40.0]
    elif mode == "quick":
        p1_values = [60.0, 80.0, 100.0, 150.0]
        t1_values = [0.0, 20.0, 40.0, 60.0, 100.0]
        p2_values = [6.0, 10.0, 20.0, 40.0, 80.0]
    else:
        p1_values = g["p1_bar"]
        t1_values = g["t1_C"]
        p2_values = g["p2_bar"]

    cases = [
        item
        for item in itertools.product(p1_values, t1_values, p2_values)
        if item[2] < 0.995 * item[0]
    ]

    n_cases = len(cases)
    print(f"[capability] Post-choke P-H cases: {n_cases}", flush=True)

    for case_no, (p1_bar, t1_C, p2_bar) in enumerate(cases, start=1):
        if case_no == 1 or case_no % 50 == 0 or case_no == n_cases:
            print(
                f"[capability] Post-choke {case_no}/{n_cases}: "
                f"P1={p1_bar:g} bar, T1={t1_C:g} C, P2={p2_bar:g} bar",
                flush=True,
            )

        case_id = f"POSTCHOKE_P1{p1_bar:g}_T1{t1_C:g}_P2{p2_bar:g}"
        t0 = time.time()

        row = base_row(
            "post_choke",
            case_id,
            upstream_pressure_bar=p1_bar,
            upstream_temperature_C=t1_C,
            downstream_pressure_bar=p2_bar,
            post_choke_pressure_bar=p2_bar,
        )

        try:
            upstream = props.props(p1_bar * BAR_TO_PA, t1_C + 273.15)
            h1 = float(upstream["enthalpy_J_kg"])

            post = props.props_ph(p2_bar * BAR_TO_PA, h1)

            T2_C = f(post.get("temperature_K")) - 273.15
            rho2 = f(post.get("rho_kg_m3"))
            q2 = f(post.get("quality_mass"))
            phase2 = str(post.get("phase_label", ""))

            risk_label, risk_score = hydrate_risk_proxy_label(
                post_choke_temperature_C=T2_C,
                post_choke_pressure_bar=p2_bar,
                quality_mass=q2,
            )

            row.update(
                phase_label=phase2,
                quality_mass=q2,
                density_kg_m3=rho2,
                enthalpy_J_kg=h1,
                post_choke_temperature_C=T2_C,
                post_choke_density_kg_m3=rho2,
                post_choke_enthalpy_J_kg=h1,
                post_choke_quality_mass=q2,
                post_choke_phase_label=phase2,
                post_choke_deltaT_C=T2_C - t1_C,
                hydrate_risk_proxy=risk_label,
                hydrate_risk_score=risk_score,
            )

            if not math.isfinite(T2_C) or not math.isfinite(rho2) or rho2 <= 0.0:
                row["status"] = "warning"
                row["failure_reason"] = "INVALID_POST_CHOKE_STATE"

        except Exception as exc:
            row["status"] = "fail"
            row["failure_reason"] = classify_exception(exc)
            row["message"] = str(exc)[:500]

        row["runtime_s"] = time.time() - t0
        rows.append(row)

        if case_no == 1 or case_no % 50 == 0 or case_no == n_cases:
            print(
                f"[capability] Post-choke done {case_no}/{n_cases}: "
                f"status={row['status']} reason={row['failure_reason']} "
                f"T2={row['post_choke_temperature_C']:.3g} C "
                f"risk={row['hydrate_risk_proxy']} "
                f"runtime={row['runtime_s']:.3f}s",
                flush=True,
            )

    return rows

def run_wellbore_sweep(mode: str) -> List[Dict[str, Any]]:
    g = grids(mode)
    rows: List[Dict[str, Any]] = []

    if mode == "smoke":
        thermal_modes = ["off", "overall_U"]
        t_head_values = [20.0, 60.0]
    elif mode == "quick":
        thermal_modes = ["off", "overall_U", "layered"]
        t_head_values = [20.0, 60.0]
    else:
        thermal_modes = ["off", "overall_U", "layered"]
        t_head_values = [5.0, 20.0, 40.0, 60.0]

    cases = list(
        itertools.product(
            g["thp_bar"],
            g["q_sm3_day"],
            g["tvd_m"],
            g["diameter_m"],
            g["n_segments"],
            thermal_modes,
            t_head_values,
        )
    )

    n_cases = len(cases)
    print(f"[capability] Wellbore cases: {n_cases}", flush=True)

    for case_no, (
        thp_bar,
        q_sm3_day,
        tvd_m,
        diameter_m,
        n_segments,
        thermal_mode,
        t_head_C,
    ) in enumerate(cases, start=1):

        if case_no == 1 or case_no % 10 == 0 or case_no == n_cases:
            print(
                f"[capability] Wellbore {case_no}/{n_cases}: "
                f"THP={thp_bar:g} bar, q={q_sm3_day:g} sm3/d, "
                f"TVD={tvd_m:g} m, D={diameter_m:g} m, "
                f"N={n_segments:g}, thermal={thermal_mode}, T_head={t_head_C:g} C",
                flush=True,
            )

        case_id = (
            f"WB_THP{thp_bar:g}_Q{q_sm3_day:g}_TVD{tvd_m:g}_D{diameter_m:g}"
            f"_N{n_segments:g}_{thermal_mode}_THEAD{t_head_C:g}"
        )

        t0 = time.time()
        row = base_row(
            "wellbore",
            case_id,
            thp_bar=thp_bar,
            q_sm3_day=q_sm3_day,
            tvd_m=tvd_m,
            diameter_m=diameter_m,
            thermal_model=thermal_mode,
            n_segments=n_segments,
            temperature_C=t_head_C,
        )

        try:
            if thermal_mode == "off":
                thermal = ThermalConfig(enabled=False)
            elif thermal_mode == "overall_U":
                thermal = ThermalConfig(
                    enabled=True,
                    model="overall_U",
                    overall_U_W_m2K=4.0,
                    surface_temperature_C=32.0,
                    geothermal_gradient_C_per_m=0.03,
                    max_abs_dT_per_segment_C=10.0,
                )
            else:
                thermal = ThermalConfig(
                    enabled=True,
                    model="layered",
                    surface_temperature_C=32.0,
                    geothermal_gradient_C_per_m=0.03,
                    max_abs_dT_per_segment_C=10.0,
                    elapsed_time_days=30.0,
                )

            wb = CO2WellboreCalculator(
                geometry=WellboreGeometry(
                    tvd_m=tvd_m,
                    diameter_m=diameter_m,
                    roughness_m=1.5e-5,
                    n_segments=int(n_segments),
                ),
                thermal=thermal,
                drift_flux=DriftFluxConfig(enabled=True),
            )

            profile = wb.profile_from_thp_and_rate(
                thp_bar=thp_bar,
                q_sm3_day=q_sm3_day,
                wellhead_temperature_C=t_head_C,
            )

            row.update(
                mass_rate_kg_s=wb.mass_rate_from_sm3_day(q_sm3_day),
                bhp_bar=f(profile["pressure_bar"].iloc[-1]),
                bht_C=f(profile["temperature_C"].iloc[-1]),
                min_pressure_bar=f(profile["pressure_bar"].min()),
                max_pressure_bar=f(profile["pressure_bar"].max()),
                max_velocity_m_s=f(profile["velocity_m_s"].abs().max()),
                max_reynolds=f(profile["reynolds"].max()),
                bottom_phase_label=str(profile["phase_label"].iloc[-1]),
                phase_label=str(profile["phase_label"].iloc[-1]),
                quality_mass=f(profile["quality_mass"].iloc[-1]),
                density_kg_m3=f(profile["density_kg_m3"].iloc[-1]),
                two_phase_fraction=f((profile["two_phase_active"] > 0.5).mean()),
            )

            if row["max_velocity_m_s"] > 50.0:
                row["status"] = "warning"
                row["failure_reason"] = "HIGH_WELLBORE_VELOCITY"

            if row["min_pressure_bar"] <= 0.1:
                row["status"] = "warning"
                row["failure_reason"] = "VERY_LOW_WELLBORE_PRESSURE"

        except Exception as exc:
            row["status"] = "fail"
            row["failure_reason"] = classify_exception(exc)
            row["message"] = str(exc)[:500]

        row["runtime_s"] = time.time() - t0
        rows.append(row)

        if case_no == 1 or case_no % 10 == 0 or case_no == n_cases:
            print(
                f"[capability] Wellbore done {case_no}/{n_cases}: "
                f"status={row['status']} reason={row['failure_reason']} "
                f"runtime={row['runtime_s']:.3f}s",
                flush=True,
            )

    return rows

def write_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    summary_path = output_dir / "capability_summary.md"

    lines: List[str] = []
    lines.append("# Capability/stress summary")
    lines.append("")
    lines.append("This file is generated by `scripts/stress/run_capability_sweep.py`.")
    lines.append("")
    lines.append("## Counts by component and status")
    lines.append("")

    counts = (
        df.groupby(["component", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["component", "status"])
    )
    lines.append(counts.to_markdown(index=False))
    lines.append("")

    lines.append("## Top failure/warning reasons")
    lines.append("")
    bad = df[df["status"].isin(["fail", "warning"])].copy()
    if len(bad):
        reason_counts = (
            bad.groupby(["component", "status", "failure_reason"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(40)
        )
        lines.append(reason_counts.to_markdown(index=False))
    else:
        lines.append("No failures or warnings.")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `ok`: component returned finite outputs inside the tested point.")
    lines.append("- `warning`: component returned outputs, but a suspicious diagnostic was triggered.")
    lines.append("- `fail`: component raised an exception or entered an unsupported state.")
    lines.append("")
    lines.append("This is not field validation. It is a capability envelope map.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "quick", "full"],
        default="quick",
        help="smoke = tiny CI run, quick = default v0.2.9 sweep, full = heavier local sweep",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/results",
        help="Directory for capability_envelope.csv and summary.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capability] mode={args.mode}")
    print("[capability] running property sweep...")
    rows = run_property_sweep(args.mode)

    print("[capability] running HEM choke sweep...")
    rows.extend(run_hem_sweep(args.mode))

    print("[capability] running wellbore sweep...")
    rows.extend(run_wellbore_sweep(args.mode))

    df = pd.DataFrame(rows)

    csv_path = output_dir / "capability_envelope.csv"
    df.to_csv(csv_path, index=False)

    summary_path = write_summary(df, output_dir)

    print(f"[capability] wrote: {csv_path}")
    print(f"[capability] wrote: {summary_path}")
    print("[capability] status counts:")
    print(df.groupby(["component", "status"]).size())


if __name__ == "__main__":
    main()
