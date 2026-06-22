#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from CoolProp.CoolProp import PropsSI

from co2_wellbore import CO2HEMChokeValve, HEMChokeConfig


FIG_DIR = Path("validation/figures")
RESULTS_DIR = Path("validation/results")


def ideal_gas_choked_capacity_kg_s(
    *,
    p0_bar: float,
    t0_C: float,
    diameter_m: float,
    opening_fraction: float,
    cd: float,
    area_exponent: float,
) -> float:
    p0 = p0_bar * 1.0e5
    t0 = t0_C + 273.15

    fluid = "CO2"
    cp = PropsSI("Cpmass", "P", p0, "T", t0, fluid)
    cv = PropsSI("Cvmass", "P", p0, "T", t0, fluid)
    molar_mass = PropsSI("M", fluid)
    r_specific = 8.31446261815324 / molar_mass

    gamma = cp / cv

    full_area = 3.141592653589793 * diameter_m**2 / 4.0
    effective_area = full_area * opening_fraction**area_exponent

    mass_flux = p0 * (
        gamma
        / (r_specific * t0)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
    ) ** 0.5

    return cd * effective_area * mass_flux


def run_hem_capacity(
    *,
    p0_bar: float,
    t0_C: float,
    p2_bar: float,
    diameter_m: float,
    opening_fraction: float,
    cd: float,
    area_exponent: float,
) -> float:
    valve = CO2HEMChokeValve(
        config=HEMChokeConfig(
            full_diameter_m=diameter_m,
            opening_fraction=opening_fraction,
            discharge_coefficient=cd,
            area_exponent=area_exponent,
            min_pressure_bar=0.05,
            n_pressure_samples=220,
        )
    )

    out = valve.capacity_result(
        upstream_pressure_bar=p0_bar,
        upstream_temperature_C=t0_C,
        downstream_pressure_bar=p2_bar,
    )

    return float(out["mass_flow_capacity_kg_s"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    diameter_m = 0.067
    opening_fraction = 0.20
    cd = 0.84
    area_exponent = 1.0

    # Gas-like states: low/moderate pressure and high temperature.
    # The point is not to pretend CO2 is always ideal, but to verify the limiting behavior.
    cases = [
        {"case_id": "P5_T120", "p0_bar": 5.0, "t0_C": 120.0, "p2_bar": 0.5},
        {"case_id": "P10_T150", "p0_bar": 10.0, "t0_C": 150.0, "p2_bar": 1.0},
        {"case_id": "P20_T200", "p0_bar": 20.0, "t0_C": 200.0, "p2_bar": 2.0},
        {"case_id": "P30_T250", "p0_bar": 30.0, "t0_C": 250.0, "p2_bar": 3.0},
    ]

    rows = []

    for c in cases:
        p0_bar = float(c["p0_bar"])
        t0_C = float(c["t0_C"])
        p2_bar = float(c["p2_bar"])

        hem = run_hem_capacity(
            p0_bar=p0_bar,
            t0_C=t0_C,
            p2_bar=p2_bar,
            diameter_m=diameter_m,
            opening_fraction=opening_fraction,
            cd=cd,
            area_exponent=area_exponent,
        )

        ideal = ideal_gas_choked_capacity_kg_s(
            p0_bar=p0_bar,
            t0_C=t0_C,
            diameter_m=diameter_m,
            opening_fraction=opening_fraction,
            cd=cd,
            area_exponent=area_exponent,
        )

        rel_diff_percent = abs(hem - ideal) / max(abs(ideal), 1.0e-30) * 100.0

        p0 = p0_bar * 1.0e5
        t0 = t0_C + 273.15
        rho = PropsSI("Dmass", "P", p0, "T", t0, "CO2")
        molar_mass = PropsSI("M", "CO2")
        r_specific = 8.31446261815324 / molar_mass
        z_factor = p0 / (rho * r_specific * t0)

        rows.append(
            {
                "case_id": c["case_id"],
                "upstream_pressure_bar": p0_bar,
                "upstream_temperature_C": t0_C,
                "downstream_pressure_bar": p2_bar,
                "z_factor": z_factor,
                "hem_capacity_kg_s": hem,
                "ideal_gas_capacity_kg_s": ideal,
                "relative_difference_percent": rel_diff_percent,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "hem_ideal_gas_limit.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["case_id"], df["hem_capacity_kg_s"], marker="o", label="HEM real-fluid model")
    plt.plot(df["case_id"], df["ideal_gas_capacity_kg_s"], marker="s", linestyle="--", label="Ideal-gas choked-flow limit")
    plt.xlabel("Gas-like CO₂ verification case")
    plt.ylabel("Mass-flow capacity [kg/s]")
    plt.title("HEM choke verification: ideal-gas limiting behavior")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIG_DIR / "hem_ideal_gas_limit.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["case_id"], df["relative_difference_percent"])
    plt.xlabel("Gas-like CO₂ verification case")
    plt.ylabel("Relative difference [%]")
    plt.title("HEM choke verification: ideal-gas limit error")
    plt.grid(True, axis="y", alpha=0.3)

    ymax = max(float(df["relative_difference_percent"].max()) * 1.20, 1.0e-12)
    plt.ylim(0.0, ymax)

    for bar, value in zip(bars, df["relative_difference_percent"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{float(value):.4f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    err_fig_path = FIG_DIR / "hem_ideal_gas_limit_error.png"
    plt.savefig(err_fig_path, dpi=220)
    plt.close()

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {fig_path}")
    print(f"[OK] Wrote {err_fig_path}")


if __name__ == "__main__":
    main()
