#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from co2_wellbore import (
    CO2HEMChokeValve,
    CO2WellboreCalculator,
    CoolPropCO2,
    HEMChokeConfig,
    ThermalConfig,
    WellboreGeometry,
)
from co2_wellbore.constants import BAR_TO_PA, G, PA_TO_BAR


FIG_DIR = Path("validation/figures")
RESULTS_DIR = Path("validation/results")


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def make_co2_property_figures() -> list[Path]:
    props = CoolPropCO2()

    rows: list[dict[str, float | str]] = []

    pressure_grid_bar = np.linspace(20.0, 140.0, 49)
    temperature_cases_C = [40.0, 60.0, 80.0]

    for t_C in temperature_cases_C:
        for p_bar in pressure_grid_bar:
            out = props.props(float(p_bar) * BAR_TO_PA, float(t_C) + 273.15)
            rows.append(
                {
                    "pressure_bar": float(p_bar),
                    "temperature_C": float(t_C),
                    "density_kg_m3": float(out["rho_kg_m3"]),
                    "enthalpy_J_kg": float(out["enthalpy_J_kg"]),
                    "viscosity_Pa_s": float(out["viscosity_Pa_s"]),
                    "phase_label": str(out["phase_label"]),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "co2_property_sanity.csv", index=False)

    figures: list[Path] = []

    plt.figure()
    for t_C, group in df.groupby("temperature_C"):
        plt.plot(group["pressure_bar"], group["density_kg_m3"], marker="o", markersize=3, label=f"{t_C:g} °C")
    plt.xlabel("Pressure [bar]")
    plt.ylabel("Density [kg/m³]")
    plt.title("CO₂ property verification: density trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    figures.append(save_current_figure("co2_density_vs_pressure.png"))

    plt.figure()
    for t_C, group in df.groupby("temperature_C"):
        plt.plot(group["pressure_bar"], group["enthalpy_J_kg"] / 1000.0, marker="o", markersize=3, label=f"{t_C:g} °C")
    plt.xlabel("Pressure [bar]")
    plt.ylabel("Specific enthalpy [kJ/kg]")
    plt.title("CO₂ property verification: enthalpy trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    figures.append(save_current_figure("co2_enthalpy_vs_pressure.png"))

    return figures


def make_wellbore_pressure_figure() -> list[Path]:
    wb = CO2WellboreCalculator(
        geometry=WellboreGeometry(
            tvd_m=1000.0,
            diameter_m=0.10,
            roughness_m=1.5e-5,
            n_segments=80,
        ),
        thermal=ThermalConfig(
            enabled=False,
            surface_temperature_C=30.0,
            geothermal_gradient_C_per_m=0.03,
        ),
    )

    thp_bar = 100.0
    q_sm3_day = 1.0

    profile = wb.profile_from_thp_and_rate(
        thp_bar=thp_bar,
        q_sm3_day=q_sm3_day,
        wellhead_temperature_C=60.0,
    )

    mean_rho = float(profile["rho_hydro_kg_m3"].mean())
    profile["hydrostatic_estimate_bar"] = (
        thp_bar + mean_rho * G * profile["depth_m"] * PA_TO_BAR
    )

    profile.to_csv(RESULTS_DIR / "wellbore_pressure_hydrostatic_limit.csv", index=False)

    plt.figure()
    plt.plot(profile["depth_m"], profile["pressure_bar"], marker="o", markersize=3, label="Wellbore model")
    plt.plot(profile["depth_m"], profile["hydrostatic_estimate_bar"], linestyle="--", label="Hydrostatic estimate")
    plt.xlabel("Depth [m]")
    plt.ylabel("Pressure [bar]")
    plt.title("Wellbore pressure verification: tiny-flow hydrostatic limit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return [save_current_figure("wellbore_pressure_hydrostatic_limit.png")]


def make_wellbore_thermal_figure() -> list[Path]:
    rows: list[dict[str, float]] = []

    u_values = [1.0e-6, 0.1, 1.0, 10.0, 50.0, 200.0]

    for u in u_values:
        wb = CO2WellboreCalculator(
            geometry=WellboreGeometry(
                tvd_m=800.0,
                diameter_m=0.10,
                roughness_m=1.5e-5,
                n_segments=80,
            ),
            thermal=ThermalConfig(
                enabled=True,
                model="overall_U",
                overall_U_W_m2K=float(u),
                surface_temperature_C=30.0,
                geothermal_gradient_C_per_m=0.03,
                max_abs_dT_per_segment_C=10.0,
            ),
        )

        profile = wb.profile_from_thp_and_rate(
            thp_bar=100.0,
            q_sm3_day=100000.0,
            wellhead_temperature_C=20.0,
        )

        rows.append(
            {
                "overall_U_W_m2K": float(u),
                "bottom_temperature_C": float(profile["temperature_C"].iloc[-1]),
                "geothermal_bottom_temperature_C": float(
                    wb.thermal.geothermal_temperature_C(wb.geometry.tvd_m)
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "wellbore_thermal_response.csv", index=False)

    plt.figure()
    plt.plot(df["overall_U_W_m2K"], df["bottom_temperature_C"], marker="o", label="Bottomhole temperature")
    plt.axhline(
        float(df["geothermal_bottom_temperature_C"].iloc[0]),
        linestyle="--",
        label="Geothermal bottom temperature",
    )
    plt.xscale("log")
    plt.xlabel("Overall heat-transfer coefficient U [W/m²/K]")
    plt.ylabel("Temperature [°C]")
    plt.title("Wellbore thermal verification: response to heat transfer")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return [save_current_figure("wellbore_thermal_response.png")]


def hem_valve(opening_fraction: float = 0.20, samples: int = 220) -> CO2HEMChokeValve:
    return CO2HEMChokeValve(
        config=HEMChokeConfig(
            full_diameter_m=0.067,
            opening_fraction=float(opening_fraction),
            discharge_coefficient=0.84,
            area_exponent=1.0,
            min_pressure_bar=10.0,
            n_pressure_samples=int(samples),
        )
    )


def make_hem_opening_figure() -> list[Path]:
    rows: list[dict[str, float | str]] = []

    for opening_percent in [2, 5, 10, 15, 20, 30, 40, 60, 80, 100]:
        valve = hem_valve(opening_fraction=opening_percent / 100.0)
        out = valve.capacity_result(
            upstream_pressure_bar=60.0,
            upstream_temperature_C=60.0,
            downstream_pressure_bar=20.0,
        )
        rows.append(
            {
                "opening_percent": float(opening_percent),
                "capacity_kg_s": float(out["mass_flow_capacity_kg_s"]),
                "critical_pressure_bar": float(out["critical_pressure_bar"]),
                "critical_mass_flux_kg_m2_s": float(out["critical_mass_flux_kg_m2_s"]),
                "flow_regime": str(out["flow_regime"]),
                "critical_search_method": str(out["critical_search_method"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hem_capacity_vs_opening.csv", index=False)

    plt.figure()
    plt.plot(df["opening_percent"], df["capacity_kg_s"], marker="o")
    plt.xlabel("Choke opening [%]")
    plt.ylabel("Mass-flow capacity [kg/s]")
    plt.title("HEM choke verification: capacity response to opening")
    plt.grid(True, alpha=0.3)

    return [save_current_figure("hem_capacity_vs_opening.png")]


def make_hem_downstream_pressure_figure() -> list[Path]:
    valve = hem_valve(opening_fraction=0.20)

    rows: list[dict[str, float | str]] = []

    for p2_bar in np.linspace(12.0, 55.0, 45):
        out = valve.capacity_result(
            upstream_pressure_bar=60.0,
            upstream_temperature_C=60.0,
            downstream_pressure_bar=float(p2_bar),
        )
        rows.append(
            {
                "downstream_pressure_bar": float(p2_bar),
                "capacity_kg_s": float(out["mass_flow_capacity_kg_s"]),
                "critical_pressure_bar": float(out["critical_pressure_bar"]),
                "flow_regime": str(out["flow_regime"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hem_capacity_vs_downstream_pressure.csv", index=False)

    pcrit = float(df["critical_pressure_bar"].iloc[0])

    plt.figure()
    plt.plot(df["downstream_pressure_bar"], df["capacity_kg_s"], marker="o", markersize=3)
    plt.axvline(pcrit, linestyle="--", label=f"Critical pressure ≈ {pcrit:.2f} bar")
    plt.xlabel("Downstream pressure [bar]")
    plt.ylabel("Mass-flow capacity [kg/s]")
    plt.title("HEM choke verification: choked and subcritical regimes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return [save_current_figure("hem_capacity_vs_downstream_pressure.png")]


def make_hem_convergence_figures() -> list[Path]:
    rows: list[dict[str, float | str]] = []

    for samples in [30, 50, 100, 220, 500, 1000]:
        valve = hem_valve(opening_fraction=0.20, samples=samples)
        out = valve.capacity_result(
            upstream_pressure_bar=60.0,
            upstream_temperature_C=60.0,
            downstream_pressure_bar=20.0,
        )
        rows.append(
            {
                "samples": float(samples),
                "critical_pressure_bar": float(out["critical_pressure_bar"]),
                "critical_mass_flux_kg_m2_s": float(out["critical_mass_flux_kg_m2_s"]),
                "capacity_kg_s": float(out["mass_flow_capacity_kg_s"]),
                "critical_search_method": str(out["critical_search_method"]),
            }
        )

    df = pd.DataFrame(rows)
    cap_ref = float(df.loc[df["samples"] == 1000.0, "capacity_kg_s"].iloc[0])
    df["relative_capacity_error_vs_1000"] = (
        (df["capacity_kg_s"] - cap_ref).abs() / max(abs(cap_ref), 1.0e-30)
    )

    df.to_csv(RESULTS_DIR / "hem_critical_pressure_convergence.csv", index=False)

    figures: list[Path] = []

    p_ref = float(df.loc[df["samples"] == 1000.0, "critical_pressure_bar"].iloc[0])
    df["relative_pcrit_error_vs_1000"] = (
        (df["critical_pressure_bar"] - p_ref).abs() / max(abs(p_ref), 1.0e-30)
    )

    plt.figure()
    plt.plot(df["samples"], 100.0 * df["relative_pcrit_error_vs_1000"], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Pressure samples used for optimizer bracketing")
    plt.ylabel("Critical-pressure relative error vs 1000 samples [%]")
    plt.title("HEM numerical robustness: critical-pressure error")
    plt.grid(True, alpha=0.3)
    figures.append(save_current_figure("hem_critical_pressure_convergence.png"))

    plt.figure()
    plt.plot(df["samples"], 100.0 * df["relative_capacity_error_vs_1000"], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Pressure samples used for optimizer bracketing")
    plt.ylabel("Capacity relative error vs 1000 samples [%]")
    plt.title("HEM numerical robustness: capacity error")
    plt.grid(True, alpha=0.3)
    figures.append(save_current_figure("hem_capacity_convergence.png"))

    return figures


def write_figure_index(figures: list[Path]) -> Path:
    path = FIG_DIR / "README.md"

    lines = [
        "# Validation figures",
        "",
        "These figures are generated by:",
        "",
        "```bash",
        "python scripts/validation/generate_validation_figures.py",
        "```",
        "",
        "The figures document sanity checks and numerical robustness for the external CO₂ wellbore and HEM choke model.",
        "",
        "The reservoir simulator is not validated here.",
        "",
        "## Generated figures",
        "",
    ]

    for fig in figures:
        lines.append(f"- `{fig.name}`")

    lines.append("")
    lines.append("## Generated data tables")
    lines.append("")
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        lines.append(f"- `validation/results/{csv_path.name}`")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()

    figures: list[Path] = []
    figures.extend(make_co2_property_figures())
    figures.extend(make_wellbore_pressure_figure())
    figures.extend(make_wellbore_thermal_figure())
    figures.extend(make_hem_opening_figure())
    figures.extend(make_hem_downstream_pressure_figure())
    figures.extend(make_hem_convergence_figures())

    index = write_figure_index(figures)

    print("[OK] Generated validation figures:")
    for fig in figures:
        print(f"  - {fig}")

    print(f"[OK] Figure index: {index}")
    print(f"[OK] Data tables: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
