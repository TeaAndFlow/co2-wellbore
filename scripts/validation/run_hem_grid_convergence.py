#!/usr/bin/env python3
from __future__ import annotations

from co2_wellbore import CO2HEMChokeValve, HEMChokeConfig


def run(samples: int):
    valve = CO2HEMChokeValve(
        config=HEMChokeConfig(
            full_diameter_m=0.067,
            opening_fraction=0.20,
            discharge_coefficient=0.84,
            area_exponent=1.0,
            min_pressure_bar=10.0,
            n_pressure_samples=samples,
        )
    )
    return valve.capacity_result(
        upstream_pressure_bar=60.0,
        upstream_temperature_C=60.0,
        downstream_pressure_bar=20.0,
    )


def main() -> None:
    sample_list = [50, 100, 220, 500, 1000]
    results = [(n, run(n)) for n in sample_list]

    ref_cap = float(results[-1][1]["mass_flow_capacity_kg_s"])

    print("# HEM critical-pressure grid convergence")
    print()
    print("Case:")
    print("- upstream pressure: 60 bar")
    print("- upstream temperature: 60 C")
    print("- downstream pressure: 20 bar")
    print("- valve diameter: 0.067 m")
    print("- opening: 20 %")
    print()
    print(
        "| samples | method | pcrit bar | Gcrit kg/m2/s | capacity kg/s | rel err vs 1000 |"
    )
    print(
        "|---:|---|---:|---:|---:|---:|"
    )

    for n, out in results:
        cap = float(out["mass_flow_capacity_kg_s"])
        rel = abs(cap - ref_cap) / max(ref_cap, 1.0e-30)

        print(
            f"| {n} "
            f"| {out['critical_search_method']} "
            f"| {float(out['critical_pressure_bar']):.6f} "
            f"| {float(out['critical_mass_flux_kg_m2_s']):.6f} "
            f"| {cap:.9f} "
            f"| {rel:.6e} |"
        )


if __name__ == "__main__":
    main()
