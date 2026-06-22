#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from co2_wellbore import CoolPropCO2
from co2_wellbore.constants import BAR_TO_PA


REF_PATH = Path("validation/reference/co2_nist_reference_points.csv")
FIG_DIR = Path("validation/figures")
RESULTS_DIR = Path("validation/results")


PROPERTY_LABELS = {
    "density_kg_m3": "Density",
    "enthalpy_J_kg": "Enthalpy",
    "viscosity_Pa_s": "Viscosity",
    "cp_J_kgK": "Cp",
}

PROPERTY_UNITS = {
    "density_kg_m3": "kg/m³",
    "enthalpy_J_kg": "J/kg",
    "viscosity_Pa_s": "Pa·s",
    "cp_J_kgK": "J/kg/K",
}


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not REF_PATH.exists():
        print(f"[SKIP] Missing reference file: {REF_PATH}")
        return

    ref = pd.read_csv(REF_PATH)

    required = ["case_id", "pressure_bar", "temperature_C"]
    for col in required:
        if col not in ref.columns:
            raise ValueError(f"Missing required column in {REF_PATH}: {col}")

    property_cols = [
        "density_kg_m3",
        "enthalpy_J_kg",
        "viscosity_Pa_s",
        "cp_J_kgK",
    ]

    available_property_cols = [
        c for c in property_cols if c in ref.columns and ref[c].notna().any()
    ]

    if not available_property_cols:
        print("[SKIP] Reference CSV exists, but no reference values are filled.")
        return

    props = CoolPropCO2()
    rows: list[dict[str, float | str]] = []

    for _, r in ref.iterrows():
        p_bar = float(r["pressure_bar"])
        t_C = float(r["temperature_C"])

        cp_out = props.props(p_bar * BAR_TO_PA, t_C + 273.15)

        model = {
            "density_kg_m3": float(cp_out["rho_kg_m3"]),
            "enthalpy_J_kg": float(cp_out["enthalpy_J_kg"]),
            "viscosity_Pa_s": float(cp_out["viscosity_Pa_s"]),
            "cp_J_kgK": float(cp_out["cp_J_kgK"]),
        }

        for prop_name in available_property_cols:
            ref_value = r.get(prop_name)

            if pd.isna(ref_value):
                continue

            ref_value = float(ref_value)
            model_value = float(model[prop_name])

            abs_error = model_value - ref_value
            rel_error_percent = abs(abs_error) / max(abs(ref_value), 1.0e-30) * 100.0

            rows.append(
                {
                    "case_id": str(r["case_id"]),
                    "pressure_bar": p_bar,
                    "temperature_C": t_C,
                    "property": prop_name,
                    "property_label": PROPERTY_LABELS[prop_name],
                    "unit": PROPERTY_UNITS[prop_name],
                    "reference_value": ref_value,
                    "model_value": model_value,
                    "absolute_error": abs_error,
                    "relative_error_percent": rel_error_percent,
                    "source": str(r.get("source", "")),
                }
            )

    out = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "co2_reference_property_comparison.csv"
    out.to_csv(out_csv, index=False)

    # Figure 1: maximum relative error by property.
    summary = (
        out.groupby(["property", "property_label"], as_index=False)
        .agg(
            max_relative_error_percent=("relative_error_percent", "max"),
            mean_relative_error_percent=("relative_error_percent", "mean"),
        )
        .sort_values("max_relative_error_percent", ascending=True)
    )
    summary.to_csv(RESULTS_DIR / "co2_reference_property_error_summary.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.barh(summary["property_label"], summary["max_relative_error_percent"])
    plt.xlabel("Maximum absolute relative error [%]")
    plt.ylabel("Property")
    plt.title("CO₂ property verification against NIST reference points")
    plt.grid(True, axis="x", alpha=0.3)
    xmax = max(float(summary["max_relative_error_percent"].max()) * 1.18, 1.0e-12)
    plt.xlim(0.0, xmax)

    for i, value in enumerate(summary["max_relative_error_percent"]):
        plt.text(
            value,
            i,
            f" {value:.4f}%",
            va="center",
        )

    savefig(FIG_DIR / "co2_reference_property_max_errors.png")

    # Figure 2+: parity plots by property.
    for prop_name in available_property_cols:
        prop_df = out[out["property"] == prop_name].copy()
        if prop_df.empty:
            continue

        label = PROPERTY_LABELS[prop_name]
        unit = PROPERTY_UNITS[prop_name]

        x = prop_df["reference_value"]
        y = prop_df["model_value"]

        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        pad = 0.05 * max(abs(hi - lo), 1.0e-30)

        plt.figure(figsize=(6.8, 5.2))
        plt.scatter(x, y)

        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", label="1:1 line")

        for _, row in prop_df.iterrows():
            plt.annotate(
                str(row["case_id"]),
                (float(row["reference_value"]), float(row["model_value"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        plt.xlabel(f"NIST reference {label.lower()} [{unit}]")
        plt.ylabel(f"Model {label.lower()} [{unit}]")
        plt.title(f"CO₂ {label.lower()} parity against frozen NIST points")
        plt.legend()
        plt.grid(True, alpha=0.3)

        savefig(FIG_DIR / f"co2_reference_{prop_name}_parity.png")

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {RESULTS_DIR / 'co2_reference_property_error_summary.csv'}")
    print(f"[OK] Wrote improved reference-property figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
