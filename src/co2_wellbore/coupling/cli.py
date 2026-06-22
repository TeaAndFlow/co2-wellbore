from __future__ import annotations

import argparse

from .orchestrator import run_orchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OPM Flow restart coupling with CoolProp/ManyWells CO2 wellbore and THP/valve pressure control"
    )

    p.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Folder containing Base.DATA and INCLUDE/ files. "
            "If omitted, defaults to the repository data/ directory."
        ),
    )
    p.add_argument("--deck", default="Base.DATA", help="Deck filename or absolute path")
    p.add_argument("--output-root", default="runs_restart_coupled_coolprop_manywells")
    p.add_argument("--figures-dir", default="figures", help="Directory where informative 07 figures will be saved")
    p.add_argument("--case-id", default="Base_COOLPROP_MANYWELLS_THPVALVE")
    p.add_argument("--well-name", default="INJ1")

    p.add_argument("--run-flow", action="store_true", help="Actually run OPM Flow")
    p.add_argument("--dry-run", action="store_true", help="Do not run OPM; use synthetic reservoir values for checking script logic")
    p.add_argument("--flow-exe", default="flow")
    p.add_argument("--mpi-nproc", type=int, default=1)
    p.add_argument("--flow-timeout-sec", type=float, default=1800.0)

    # VTK/PVTU output options for ML dataset collection.
    # These are passed directly to OPM Flow as command-line switches.
    p.add_argument("--enable-vtk-output", action="store_true",
                   help="Pass --enable-vtk-output=true and VTK array switches to OPM Flow")
    p.add_argument("--vtk-write-temperature", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request temperature output if supported by Flow")
    p.add_argument("--vtk-write-pressures", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request pressure output")
    p.add_argument("--vtk-write-saturations", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request saturation output")

    p.add_argument("--n-steps", type=int, default=36)
    p.add_argument("--step-days", type=float, default=30.0)
    p.add_argument("--rate-sm3-day", type=float, default=100000.0)
    p.add_argument("--rate-schedule", type=float, nargs="*", default=None, help="Demand/cap q(t) list, sm3/day, one value per step")
    p.add_argument("--rate-schedule-is-cap", action=argparse.BooleanOptionalAction, default=True,
                   help="In pressure-control modes, treat rate schedule as an upper demand/cap rather than an exact target")
    p.add_argument("--bhp-limit-bar", type=float, default=240.0)

    p.add_argument(
        "--pressure-control-mode",
        choices=["fixed_rate", "thp_limit", "opening_choke"],
        default="thp_limit",
        help="fixed_rate = old robust baseline; thp_limit = reduce q until THP limit is satisfied",
    )

    p.add_argument(
        "--opening-schedule-percent",
        type=float,
        nargs="*",
        default=None,
        help="Choke opening schedule in percent, one value per coupling step.",
    )

    p.add_argument("--upstream-pressure-bar", type=float, default=98.0)
    p.add_argument("--upstream-temperature-C", type=float, default=4.0)
    p.add_argument("--kv-full-m3-h", type=float, default=118.0)
    p.add_argument("--choke-model", choices=["legacy_kv", "hem"], default="hem",
                   help="legacy_kv = fluids/IEC liquid-like Kv model; hem = real-fluid HEM choke backend")
    p.add_argument("--valve-FL", type=float, default=0.90)
    p.add_argument("--valve-Fd", type=float, default=1.00)
    p.add_argument("--valve-diameter-m", type=float, default=0.16,
                   help="For HEM: physical full-open choke diameter. For legacy_kv: valve diameter for IEC sizing.")
    p.add_argument("--hem-discharge-coefficient", type=float, default=0.84)
    p.add_argument("--hem-area-exponent", type=float, default=1.0,
                   help="HEM effective area law: A_eff = A_full * opening_fraction ** exponent")
    p.add_argument("--hem-min-pressure-bar", type=float, default=6.0,
                   help="Minimum pressure for HEM throat search; keep above CO2 triple-region problems")
    p.add_argument("--hem-pressure-samples", type=int, default=220)
    p.add_argument("--hem-flow-tolerance-kg-s", type=float, default=0.05,
                   help="HEM choke residual tolerance: |m_actual - m_capacity| in kg/s")
    p.add_argument("--kv-tolerance-m3-h", type=float, default=0.05,
                   help="Legacy Kv tolerance. Ignored by HEM residual; HEM uses --hem-flow-tolerance-kg-s.")
    p.add_argument("--thp-target-bar", type=float, default=80.0,
                   help="Target/maximum downstream THP for thp_limit mode")
    p.add_argument("--thp-tolerance-bar", type=float, default=0.25)
    p.add_argument("--q-min-sm3-day", type=float, default=0.0)
    p.add_argument("--q-max-sm3-day", type=float, default=300000.0)

    p.add_argument("--initial-wtemp-C", type=float, default=40.0, help="First WTEMP guess if not reading from deck")
    p.add_argument("--use-deck-wtemp-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--carry-wtemp-between-steps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-coupling-iter", type=int, default=12,
                   help="Total restart evaluations per step for both pressure and temperature convergence")
    p.add_argument("--temperature-relaxation", type=float, default=0.7)
    p.add_argument(
        "--wtemp-max-step-C",
        type=float,
        default=10.0,
        help="Maximum WTEMP update per coupling iteration.",
    )
    p.add_argument(
        "--wtemp-min-C",
        type=float,
        default=-20.0,
        help="Lower bound for WTEMP coupling guesses.",
    )
    p.add_argument(
        "--wtemp-max-C",
        type=float,
        default=150.0,
        help="Upper bound for WTEMP coupling guesses.",
    )
    p.add_argument("--t-tolerance-C", type=float, default=0.5)
    p.add_argument("--p-tolerance-bar", type=float, default=0.05,
                   help="Wellbore inversion tolerance; not the THP-control tolerance")
    p.add_argument("--q-tolerance-sm3-day", type=float, default=100.0)
    p.add_argument("--allow-force-accept", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, accept the last iteration at max iterations; default False for strict science runs")
    p.add_argument("--accept-last-feasible", action=argparse.BooleanOptionalAction, default=True,
                   help="If strict root convergence fails, accept the last pressure-feasible row once temperature is converged")
    p.add_argument("--accept-capacity-surplus", action=argparse.BooleanOptionalAction, default=True,
                   help="In opening_choke mode, accept a feasible point when Kv_required < Kv_effective and the next higher rate is unsafe")
    p.add_argument("--capacity-surplus-q-width-sm3-day", type=float, default=5000.0,
                   help="If the q bracket is narrower than this and Kv_required < Kv_effective, accept as non-binding choke capacity")

    p.add_argument("--reference-thp-bar", type=float, default=100.0, help="Only for dry-run fallback and inversion fallback")
    p.add_argument("--thp-min-bar", type=float, default=1.0)
    p.add_argument("--thp-max-bar", type=float, default=350.0)
    p.add_argument("--t-head-C", type=float, default=5.0, help="External wellhead/tubing-head temperature for the calculator")

    p.add_argument(
        "--wellhead-enthalpy-J-kg",
        type=float,
        default=None,
        help="Post-choke inlet enthalpy for wellbore P-H flash. Best option for isenthalpic choke boundary.",
    )
    p.add_argument(
        "--wellhead-quality-mass",
        type=float,
        default=None,
        help="Post-choke saturated CO2 mass quality at wellhead. Use only if you know x after choke.",
    )

    # CoolProp/ManyWells wellbore parameters matching the Base.DATA-like case.
    p.add_argument("--tvd-m", type=float, default=1600.0)
    p.add_argument("--diameter-m", type=float, default=0.10)
    p.add_argument("--roughness-m", type=float, default=1.5e-5)
    p.add_argument("--n-segments", type=int, default=120)
    p.add_argument("--std-pressure-bar", type=float, default=1.01325)
    p.add_argument("--std-temperature-C", type=float, default=15.0)
    p.add_argument("--surface-temperature-C", type=float, default=32.0)
    p.add_argument("--geothermal-gradient-C-per-m", type=float, default=0.03)
    p.add_argument("--relaxation-length-m", type=float, default=700.0)

    p.add_argument("--heat-transfer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overall-U-W-m2K", type=float, default=4.0)
    p.add_argument("--thermal-model", choices=["overall_U", "layered"], default="overall_U",
                   help="overall_U = old fixed U; layered = tubing/cement/formation resistance model")
    p.add_argument("--tubing-outer-diameter-m", type=float, default=0.1143)
    p.add_argument("--cement-outer-diameter-m", type=float, default=0.2160)
    p.add_argument("--inner-heat-transfer-coefficient-W-m2K", type=float, default=1000.0)
    p.add_argument("--tubing-k-W-mK", type=float, default=45.0)
    p.add_argument("--cement-k-W-mK", type=float, default=1.0)
    p.add_argument("--formation-k-W-mK", type=float, default=2.5)
    p.add_argument("--formation-heat-capacity-J-m3K", type=float, default=2.2e6)
    p.add_argument("--transient-formation-thermal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-thermal-time-days", type=float, default=0.01)
    p.add_argument("--max-formation-radius-m", type=float, default=30.0)
    p.add_argument("--max-abs-dT-per-segment-C", type=float, default=10.0)
    p.add_argument("--max-abs-jt-K-per-Pa", type=float, default=2.0e-5)
    p.add_argument("--include-joule-thomson-in-well", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--two-phase", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable CoolProp-saturation-triggered drift-flux closure; usually inactive for dense/supercritical storage states")
    p.add_argument("--drift-flux-C0", type=float, default=1.20)
    p.add_argument("--drift-velocity-coeff", type=float, default=0.35)
    p.add_argument("--surface-tension-N-m", type=float, default=0.01)
    p.add_argument("--friction-density", choices=["homogeneous", "hydrostatic"], default="homogeneous")

    p.add_argument("--inner-solver-max-iter", type=int, default=80)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.run_flow:
        args.dry_run = True
    if args.dry_run:
        print("[INFO] dry-run mode: OPM Flow will not be executed; this only checks deck/script logic.")
    run_orchestrator(args)