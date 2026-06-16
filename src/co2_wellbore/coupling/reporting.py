from __future__ import annotations

import math
from typing import Any

def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out
    except Exception:
        return default


def _fmt(value: Any, ndigits: int = 3, unit: str = "") -> str:
    x = _safe_float(value)
    if math.isnan(x):
        s = "nan"
    elif math.isinf(x):
        s = "+inf" if x > 0 else "-inf"
    else:
        s = f"{x:.{ndigits}f}"
    return f"{s} {unit}".rstrip()


def _status(flag: bool) -> str:
    return "OK" if bool(flag) else "NOT OK"


def _ratio(residual: Any, tolerance: Any) -> str:
    r = abs(_safe_float(residual))
    t = abs(_safe_float(tolerance))
    if not math.isfinite(r) or not math.isfinite(t) or t <= 0.0:
        return "n/a"
    return f"{r / t:.2f} x tol"


def _opening_choke_reason(valve_resid: float) -> str:
    if math.isinf(float(valve_resid)):
        return (
            "INFEASIBLE: required THP is above upstream pressure. "
            "Surface system cannot deliver this state; decrease trial rate."
        )
    if valve_resid > 0.0:
        return (
            "Trial rate is too high for this opening: "
            "Kv_required > Kv_effective. Move upper rate bracket downward."
        )
    if valve_resid < 0.0:
        return (
            "Trial rate is too low for this opening: "
            "Kv_required < Kv_effective. Move lower rate bracket upward."
        )
    return "Choke capacity is exactly matched."


def _opening_choke_flow_reason(flow_resid_kg_s: float) -> str:
    """Explain HEM mass-flow residual sign.

    R_m = m_actual - m_capacity.
    """
    r = float(flow_resid_kg_s)
    if math.isinf(r):
        return (
            "INFEASIBLE: required THP is above upstream pressure or HEM capacity is unavailable. "
            "Decrease trial rate."
        )
    if r > 0.0:
        return (
            "Trial rate is too high for this opening: "
            "actual mass rate exceeds HEM choke capacity. Move upper rate bracket downward."
        )
    if r < 0.0:
        return (
            "Trial rate is below HEM choke capacity: "
            "choke has spare capacity. Move lower rate bracket upward, or accept as capacity surplus."
        )
    return "HEM choke mass-flow capacity is exactly matched."


def print_solver_iteration_block(
    *,
    step: int,
    iteration: int,
    max_iter: int,
    mode: str,
    root: str,
    flow_ok: bool,
    flow_seconds: float,
    reservoir_source: str,
    q_target: float,
    q_actual: float,
    q_lo: float,
    q_hi: float,
    opening_percent: float,
    kv_eff: float,
    kv_required: float,
    valve_resid: float,
    kv_tol: float,
    bhp_opm: float,
    thp_calc: float,
    bhp_calc: float,
    inversion_resid: float,
    p_tol: float,
    wtemp_used: float,
    bht_calc: float,
    temp_resid: float,
    t_tol: float,
    rate_resid: float,
    q_tol: float,
    t2: float,
    quality: float,
    phase: str,
    pressure_converged: bool,
    temperature_converged: bool,
    rate_converged: bool,
    choke_model: str = "legacy_kv",
    choke_capacity_kg_s: float = float("nan"),
    choke_actual_mass_rate_kg_s: float = float("nan"),
    choke_capacity_residual_kg_s: float = float("nan"),
    hem_flow_tolerance_kg_s: float = float("nan"),
    flow_regime: str = "",
) -> None:
    converged = bool(pressure_converged and temperature_converged and rate_converged)
    is_hem = bool(mode == "opening_choke" and str(choke_model).lower() == "hem")

    if mode == "opening_choke" and is_hem:
        pressure_name = "CHOKE FLOW"
        pressure_equation = "R_m = m_actual - m_capacity_HEM(opening, P1, T1, THP)"
        pressure_value = choke_capacity_residual_kg_s
        pressure_tol = hem_flow_tolerance_kg_s
        pressure_unit = "kg/s"
        pressure_reason = _opening_choke_flow_reason(float(choke_capacity_residual_kg_s))
    elif mode == "opening_choke":
        pressure_name = "CHOKE CAPACITY"
        pressure_equation = "R_Kv = Kv_required(q, THP_required) - Kv_effective(opening)"
        pressure_value = valve_resid
        pressure_tol = kv_tol
        pressure_unit = "m3/h"
        pressure_reason = _opening_choke_reason(float(valve_resid))
    elif mode == "thp_limit":
        pressure_name = "THP LIMIT"
        pressure_equation = "R_THP = THP_required - THP_target"
        pressure_value = thp_calc
        pressure_tol = p_tol
        pressure_unit = "bar"
        pressure_reason = "THP-limit control mode."
    else:
        pressure_name = "BHP INVERSION"
        pressure_equation = "R_BHP = BHP_calc - WBHP_OPM"
        pressure_value = inversion_resid
        pressure_tol = p_tol
        pressure_unit = "bar"
        pressure_reason = "Fixed-rate pressure reconstruction mode."

    decision = "ACCEPT STEP" if converged else "CONTINUE ITERATION"

    print()
    print("=" * 96)
    print(f"COUPLING NONLINEAR ITERATION REPORT")
    print("-" * 96)
    print(f"Step / Iteration     : {step} / {iteration} of {max_iter}")
    print(f"Segment root         : {root}")
    print(f"Mode                 : {mode}")
    print(f"Choke model          : {choke_model}")
    print(f"Decision             : {decision}")
    print("-" * 96)
    print("1) OPM RESERVOIR EVALUATION")
    print(f"   Flow status        : {'OK' if flow_ok else 'FAILED'}  ({_fmt(flow_seconds, 2, 's')})")
    print(f"   Result source      : {reservoir_source}")
    print(f"   Trial rate target  : {_fmt(q_target, 3, 'sm3/day')}")
    print(f"   OPM actual rate    : {_fmt(q_actual, 3, 'sm3/day')}")
    print(f"   Rate residual      : {_fmt(rate_resid, 3, 'sm3/day')}  "
          f"|tol|={_fmt(q_tol, 3)}  [{_status(rate_converged)}; {_ratio(rate_resid, q_tol)}]")
    print(f"   WBHP from OPM      : {_fmt(bhp_opm, 3, 'bar')}")
    print("-" * 96)
    print("2) EXTERNAL WELLBORE INVERSION")
    print("   Search variable    : THP_required")
    print("   Equation           : BHP_wellbore(THP, q_OPM, h_choke) = WBHP_OPM")
    print(f"   THP required       : {_fmt(thp_calc, 3, 'bar')}")
    print(f"   BHP calculated     : {_fmt(bhp_calc, 3, 'bar')}")
    print(f"   BHP inversion res. : {_fmt(inversion_resid, 4, 'bar')}  "
          f"|tol|={_fmt(p_tol, 4)}  [{_ratio(inversion_resid, p_tol)}]")
    print("-" * 96)
    print("3) SURFACE CHOKE / VALVE CONSTRAINT")
    print(f"   Opening            : {_fmt(opening_percent, 2, '%')}")

    if is_hem:
        print(f"   Flow regime        : {flow_regime}")
        print(f"   Actual mass rate   : {_fmt(choke_actual_mass_rate_kg_s, 5, 'kg/s')}")
        print(f"   HEM capacity       : {_fmt(choke_capacity_kg_s, 5, 'kg/s')}")
        print(f"   Residual equation  : {pressure_equation}")
        print(f"   {pressure_name:18s}: {_fmt(pressure_value, 5, pressure_unit)}  "
              f"|tol|={_fmt(pressure_tol, 5)}  [{_status(pressure_converged)}; {_ratio(pressure_value, pressure_tol)}]")
    else:
        print(f"   Kv effective       : {_fmt(kv_eff, 4, 'm3/h')}")
        print(f"   Kv required        : {_fmt(kv_required, 4, 'm3/h')}")
        print(f"   Residual equation  : {pressure_equation}")
        print(f"   {pressure_name:18s}: {_fmt(pressure_value, 4, pressure_unit)}  "
              f"|tol|={_fmt(pressure_tol, 4)}  [{_status(pressure_converged)}; {_ratio(pressure_value, pressure_tol)}]")

    print(f"   Reason             : {pressure_reason}")
    print(f"   Post-choke T       : {_fmt(t2, 3, 'C')}")
    print(f"   Post-choke quality : {_fmt(quality, 5)}")
    print(f"   Post-choke phase   : {phase}")
    print("-" * 96)
    print("4) THERMAL COUPLING")
    print("   Equation           : WTEMP_OPM -> BHT_wellbore consistency")
    print(f"   WTEMP used in OPM  : {_fmt(wtemp_used, 3, 'C')}")
    print(f"   BHT calculated     : {_fmt(bht_calc, 3, 'C')}")
    print(f"   Temperature resid. : {_fmt(temp_resid, 3, 'C')}  "
          f"|tol|={_fmt(t_tol, 3)}  [{_status(temperature_converged)}; {_ratio(temp_resid, t_tol)}]")
    print("-" * 96)
    print("5) RATE SEARCH STATE")
    print(f"   Current bracket    : [{_fmt(q_lo, 3)}, {_fmt(q_hi, 3)}] sm3/day")
    print(f"   Search method      : bracketed secant / false-position when both residuals exist, otherwise bisection")
    print(f"   Pressure OK        : {_status(pressure_converged)}")
    print(f"   Temperature OK     : {_status(temperature_converged)}")
    print(f"   Rate OK            : {_status(rate_converged)}")
    print("=" * 96)
    print()
