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


def print_solver_iteration_block(**kw) -> None:
    """Print one compact, user-friendly coupling iteration line.

    Full diagnostics are still written by orchestrator.py to CSV/JSON.
    Terminal output must stay readable during long OPM runs.
    """
    import math

    def get(name: str, default=None):
        return kw.get(name, default)

    def f(name: str, fmt: str = ".3f", default: str = "nan") -> str:
        try:
            x = float(get(name))
            if not math.isfinite(x):
                return default
            return format(x, fmt)
        except Exception:
            return default

    def flag(name: str) -> str:
        return "OK" if bool(get(name, False)) else "NO"

    step = int(get("step", -1))
    max_steps = int(get("max_steps", -1))
    iteration = int(get("iteration", -1))
    max_iter = int(get("max_iter", -1))

    pressure_ok = flag("pressure_converged")
    temperature_ok = flag("temperature_converged")
    rate_ok = flag("rate_converged")

    accepted = pressure_ok == "OK" and temperature_ok == "OK" and rate_ok == "OK"
    decision = "accept" if accepted else "continue"

    choke_model = str(get("choke_model", "")).lower()
    if choke_model == "hem":
        choke_text = f"choke={f('choke_capacity_residual_kg_s', '+.3f')} kg/s"
    else:
        choke_text = f"dKv={f('valve_resid', '+.4f')}"

    if max_steps > 0:
        step_text = f"{step:02d}/{max_steps:02d}"
    else:
        step_text = f"{step:02d}"

    print(
        f"[{step_text} it {iteration:02d}/{max_iter:02d}] "
        f"open={f('opening_percent', '.1f')}% "
        f"q={f('q_actual', '.0f')} sm3/d | "
        f"WBHP={f('bhp_opm', '.1f')} "
        f"THP={f('thp_calc', '.2f')} | "
        f"dT={f('temp_resid', '+.2f')}C | "
        f"{choke_text} | "
        f"P:{pressure_ok} T:{temperature_ok} -> {decision}",
        flush=True,
    )