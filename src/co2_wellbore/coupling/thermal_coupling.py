from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ThermalUpdateState:
    """State for solving WTEMP consistency inside one coupling step."""

    previous_wtemp_C: Optional[float] = None
    previous_residual_C: Optional[float] = None

    lower_wtemp_C: Optional[float] = None
    lower_residual_C: Optional[float] = None

    upper_wtemp_C: Optional[float] = None
    upper_residual_C: Optional[float] = None


def propose_next_wtemp(
    *,
    state: ThermalUpdateState,
    current_wtemp_C: float,
    residual_C: float,
    relaxation: float,
    max_step_C: float,
    min_wtemp_C: float,
    max_wtemp_C: float,
) -> Tuple[float, Dict[str, float | str]]:
    """Propose next WTEMP for residual R = BHT_calc - WTEMP.

    Sign convention:
        residual > 0  => BHT_calc > WTEMP, current WTEMP is too low
        residual < 0  => BHT_calc < WTEMP, current WTEMP is too high

    Strategy:
        1. Maintain a sign-change bracket when possible.
        2. Use bracketed false-position/secant inside the bracket.
        3. Fall back to secant if only previous point exists.
        4. Fall back to relaxed fixed-point update.
        5. Always clamp update size and physical WTEMP bounds.
    """
    w = float(current_wtemp_C)
    r = float(residual_C)

    if r > 0.0:
        state.lower_wtemp_C = w
        state.lower_residual_C = r
    elif r < 0.0:
        state.upper_wtemp_C = w
        state.upper_residual_C = r

    method = "relaxed_fixed_point"
    candidate = w + float(relaxation) * r

    has_bracket = (
        state.lower_wtemp_C is not None
        and state.upper_wtemp_C is not None
        and state.lower_residual_C is not None
        and state.upper_residual_C is not None
    )

    if has_bracket:
        lo_w = float(state.lower_wtemp_C)
        hi_w = float(state.upper_wtemp_C)
        lo_r = float(state.lower_residual_C)
        hi_r = float(state.upper_residual_C)

        if hi_w < lo_w:
            lo_w, hi_w = hi_w, lo_w
            lo_r, hi_r = hi_r, lo_r

        denom = hi_r - lo_r
        if abs(denom) > 1.0e-12:
            candidate = lo_w + (0.0 - lo_r) * (hi_w - lo_w) / denom

            width = max(hi_w - lo_w, 0.0)
            if width > 1.0e-9:
                eps = 0.05 * width
                candidate = min(max(candidate, lo_w + eps), hi_w - eps)

            method = "bracketed_false_position"

    elif state.previous_wtemp_C is not None and state.previous_residual_C is not None:
        prev_w = float(state.previous_wtemp_C)
        prev_r = float(state.previous_residual_C)
        denom = r - prev_r

        if abs(denom) > 1.0e-12:
            candidate = w - r * (w - prev_w) / denom
            method = "secant"

    delta_raw = candidate - w

    max_step = abs(float(max_step_C))
    if max_step > 0.0 and abs(delta_raw) > max_step:
        candidate = w + math.copysign(max_step, delta_raw)
        method = method + "_clipped"

    candidate = min(max(candidate, float(min_wtemp_C)), float(max_wtemp_C))

    state.previous_wtemp_C = w
    state.previous_residual_C = r

    info: Dict[str, float | str] = {
        "thermal_update_method": method,
        "thermal_residual_C": r,
        "thermal_wtemp_current_C": w,
        "thermal_wtemp_next_C": float(candidate),
        "thermal_wtemp_delta_C": float(candidate - w),
    }

    return float(candidate), info