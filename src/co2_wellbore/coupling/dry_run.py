from __future__ import annotations

import math
from typing import Optional

from co2_wellbore import CO2WellboreCalculator

from .deck_editing import StepControl
from .models import ReservoirResult


def dry_run_reservoir_result(
    wb: CO2WellboreCalculator,
    control: StepControl,
    t_head_C: float,
    thp_reference_bar: float,
    prev_cum: float,
    cumulative_days: float,
    wellhead_enthalpy_J_kg: Optional[float] = None,
    wellhead_quality_mass: Optional[float] = None,
) -> ReservoirResult:
    """Synthetic reservoir result for debugging deck/script logic only."""
    bhp = wb.bhp_from_thp_and_rate(
        thp_reference_bar,
        q_sm3_day=control.rate_target_sm3_day,
        wellhead_temperature_C=t_head_C,
        wellhead_enthalpy_J_kg=wellhead_enthalpy_J_kg,
        wellhead_quality_mass=wellhead_quality_mass,
    )

    return ReservoirResult(
        time_days=float(cumulative_days),
        rate_sm3_day=float(control.rate_target_sm3_day),
        bhp_bar=min(float(bhp), float(control.bhp_limit_bar)),
        thp_bar=float("nan"),
        cum_gas_sm3=prev_cum + control.rate_target_sm3_day * control.step_days,
        reservoir_pressure_bar=float("nan"),
        source="dry_run_synthetic",
    )