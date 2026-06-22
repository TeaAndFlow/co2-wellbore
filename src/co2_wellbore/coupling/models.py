from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReservoirResult:
    time_days: float
    rate_sm3_day: float
    bhp_bar: float
    thp_bar: float
    cum_gas_sm3: float
    reservoir_pressure_bar: float
    source: str