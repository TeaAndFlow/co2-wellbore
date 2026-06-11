"""Configuration dataclasses for the CO2 wellbore model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WellboreGeometry:
    """Vertical injection well geometry."""

    tvd_m: float = 1600.0
    diameter_m: float = 0.10
    roughness_m: float = 1.5e-5
    n_segments: int = 120
    std_pressure_bar: float = 1.01325
    std_temperature_C: float = 15.0
    min_reynolds: float = 1.0e-12

    def validate(self) -> None:
        if self.tvd_m <= 0.0:
            raise ValueError("tvd_m must be positive.")
        if self.diameter_m <= 0.0:
            raise ValueError("diameter_m must be positive.")
        if self.n_segments < 1:
            raise ValueError("n_segments must be >= 1.")
        if self.roughness_m < 0.0:
            raise ValueError("roughness_m must be non-negative.")


@dataclass
class ThermalConfig:
    """Simplified steady 1D heat exchange with geothermal surroundings."""

    enabled: bool = True
    overall_U_W_m2K: float = 4.0
    heat_transfer_diameter_m: Optional[float] = None
    surface_temperature_C: float = 32.0
    geothermal_gradient_C_per_m: float = 0.03
    max_abs_dT_per_segment_C: float = 10.0

    def validate(self) -> None:
        if self.overall_U_W_m2K < 0.0:
            raise ValueError("overall_U_W_m2K must be non-negative.")
        if self.geothermal_gradient_C_per_m < 0.0:
            raise ValueError("geothermal_gradient_C_per_m must be non-negative.")
        if self.heat_transfer_diameter_m is not None and self.heat_transfer_diameter_m <= 0.0:
            raise ValueError("heat_transfer_diameter_m must be positive when provided.")

    def geothermal_temperature_C(self, depth_m: float) -> float:
        return float(
            self.surface_temperature_C + self.geothermal_gradient_C_per_m * max(depth_m, 0.0)
        )


@dataclass
class DriftFluxConfig:
    """Reduced drift-flux closure for saturated two-phase CO2 states."""

    enabled: bool = True
    C0: float = 1.20
    drift_velocity_coeff: float = 0.35
    surface_tension_N_m: float = 0.01
    min_gas_holdup: float = 1.0e-5
    max_gas_holdup: float = 0.99999
    friction_density: str = "homogeneous"  # "homogeneous" or "hydrostatic"

    def validate(self) -> None:
        if self.C0 <= 0.0:
            raise ValueError("C0 must be positive.")
        if self.drift_velocity_coeff < 0.0:
            raise ValueError("drift_velocity_coeff must be non-negative.")
        if self.surface_tension_N_m <= 0.0:
            raise ValueError("surface_tension_N_m must be positive.")
        if not (0.0 <= self.min_gas_holdup < self.max_gas_holdup <= 1.0):
            raise ValueError("Gas holdup bounds must satisfy 0 <= min < max <= 1.")
        if self.friction_density not in {"homogeneous", "hydrostatic"}:
            raise ValueError("friction_density must be 'homogeneous' or 'hydrostatic'.")


@dataclass
class SolverConfig:
    """Numerical solver settings."""

    thp_bounds_bar: Tuple[float, float] = (1.0, 1000.0)
    q_bounds_sm3_day: Tuple[float, float] = (0.0, 1.0e7)
    tol_pressure_bar: float = 1.0e-4
    tol_rate_sm3_day: float = 1.0
    max_iter: int = 80

    def validate(self) -> None:
        if self.thp_bounds_bar[0] >= self.thp_bounds_bar[1]:
            raise ValueError("thp_bounds_bar must be increasing.")
        if self.q_bounds_sm3_day[0] > self.q_bounds_sm3_day[1]:
            raise ValueError("q_bounds_sm3_day must be increasing.")
        if self.tol_pressure_bar <= 0.0:
            raise ValueError("tol_pressure_bar must be positive.")
        if self.tol_rate_sm3_day <= 0.0:
            raise ValueError("tol_rate_sm3_day must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
