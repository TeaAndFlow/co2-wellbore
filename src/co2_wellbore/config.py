"""Configuration dataclasses for the CO2 wellbore model."""

from __future__ import annotations

import math
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
    """1D heat exchange with optional layered radial thermal resistance.

    model:
        "overall_U" keeps the old behavior.
        "layered" computes U from tubing/cement/formation resistances.

    The layered model is still a reduced engineering model, not a full
    transient thermal simulator. But it is much more physical than one
    constant overall_U.
    """

    enabled: bool = True

    # Old/simple mode.
    overall_U_W_m2K: float = 4.0

    heat_transfer_diameter_m: Optional[float] = None
    surface_temperature_C: float = 32.0
    geothermal_gradient_C_per_m: float = 0.03
    max_abs_dT_per_segment_C: float = 10.0

    # New thermal model selector.
    model: str = "overall_U"  # "overall_U" or "layered"

    # Layered radial thermal model.
    tubing_outer_diameter_m: float = 0.1143
    cement_outer_diameter_m: float = 0.2160

    inner_heat_transfer_coefficient_W_m2K: float = 1000.0
    tubing_k_W_mK: float = 45.0
    cement_k_W_mK: float = 1.0
    formation_k_W_mK: float = 2.5
    formation_heat_capacity_J_m3K: float = 2.2e6

    # Approximate transient radial heat penetration into formation.
    transient_formation: bool = True
    min_thermal_time_days: float = 0.01
    max_formation_radius_m: float = 30.0

    # Updated by the orchestrator at every coupling step.
    elapsed_time_days: Optional[float] = None

    def validate(self) -> None:
        if self.model not in {"overall_U", "layered"}:
            raise ValueError("thermal model must be 'overall_U' or 'layered'.")
        if self.overall_U_W_m2K < 0.0:
            raise ValueError("overall_U_W_m2K must be non-negative.")
        if self.geothermal_gradient_C_per_m < 0.0:
            raise ValueError("geothermal_gradient_C_per_m must be non-negative.")
        if self.heat_transfer_diameter_m is not None and self.heat_transfer_diameter_m <= 0.0:
            raise ValueError("heat_transfer_diameter_m must be positive when provided.")
        if self.tubing_outer_diameter_m <= 0.0:
            raise ValueError("tubing_outer_diameter_m must be positive.")
        if self.cement_outer_diameter_m <= 0.0:
            raise ValueError("cement_outer_diameter_m must be positive.")
        if self.inner_heat_transfer_coefficient_W_m2K < 0.0:
            raise ValueError("inner_heat_transfer_coefficient_W_m2K must be non-negative.")
        if self.tubing_k_W_mK <= 0.0:
            raise ValueError("tubing_k_W_mK must be positive.")
        if self.cement_k_W_mK <= 0.0:
            raise ValueError("cement_k_W_mK must be positive.")
        if self.formation_k_W_mK <= 0.0:
            raise ValueError("formation_k_W_mK must be positive.")
        if self.formation_heat_capacity_J_m3K <= 0.0:
            raise ValueError("formation_heat_capacity_J_m3K must be positive.")
        if self.min_thermal_time_days <= 0.0:
            raise ValueError("min_thermal_time_days must be positive.")
        if self.max_formation_radius_m <= 0.0:
            raise ValueError("max_formation_radius_m must be positive.")

    def geothermal_temperature_C(self, depth_m: float) -> float:
        return float(
            self.surface_temperature_C + self.geothermal_gradient_C_per_m * max(depth_m, 0.0)
        )

    def effective_overall_U_W_m2K(
        self,
        depth_m: float,
        flow_diameter_m: float,
    ) -> float:
        """Return effective U [W/m2/K] based on selected thermal model.

        The returned U is referenced to the heat-transfer area pi * D_flow * dz,
        matching the existing enthalpy update in calculator.py.
        """
        if not self.enabled:
            return 0.0

        if self.model == "overall_U":
            return float(self.overall_U_W_m2K)

        D_i = float(self.heat_transfer_diameter_m or flow_diameter_m)
        D_i = max(D_i, 1.0e-6)

        r_i = 0.5 * D_i
        r_tub_o = max(0.5 * float(self.tubing_outer_diameter_m), r_i * 1.001)
        r_cem_o = max(0.5 * float(self.cement_outer_diameter_m), r_tub_o * 1.001)

        # Internal convection resistance per meter.
        h_i = float(self.inner_heat_transfer_coefficient_W_m2K)
        R_inner = 0.0 if h_i <= 0.0 else 1.0 / max(h_i * 2.0 * math.pi * r_i, 1.0e-30)

        # Wall / cement radial conduction resistance per meter.
        R_tubing = math.log(r_tub_o / r_i) / (2.0 * math.pi * self.tubing_k_W_mK)
        R_cement = math.log(r_cem_o / r_tub_o) / (2.0 * math.pi * self.cement_k_W_mK)

        # Formation resistance. If transient is enabled, grow thermal radius with time.
        if self.transient_formation:
            t_days = self.elapsed_time_days
            if t_days is None or not math.isfinite(float(t_days)):
                t_days = self.min_thermal_time_days
            t_s = max(float(t_days), self.min_thermal_time_days) * 86400.0
            alpha = self.formation_k_W_mK / max(self.formation_heat_capacity_J_m3K, 1.0e-30)
            penetration = 2.0 * math.sqrt(max(alpha * t_s, 0.0))
            r_ext = r_cem_o + penetration
            r_ext = min(max(r_ext, r_cem_o * 1.01), float(self.max_formation_radius_m))
        else:
            r_ext = max(float(self.max_formation_radius_m), r_cem_o * 1.01)

        R_formation = math.log(r_ext / r_cem_o) / (2.0 * math.pi * self.formation_k_W_mK)

        R_total = max(R_inner + R_tubing + R_cement + R_formation, 1.0e-30)

        # q' = DeltaT / R_total = U * pi * D_i * DeltaT
        U = 1.0 / (R_total * math.pi * D_i)
        return float(max(U, 0.0))


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
