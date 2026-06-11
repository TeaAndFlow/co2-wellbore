"""Reduced drift-flux closure for saturated two-phase CO2 states."""

from __future__ import annotations

import math
from typing import Dict

from .config import DriftFluxConfig
from .constants import G
from .correlations import clamp


def build_single_phase_flow_props(base: Dict[str, float | str]) -> Dict[str, float | str]:
    """Wrap single-phase property data into the common flow-property dictionary."""
    rho = float(base["rho_kg_m3"])
    mu = float(base.get("viscosity_Pa_s", float("nan")))

    if not math.isfinite(mu):
        mu = 1.0e-5

    return {
        **base,
        "two_phase_active": 0.0,
        "rho_hydro_kg_m3": rho,
        "rho_friction_kg_m3": rho,
        "rho_flow_kg_m3": rho,
        "gas_holdup": 0.0,
        "liquid_holdup": 0.0,
        "rho_vapor_kg_m3": float("nan"),
        "rho_liquid_kg_m3": float("nan"),
        "viscosity_mix_Pa_s": mu,
    }


def apply_drift_flux_closure(
    base: Dict[str, float | str],
    vapor: Dict[str, float],
    liquid: Dict[str, float],
    mass_rate_kg_s: float,
    area_m2: float,
    config: DriftFluxConfig,
) -> Dict[str, float | str]:
    """Apply a reduced vertical-well drift-flux closure to a saturated CO2 state."""
    q_mass = float(base.get("quality_mass", float("nan")))

    if not config.enabled or not math.isfinite(q_mass) or not (0.0 < q_mass < 1.0):
        return build_single_phase_flow_props(base)

    rho_v = max(float(vapor["rho_kg_m3"]), 1.0e-12)
    rho_l = max(float(liquid["rho_kg_m3"]), 1.0e-12)

    mu_v = max(float(vapor["viscosity_Pa_s"]), 1.0e-12)
    mu_l = max(float(liquid["viscosity_Pa_s"]), 1.0e-12)

    x = clamp(q_mass, 0.0, 1.0)

    m_g = x * mass_rate_kg_s
    m_l = (1.0 - x) * mass_rate_kg_s

    qg = m_g / rho_v
    ql = m_l / rho_l

    jg = qg / max(area_m2, 1.0e-30)
    jl = ql / max(area_m2, 1.0e-30)
    j_total = jg + jl

    delta_rho = max(rho_l - rho_v, 0.0)
    vgj = (
        config.drift_velocity_coeff
        * ((G * config.surface_tension_N_m * delta_rho) / max(rho_l**2, 1.0e-30)) ** 0.25
    )

    alpha_g = jg / max(config.C0 * j_total + vgj, 1.0e-30)
    alpha_g = clamp(alpha_g, config.min_gas_holdup, config.max_gas_holdup)
    alpha_l = 1.0 - alpha_g

    rho_hydro = alpha_g * rho_v + alpha_l * rho_l
    rho_hom = 1.0 / max(x / rho_v + (1.0 - x) / rho_l, 1.0e-30)

    if config.friction_density == "homogeneous":
        rho_friction = rho_hom
    else:
        rho_friction = rho_hydro

    mu_mix = max(mu_v**alpha_g * mu_l**alpha_l, 1.0e-8)

    return {
        **base,
        "two_phase_active": 1.0,
        "rho_hydro_kg_m3": float(rho_hydro),
        "rho_friction_kg_m3": float(rho_friction),
        "rho_flow_kg_m3": float(rho_hom),
        "gas_holdup": float(alpha_g),
        "liquid_holdup": float(alpha_l),
        "rho_vapor_kg_m3": float(rho_v),
        "rho_liquid_kg_m3": float(rho_l),
        "viscosity_mix_Pa_s": float(mu_mix),
    }
