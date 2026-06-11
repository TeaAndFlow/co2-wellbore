#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co2_wellbore_coolprop_manywells.py

A stronger open-source research CO2 injection wellbore calculator for OPM/ECL coupling.

Design goal
-----------
This module is meant to replace/upgrade the earlier PR-EOS MVP calculator:

    co2_wellbore_pr_phase.py
    co2_wellbore_pr_advanced.py

What is stronger here
---------------------
1) CO2 properties from CoolProp/Span-Wagner instead of the local PR-EOS placeholders:
      density, viscosity, cp, enthalpy, phase indicator, Joule-Thomson coefficient.
2) ManyWells-inspired architecture:
      - explicit wellbore state columns for P, T, rho, phase, velocity, holdup;
      - optional drift-flux two-phase closure when CoolProp reports a saturated state;
      - pressure-controlled/injectivity-controlled rate solver.
3) Better pressure-control interface for OPM restart coupling:
      - fixed RATE mode still supported;
      - required THP for a target OPM WBHP;
      - THP-limited rate solver using an injectivity proxy;
      - VFPINJ table generator from the same physics.

Important warning
-----------------
This is still not LedaFlow/OLGA/PipeSim. It is a transparent research calculator.
The single-phase/supercritical CO2 part is much stronger than the previous PR MVP because
it uses CoolProp. The two-phase drift-flux block is still an engineering closure; it is not
a full transient non-equilibrium multiphase flow simulator.

Install
-------
    pip install CoolProp numpy pandas

Typical use
-----------
    from co2_wellbore_coolprop_manywells import *

    calc = CO2CoolPropManyWellsCalculator(
        WellboreGeometry(tvd_m=1600.0, diameter_m=0.10, roughness_m=1.5e-5, n_segments=120),
        ThermalConfig(enabled=True, surface_temperature_C=32.0, geothermal_gradient_C_per_m=0.03),
    )

    prof = calc.profile_from_thp_and_rate(thp_bar=60.0, q_sm3_day=100000.0, wellhead_temperature_C=40.0)
    print(prof.tail())
    print(calc.bhp_from_thp_and_rate(60.0, q_sm3_day=100000.0, wellhead_temperature_C=40.0))

Authoring note
--------------
Written for Vasilii Anisimov's OPM Flow CO2STORE/THERMAL restart-coupling workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

try:  # optional at import time, required at runtime
    from CoolProp.CoolProp import PropsSI  # type: ignore
except Exception:  # pragma: no cover
    PropsSI = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


G = 9.80665
BAR_TO_PA = 1.0e5
PA_TO_BAR = 1.0e-5
DAY_TO_S = 86400.0


# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------


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
    """Steady 1D heat exchange with geothermal surroundings."""

    enabled: bool = True
    overall_U_W_m2K: float = 4.0
    heat_transfer_diameter_m: Optional[float] = None
    surface_temperature_C: float = 32.0
    geothermal_gradient_C_per_m: float = 0.03
    relaxation_length_m: float = 700.0
    include_joule_thomson: bool = True
    max_abs_dT_per_segment_C: float = 10.0
    max_abs_jt_K_per_Pa: float = 2.0e-5

    def geothermal_temperature_C(self, depth_m: float) -> float:
        return float(self.surface_temperature_C + self.geothermal_gradient_C_per_m * max(depth_m, 0.0))


@dataclass
class DriftFluxConfig:
    """ManyWells-inspired reduced two-phase closure.

    This is activated only when CoolProp reports a saturated CO2 quality 0 < Q < 1.
    For most deep CO2 storage injection cases the state should be dense/supercritical
    and this block stays inactive.
    """

    enabled: bool = True
    C0: float = 1.20
    drift_velocity_coeff: float = 0.35
    surface_tension_N_m: float = 0.01
    min_gas_holdup: float = 1.0e-5
    max_gas_holdup: float = 0.99999
    friction_density: str = "homogeneous"  # "homogeneous" or "hydrostatic"

@dataclass
class SolverConfig:
    thp_bounds_bar: Tuple[float, float] = (1.0, 1000.0)
    q_bounds_sm3_day: Tuple[float, float] = (0.0, 1.0e7)
    tol_pressure_bar: float = 1.0e-4
    tol_rate_sm3_day: float = 1.0
    max_iter: int = 80


# -----------------------------------------------------------------------------
# CoolProp property backend
# -----------------------------------------------------------------------------


class CoolPropCO2:
    """Thin, safe wrapper around CoolProp for pure CO2.

    Public units:
        P in Pa internally for property calls.
        T in K internally.
        returned density kg/m3, viscosity Pa*s, cp J/kg/K, enthalpy J/kg.
    """

    fluid: str = "CO2"

    def _require_coolprop(self) -> None:
        if PropsSI is None:
            raise ImportError(
                "CoolProp is required for this calculator. Install it with: pip install CoolProp"
            )

    def props(self, P_Pa: float, T_K: float) -> Dict[str, float | str]:
        self._require_coolprop()
        if P_Pa <= 0.0 or T_K <= 0.0:
            raise ValueError("Pressure and temperature must be positive.")

        # CoolProp may complain exactly on saturation if using P,T. For wellbore integration,
        # slight off-saturation states are normal; saturated states are handled separately.
        rho = float(PropsSI("D", "P", P_Pa, "T", T_K, self.fluid))
        mu = float(PropsSI("V", "P", P_Pa, "T", T_K, self.fluid))
        cp = float(PropsSI("Cpmass", "P", P_Pa, "T", T_K, self.fluid))
        h = float(PropsSI("Hmass", "P", P_Pa, "T", T_K, self.fluid))
        phase_i = int(PropsSI("Phase", "P", P_Pa, "T", T_K, self.fluid))

        q = self.quality(P_Pa, T_K)
        return {
            "rho_kg_m3": rho,
            "viscosity_Pa_s": mu,
            "cp_J_kgK": cp,
            "enthalpy_J_kg": h,
            "quality_mass": q,
            "phase_index": float(phase_i),
            "phase_label": self.phase_label(P_Pa, T_K, phase_i, q),
        }

    def quality(self, P_Pa: float, T_K: float) -> float:
        self._require_coolprop()
        try:
            q = float(PropsSI("Q", "P", P_Pa, "T", T_K, self.fluid))
            return q if math.isfinite(q) else float("nan")
        except Exception:
            return float("nan")

    def saturated_props(self, P_Pa: float, quality_mass: float) -> Dict[str, float]:
        self._require_coolprop()
        q = min(max(float(quality_mass), 0.0), 1.0)
        return {
            "T_K": float(PropsSI("T", "P", P_Pa, "Q", q, self.fluid)),
            "rho_kg_m3": float(PropsSI("D", "P", P_Pa, "Q", q, self.fluid)),
            "viscosity_Pa_s": float(PropsSI("V", "P", P_Pa, "Q", q, self.fluid)),
            "enthalpy_J_kg": float(PropsSI("Hmass", "P", P_Pa, "Q", q, self.fluid)),
        }

    def density(self, P_Pa: float, T_K: float) -> float:
        self._require_coolprop()
        return float(PropsSI("D", "P", P_Pa, "T", T_K, self.fluid))

    def enthalpy(self, P_Pa: float, T_K: float) -> float:
        self._require_coolprop()
        return float(PropsSI("Hmass", "P", P_Pa, "T", T_K, self.fluid))

    def temperature_from_ph(self, P_Pa: float, h_J_kg: float) -> float:
        self._require_coolprop()
        return float(PropsSI("T", "P", P_Pa, "Hmass", h_J_kg, self.fluid))

    def enthalpy_from_boundary(
        self,
        P_Pa: float,
        T_K: Optional[float] = None,
        h_J_kg: Optional[float] = None,
        quality_mass: Optional[float] = None,
    ) -> float:
        """Build a physically defined inlet enthalpy.

        Priority:
        1) h_J_kg directly: best for post-choke isenthalpic boundary.
        2) quality_mass at given P: saturated liquid/vapor mixture.
        3) P,T single-phase state.

        Important: for pure CO2, P,T exactly on saturation line is ambiguous.
        In that case user must provide h_J_kg or quality_mass.
        """
        self._require_coolprop()

        if P_Pa <= 0.0:
            raise ValueError("Boundary pressure must be positive.")

        if h_J_kg is not None:
            h = float(h_J_kg)
            if not math.isfinite(h):
                raise ValueError("wellhead_enthalpy_J_kg must be finite.")
            return h

        if quality_mass is not None:
            q = min(max(float(quality_mass), 0.0), 1.0)
            return float(PropsSI("Hmass", "P", P_Pa, "Q", q, self.fluid))

        if T_K is None:
            raise ValueError("Provide T_K, h_J_kg, or quality_mass for inlet state.")

        if T_K <= 0.0:
            raise ValueError("Boundary temperature must be positive.")

        # Do not silently guess quality at saturation.
        try:
            psat = float(PropsSI("P", "T", T_K, "Q", 0.0, self.fluid))
            if math.isfinite(psat) and psat > 0.0:
                rel = abs(P_Pa - psat) / psat
                if rel < 5.0e-4:
                    raise ValueError(
                        "Boundary P,T lies on/near CO2 saturation line. "
                        "For pure CO2 this does not define quality. "
                        "Provide wellhead_enthalpy_J_kg or wellhead_quality_mass."
                    )
        except ValueError:
            raise
        except Exception:
            pass

        return float(PropsSI("Hmass", "P", P_Pa, "T", T_K, self.fluid))

    def props_ph(self, P_Pa: float, h_J_kg: float) -> Dict[str, float | str]:
        """CoolProp flash using P,Hmass.

        This is much stronger than P,T near two-phase conditions because
        P,H defines quality in saturated two-phase pure CO2.
        """
        self._require_coolprop()
        if P_Pa <= 0.0:
            raise ValueError("Pressure must be positive.")
        if not math.isfinite(float(h_J_kg)):
            raise ValueError("Enthalpy must be finite.")

        # CoolProp returns Q in [0,1] for two-phase, often -1 outside.
        try:
            q_raw = float(PropsSI("Q", "P", P_Pa, "Hmass", h_J_kg, self.fluid))
        except Exception:
            q_raw = float("nan")

        if math.isfinite(q_raw) and 0.0 <= q_raw <= 1.0:
            liquid = self.saturated_props(P_Pa, 0.0)
            vapor = self.saturated_props(P_Pa, 1.0)
            h_l = float(liquid["enthalpy_J_kg"])
            h_v = float(vapor["enthalpy_J_kg"])
            denom = max(h_v - h_l, 1.0e-12)
            q = min(max((float(h_J_kg) - h_l) / denom, 0.0), 1.0)

            T_K = float(liquid["T_K"])
            rho_l = max(float(liquid["rho_kg_m3"]), 1.0e-12)
            rho_v = max(float(vapor["rho_kg_m3"]), 1.0e-12)
            rho_hom = 1.0 / max(q / rho_v + (1.0 - q) / rho_l, 1.0e-30)

            return {
                "temperature_K": T_K,
                "rho_kg_m3": float(rho_hom),
                "viscosity_Pa_s": float("nan"),
                "cp_J_kgK": float("nan"),
                "enthalpy_J_kg": float(h_J_kg),
                "quality_mass": float(q),
                "phase_index": float("nan"),
                "phase_label": "two_phase_saturated",
            }

        T_K = float(PropsSI("T", "P", P_Pa, "Hmass", h_J_kg, self.fluid))
        rho = float(PropsSI("D", "P", P_Pa, "Hmass", h_J_kg, self.fluid))
        mu = float(PropsSI("V", "P", P_Pa, "Hmass", h_J_kg, self.fluid))
        cp = float(PropsSI("Cpmass", "P", P_Pa, "Hmass", h_J_kg, self.fluid))

        try:
            phase_i = int(PropsSI("Phase", "P", P_Pa, "Hmass", h_J_kg, self.fluid))
        except Exception:
            phase_i = -999

        return {
            "temperature_K": T_K,
            "rho_kg_m3": rho,
            "viscosity_Pa_s": mu,
            "cp_J_kgK": cp,
            "enthalpy_J_kg": float(h_J_kg),
            "quality_mass": float("nan"),
            "phase_index": float(phase_i),
            "phase_label": self.phase_label(P_Pa, T_K, phase_i, float("nan")),
        }

    def joule_thomson_K_per_Pa(self, P_Pa: float, T_K: float, max_abs: float = 2.0e-5) -> float:
        """Finite-difference JT coefficient mu_JT = (dT/dP)_h.

        CoolProp derivative-string availability varies by version; this robust method
        uses a small pressure perturbation at constant enthalpy instead.
        """
        self._require_coolprop()
        h0 = self.enthalpy(P_Pa, T_K)
        dP = max(1000.0, 1.0e-4 * P_Pa)  # Pa
        try:
            T2 = self.temperature_from_ph(P_Pa + dP, h0)
            mu = (T2 - T_K) / dP
        except Exception:
            # Fallback to central difference if upper perturbation fails.
            T_lo = self.temperature_from_ph(max(P_Pa - dP, 1.0e3), h0)
            T_hi = self.temperature_from_ph(P_Pa + dP, h0)
            mu = (T_hi - T_lo) / (2.0 * dP)
        return float(min(max(mu, -max_abs), max_abs))

    def phase_label(self, P_Pa: float, T_K: float, phase_index: Optional[int] = None, q: Optional[float] = None) -> str:
        self._require_coolprop()
        # Stable human-readable labels without depending on CoolProp's enum names.
        qv = self.quality(P_Pa, T_K) if q is None else q
        try:
            Tcrit = float(PropsSI("Tcrit", self.fluid))
            Pcrit = float(PropsSI("pcrit", self.fluid))
        except Exception:
            Tcrit, Pcrit = 304.1282, 7.3773e6
        if qv is not None and math.isfinite(float(qv)) and 0.0 <= float(qv) <= 1.0:
            if 0.0 < float(qv) < 1.0:
                return "two_phase_saturated"
            if abs(float(qv)) < 1e-12:
                return "saturated_liquid"
            if abs(float(qv) - 1.0) < 1e-12:
                return "saturated_vapor"
        if T_K >= Tcrit and P_Pa >= Pcrit:
            return "supercritical_dense"
        if T_K >= Tcrit and P_Pa < Pcrit:
            return "superheated_gas"
        if T_K < Tcrit and P_Pa >= Pcrit:
            return "compressed_liquid_or_dense"
        return "single_phase"


# -----------------------------------------------------------------------------
# Utility correlations
# -----------------------------------------------------------------------------


def darcy_friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    """Darcy friction factor using laminar 64/Re or Haaland turbulent approximation."""
    Re = max(float(reynolds), 1.0e-12)
    if Re < 2300.0:
        return 64.0 / Re
    epsD = max(float(roughness_m) / max(float(diameter_m), 1.0e-12), 0.0)
    return float((-1.8 * math.log10((epsD / 3.7) ** 1.11 + 6.9 / Re)) ** -2)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


# -----------------------------------------------------------------------------
# Main calculator
# -----------------------------------------------------------------------------


class CO2CoolPropManyWellsCalculator:
    """CO2 injection wellbore calculator with CoolProp properties and ManyWells-style closures."""

    def __init__(
        self,
        geometry: Optional[WellboreGeometry] = None,
        thermal: Optional[ThermalConfig] = None,
        drift_flux: Optional[DriftFluxConfig] = None,
        solver: Optional[SolverConfig] = None,
        properties: Optional[CoolPropCO2] = None,
    ) -> None:
        self.geometry = geometry or WellboreGeometry()
        self.geometry.validate()
        self.thermal = thermal or ThermalConfig()
        self.drift_flux = drift_flux or DriftFluxConfig()
        self.solver = solver or SolverConfig()
        self.props = properties or CoolPropCO2()

    # ------------------------- Rate conversion -------------------------
    def standard_density_kg_m3(self) -> float:
        P = self.geometry.std_pressure_bar * BAR_TO_PA
        T = self.geometry.std_temperature_C + 273.15
        return self.props.density(P, T)

    def mass_rate_from_sm3_day(self, q_sm3_day: float) -> float:
        return float(q_sm3_day) * self.standard_density_kg_m3() / DAY_TO_S

    def sm3_day_from_mass_rate(self, mass_rate_kg_s: float) -> float:
        return float(mass_rate_kg_s) * DAY_TO_S / self.standard_density_kg_m3()

    # ------------------------- Phase/holdup model -------------------------
    def _flow_props(self, P_Pa: float, T_K: float, mass_rate_kg_s: float, area_m2: float) -> Dict[str, float | str]:
        base = self.props.props(P_Pa, T_K)
        q_mass = float(base.get("quality_mass", float("nan")))

        # Default: single-phase/supercritical.
        if not self.drift_flux.enabled or not math.isfinite(q_mass) or not (0.0 < q_mass < 1.0):
            rho = float(base["rho_kg_m3"])
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
                "viscosity_mix_Pa_s": float(base["viscosity_Pa_s"]),
            }

        # Saturated two-phase closure, ManyWells-style drift-flux.
        vapor = self.props.saturated_props(P_Pa, 1.0)
        liquid = self.props.saturated_props(P_Pa, 0.0)
        rho_v = max(float(vapor["rho_kg_m3"]), 1.0e-12)
        rho_l = max(float(liquid["rho_kg_m3"]), 1.0e-12)
        mu_v = max(float(vapor["viscosity_Pa_s"]), 1.0e-12)
        mu_l = max(float(liquid["viscosity_Pa_s"]), 1.0e-12)
        x = clamp(q_mass, 0.0, 1.0)

        m_g = x * mass_rate_kg_s
        m_l = (1.0 - x) * mass_rate_kg_s
        Qg = m_g / rho_v
        Ql = m_l / rho_l
        Jg = Qg / max(area_m2, 1.0e-30)
        Jl = Ql / max(area_m2, 1.0e-30)
        J = Jg + Jl

        # Drift-flux alpha = Jg / (C0 J + Vgj). This is a vertical-well reduced closure.
        delta_rho = max(rho_l - rho_v, 0.0)
        Vgj = self.drift_flux.drift_velocity_coeff * (
            (G * self.drift_flux.surface_tension_N_m * delta_rho) / max(rho_l**2, 1.0e-30)
        ) ** 0.25
        alpha_g = Jg / max(self.drift_flux.C0 * J + Vgj, 1.0e-30)
        alpha_g = clamp(alpha_g, self.drift_flux.min_gas_holdup, self.drift_flux.max_gas_holdup)
        alpha_l = 1.0 - alpha_g

        rho_hydro = alpha_g * rho_v + alpha_l * rho_l
        rho_hom = 1.0 / max(x / rho_v + (1.0 - x) / rho_l, 1.0e-30)
        rho_friction = rho_hom if self.drift_flux.friction_density == "homogeneous" else rho_hydro
        # Simple log/volume-like viscosity mixing. This is not a substitute for validation.
        mu_mix = max(mu_v ** alpha_g * mu_l ** alpha_l, 1.0e-8)

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

    def _flow_props_ph(
        self,
        P_Pa: float,
        h_J_kg: float,
        mass_rate_kg_s: float,
        area_m2: float,
    ) -> Dict[str, float | str]:
        """Flow properties from P-H flash.

        This is the preferred path for post-choke/two-phase CO2.
        """
        base = self.props.props_ph(P_Pa, h_J_kg)
        q_mass = float(base.get("quality_mass", float("nan")))

        if not self.drift_flux.enabled or not math.isfinite(q_mass) or not (0.0 < q_mass < 1.0):
            rho = float(base["rho_kg_m3"])
            mu = float(base["viscosity_Pa_s"])
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

        vapor = self.props.saturated_props(P_Pa, 1.0)
        liquid = self.props.saturated_props(P_Pa, 0.0)

        rho_v = max(float(vapor["rho_kg_m3"]), 1.0e-12)
        rho_l = max(float(liquid["rho_kg_m3"]), 1.0e-12)
        mu_v = max(float(vapor["viscosity_Pa_s"]), 1.0e-12)
        mu_l = max(float(liquid["viscosity_Pa_s"]), 1.0e-12)
        x = clamp(q_mass, 0.0, 1.0)

        m_g = x * mass_rate_kg_s
        m_l = (1.0 - x) * mass_rate_kg_s
        Qg = m_g / rho_v
        Ql = m_l / rho_l
        Jg = Qg / max(area_m2, 1.0e-30)
        Jl = Ql / max(area_m2, 1.0e-30)
        J = Jg + Jl

        delta_rho = max(rho_l - rho_v, 0.0)
        Vgj = self.drift_flux.drift_velocity_coeff * (
            (G * self.drift_flux.surface_tension_N_m * delta_rho) / max(rho_l**2, 1.0e-30)
        ) ** 0.25

        alpha_g = Jg / max(self.drift_flux.C0 * J + Vgj, 1.0e-30)
        alpha_g = clamp(alpha_g, self.drift_flux.min_gas_holdup, self.drift_flux.max_gas_holdup)
        alpha_l = 1.0 - alpha_g

        rho_hydro = alpha_g * rho_v + alpha_l * rho_l
        rho_hom = 1.0 / max(x / rho_v + (1.0 - x) / rho_l, 1.0e-30)
        rho_friction = rho_hom if self.drift_flux.friction_density == "homogeneous" else rho_hydro
        mu_mix = max(mu_v ** alpha_g * mu_l ** alpha_l, 1.0e-8)

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

    # ------------------------- Wellbore integration -------------------------
    def profile_from_thp_and_rate(
        self,
        thp_bar: float,
        q_sm3_day: Optional[float] = None,
        mass_rate_kg_s: Optional[float] = None,
        wellhead_temperature_C: float = 40.0,
        wellhead_enthalpy_J_kg: Optional[float] = None,
        wellhead_quality_mass: Optional[float] = None,
        return_dataframe: bool = True,
    ):
        """Integrate downward using P-H flash.

        Sign convention:
            positive injection rate flows downward.

        Momentum:
            dP_down = hydrostatic - friction + acceleration_term

        Energy:
            enthalpy is updated by heat exchange.
            Joule-Thomson cooling/heating appears naturally through P-H flash.
        """
        if mass_rate_kg_s is None:
            if q_sm3_day is None:
                raise ValueError("Provide q_sm3_day or mass_rate_kg_s.")
            mass_rate_kg_s = self.mass_rate_from_sm3_day(q_sm3_day)

        mass_rate_kg_s = float(mass_rate_kg_s)
        if mass_rate_kg_s < 0.0:
            raise ValueError("Use positive injection rate magnitude.")

        cfg = self.geometry
        hcfg = self.thermal
        area = math.pi * cfg.diameter_m**2 / 4.0
        dz = cfg.tvd_m / cfg.n_segments
        heat_D = cfg.diameter_m if hcfg.heat_transfer_diameter_m is None else hcfg.heat_transfer_diameter_m

        P = float(thp_bar) * BAR_TO_PA
        T0_K = float(wellhead_temperature_C) + 273.15

        h = self.props.enthalpy_from_boundary(
            P_Pa=P,
            T_K=T0_K,
            h_J_kg=wellhead_enthalpy_J_kg,
            quality_mass=wellhead_quality_mass,
        )

        rows: List[Dict[str, float | str]] = []

        cum_hydro = 0.0
        cum_friction = 0.0     # signed: friction is negative for downward injection
        cum_accel = 0.0
        cum_heat_J_kg = 0.0

        last_velocity: Optional[float] = None

        for i in range(cfg.n_segments + 1):
            depth = min(i * dz, cfg.tvd_m)

            fp = self._flow_props_ph(P, h, mass_rate_kg_s, area)

            T_K = float(fp["temperature_K"])
            rho_flow = max(float(fp["rho_flow_kg_m3"]), 1.0e-12)
            rho_h = max(float(fp["rho_hydro_kg_m3"]), 1.0e-12)
            rho_f = max(float(fp["rho_friction_kg_m3"]), 1.0e-12)
            mu = max(float(fp["viscosity_mix_Pa_s"]), 1.0e-12)

            cp_raw = float(fp.get("cp_J_kgK", float("nan")))
            cp_eff = cp_raw if math.isfinite(cp_raw) and cp_raw > 1.0 else 2000.0

            q_actual_m3_s = mass_rate_kg_s / rho_flow
            velocity = q_actual_m3_s / max(area, 1.0e-30)
            Re = rho_f * abs(velocity) * cfg.diameter_m / mu
            f = darcy_friction_factor(max(Re, cfg.min_reynolds), cfg.roughness_m, cfg.diameter_m)

            rows.append({
                "segment": float(i),
                "depth_m": float(depth),
                "pressure_bar": float(P * PA_TO_BAR),
                "temperature_C": float(T_K - 273.15),
                "geothermal_temperature_C": float(hcfg.geothermal_temperature_C(depth)),
                "phase_label": str(fp["phase_label"]),
                "quality_mass": float(fp["quality_mass"]),
                "two_phase_active": float(fp["two_phase_active"]),
                "density_kg_m3": rho_flow,
                "rho_hydro_kg_m3": rho_h,
                "rho_friction_kg_m3": rho_f,
                "rho_vapor_kg_m3": float(fp["rho_vapor_kg_m3"]),
                "rho_liquid_kg_m3": float(fp["rho_liquid_kg_m3"]),
                "gas_holdup": float(fp["gas_holdup"]),
                "liquid_holdup": float(fp["liquid_holdup"]),
                "viscosity_Pa_s": mu,
                "cp_J_kgK": cp_eff,
                "enthalpy_J_kg": float(h),
                "mass_rate_kg_s": mass_rate_kg_s,
                "std_rate_sm3_day": self.sm3_day_from_mass_rate(mass_rate_kg_s),
                "actual_rate_m3_s": q_actual_m3_s,
                "velocity_m_s": velocity,
                "reynolds": Re,
                "friction_factor": f,
                "cum_dP_hydro_bar": cum_hydro * PA_TO_BAR,
                "cum_dP_friction_bar": cum_friction * PA_TO_BAR,
                "cum_dP_acceleration_bar": cum_accel * PA_TO_BAR,
                "cum_dP_total_bar": (cum_hydro + cum_friction + cum_accel) * PA_TO_BAR,
                "cum_heat_J_kg": cum_heat_J_kg,
                "energy_formulation": "P-H flash",
            })

            if i == cfg.n_segments:
                break

            # Pressure terms.
            dP_h = rho_h * G * dz
            dP_f_positive_loss = f * (dz / cfg.diameter_m) * 0.5 * rho_f * velocity**2

            # Steady acceleration term: pressure decreases if velocity increases downward.
            dP_acc = 0.0
            if last_velocity is not None:
                dP_acc = -rho_flow * velocity * (velocity - last_velocity)
            last_velocity = velocity

            dP_total = dP_h - dP_f_positive_loss + dP_acc

            # Heat exchange updates enthalpy, not temperature directly.
            if hcfg.enabled and mass_rate_kg_s > 0.0:
                depth_mid = min(depth + 0.5 * dz, cfg.tvd_m)
                T_amb_K = hcfg.geothermal_temperature_C(depth_mid) + 273.15
                dh_heat = (
                    hcfg.overall_U_W_m2K
                    * math.pi
                    * heat_D
                    * dz
                    * (T_amb_K - T_K)
                    / max(mass_rate_kg_s, 1.0e-30)
                )
            else:
                dh_heat = 0.0

            # Numerical limiter. In two-phase this is an enthalpy limiter, not a dT law.
            max_abs_dh = max(abs(hcfg.max_abs_dT_per_segment_C) * cp_eff, 5.0e4)
            dh_heat = clamp(dh_heat, -max_abs_dh, max_abs_dh)

            P = max(P + dP_total, 1.0e3)
            h = h + dh_heat

            cum_hydro += dP_h
            cum_friction += -dP_f_positive_loss
            cum_accel += dP_acc
            cum_heat_J_kg += dh_heat

        if return_dataframe:
            if pd is None:
                raise ImportError("pandas is required for return_dataframe=True")
            return pd.DataFrame(rows)
        return rows

    def bhp_from_thp_and_rate(
        self,
        thp_bar: float,
        q_sm3_day: Optional[float] = None,
        mass_rate_kg_s: Optional[float] = None,
        wellhead_enthalpy_J_kg: Optional[float] = None,
        wellhead_quality_mass: Optional[float] = None,
        wellhead_temperature_C: float = 40.0,
    ) -> float:
        rows = self.profile_from_thp_and_rate(
            thp_bar=thp_bar,
            q_sm3_day=q_sm3_day,
            mass_rate_kg_s=mass_rate_kg_s,
            wellhead_temperature_C=wellhead_temperature_C,
            wellhead_enthalpy_J_kg=wellhead_enthalpy_J_kg,
            wellhead_quality_mass=wellhead_quality_mass,
            return_dataframe=False,
        )
        return float(rows[-1]["pressure_bar"])

    def bht_from_thp_and_rate(
        self,
        thp_bar: float,
        q_sm3_day: Optional[float] = None,
        mass_rate_kg_s: Optional[float] = None,
        wellhead_enthalpy_J_kg: Optional[float] = None,
        wellhead_quality_mass: Optional[float] = None,
        wellhead_temperature_C: float = 40.0,
    ) -> float:
        rows = self.profile_from_thp_and_rate(
            thp_bar=thp_bar,
            q_sm3_day=q_sm3_day,
            mass_rate_kg_s=mass_rate_kg_s,
            wellhead_temperature_C=wellhead_temperature_C,
            wellhead_enthalpy_J_kg=wellhead_enthalpy_J_kg,
            wellhead_quality_mass=wellhead_quality_mass,
            return_dataframe=False,
        )
        return float(rows[-1]["temperature_C"])

    # ------------------------- Solvers for coupling -------------------------
    def required_thp_for_target_bhp(
        self,
        target_bhp_bar: float,
        q_sm3_day: Optional[float] = None,
        mass_rate_kg_s: Optional[float] = None,
        wellhead_enthalpy_J_kg: Optional[float] = None,
        wellhead_quality_mass: Optional[float] = None,
        wellhead_temperature_C: float = 40.0,
        thp_bounds_bar: Optional[Tuple[float, float]] = None,
        tol_bar: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> float:
        """Invert wellbore model: find THP required to reproduce a target BHP at fixed rate."""
        lo, hi = thp_bounds_bar or self.solver.thp_bounds_bar
        tol = self.solver.tol_pressure_bar if tol_bar is None else float(tol_bar)
        nmax = self.solver.max_iter if max_iter is None else int(max_iter)

        def residual(thp: float) -> float:
            return self.bhp_from_thp_and_rate(
                thp,
                q_sm3_day=q_sm3_day,
                mass_rate_kg_s=mass_rate_kg_s,
                wellhead_temperature_C=wellhead_temperature_C,
                wellhead_enthalpy_J_kg=wellhead_enthalpy_J_kg,
                wellhead_quality_mass=wellhead_quality_mass,
            ) - float(target_bhp_bar)

        f_lo = residual(lo)
        f_hi = residual(hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(
                f"Target BHP={target_bhp_bar} bar is not bracketed by THP bounds {lo:g}-{hi:g} bar. "
                f"Residuals: {f_lo:.6g}, {f_hi:.6g} bar."
            )

        for _ in range(nmax):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if abs(f_mid) <= tol:
                return float(mid)
            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return float(0.5 * (lo + hi))

    def rate_from_thp_and_injectivity(
        self,
        thp_bar: float,
        reservoir_pressure_bar: float,
        injectivity_sm3_day_per_bar: float,
        wellhead_temperature_C: float = 40.0,
        q_bounds_sm3_day: Optional[Tuple[float, float]] = None,
        tol_sm3_day: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Solve q = J * max(BHP(thp, q) - p_res, 0) for injection.

        Returns:
            q_sm3_day, bhp_bar, residual_sm3_day

        This is the simplest pressure-controlled closure for replacing pure RATE-control.
        It can be used as an external controller around OPM restart RATE runs.
        """
        J = float(injectivity_sm3_day_per_bar)
        if J <= 0.0:
            raise ValueError("injectivity_sm3_day_per_bar must be positive.")
        lo, hi = q_bounds_sm3_day or self.solver.q_bounds_sm3_day
        tol = self.solver.tol_rate_sm3_day if tol_sm3_day is None else float(tol_sm3_day)
        nmax = self.solver.max_iter if max_iter is None else int(max_iter)

        def residual(q: float) -> float:
            bhp = self.bhp_from_thp_and_rate(thp_bar, q_sm3_day=q, wellhead_temperature_C=wellhead_temperature_C)
            return q - J * max(bhp - float(reservoir_pressure_bar), 0.0)

        r_lo = residual(lo)
        r_hi = residual(hi)
        if r_lo * r_hi > 0.0:
            # Robust endpoint fallback; also useful if there is no injection at this THP.
            q_best = lo if abs(r_lo) <= abs(r_hi) else hi
            bhp_best = self.bhp_from_thp_and_rate(thp_bar, q_sm3_day=q_best, wellhead_temperature_C=wellhead_temperature_C)
            return float(q_best), float(bhp_best), float(residual(q_best))

        for _ in range(nmax):
            mid = 0.5 * (lo + hi)
            r_mid = residual(mid)
            if abs(r_mid) <= tol:
                bhp = self.bhp_from_thp_and_rate(thp_bar, q_sm3_day=mid, wellhead_temperature_C=wellhead_temperature_C)
                return float(mid), float(bhp), float(r_mid)
            if r_lo * r_mid <= 0.0:
                hi, r_hi = mid, r_mid
            else:
                lo, r_lo = mid, r_mid

        q = 0.5 * (lo + hi)
        bhp = self.bhp_from_thp_and_rate(thp_bar, q_sm3_day=q, wellhead_temperature_C=wellhead_temperature_C)
        return float(q), float(bhp), float(residual(q))

    def rate_limited_by_thp(
        self,
        thp_limit_bar: float,
        reservoir_bhp_for_rate_callable,
        q_bounds_sm3_day: Optional[Tuple[float, float]] = None,
        wellhead_temperature_C: float = 40.0,
        tol_thp_bar: float = 0.2,
        max_iter: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """External OPM-coupling helper: choose q so required THP equals a limit.

        reservoir_bhp_for_rate_callable(q) must run or approximate the reservoir and return WBHP(q) in bar.
        This is the key algorithmic replacement for pure RATE-control.

        Returns:
            q_sm3_day, required_thp_bar, pressure_residual_bar
        """
        lo, hi = q_bounds_sm3_day or self.solver.q_bounds_sm3_day
        nmax = self.solver.max_iter if max_iter is None else int(max_iter)

        def thp_residual(q: float) -> Tuple[float, float]:
            bhp = float(reservoir_bhp_for_rate_callable(float(q)))
            thp_req = self.required_thp_for_target_bhp(
                bhp, q_sm3_day=float(q), wellhead_temperature_C=wellhead_temperature_C
            )
            return thp_req - float(thp_limit_bar), thp_req

        r_lo, thp_lo = thp_residual(lo)
        r_hi, thp_hi = thp_residual(hi)
        if r_lo * r_hi > 0.0:
            # If both are below limit, hi is feasible. If both are above limit, lo is least-bad.
            if r_hi <= 0.0:
                return float(hi), float(thp_hi), float(r_hi)
            return float(lo), float(thp_lo), float(r_lo)

        last_thp = float("nan")
        last_r = float("nan")
        for _ in range(nmax):
            mid = 0.5 * (lo + hi)
            r_mid, thp_mid = thp_residual(mid)
            last_thp, last_r = thp_mid, r_mid
            if abs(r_mid) <= tol_thp_bar:
                return float(mid), float(thp_mid), float(r_mid)
            if r_lo * r_mid <= 0.0:
                hi, r_hi = mid, r_mid
            else:
                lo, r_lo = mid, r_mid
        return float(0.5 * (lo + hi)), float(last_thp), float(last_r)

    # ------------------------- OPM helper -------------------------
    def make_vfpinj_table(
        self,
        table_id: int,
        rates_sm3_day: Sequence[float],
        thp_values_bar: Sequence[float],
        wellhead_temperature_C: float = 40.0,
        reference_depth_m: Optional[float] = None,
        flo_phase: str = "GAS",
        decimals: int = 4,
    ) -> str:
        """Generate an ECL/OPM-style VFPINJ table from this calculator.

        Check exact VFPINJ syntax compatibility with your OPM Flow version before production use.
        """
        ref = self.geometry.tvd_m if reference_depth_m is None else float(reference_depth_m)
        rates = [float(x) for x in rates_sm3_day]
        thps = [float(x) for x in thp_values_bar]
        if any(r <= 0.0 for r in rates):
            raise ValueError("rates_sm3_day must contain positive values.")
        if any(t <= 0.0 for t in thps):
            raise ValueError("thp_values_bar must contain positive values.")
        if rates != sorted(rates):
            raise ValueError("rates_sm3_day must be monotonically increasing.")
        if thps != sorted(thps):
            raise ValueError("thp_values_bar must be monotonically increasing.")

        fmt = f"{{:.{decimals}f}}"
        lines: List[str] = []
        lines.append("VFPINJ")
        lines.append(f"  {int(table_id)}  {fmt.format(ref)}  {flo_phase.upper()}  THP  1*  BHP /")
        lines.append("  " + "  ".join(fmt.format(r) for r in rates) + " /")
        lines.append("  " + "  ".join(fmt.format(t) for t in thps) + " /")
        for thp in thps:
            bhps = [
                self.bhp_from_thp_and_rate(thp, q_sm3_day=q, wellhead_temperature_C=wellhead_temperature_C)
                for q in rates
            ]
            lines.append("  " + "  ".join(fmt.format(b) for b in bhps) + " /")
        lines.append("/")
        return "\n".join(lines)

    def config_dict(self) -> Dict[str, object]:
        return {
            "geometry": asdict(self.geometry),
            "thermal": asdict(self.thermal),
            "drift_flux": asdict(self.drift_flux),
            "solver": asdict(self.solver),
            "property_backend": "CoolProp::CO2",
        }


if __name__ == "__main__":
    # Base.DATA-like smoke example. Requires: pip install CoolProp pandas
    calc = CO2CoolPropManyWellsCalculator(
        geometry=WellboreGeometry(tvd_m=1600.0, diameter_m=0.10, roughness_m=1.5e-5, n_segments=120),
        thermal=ThermalConfig(
            enabled=True,
            overall_U_W_m2K=4.0,
            surface_temperature_C=32.0,
            geothermal_gradient_C_per_m=0.03,
            include_joule_thomson=True,
        ),
        drift_flux=DriftFluxConfig(enabled=True),
    )
    prof = calc.profile_from_thp_and_rate(thp_bar=60.0, q_sm3_day=100000.0, wellhead_temperature_C=40.0)
    print(prof.tail())
    print(f"BHP = {prof['pressure_bar'].iloc[-1]:.3f} bar")
    print(f"BHT = {prof['temperature_C'].iloc[-1]:.3f} C")
