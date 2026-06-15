"""Surface choke / control-valve model for CO2 injection.

This module uses:
- fluids.control_valve for IEC-style liquid-like control-valve sizing/rating
- CoolProp for CO2 thermodynamic state and isenthalpic downstream flash

Important:
This first version is intended for dense/liquid-like upstream CO2.
If the upstream state is gas/two-phase, the liquid valve equation is not enough
and we should later add gas/two-phase valve models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

try:
    from fluids.control_valve import size_control_valve_l
except Exception:  # pragma: no cover
    size_control_valve_l = None  # type: ignore

try:
    from CoolProp.CoolProp import PropsSI
except Exception:  # pragma: no cover
    PropsSI = None  # type: ignore

from .constants import BAR_TO_PA, PA_TO_BAR
from .properties import CoolPropCO2


@dataclass
class ChokeConfig:
    """Configuration for a surface choke/control valve.

    kv_full_m3_h:
        Full-open valve coefficient Kv in m3/h.

    opening_fraction:
        Valve opening from 0 to 1. In this first version:
        Kv_effective = kv_full_m3_h * opening_fraction.

    Note:
        Kv is not the same as the dimensionless Cv used in some reservoir/OCD equations.
        Here we use the engineering valve coefficient expected by fluids.control_valve.
    """

    kv_full_m3_h: float = 110.0
    opening_fraction: float = 0.20

    FL: float = 0.90
    Fd: float = 1.00

    inlet_diameter_m: Optional[float] = None
    outlet_diameter_m: Optional[float] = None
    valve_diameter_m: Optional[float] = None

    min_downstream_pressure_bar: float = 1.0
    min_liquid_like_density_kg_m3: float = 200.0

    tol_kv: float = 1.0e-5
    max_iter: int = 80

    def validate(self) -> None:
        if self.kv_full_m3_h <= 0.0:
            raise ValueError("kv_full_m3_h must be positive.")
        if not (0.0 < self.opening_fraction <= 1.0):
            raise ValueError("opening_fraction must satisfy 0 < opening_fraction <= 1.")
        if self.FL <= 0.0:
            raise ValueError("FL must be positive.")
        if self.Fd <= 0.0:
            raise ValueError("Fd must be positive.")
        if self.min_downstream_pressure_bar <= 0.0:
            raise ValueError("min_downstream_pressure_bar must be positive.")
        if self.min_liquid_like_density_kg_m3 <= 0.0:
            raise ValueError("min_liquid_like_density_kg_m3 must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if self.tol_kv <= 0.0:
            raise ValueError("tol_kv must be positive.")

    @property
    def effective_kv_m3_h(self) -> float:
        return float(self.kv_full_m3_h * self.opening_fraction)


class CO2ChokeValve:
    """Dense/liquid-like CO2 choke model.

    The pressure drop is solved by inverting fluids.control_valve.size_control_valve_l:
        given P1, T1, mass rate and effective Kv -> find P2.

    Then the downstream temperature/quality is obtained from an isenthalpic P-H flash:
        h2 = h1
        T2 = T(P2, h1)
    """

    def __init__(
        self,
        properties: Optional[CoolPropCO2] = None,
        config: Optional[ChokeConfig] = None,
    ) -> None:
        self.props = properties or CoolPropCO2()
        self.config = config or ChokeConfig()
        self.config.validate()

    def _require_libraries(self) -> None:
        if size_control_valve_l is None:
            raise ImportError("fluids is required. Install with: pip install fluids")
        if PropsSI is None:
            raise ImportError("CoolProp is required. Install with: pip install CoolProp")

    def _co2_psat_or_zero(self, temperature_K: float) -> float:
        """Return CO2 saturation pressure if it exists at T, otherwise 0.

        For T >= Tcrit, saturation pressure is not defined.
        The liquid-valve model is most meaningful below Tcrit and above Psat.
        """
        self._require_libraries()

        tcrit = float(PropsSI("Tcrit", "CO2"))
        if temperature_K >= tcrit:
            return 0.0

        return float(PropsSI("P", "T", temperature_K, "Q", 0.0, "CO2"))

    def required_kv_liquid_like(
        self,
        upstream_pressure_bar: float,
        downstream_pressure_bar: float,
        upstream_temperature_C: float,
        mass_rate_kg_s: float,
    ) -> float:
        """Return required Kv [m3/h] for given P1/P2/T1/mass-rate.

        This uses fluids.control_valve.size_control_valve_l.
        """
        self._require_libraries()

        if upstream_pressure_bar <= downstream_pressure_bar:
            raise ValueError("upstream_pressure_bar must be greater than downstream_pressure_bar.")
        if mass_rate_kg_s <= 0.0:
            raise ValueError("mass_rate_kg_s must be positive.")

        p1 = float(upstream_pressure_bar) * BAR_TO_PA
        p2 = float(downstream_pressure_bar) * BAR_TO_PA
        t1 = float(upstream_temperature_C) + 273.15

        up = self.props.props(p1, t1)
        rho = float(up["rho_kg_m3"])
        mu = float(up["viscosity_Pa_s"])

        if rho < self.config.min_liquid_like_density_kg_m3:
            raise ValueError(
                f"Upstream CO2 density is {rho:.3f} kg/m3. "
                "This first choke model expects dense/liquid-like upstream CO2. "
                "For gas/two-phase upstream state we should add gas/two-phase valve logic."
            )

        q_actual_m3_s = float(mass_rate_kg_s) / rho
        psat = self._co2_psat_or_zero(t1)
        pc = float(PropsSI("Pcrit", "CO2"))

        kv_required = size_control_valve_l(
            rho=rho,
            Psat=psat,
            Pc=pc,
            mu=mu,
            P1=p1,
            P2=p2,
            Q=q_actual_m3_s,
            D1=self.config.inlet_diameter_m,
            D2=self.config.outlet_diameter_m,
            d=self.config.valve_diameter_m,
            FL=self.config.FL,
            Fd=self.config.Fd,
            allow_choked=True,
            allow_laminar=True,
            full_output=False,
        )

        return float(kv_required)

    def solve_downstream_pressure_bar(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        mass_rate_kg_s: float,
    ) -> float:
        """Find P2 so that required Kv equals effective valve Kv."""
        p1_bar = float(upstream_pressure_bar)
        if p1_bar <= self.config.min_downstream_pressure_bar:
            raise ValueError("upstream pressure must exceed min_downstream_pressure_bar.")

        kv_target = self.config.effective_kv_m3_h

        lo = float(self.config.min_downstream_pressure_bar)
        hi = p1_bar * 0.999999

        def f(p2_bar: float) -> float:
            return (
                self.required_kv_liquid_like(
                    upstream_pressure_bar=p1_bar,
                    downstream_pressure_bar=p2_bar,
                    upstream_temperature_C=upstream_temperature_C,
                    mass_rate_kg_s=mass_rate_kg_s,
                )
                - kv_target
            )

        f_lo = f(lo)
        f_hi = f(hi)

        if f_lo * f_hi > 0.0:
            raise ValueError(
                "Could not bracket downstream pressure for this choke. "
                f"Try changing kv_full_m3_h/opening_fraction. "
                f"f(min P2)={f_lo:.6g}, f(nearly P1)={f_hi:.6g}, "
                f"Kv_effective={kv_target:.6g} m3/h."
            )

        for _ in range(self.config.max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)

            if abs(f_mid) <= self.config.tol_kv:
                return float(mid)

            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return float(0.5 * (lo + hi))

    def downstream_state_at_pressure(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        downstream_pressure_bar: float,
        mass_rate_kg_s: float,
    ) -> Dict[str, float | str]:
        """Calculate post-choke state for a prescribed downstream pressure.

        This is useful when the choke/valve opening is known from another model
        or when we want to reproduce a target THP case.

        Physics:
            h2 = h1
            P2 is prescribed
            T2, quality and phase are obtained from CoolProp P-H flash.
        """
        self._require_libraries()

        if mass_rate_kg_s <= 0.0:
            raise ValueError("mass_rate_kg_s must be positive.")
        if upstream_pressure_bar <= downstream_pressure_bar:
            raise ValueError("upstream_pressure_bar must be greater than downstream_pressure_bar.")

        p1 = float(upstream_pressure_bar) * BAR_TO_PA
        t1 = float(upstream_temperature_C) + 273.15
        p2 = float(downstream_pressure_bar) * BAR_TO_PA

        h1 = self.props.enthalpy(p1, t1)
        down = self.props.props_ph(p2, h1)

        kv_required = self.required_kv_liquid_like(
            upstream_pressure_bar=upstream_pressure_bar,
            downstream_pressure_bar=downstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            mass_rate_kg_s=mass_rate_kg_s,
        )

        return {
            "upstream_pressure_bar": float(upstream_pressure_bar),
            "upstream_temperature_C": float(upstream_temperature_C),
            "upstream_enthalpy_J_kg": float(h1),
            "downstream_pressure_bar": float(downstream_pressure_bar),
            "downstream_temperature_C": float(down["temperature_K"]) - 273.15,
            "downstream_enthalpy_J_kg": float(h1),
            "quality_mass": float(down["quality_mass"]),
            "phase_label": str(down["phase_label"]),
            "kv_required_m3_h": float(kv_required),
            "effective_kv_m3_h": float(self.config.effective_kv_m3_h),
            "opening_fraction": float(self.config.opening_fraction),
            "mass_rate_kg_s": float(mass_rate_kg_s),
        }

    def downstream_state(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        mass_rate_kg_s: float,
    ) -> Dict[str, float | str]:
        """Calculate post-choke state.

        Returns pressure, temperature, enthalpy, quality and phase label.
        The downstream enthalpy equals upstream enthalpy.
        """
        self._require_libraries()

        p1 = float(upstream_pressure_bar) * BAR_TO_PA
        t1 = float(upstream_temperature_C) + 273.15
        h1 = self.props.enthalpy(p1, t1)

        p2_bar = self.solve_downstream_pressure_bar(
            upstream_pressure_bar=upstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            mass_rate_kg_s=mass_rate_kg_s,
        )
        p2 = p2_bar * BAR_TO_PA

        down = self.props.props_ph(p2, h1)

        return {
            "upstream_pressure_bar": float(upstream_pressure_bar),
            "upstream_temperature_C": float(upstream_temperature_C),
            "upstream_enthalpy_J_kg": float(h1),
            "downstream_pressure_bar": float(p2_bar),
            "downstream_temperature_C": float(down["temperature_K"]) - 273.15,
            "downstream_enthalpy_J_kg": float(h1),
            "quality_mass": float(down["quality_mass"]),
            "phase_label": str(down["phase_label"]),
            "effective_kv_m3_h": float(self.config.effective_kv_m3_h),
            "opening_fraction": float(self.config.opening_fraction),
            "mass_rate_kg_s": float(mass_rate_kg_s),
        }
