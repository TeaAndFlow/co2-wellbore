"""Homogeneous Equilibrium Model (HEM) choke backend for CO2.

This module implements a real-fluid HEM choke capacity model using CoolProp.

Purpose:
    Replace liquid-like IEC/Kv sizing for CO2 choke coupling.

Main idea:
    1. Upstream state is P1,T1.
    2. Capacity is computed from isentropic expansion to a throat/static pressure.
    3. Downstream thermodynamic state is still reported as isenthalpic P-H flash,
       which is appropriate for throttling/JT temperature after the valve.
    4. For compatibility with the existing 07 coupling script, the method
       required_kv_liquid_like(...) returns a HEM-equivalent required Kv.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, List

try:
    from CoolProp.CoolProp import PropsSI
except Exception:  # pragma: no cover
    PropsSI = None  # type: ignore

from .constants import BAR_TO_PA, PA_TO_BAR
from .properties import CoolPropCO2


@dataclass
class HEMChokeConfig:
    """Configuration for the HEM CO2 choke model.

    kv_full_m3_h:
        Reporting/equivalent full-open Kv scale.
        In HEM mode this does not define the physics directly.

    full_diameter_m:
        Physical full-open choke/valve diameter used for area.

    opening_fraction:
        Valve opening from 0 to 1.

    discharge_coefficient:
        Cd multiplying mass flux * area.

    area_exponent:
        Effective area model:
            A_eff = A_full * opening_fraction ** area_exponent
        Use 1.0 as first physical baseline.

    min_pressure_bar:
        Lower bound for HEM throat-pressure search.
        For pure CO2, keep this above the triple region. 6 bar is safer
        than allowing CoolProp to explore invalid ~4-5 bar P-H states.
    """

    kv_full_m3_h: float = 118.0
    full_diameter_m: float = 0.16
    opening_fraction: float = 0.20
    discharge_coefficient: float = 0.84
    area_exponent: float = 1.0
    min_pressure_bar: float = 6.0
    n_pressure_samples: int = 220
    pressure_eps_fraction: float = 1.0e-5

    def validate(self) -> None:
        if self.kv_full_m3_h <= 0.0:
            raise ValueError("kv_full_m3_h must be positive.")
        if self.full_diameter_m <= 0.0:
            raise ValueError("full_diameter_m must be positive.")
        if not (0.0 < self.opening_fraction <= 1.0):
            raise ValueError("opening_fraction must satisfy 0 < opening_fraction <= 1.")
        if not (0.0 < self.discharge_coefficient <= 1.5):
            raise ValueError("discharge_coefficient should be in a reasonable positive range.")
        if self.area_exponent <= 0.0:
            raise ValueError("area_exponent must be positive.")
        if self.min_pressure_bar <= 0.0:
            raise ValueError("min_pressure_bar must be positive.")
        if self.n_pressure_samples < 20:
            raise ValueError("n_pressure_samples must be >= 20.")

    @property
    def full_area_m2(self) -> float:
        return math.pi * self.full_diameter_m**2 / 4.0

    @property
    def effective_area_m2(self) -> float:
        return self.full_area_m2 * self.opening_fraction**self.area_exponent

    @property
    def effective_kv_m3_h(self) -> float:
        """Equivalent Kv scale for compatibility with existing reports."""
        return self.kv_full_m3_h * self.opening_fraction


class CO2HEMChokeValve:
    """Real-fluid HEM choke model for CO2.

    Capacity:
        Uses isentropic expansion from upstream stagnation state to throat/static
        pressure:
            h0 = h(P1,T1)
            s0 = s(P1,T1)
            v = sqrt(2 * (h0 - h(P*,s0)))
            G = rho(P*,s0) * v
            m_dot = Cd * A_eff * G

    Downstream state:
        For reporting and wellbore inlet enthalpy, throttling remains
        isenthalpic:
            h2 = h1
            T2 = T(P2,h1)
    """

    def __init__(
        self,
        properties: Optional[CoolPropCO2] = None,
        config: Optional[HEMChokeConfig] = None,
    ) -> None:
        self.props = properties or CoolPropCO2()
        self.config = config or HEMChokeConfig()
        self.config.validate()

    def _require_libraries(self) -> None:
        if PropsSI is None:
            raise ImportError("CoolProp is required. Install with: pip install CoolProp")

    def _upstream_hs(self, upstream_pressure_bar: float, upstream_temperature_C: float) -> Tuple[float, float]:
        self._require_libraries()
        p1 = float(upstream_pressure_bar) * BAR_TO_PA
        t1 = float(upstream_temperature_C) + 273.15
        h1 = float(PropsSI("Hmass", "P", p1, "T", t1, "CO2"))
        s1 = float(PropsSI("Smass", "P", p1, "T", t1, "CO2"))
        return h1, s1

    def _mass_flux_at_static_pressure(
        self,
        *,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        static_pressure_bar: float,
    ) -> Optional[Dict[str, float | str]]:
        """Return HEM mass flux at a candidate static/throat pressure."""
        self._require_libraries()

        p1 = float(upstream_pressure_bar) * BAR_TO_PA
        p = float(static_pressure_bar) * BAR_TO_PA

        if p <= 0.0 or p >= p1:
            return None

        h0, s0 = self._upstream_hs(upstream_pressure_bar, upstream_temperature_C)

        try:
            h = float(PropsSI("Hmass", "P", p, "Smass", s0, "CO2"))
            rho = float(PropsSI("Dmass", "P", p, "Smass", s0, "CO2"))
            temp = float(PropsSI("T", "P", p, "Smass", s0, "CO2"))
            try:
                q = float(PropsSI("Q", "P", p, "Smass", s0, "CO2"))
            except Exception:
                q = float("nan")
        except Exception:
            return None

        dh = h0 - h
        if not math.isfinite(dh) or dh <= 0.0:
            return None
        if not math.isfinite(rho) or rho <= 0.0:
            return None

        velocity = math.sqrt(max(2.0 * dh, 0.0))
        mass_flux = rho * velocity

        if not math.isfinite(mass_flux) or mass_flux <= 0.0:
            return None

        if math.isfinite(q) and 0.0 <= q <= 1.0:
            if 0.0 < q < 1.0:
                phase = "two_phase_throat"
            elif q <= 0.0:
                phase = "liquid_or_dense_throat"
            else:
                phase = "vapor_throat"
        else:
            phase = "single_phase_throat"

        return {
            "static_pressure_bar": float(static_pressure_bar),
            "temperature_C": float(temp - 273.15),
            "enthalpy_J_kg": float(h),
            "entropy_J_kgK": float(s0),
            "density_kg_m3": float(rho),
            "quality_mass": float(q),
            "velocity_m_s": float(velocity),
            "mass_flux_kg_m2_s": float(mass_flux),
            "phase_label": phase,
        }

    def _pressure_grid_bar(self, p_min_bar: float, p_max_bar: float, n: int) -> List[float]:
        """Log-spaced pressure grid, robust for wide pressure ranges."""
        p_min = max(float(p_min_bar), 1.0e-9)
        p_max = max(float(p_max_bar), p_min * 1.0001)

        log_min = math.log(p_min)
        log_max = math.log(p_max)

        return [
            math.exp(log_min + (log_max - log_min) * i / max(int(n) - 1, 1))
            for i in range(int(n))
        ]

    def capacity_result(
        self,
        *,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        downstream_pressure_bar: float,
    ) -> Dict[str, float | str]:
        """Return HEM choke capacity and state information for P1,T1,P2."""
        self._require_libraries()

        p1_bar = float(upstream_pressure_bar)
        p2_bar = float(downstream_pressure_bar)

        if p1_bar <= p2_bar:
            raise ValueError("upstream_pressure_bar must be greater than downstream_pressure_bar.")
        if p2_bar < self.config.min_pressure_bar:
            raise ValueError(
                f"downstream_pressure_bar={p2_bar:.6g} is below HEM minimum "
                f"{self.config.min_pressure_bar:.6g} bar."
            )

        p_hi = p1_bar * (1.0 - self.config.pressure_eps_fraction)
        p_lo = max(float(self.config.min_pressure_bar), 1.0e-6)

        candidates: List[Dict[str, float | str]] = []
        for p_bar in self._pressure_grid_bar(p_lo, p_hi, self.config.n_pressure_samples):
            r = self._mass_flux_at_static_pressure(
                upstream_pressure_bar=p1_bar,
                upstream_temperature_C=upstream_temperature_C,
                static_pressure_bar=p_bar,
            )
            if r is not None:
                candidates.append(r)

        if not candidates:
            raise ValueError("HEM failed: no valid throat pressure candidates.")

        critical = max(candidates, key=lambda d: float(d["mass_flux_kg_m2_s"]))
        pcrit_bar = float(critical["static_pressure_bar"])
        gcrit = float(critical["mass_flux_kg_m2_s"])

        down_candidate = self._mass_flux_at_static_pressure(
            upstream_pressure_bar=p1_bar,
            upstream_temperature_C=upstream_temperature_C,
            static_pressure_bar=max(p2_bar, p_lo),
        )

        if down_candidate is None:
            raise ValueError("HEM failed at prescribed downstream pressure.")

        g_down = float(down_candidate["mass_flux_kg_m2_s"])

        if p2_bar <= pcrit_bar:
            selected = critical
            selected_g = gcrit
            flow_regime = "critical"
        else:
            selected = down_candidate
            selected_g = g_down
            flow_regime = "subcritical"

        mass_capacity = (
            self.config.discharge_coefficient
            * self.config.effective_area_m2
            * selected_g
        )

        h1, _ = self._upstream_hs(p1_bar, upstream_temperature_C)
        down = self.props.props_ph(p2_bar * BAR_TO_PA, h1)

        return {
            "model": "HEM",
            "flow_regime": flow_regime,
            "upstream_pressure_bar": float(p1_bar),
            "upstream_temperature_C": float(upstream_temperature_C),
            "upstream_enthalpy_J_kg": float(h1),
            "downstream_pressure_bar": float(p2_bar),
            "downstream_temperature_C": float(down["temperature_K"]) - 273.15,
            "downstream_enthalpy_J_kg": float(h1),
            "quality_mass": float(down["quality_mass"]),
            "phase_label": str(down["phase_label"]),
            "critical_pressure_bar": float(pcrit_bar),
            "critical_mass_flux_kg_m2_s": float(gcrit),
            "selected_throat_pressure_bar": float(selected["static_pressure_bar"]),
            "selected_throat_temperature_C": float(selected["temperature_C"]),
            "selected_throat_quality_mass": float(selected["quality_mass"]),
            "selected_mass_flux_kg_m2_s": float(selected_g),
            "selected_throat_phase_label": str(selected["phase_label"]),
            "full_area_m2": float(self.config.full_area_m2),
            "effective_area_m2": float(self.config.effective_area_m2),
            "opening_fraction": float(self.config.opening_fraction),
            "discharge_coefficient": float(self.config.discharge_coefficient),
            "mass_flow_capacity_kg_s": float(mass_capacity),
            "effective_kv_m3_h": float(self.config.effective_kv_m3_h),
        }

    def mass_flow_capacity_kg_s(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        downstream_pressure_bar: float,
    ) -> float:
        result = self.capacity_result(
            upstream_pressure_bar=upstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            downstream_pressure_bar=downstream_pressure_bar,
        )
        return float(result["mass_flow_capacity_kg_s"])

    def required_kv_liquid_like(
        self,
        upstream_pressure_bar: float,
        downstream_pressure_bar: float,
        upstream_temperature_C: float,
        mass_rate_kg_s: float,
    ) -> float:
        """Return HEM-equivalent required Kv for compatibility with 07 script.

        This is NOT IEC liquid Kv. It is an equivalent scale:
            required area from HEM / full area -> required opening -> equivalent Kv.
        """
        if mass_rate_kg_s <= 0.0:
            raise ValueError("mass_rate_kg_s must be positive.")

        result = self.capacity_result(
            upstream_pressure_bar=upstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            downstream_pressure_bar=downstream_pressure_bar,
        )

        g = float(result["selected_mass_flux_kg_m2_s"])
        cd = float(self.config.discharge_coefficient)

        if not math.isfinite(g) or g <= 0.0:
            raise ValueError("HEM selected mass flux is not positive.")

        required_area = float(mass_rate_kg_s) / max(cd * g, 1.0e-30)
        area_fraction = required_area / max(self.config.full_area_m2, 1.0e-30)

        if area_fraction <= 0.0:
            required_opening = 0.0
        else:
            required_opening = area_fraction ** (1.0 / self.config.area_exponent)

        return float(self.config.kv_full_m3_h * required_opening)

    def downstream_state_at_pressure(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        downstream_pressure_bar: float,
        mass_rate_kg_s: float,
    ) -> Dict[str, float | str]:
        """Calculate post-choke state and HEM capacity at prescribed P2."""
        result = self.capacity_result(
            upstream_pressure_bar=upstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            downstream_pressure_bar=downstream_pressure_bar,
        )
        result["mass_rate_kg_s"] = float(mass_rate_kg_s)
        result["capacity_minus_actual_kg_s"] = (
            float(result["mass_flow_capacity_kg_s"]) - float(mass_rate_kg_s)
        )
        result["required_kv_equivalent_m3_h"] = self.required_kv_liquid_like(
            upstream_pressure_bar=upstream_pressure_bar,
            downstream_pressure_bar=downstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            mass_rate_kg_s=mass_rate_kg_s,
        )
        return result

    def downstream_state(
        self,
        upstream_pressure_bar: float,
        upstream_temperature_C: float,
        mass_rate_kg_s: float,
    ) -> Dict[str, float | str]:
        """Find downstream pressure where HEM-equivalent required Kv matches effective Kv."""
        p1_bar = float(upstream_pressure_bar)
        lo = max(float(self.config.min_pressure_bar), 1.0e-6)
        hi = p1_bar * 0.999

        target = float(self.config.effective_kv_m3_h)

        def f(p2_bar: float) -> float:
            return self.required_kv_liquid_like(
                upstream_pressure_bar=upstream_pressure_bar,
                downstream_pressure_bar=p2_bar,
                upstream_temperature_C=upstream_temperature_C,
                mass_rate_kg_s=mass_rate_kg_s,
            ) - target

        f_lo = f(lo)
        f_hi = f(hi)

        if f_lo * f_hi > 0.0:
            raise ValueError(
                "Could not bracket HEM downstream pressure. "
                f"f(min P2)={f_lo:.6g}, f(nearly P1)={f_hi:.6g}, "
                f"Kv_effective={target:.6g}."
            )

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)

            if abs(f_mid) <= 1.0e-5:
                return self.downstream_state_at_pressure(
                    upstream_pressure_bar=upstream_pressure_bar,
                    upstream_temperature_C=upstream_temperature_C,
                    downstream_pressure_bar=mid,
                    mass_rate_kg_s=mass_rate_kg_s,
                )

            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid

        mid = 0.5 * (lo + hi)
        return self.downstream_state_at_pressure(
            upstream_pressure_bar=upstream_pressure_bar,
            upstream_temperature_C=upstream_temperature_C,
            downstream_pressure_bar=mid,
            mass_rate_kg_s=mass_rate_kg_s,
        )
