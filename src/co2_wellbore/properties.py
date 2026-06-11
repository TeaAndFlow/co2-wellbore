"""CO2 thermodynamic and transport property backend."""

from __future__ import annotations

import math
from typing import Dict, Optional

try:  # optional at import time, required at runtime
    from CoolProp.CoolProp import PropsSI  # type: ignore
except Exception:  # pragma: no cover
    PropsSI = None  # type: ignore


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
        3) P,T single-phase or supercritical state.

        Important:
        For pure CO2, P,T exactly on the saturation line is ambiguous.
        In that case the user must provide h_J_kg or quality_mass.

        The saturation-line check is only valid below the critical temperature.
        Above the critical temperature there is no saturation line, so the check
        must be skipped.
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

        # Saturation P(T) exists only below the critical temperature.
        # For T >= Tcrit, skip this check and treat P,T as a valid
        # single-phase/supercritical boundary state.
        try:
            Tcrit = float(PropsSI("Tcrit", self.fluid))
        except Exception:
            Tcrit = 304.1282

        if T_K < Tcrit:
            try:
                psat = float(PropsSI("P", "T", T_K, "Q", 0.0, self.fluid))
            except Exception:
                psat = float("nan")

            if math.isfinite(psat) and psat > 0.0:
                rel = abs(P_Pa - psat) / psat
                if rel < 5.0e-4:
                    raise ValueError(
                        "Boundary P,T lies on/near CO2 saturation line. "
                        "For pure CO2 this does not define quality. "
                        "Provide wellhead_enthalpy_J_kg or wellhead_quality_mass."
                    )

        return float(PropsSI("Hmass", "P", P_Pa, "T", T_K, self.fluid))

    def props_ph(self, P_Pa: float, h_J_kg: float) -> Dict[str, float | str]:
        """CoolProp flash using P,Hmass."""
        self._require_coolprop()
        if P_Pa <= 0.0:
            raise ValueError("Pressure must be positive.")
        if not math.isfinite(float(h_J_kg)):
            raise ValueError("Enthalpy must be finite.")

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
        """Finite-difference JT coefficient mu_JT = (dT/dP)_h."""
        self._require_coolprop()
        h0 = self.enthalpy(P_Pa, T_K)
        dP = max(1000.0, 1.0e-4 * P_Pa)
        try:
            T2 = self.temperature_from_ph(P_Pa + dP, h0)
            mu = (T2 - T_K) / dP
        except Exception:
            T_lo = self.temperature_from_ph(max(P_Pa - dP, 1.0e3), h0)
            T_hi = self.temperature_from_ph(P_Pa + dP, h0)
            mu = (T_hi - T_lo) / (2.0 * dP)
        return float(min(max(mu, -max_abs), max_abs))

    def phase_label(
        self,
        P_Pa: float,
        T_K: float,
        phase_index: Optional[int] = None,
        q: Optional[float] = None,
    ) -> str:
        self._require_coolprop()
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
