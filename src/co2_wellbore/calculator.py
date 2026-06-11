"""Main CO2 injection wellbore calculator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .config import DriftFluxConfig, SolverConfig, ThermalConfig, WellboreGeometry
from .constants import BAR_TO_PA, DAY_TO_S, G, PA_TO_BAR
from .correlations import clamp, darcy_friction_factor
from .drift_flux import apply_drift_flux_closure, build_single_phase_flow_props
from .properties import CoolPropCO2


class CO2WellboreCalculator:
    """CO2 injection wellbore calculator with CoolProp properties."""

    def __init__(
        self,
        geometry: Optional[WellboreGeometry] = None,
        thermal: Optional[ThermalConfig] = None,
        drift_flux: Optional[DriftFluxConfig] = None,
        solver: Optional[SolverConfig] = None,
        properties: Optional[CoolPropCO2] = None,
    ) -> None:
        self.geometry = geometry or WellboreGeometry()
        self.thermal = thermal or ThermalConfig()
        self.drift_flux = drift_flux or DriftFluxConfig()
        self.solver = solver or SolverConfig()
        self.props = properties or CoolPropCO2()

        self.geometry.validate()
        self.thermal.validate()
        self.drift_flux.validate()
        self.solver.validate()

    def standard_density_kg_m3(self) -> float:
        """Return CO2 density at configured standard conditions."""
        pressure = self.geometry.std_pressure_bar * BAR_TO_PA
        temperature = self.geometry.std_temperature_C + 273.15
        return self.props.density(pressure, temperature)

    def mass_rate_from_sm3_day(self, q_sm3_day: float) -> float:
        """Convert standard m3/day to kg/s."""
        return float(q_sm3_day) * self.standard_density_kg_m3() / DAY_TO_S

    def sm3_day_from_mass_rate(self, mass_rate_kg_s: float) -> float:
        """Convert kg/s to standard m3/day."""
        return float(mass_rate_kg_s) * DAY_TO_S / self.standard_density_kg_m3()

    def _flow_props_ph(
        self,
        p_pa: float,
        h_j_kg: float,
        mass_rate_kg_s: float,
        area_m2: float,
    ) -> Dict[str, float | str]:
        """Flow properties from P-H flash."""
        base = self.props.props_ph(p_pa, h_j_kg)
        q_mass = float(base.get("quality_mass", float("nan")))

        if not math.isfinite(q_mass) or not (0.0 < q_mass < 1.0):
            return build_single_phase_flow_props(base)

        vapor = self.props.saturated_props(p_pa, 1.0)
        liquid = self.props.saturated_props(p_pa, 0.0)

        return apply_drift_flux_closure(
            base=base,
            vapor=vapor,
            liquid=liquid,
            mass_rate_kg_s=mass_rate_kg_s,
            area_m2=area_m2,
            config=self.drift_flux,
        )

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
        heat_diameter = (
            cfg.diameter_m
            if hcfg.heat_transfer_diameter_m is None
            else hcfg.heat_transfer_diameter_m
        )

        pressure = float(thp_bar) * BAR_TO_PA
        wellhead_temperature_K = float(wellhead_temperature_C) + 273.15

        enthalpy = self.props.enthalpy_from_boundary(
            P_Pa=pressure,
            T_K=wellhead_temperature_K,
            h_J_kg=wellhead_enthalpy_J_kg,
            quality_mass=wellhead_quality_mass,
        )

        rows: List[Dict[str, float | str]] = []

        cum_hydro = 0.0
        cum_friction = 0.0
        cum_accel = 0.0
        cum_heat_J_kg = 0.0
        last_velocity: Optional[float] = None

        for i in range(cfg.n_segments + 1):
            depth = min(i * dz, cfg.tvd_m)

            flow_props = self._flow_props_ph(pressure, enthalpy, mass_rate_kg_s, area)

            temperature_K = float(flow_props["temperature_K"])
            rho_flow = max(float(flow_props["rho_flow_kg_m3"]), 1.0e-12)
            rho_hydro = max(float(flow_props["rho_hydro_kg_m3"]), 1.0e-12)
            rho_friction = max(float(flow_props["rho_friction_kg_m3"]), 1.0e-12)
            viscosity = max(float(flow_props["viscosity_mix_Pa_s"]), 1.0e-12)

            cp_raw = float(flow_props.get("cp_J_kgK", float("nan")))
            cp_eff = cp_raw if math.isfinite(cp_raw) and cp_raw > 1.0 else 2000.0

            q_actual_m3_s = mass_rate_kg_s / rho_flow
            velocity = q_actual_m3_s / max(area, 1.0e-30)
            reynolds = rho_friction * abs(velocity) * cfg.diameter_m / viscosity
            friction_factor = darcy_friction_factor(
                max(reynolds, cfg.min_reynolds), cfg.roughness_m, cfg.diameter_m
            )

            rows.append(
                {
                    "segment": float(i),
                    "depth_m": float(depth),
                    "pressure_bar": float(pressure * PA_TO_BAR),
                    "temperature_C": float(temperature_K - 273.15),
                    "geothermal_temperature_C": float(hcfg.geothermal_temperature_C(depth)),
                    "phase_label": str(flow_props["phase_label"]),
                    "quality_mass": float(flow_props["quality_mass"]),
                    "two_phase_active": float(flow_props["two_phase_active"]),
                    "density_kg_m3": rho_flow,
                    "rho_hydro_kg_m3": rho_hydro,
                    "rho_friction_kg_m3": rho_friction,
                    "rho_vapor_kg_m3": float(flow_props["rho_vapor_kg_m3"]),
                    "rho_liquid_kg_m3": float(flow_props["rho_liquid_kg_m3"]),
                    "gas_holdup": float(flow_props["gas_holdup"]),
                    "liquid_holdup": float(flow_props["liquid_holdup"]),
                    "viscosity_Pa_s": viscosity,
                    "cp_J_kgK": cp_eff,
                    "enthalpy_J_kg": float(enthalpy),
                    "mass_rate_kg_s": mass_rate_kg_s,
                    "std_rate_sm3_day": self.sm3_day_from_mass_rate(mass_rate_kg_s),
                    "actual_rate_m3_s": q_actual_m3_s,
                    "velocity_m_s": velocity,
                    "reynolds": reynolds,
                    "friction_factor": friction_factor,
                    "cum_dP_hydro_bar": cum_hydro * PA_TO_BAR,
                    "cum_dP_friction_bar": cum_friction * PA_TO_BAR,
                    "cum_dP_acceleration_bar": cum_accel * PA_TO_BAR,
                    "cum_dP_total_bar": (cum_hydro + cum_friction + cum_accel) * PA_TO_BAR,
                    "cum_heat_J_kg": cum_heat_J_kg,
                    "energy_formulation": "P-H flash",
                }
            )

            if i == cfg.n_segments:
                break

            dP_hydro = rho_hydro * G * dz
            dP_friction_positive_loss = (
                friction_factor * (dz / cfg.diameter_m) * 0.5 * rho_friction * velocity**2
            )

            dP_acc = 0.0
            if last_velocity is not None:
                dP_acc = -rho_flow * velocity * (velocity - last_velocity)
            last_velocity = velocity

            dP_total = dP_hydro - dP_friction_positive_loss + dP_acc

            if hcfg.enabled and mass_rate_kg_s > 0.0:
                depth_mid = min(depth + 0.5 * dz, cfg.tvd_m)
                ambient_temperature_K = hcfg.geothermal_temperature_C(depth_mid) + 273.15
                dh_heat = (
                    hcfg.overall_U_W_m2K
                    * math.pi
                    * heat_diameter
                    * dz
                    * (ambient_temperature_K - temperature_K)
                    / max(mass_rate_kg_s, 1.0e-30)
                )
            else:
                dh_heat = 0.0

            max_abs_dh = max(abs(hcfg.max_abs_dT_per_segment_C) * cp_eff, 5.0e4)
            dh_heat = clamp(dh_heat, -max_abs_dh, max_abs_dh)

            pressure = max(pressure + dP_total, 1.0e3)
            enthalpy = enthalpy + dh_heat

            cum_hydro += dP_hydro
            cum_friction += -dP_friction_positive_loss
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
        """Return bottomhole pressure in bar for a given THP and injection rate."""
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
        """Return bottomhole temperature in C for a given THP and injection rate."""
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
        """Invert wellbore model: find THP required to reproduce a target BHP."""
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
                f"Target BHP={target_bhp_bar} bar is not bracketed by THP bounds "
                f"{lo:g}-{hi:g} bar. Residuals: {f_lo:.6g}, {f_hi:.6g} bar."
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
        """Solve q = J * max(BHP(thp, q) - p_res, 0) for injection."""
        injectivity = float(injectivity_sm3_day_per_bar)
        if injectivity <= 0.0:
            raise ValueError("injectivity_sm3_day_per_bar must be positive.")
        lo, hi = q_bounds_sm3_day or self.solver.q_bounds_sm3_day
        tol = self.solver.tol_rate_sm3_day if tol_sm3_day is None else float(tol_sm3_day)
        nmax = self.solver.max_iter if max_iter is None else int(max_iter)

        def residual(rate: float) -> float:
            bhp = self.bhp_from_thp_and_rate(
                thp_bar, q_sm3_day=rate, wellhead_temperature_C=wellhead_temperature_C
            )
            return rate - injectivity * max(bhp - float(reservoir_pressure_bar), 0.0)

        r_lo = residual(lo)
        r_hi = residual(hi)
        if r_lo * r_hi > 0.0:
            q_best = lo if abs(r_lo) <= abs(r_hi) else hi
            bhp_best = self.bhp_from_thp_and_rate(
                thp_bar, q_sm3_day=q_best, wellhead_temperature_C=wellhead_temperature_C
            )
            return float(q_best), float(bhp_best), float(residual(q_best))

        for _ in range(nmax):
            mid = 0.5 * (lo + hi)
            r_mid = residual(mid)
            if abs(r_mid) <= tol:
                bhp = self.bhp_from_thp_and_rate(
                    thp_bar, q_sm3_day=mid, wellhead_temperature_C=wellhead_temperature_C
                )
                return float(mid), float(bhp), float(r_mid)
            if r_lo * r_mid <= 0.0:
                hi, r_hi = mid, r_mid
            else:
                lo, r_lo = mid, r_mid

        rate = 0.5 * (lo + hi)
        bhp = self.bhp_from_thp_and_rate(
            thp_bar, q_sm3_day=rate, wellhead_temperature_C=wellhead_temperature_C
        )
        return float(rate), float(bhp), float(residual(rate))

    def rate_limited_by_thp(
        self,
        thp_limit_bar: float,
        reservoir_bhp_for_rate_callable,
        q_bounds_sm3_day: Optional[Tuple[float, float]] = None,
        wellhead_temperature_C: float = 40.0,
        tol_thp_bar: float = 0.2,
        max_iter: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Choose q so required THP equals a limit.

        reservoir_bhp_for_rate_callable(q) must run or approximate the reservoir
        and return WBHP(q) in bar.
        """
        lo, hi = q_bounds_sm3_day or self.solver.q_bounds_sm3_day
        nmax = self.solver.max_iter if max_iter is None else int(max_iter)

        def thp_residual(rate: float) -> Tuple[float, float]:
            bhp = float(reservoir_bhp_for_rate_callable(float(rate)))
            thp_req = self.required_thp_for_target_bhp(
                bhp, q_sm3_day=float(rate), wellhead_temperature_C=wellhead_temperature_C
            )
            return thp_req - float(thp_limit_bar), thp_req

        r_lo, thp_lo = thp_residual(lo)
        r_hi, thp_hi = thp_residual(hi)
        if r_lo * r_hi > 0.0:
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

    def config_dict(self) -> Dict[str, object]:
        """Return calculator configuration as a dictionary."""
        return {
            "geometry": asdict(self.geometry),
            "thermal": asdict(self.thermal),
            "drift_flux": asdict(self.drift_flux),
            "solver": asdict(self.solver),
            "property_backend": "CoolProp::CO2",
        }
