from __future__ import annotations

import argparse

from co2_wellbore import (
    CO2WellboreCalculator,
    DriftFluxConfig,
    ThermalConfig,
    WellboreGeometry,
)


def build_advanced_wellbore(args: argparse.Namespace) -> CO2WellboreCalculator:
    """Build the package CO2 wellbore calculator from CLI/config args."""
    geometry = WellboreGeometry(
        tvd_m=float(args.tvd_m),
        diameter_m=float(args.diameter_m),
        roughness_m=float(args.roughness_m),
        n_segments=int(args.n_segments),
    )

    thermal = ThermalConfig(
        enabled=bool(args.heat_transfer),
        overall_U_W_m2K=float(args.overall_U_W_m2K),
        heat_transfer_diameter_m=float(args.diameter_m),
        surface_temperature_C=float(args.surface_temperature_C),
        geothermal_gradient_C_per_m=float(args.geothermal_gradient_C_per_m),
        max_abs_dT_per_segment_C=float(args.max_abs_dT_per_segment_C),
        model=str(args.thermal_model),
        tubing_outer_diameter_m=float(args.tubing_outer_diameter_m),
        cement_outer_diameter_m=float(args.cement_outer_diameter_m),
        inner_heat_transfer_coefficient_W_m2K=float(args.inner_heat_transfer_coefficient_W_m2K),
        tubing_k_W_mK=float(args.tubing_k_W_mK),
        cement_k_W_mK=float(args.cement_k_W_mK),
        formation_k_W_mK=float(args.formation_k_W_mK),
        formation_heat_capacity_J_m3K=float(args.formation_heat_capacity_J_m3K),
        transient_formation=bool(args.transient_formation_thermal),
        min_thermal_time_days=float(args.min_thermal_time_days),
        max_formation_radius_m=float(args.max_formation_radius_m),
    )

    drift_flux = DriftFluxConfig(
        enabled=bool(args.two_phase),
        C0=float(args.drift_flux_C0),
        drift_velocity_coeff=float(args.drift_velocity_coeff),
        surface_tension_N_m=float(args.surface_tension_N_m),
        friction_density=str(args.friction_density),
    )

    return CO2WellboreCalculator(
        geometry=geometry,
        thermal=thermal,
        drift_flux=drift_flux,
    )