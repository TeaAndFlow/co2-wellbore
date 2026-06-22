import math

from co2_wellbore import CO2WellboreCalculator, ThermalConfig, WellboreGeometry
from co2_wellbore.constants import G, PA_TO_BAR


def make_wellbore():
    return CO2WellboreCalculator(
        geometry=WellboreGeometry(
            tvd_m=1000.0,
            diameter_m=0.10,
            roughness_m=1.5e-5,
            n_segments=80,
        ),
        thermal=ThermalConfig(
            enabled=False,
            surface_temperature_C=30.0,
            geothermal_gradient_C_per_m=0.03,
        ),
    )


def test_tiny_flow_pressure_is_close_to_hydrostatic_limit():
    wb = make_wellbore()

    thp_bar = 100.0
    profile = wb.profile_from_thp_and_rate(
        thp_bar=thp_bar,
        q_sm3_day=1.0,
        wellhead_temperature_C=60.0,
    )

    bhp_bar = float(profile["pressure_bar"].iloc[-1])
    mean_rho = float(profile["rho_hydro_kg_m3"].mean())
    hydro_bar = mean_rho * G * wb.geometry.tvd_m * PA_TO_BAR

    assert bhp_bar > thp_bar
    assert hydro_bar > 0.0

    # This is a limiting-case sanity check, not an exact analytical solution.
    rel_err = abs((bhp_bar - thp_bar) - hydro_bar) / hydro_bar
    assert rel_err < 0.30


def test_increasing_thp_increases_bhp():
    wb = make_wellbore()

    bhp_100 = wb.bhp_from_thp_and_rate(
        thp_bar=100.0,
        q_sm3_day=50000.0,
        wellhead_temperature_C=60.0,
    )
    bhp_120 = wb.bhp_from_thp_and_rate(
        thp_bar=120.0,
        q_sm3_day=50000.0,
        wellhead_temperature_C=60.0,
    )

    assert math.isfinite(bhp_100)
    assert math.isfinite(bhp_120)
    assert bhp_120 > bhp_100


def test_wellbore_profile_contains_physical_positive_properties():
    wb = make_wellbore()

    profile = wb.profile_from_thp_and_rate(
        thp_bar=100.0,
        q_sm3_day=100000.0,
        wellhead_temperature_C=60.0,
    )

    assert len(profile) == wb.geometry.n_segments + 1
    assert profile["pressure_bar"].notna().all()
    assert profile["temperature_C"].notna().all()
    assert (profile["density_kg_m3"] > 0.0).all()
    assert (profile["viscosity_Pa_s"] > 0.0).all()
