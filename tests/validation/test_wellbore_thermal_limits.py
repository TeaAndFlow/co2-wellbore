import math

from co2_wellbore import CO2WellboreCalculator, ThermalConfig, WellboreGeometry


def make_wellbore(overall_U, enabled=True):
    return CO2WellboreCalculator(
        geometry=WellboreGeometry(
            tvd_m=800.0,
            diameter_m=0.10,
            roughness_m=1.5e-5,
            n_segments=80,
        ),
        thermal=ThermalConfig(
            enabled=enabled,
            model="overall_U",
            overall_U_W_m2K=overall_U,
            surface_temperature_C=30.0,
            geothermal_gradient_C_per_m=0.03,
            max_abs_dT_per_segment_C=10.0,
        ),
    )


def bottom_temperature(wb):
    profile = wb.profile_from_thp_and_rate(
        thp_bar=100.0,
        q_sm3_day=100000.0,
        wellhead_temperature_C=20.0,
    )
    return float(profile["temperature_C"].iloc[-1])


def test_no_heat_transfer_produces_finite_temperature():
    wb = make_wellbore(overall_U=0.0, enabled=False)
    t_bottom = bottom_temperature(wb)

    assert math.isfinite(t_bottom)


def test_high_heat_transfer_moves_temperature_toward_geothermal_trend():
    low_u = make_wellbore(overall_U=1.0e-6, enabled=True)
    high_u = make_wellbore(overall_U=200.0, enabled=True)

    t_low = bottom_temperature(low_u)
    t_high = bottom_temperature(high_u)

    geo_bottom = high_u.thermal.geothermal_temperature_C(high_u.geometry.tvd_m)

    assert math.isfinite(t_low)
    assert math.isfinite(t_high)
    assert math.isfinite(geo_bottom)

    # High U should make the bottom temperature closer to the geothermal value
    # than almost-zero U. This verifies the direction of the reduced thermal model.
    assert abs(t_high - geo_bottom) < abs(t_low - geo_bottom)
