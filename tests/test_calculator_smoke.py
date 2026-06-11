from co2_wellbore import CO2WellboreCalculator


def test_basic_profile_runs() -> None:
    calc = CO2WellboreCalculator()

    profile = calc.profile_from_thp_and_rate(
        thp_bar=60.0,
        q_sm3_day=100000.0,
        wellhead_temperature_C=40.0,
    )

    assert len(profile) > 2
    assert profile["pressure_bar"].iloc[-1] > 0.0
    assert profile["temperature_C"].iloc[-1] > -100.0
