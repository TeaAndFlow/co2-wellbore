from co2_wellbore import CO2WellboreCalculator


def test_rate_conversion_roundtrip() -> None:
    calc = CO2WellboreCalculator()

    rate_sm3_day = 100000.0
    mass_rate = calc.mass_rate_from_sm3_day(rate_sm3_day)
    converted_rate = calc.sm3_day_from_mass_rate(mass_rate)

    assert abs(converted_rate - rate_sm3_day) / rate_sm3_day < 1.0e-10
