from co2_wellbore import CO2WellboreCalculator


def main() -> None:
    calc = CO2WellboreCalculator()

    rate, bhp, residual = calc.rate_from_thp_and_injectivity(
        thp_bar=80.0,
        reservoir_pressure_bar=120.0,
        injectivity_sm3_day_per_bar=5000.0,
        wellhead_temperature_C=40.0,
    )

    print(f"Rate = {rate:.3f} sm3/day")
    print(f"BHP = {bhp:.3f} bar")
    print(f"Residual = {residual:.3f} sm3/day")


if __name__ == "__main__":
    main()
