from co2_wellbore import CO2WellboreCalculator


def main() -> None:
    calc = CO2WellboreCalculator()

    thp = calc.required_thp_for_target_bhp(
        target_bhp_bar=180.0,
        q_sm3_day=100000.0,
        wellhead_temperature_C=40.0,
    )

    print(f"Required THP = {thp:.3f} bar")


if __name__ == "__main__":
    main()
