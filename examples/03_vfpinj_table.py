from co2_wellbore import CO2WellboreCalculator
from co2_wellbore.opm import make_vfpinj_table


def main() -> None:
    calc = CO2WellboreCalculator()

    table = make_vfpinj_table(
        calculator=calc,
        table_id=1,
        rates_sm3_day=[50000.0, 100000.0, 200000.0],
        thp_values_bar=[40.0, 60.0, 80.0, 100.0],
        wellhead_temperature_C=40.0,
    )

    print(table)


if __name__ == "__main__":
    main()
