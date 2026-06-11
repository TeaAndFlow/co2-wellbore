from co2_wellbore import CO2WellboreCalculator
from co2_wellbore.opm import make_vfpinj_table


def test_vfpinj_contains_keyword() -> None:
    calc = CO2WellboreCalculator()

    text = make_vfpinj_table(
        calculator=calc,
        table_id=1,
        rates_sm3_day=[100000.0],
        thp_values_bar=[60.0],
    )

    assert "VFPINJ" in text
    assert "BHP" in text
    assert "/" in text
