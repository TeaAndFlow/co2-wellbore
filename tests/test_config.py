import pytest

from co2_wellbore import WellboreGeometry


def test_negative_tvd_fails() -> None:
    geometry = WellboreGeometry(tvd_m=-1.0)

    with pytest.raises(ValueError):
        geometry.validate()
