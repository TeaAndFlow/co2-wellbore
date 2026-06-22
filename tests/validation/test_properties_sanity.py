import math

import pytest

from co2_wellbore import CoolPropCO2
from co2_wellbore.constants import BAR_TO_PA


def test_co2_properties_are_finite_for_representative_states():
    props = CoolPropCO2()

    states = [
        # pressure_bar, temperature_C
        (20.0, 40.0),   # gas-like
        (60.0, 60.0),   # high-pressure gas / dense-ish
        (100.0, 40.0),  # supercritical/dense
        (120.0, 80.0),  # supercritical
    ]

    for p_bar, t_C in states:
        out = props.props(p_bar * BAR_TO_PA, t_C + 273.15)

        assert math.isfinite(float(out["rho_kg_m3"]))
        assert float(out["rho_kg_m3"]) > 0.0

        assert math.isfinite(float(out["enthalpy_J_kg"]))

        assert math.isfinite(float(out["viscosity_Pa_s"]))
        assert float(out["viscosity_Pa_s"]) > 0.0

        assert str(out["phase_label"]) != ""


def test_ph_flash_is_consistent_for_single_phase_state():
    props = CoolPropCO2()

    p_bar = 80.0
    t_C = 60.0

    pt = props.props(p_bar * BAR_TO_PA, t_C + 273.15)
    h = float(pt["enthalpy_J_kg"])

    ph = props.props_ph(p_bar * BAR_TO_PA, h)

    assert math.isfinite(float(ph["temperature_K"]))
    assert math.isfinite(float(ph["rho_kg_m3"]))
    assert float(ph["rho_kg_m3"]) > 0.0
    assert str(ph["phase_label"]) != ""

    # P-H flash should recover the original temperature approximately.
    assert abs(float(ph["temperature_K"]) - (t_C + 273.15)) < 1.0e-5


def test_saturated_ph_flash_returns_quality_between_zero_and_one():
    props = CoolPropCO2()

    p_bar = 50.0
    h_liq = props.saturated_props(p_bar * BAR_TO_PA, 0.0)["enthalpy_J_kg"]
    h_vap = props.saturated_props(p_bar * BAR_TO_PA, 1.0)["enthalpy_J_kg"]
    h_mid = 0.5 * (float(h_liq) + float(h_vap))

    ph = props.props_ph(p_bar * BAR_TO_PA, h_mid)

    assert str(ph["phase_label"]) == "two_phase_saturated"
    assert math.isfinite(float(ph["quality_mass"]))
    assert 0.0 <= float(ph["quality_mass"]) <= 1.0
