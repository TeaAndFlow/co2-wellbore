from __future__ import annotations

import math

from co2_wellbore.coupling.thermal_coupling import (
    ThermalUpdateState,
    propose_next_wtemp,
)


def test_relaxed_fixed_point_first_update() -> None:
    state = ThermalUpdateState()

    next_wtemp, info = propose_next_wtemp(
        state=state,
        current_wtemp_C=40.0,
        residual_C=-10.0,
        relaxation=0.7,
        max_step_C=100.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    assert math.isclose(next_wtemp, 33.0)
    assert info["thermal_update_method"] == "relaxed_fixed_point"
    assert math.isclose(float(info["thermal_wtemp_delta_C"]), -7.0)


def test_max_step_clips_large_update() -> None:
    state = ThermalUpdateState()

    next_wtemp, info = propose_next_wtemp(
        state=state,
        current_wtemp_C=40.0,
        residual_C=-100.0,
        relaxation=1.0,
        max_step_C=10.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    assert math.isclose(next_wtemp, 30.0)
    assert str(info["thermal_update_method"]).endswith("_clipped")


def test_temperature_bounds_are_respected() -> None:
    state = ThermalUpdateState()

    next_wtemp, _ = propose_next_wtemp(
        state=state,
        current_wtemp_C=0.0,
        residual_C=-100.0,
        relaxation=1.0,
        max_step_C=1000.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    assert math.isclose(next_wtemp, -20.0)


def test_bracketed_false_position_after_sign_change() -> None:
    state = ThermalUpdateState()

    propose_next_wtemp(
        state=state,
        current_wtemp_C=40.0,
        residual_C=-10.0,
        relaxation=0.7,
        max_step_C=100.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    next_wtemp, info = propose_next_wtemp(
        state=state,
        current_wtemp_C=30.0,
        residual_C=10.0,
        relaxation=0.7,
        max_step_C=100.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    assert math.isclose(next_wtemp, 35.0)
    assert info["thermal_update_method"] == "bracketed_false_position"


def test_secant_update_without_sign_change() -> None:
    state = ThermalUpdateState()

    propose_next_wtemp(
        state=state,
        current_wtemp_C=40.0,
        residual_C=-10.0,
        relaxation=0.7,
        max_step_C=100.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    next_wtemp, info = propose_next_wtemp(
        state=state,
        current_wtemp_C=35.0,
        residual_C=-5.0,
        relaxation=0.7,
        max_step_C=100.0,
        min_wtemp_C=-20.0,
        max_wtemp_C=150.0,
    )

    assert math.isclose(next_wtemp, 30.0)
    assert info["thermal_update_method"] == "secant"
