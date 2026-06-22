import math

from co2_wellbore import CO2HEMChokeValve, HEMChokeConfig


def make_valve(opening=0.20, upstream_samples=220):
    return CO2HEMChokeValve(
        config=HEMChokeConfig(
            full_diameter_m=0.067,
            opening_fraction=opening,
            discharge_coefficient=0.84,
            area_exponent=1.0,
            min_pressure_bar=10.0,
            n_pressure_samples=upstream_samples,
        )
    )


def capacity(opening=0.20, p1=60.0, t1=60.0, p2=20.0):
    valve = make_valve(opening=opening)
    return valve.capacity_result(
        upstream_pressure_bar=p1,
        upstream_temperature_C=t1,
        downstream_pressure_bar=p2,
    )


def test_hem_capacity_is_positive_and_finite():
    out = capacity()

    assert math.isfinite(float(out["mass_flow_capacity_kg_s"]))
    assert float(out["mass_flow_capacity_kg_s"]) > 0.0

    assert math.isfinite(float(out["critical_pressure_bar"]))
    assert 10.0 <= float(out["critical_pressure_bar"]) < 60.0

    assert str(out["critical_search_method"]) in {
        "grid_then_bounded_optimizer",
        "grid_fallback",
        "grid_fallback_after_optimizer_failure",
        "grid_fallback_after_invalid_optimizer_state",
        "grid_fallback_after_optimizer_guard",
    }


def test_larger_opening_gives_larger_capacity():
    small = capacity(opening=0.10)
    large = capacity(opening=0.20)

    assert float(large["mass_flow_capacity_kg_s"]) > float(small["mass_flow_capacity_kg_s"])


def test_larger_upstream_pressure_gives_larger_capacity():
    low = capacity(p1=50.0, p2=20.0)
    high = capacity(p1=70.0, p2=20.0)

    assert float(high["mass_flow_capacity_kg_s"]) > float(low["mass_flow_capacity_kg_s"])


def test_choked_capacity_is_insensitive_to_downstream_pressure_below_critical():
    a = capacity(p1=60.0, t1=60.0, p2=20.0)
    b = capacity(p1=60.0, t1=60.0, p2=25.0)

    assert str(a["flow_regime"]) == "critical"
    assert str(b["flow_regime"]) == "critical"

    ma = float(a["mass_flow_capacity_kg_s"])
    mb = float(b["mass_flow_capacity_kg_s"])

    rel = abs(ma - mb) / max(abs(ma), 1.0e-30)
    assert rel < 1.0e-6
