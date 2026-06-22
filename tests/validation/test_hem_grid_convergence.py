import math

from co2_wellbore import CO2HEMChokeValve, HEMChokeConfig


def run_case(samples: int):
    valve = CO2HEMChokeValve(
        config=HEMChokeConfig(
            full_diameter_m=0.067,
            opening_fraction=0.20,
            discharge_coefficient=0.84,
            area_exponent=1.0,
            min_pressure_bar=10.0,
            n_pressure_samples=samples,
        )
    )
    return valve.capacity_result(
        upstream_pressure_bar=60.0,
        upstream_temperature_C=60.0,
        downstream_pressure_bar=20.0,
    )


def test_hem_critical_pressure_converges_with_grid_refinement():
    # 220 is the current production/default-ish resolution.
    # 1000 is used as a high-resolution reference for this sanity test.
    coarse = run_case(220)
    reference = run_case(1000)

    cap_coarse = float(coarse["mass_flow_capacity_kg_s"])
    cap_ref = float(reference["mass_flow_capacity_kg_s"])

    assert math.isfinite(cap_coarse)
    assert math.isfinite(cap_ref)
    assert cap_ref > 0.0

    rel_err = abs(cap_coarse - cap_ref) / cap_ref

    # Scientific sanity threshold. If this fails, increase default n_pressure_samples
    # or improve optimizer bracketing.
    assert rel_err < 0.01


def test_optimizer_reports_search_metadata():
    out = run_case(220)

    assert "critical_search_method" in out
    assert "critical_optimizer_success" in out
    assert "critical_pressure_samples" in out

    assert float(out["critical_pressure_samples"]) == 220.0
