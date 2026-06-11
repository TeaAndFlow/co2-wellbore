from co2_wellbore.correlations import darcy_friction_factor


def test_laminar_friction_factor() -> None:
    friction = darcy_friction_factor(
        reynolds=1000.0,
        roughness_m=0.0,
        diameter_m=0.1,
    )

    assert abs(friction - 0.064) < 1.0e-12
