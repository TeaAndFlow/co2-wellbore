from co2_wellbore import CO2WellboreCalculator, ThermalConfig, WellboreGeometry
from co2_wellbore.diagnostics import summarize_profile


def main() -> None:
    calc = CO2WellboreCalculator(
        geometry=WellboreGeometry(
            tvd_m=1600.0,
            diameter_m=0.10,
            roughness_m=1.5e-5,
            n_segments=120,
        ),
        thermal=ThermalConfig(
            enabled=True,
            overall_U_W_m2K=4.0,
            surface_temperature_C=32.0,
            geothermal_gradient_C_per_m=0.03,
        ),
    )

    profile = calc.profile_from_thp_and_rate(
        thp_bar=60.0,
        q_sm3_day=100000.0,
        wellhead_temperature_C=40.0,
    )

    print(profile.tail())
    print(summarize_profile(profile))


if __name__ == "__main__":
    main()
