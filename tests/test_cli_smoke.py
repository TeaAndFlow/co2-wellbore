from __future__ import annotations

from co2_wellbore.coupling.cli import build_parser


def test_cli_default_base_dir_is_none() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.base_dir is None
    assert args.deck == "Base.DATA"


def test_cli_accepts_wtemp_solver_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--wtemp-max-step-C",
            "5.0",
            "--wtemp-min-C",
            "0.0",
            "--wtemp-max-C",
            "100.0",
        ]
    )

    assert args.wtemp_max_step_C == 5.0
    assert args.wtemp_min_C == 0.0
    assert args.wtemp_max_C == 100.0


def test_cli_accepts_opening_choke_hem_case() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--pressure-control-mode",
            "opening_choke",
            "--choke-model",
            "hem",
            "--opening-schedule-percent",
            "1",
            "2",
            "3",
            "--n-steps",
            "3",
        ]
    )

    assert args.pressure_control_mode == "opening_choke"
    assert args.choke_model == "hem"
    assert args.opening_schedule_percent == [1.0, 2.0, 3.0]
    assert args.n_steps == 3
