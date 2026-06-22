from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from co2_wellbore import (
    CO2ChokeValve,
    ChokeConfig,
    CO2HEMChokeValve,
    HEMChokeConfig,
)
from co2_wellbore.constants import BAR_TO_PA

from .deck_editing import (
    StepControl,
    RestartPointer,
    read_text,
    split_deck_sections,
    extract_static_well_setup,
    extract_base_producer_control,
    parse_first_wconinje_rate,
    parse_first_wtemp,
    parse_tstep_list,
    segment_root,
    build_segment_deck,
    copy_base_input_files,
    copy_restart_artifacts,
)
from .dry_run import dry_run_reservoir_result
from .flow_runner import run_flow
from .models import ReservoirResult
from .opening_choke import co2_phase_split_from_ph
from .plotting import generate_coupling_figures
from .reporting import _safe_float, print_solver_iteration_block
from .summary_reader import _read_summary_or_prt
from .wellbore_factory import build_advanced_wellbore
from .thermal_coupling import ThermalUpdateState, propose_next_wtemp

def default_data_dir() -> Path:
    """Return repository-local data/ directory.

    File location:
        src/co2_wellbore/coupling/orchestrator.py

    parents[3] is the repository root:
        coupling -> co2_wellbore -> src -> repo_root
    """
    return Path(__file__).resolve().parents[3] / "data"


def resolve_base_dir(base_dir_arg: object) -> Path:
    """Resolve user-provided base-dir or fall back to repo data/."""
    if base_dir_arg is None or str(base_dir_arg).strip() == "":
        return default_data_dir().resolve()

    return Path(str(base_dir_arg)).expanduser().resolve()

def run_orchestrator(args: argparse.Namespace) -> Path:
    base_dir = resolve_base_dir(args.base_dir)
    deck_path = (base_dir / args.deck).resolve() if not Path(args.deck).is_absolute() else Path(args.deck).resolve()
    if not deck_path.exists():
        raise FileNotFoundError(f"Deck not found: {deck_path}")

    base_text = read_text(deck_path)
    _, _, _, schedule = split_deck_sections(base_text)
    static_well_setup = extract_static_well_setup(schedule)
    producer_control = extract_base_producer_control(schedule)

    wb = build_advanced_wellbore(args)

    wb_boundary_kwargs: Dict[str, float] = {}
    if args.wellhead_enthalpy_J_kg is not None:
        wb_boundary_kwargs["wellhead_enthalpy_J_kg"] = float(args.wellhead_enthalpy_J_kg)
    if args.wellhead_quality_mass is not None:
        wb_boundary_kwargs["wellhead_quality_mass"] = float(args.wellhead_quality_mass)

    q_from_deck = parse_first_wconinje_rate(base_text, args.well_name, args.rate_sm3_day)
    wtemp_from_deck = parse_first_wtemp(base_text, args.well_name, args.initial_wtemp_C)
    tsteps = parse_tstep_list(base_text, args.step_days, args.n_steps)

    rate_schedule = [float(x) for x in args.rate_schedule] if args.rate_schedule else [float(q_from_deck)] * int(args.n_steps)
    if len(rate_schedule) < int(args.n_steps):
        rate_schedule += [rate_schedule[-1]] * (int(args.n_steps) - len(rate_schedule))

    pressure_mode = str(args.pressure_control_mode).lower()
    if pressure_mode not in {"fixed_rate", "thp_limit", "opening_choke"}:
        raise ValueError("pressure_control_mode must be fixed_rate, thp_limit or opening_choke")

    out_root = (base_dir / args.output_root).resolve()
    case_dir = out_root / args.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    iterations_live_csv = case_dir / "coupled_exchange_iterations_LIVE.csv"
    accepted_live_csv = case_dir / "coupled_exchange_accepted_LIVE.csv"
    status_live_csv = case_dir / "coupling_status_LIVE.csv"
    progress_log = case_dir / "RUN_PROGRESS.log"

    def append_status(event: str, **kwargs: Any) -> None:
        row = {"wall_time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kwargs}
        pd.DataFrame([row]).to_csv(status_live_csv, mode="a", header=not status_live_csv.exists(), index=False)
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    append_status(
        "case_initialized",
        case_dir=str(case_dir),
        deck=str(deck_path),
        n_steps=int(args.n_steps),
        pressure_control_mode=pressure_mode,
        max_coupling_iter=int(args.max_coupling_iter),
    )
    print(
        f"[CASE] {args.case_id} | mode={pressure_mode} | "
        f"steps={int(args.n_steps)} | log=normal",
        flush=True,
    )
    print("", flush=True)

    restart = RestartPointer()
    accepted_rows: List[Dict[str, object]] = []
    iteration_rows: List[Dict[str, object]] = []
    accepted_profiles: List[pd.DataFrame] = []
    prev_cum = 0.0
    cumulative_days = 0.0
    last_accepted_bht = float(wtemp_from_deck if args.use_deck_wtemp_first else args.initial_wtemp_C)

    # For pressure-controlled modes, use the rate schedule as a demand/cap unless the user disables it.
    use_rate_schedule_as_cap = bool(args.rate_schedule_is_cap)

    for k in range(1, int(args.n_steps) + 1):
        step_days = float(tsteps[k - 1] if k - 1 < len(tsteps) else args.step_days)
        cumulative_days += step_days

        # For layered/transient thermal model: current injection time controls
        # the formation heat-penetration radius used in the external wellbore.
        if hasattr(wb.thermal, "elapsed_time_days"):
            wb.thermal.elapsed_time_days = float(cumulative_days)

        q_demand = float(rate_schedule[k - 1])
        if args.opening_schedule_percent:
            if k - 1 < len(args.opening_schedule_percent):
                opening_percent = float(args.opening_schedule_percent[k - 1])
            else:
                opening_percent = float(args.opening_schedule_percent[-1])
        else:
            opening_percent = 20.0

        opening_fraction = opening_percent / 100.0

        if k == 1:
            wtemp_guess = float(wtemp_from_deck if args.use_deck_wtemp_first else args.initial_wtemp_C)
        else:
            wtemp_guess = float(last_accepted_bht if args.carry_wtemp_between_steps else args.initial_wtemp_C)

        # RATE search bracket for pressure-controlled modes.
        q_min = float(args.q_min_sm3_day)
        q_max_global = float(args.q_max_sm3_day)

        if pressure_mode == "opening_choke":
            # HEM has its own physical choke capacity, so do NOT use the old
            # empirical q-cap = 150000 * opening%. That old cap was only a
            # safety device for the legacy liquid-like Kv model.
            if str(getattr(args, "choke_model", "legacy_kv")).lower() == "hem":
                if accepted_rows:
                    previous_rate = float(accepted_rows[-1].get("reservoir_rate_sm3_day", 0.0) or 0.0)
                    previous_opening = float(accepted_rows[-1].get("opening_percent", 0.0) or 0.0)

                    if previous_rate > 0.0 and previous_opening > 0.0:
                        opening_based_q_cap = previous_rate * (float(opening_percent) / previous_opening) * 1.30
                    else:
                        opening_based_q_cap = q_max_global
                else:
                    opening_based_q_cap = q_max_global

                q_cap = min(q_max_global, opening_based_q_cap)

            else:
                # Legacy fluids/IEC Kv model: keep conservative continuation cap.
                if accepted_rows:
                    previous_rate = float(accepted_rows[-1].get("reservoir_rate_sm3_day", 0.0) or 0.0)
                    previous_opening = float(accepted_rows[-1].get("opening_percent", 0.0) or 0.0)

                    if previous_rate > 0.0 and previous_opening > 0.0:
                        ramp_q_cap = previous_rate * (float(opening_percent) / previous_opening) * 1.08
                    else:
                        ramp_q_cap = float("inf")

                    simple_q_cap = 120000.0 * float(opening_percent)
                    opening_based_q_cap = min(simple_q_cap, ramp_q_cap)
                else:
                    opening_based_q_cap = 150000.0 * float(opening_percent)

                q_cap = min(q_max_global, opening_based_q_cap)

        elif pressure_mode == "thp_limit" and use_rate_schedule_as_cap:
            q_cap = min(q_max_global, q_demand)
        else:
            q_cap = q_max_global

        q_cap = max(q_cap, q_min)
        q_lo, q_hi = q_min, q_cap
        # Residuals stored at the current RATE bracket endpoints.
        # For thp_limit they are THP_required - THP_limit.
        # Keeping these values lets us use a secant/regula-falsi step instead of pure bisection.
        q_lo_resid: Optional[float] = None
        q_hi_resid: Optional[float] = None
        q_target = q_demand if pressure_mode == "fixed_rate" else q_hi
        last_feasible_q: Optional[float] = None
        last_feasible_row: Optional[Dict[str, object]] = None
        last_feasible_profile: Optional[pd.DataFrame] = None
        last_feasible_state: Optional[RestartPointer] = None

        accepted_this_step = False
        accepted_state: Optional[RestartPointer] = None
        unsafe_high_trial_seen = False
        thermal_update_state = ThermalUpdateState()

        for it in range(1, int(args.max_coupling_iter) + 1):
            root = segment_root(k, it)
            seg_dir = case_dir / root
            seg_dir.mkdir(parents=True, exist_ok=True)
            copied: List[str] = []
            copied.extend(copy_base_input_files(base_dir, seg_dir))
            if restart.segment_dir and restart.root:
                copied.extend(copy_restart_artifacts(Path(restart.segment_dir), seg_dir, restart.root))
            # Restart already restores the simulator state at the accepted report step.
            # Do NOT replay previous TSTEPs in the restart segment.
            restart_alignment_tsteps: List[float] = []
            if restart.root is not None:
                restart_alignment_tsteps = [
                    float(tsteps[j] if j < len(tsteps) else args.step_days)
                    for j in range(k - 1)
                ]

            control = StepControl(
                step_index=k,
                iteration=it,
                step_days=step_days,
                rate_target_sm3_day=float(q_target),
                wtemp_used_C=float(wtemp_guess),
                bhp_limit_bar=float(args.bhp_limit_bar),
                restart_alignment_tsteps=restart_alignment_tsteps,
            )

            if str(getattr(args, "choke_model", "hem")).lower() == "hem":
                choke = CO2HEMChokeValve(
                    properties=wb.props,
                    config=HEMChokeConfig(
                        kv_full_m3_h=float(args.kv_full_m3_h),
                        full_diameter_m=float(args.valve_diameter_m),
                        opening_fraction=opening_fraction,
                        discharge_coefficient=float(args.hem_discharge_coefficient),
                        area_exponent=float(args.hem_area_exponent),
                        min_pressure_bar=max(float(args.hem_min_pressure_bar), float(args.thp_min_bar), 6.0),
                        n_pressure_samples=int(args.hem_pressure_samples),
                    ),
                )
            else:
                choke = CO2ChokeValve(
                    properties=wb.props,
                    config=ChokeConfig(
                        kv_full_m3_h=float(args.kv_full_m3_h),
                        opening_fraction=opening_fraction,
                        FL=float(args.valve_FL),
                        Fd=float(args.valve_Fd),
                        inlet_diameter_m=float(args.valve_diameter_m),
                        outlet_diameter_m=float(args.valve_diameter_m),
                        valve_diameter_m=float(args.valve_diameter_m),
                        min_downstream_pressure_bar=max(float(args.thp_min_bar), 6.0),
                    ),
                )

            upstream_enthalpy_J_kg = wb.props.enthalpy(
                float(args.upstream_pressure_bar) * BAR_TO_PA,
                float(args.upstream_temperature_C) + 273.15,
            )

            deck_text = build_segment_deck(
                base_deck_text=base_text,
                control=control,
                well_name=str(args.well_name),
                static_well_setup=static_well_setup,
                producer_control=producer_control,
                restart=restart,
            )
            data_path = seg_dir / f"{root}.DATA"
            data_path.write_text(deck_text, encoding="utf-8")

            append_status(
                "deck_written",
                step=k,
                iteration=it,
                root=root,
                seg_dir=str(seg_dir),
                data_file=str(data_path),
                q_target=float(q_target),
                q_lo=float(q_lo),
                q_hi=float(q_hi),
                wtemp_guess=float(wtemp_guess),
                step_days=float(step_days),
            )

            flow_ok = True
            flow_reason = "dry_run"
            flow_seconds = 0.0
            if args.run_flow and not args.dry_run:
                append_status("flow_start", step=k, iteration=it, root=root, seg_dir=str(seg_dir), log_file=str(seg_dir / "FLOW_STDOUT.log"))
                
                flow_extra_args: List[str] = []
                if bool(getattr(args, "enable_vtk_output", False)):
                    flow_extra_args.append("--enable-vtk-output=true")
                    flow_extra_args.append(f"--vtk-write-pressures={'true' if bool(args.vtk_write_pressures) else 'false'}")
                    flow_extra_args.append(f"--vtk-write-saturations={'true' if bool(args.vtk_write_saturations) else 'false'}")
                    flow_extra_args.append(f"--vtk-write-temperature={'true' if bool(args.vtk_write_temperature) else 'false'}")
                    # Keep other heavy arrays off by default.
                    flow_extra_args.append("--vtk-write-relative-permeabilities=false")
                    flow_extra_args.append("--vtk-write-porosity=false")
                    flow_extra_args.append("--vtk-write-total-mass-fractions=false")

                flow_ok, flow_reason, flow_seconds = run_flow(
                    flow_exe=str(args.flow_exe),
                    case_root=root,
                    cwd=seg_dir,
                    timeout_sec=float(args.flow_timeout_sec),
                    mpi_nproc=int(args.mpi_nproc),
                    flow_args=flow_extra_args,
                )
                append_status("flow_finished", step=k, iteration=it, root=root, flow_ok=bool(flow_ok), flow_reason=flow_reason, flow_seconds=float(flow_seconds))
                #print(f"[FLOW END] step={k} it={it} root={root} ok={flow_ok} reason={flow_reason} seconds={flow_seconds:.2f}", flush=True)
                if not flow_ok:
                    raise RuntimeError(f"OPM Flow failed in {seg_dir}: {flow_reason}. See FLOW_STDOUT.log")
                res = _read_summary_or_prt(seg_dir / root, args.well_name, control, prev_cum, cumulative_days)
                if res is None:
                    raise RuntimeError(
                        "OPM run finished but neither summary nor PRT report could be read. "
                        "Check that SEG_XXXX_ITYY.PRT exists and contains the INJECTION REPORT."
                    )
                append_status(
                    "reservoir_result_read",
                    step=k,
                    iteration=it,
                    root=root,
                    source=res.source,
                    rate_sm3_day=float(res.rate_sm3_day),
                    bhp_bar=float(res.bhp_bar),
                    thp_bar=float(res.thp_bar) if math.isfinite(float(res.thp_bar)) else "",
                )
                #print(f"[READ] step={k} it={it} source={res.source} q={res.rate_sm3_day:.3f} WBHP={res.bhp_bar:.3f} bar", flush=True)
            else:
                res = dry_run_reservoir_result(
                    wb=wb,
                    control=control,
                    t_head_C=float(args.t_head_C),
                    thp_reference_bar=float(args.reference_thp_bar),
                    prev_cum=prev_cum,
                    cumulative_days=cumulative_days,
                    **wb_boundary_kwargs,
                )

            q_actual = float(res.rate_sm3_day)
            bhp_opm = float(res.bhp_bar)

            mass_rate_kg_s = wb.mass_rate_from_sm3_day(q_actual)

            choke_model_name = str(getattr(args, "choke_model", "legacy_kv")).lower()
            choke_capacity_kg_s = float("nan")
            choke_capacity_residual_kg_s = float("nan")
            choke_flow_regime = ""
            choke_selected_throat_pressure_bar = float("nan")
            choke_selected_throat_temperature_C = float("nan")
            choke_selected_throat_quality_mass = float("nan")
            choke_selected_mass_flux_kg_m2_s = float("nan")
            choke_effective_area_m2 = float("nan")
            choke_discharge_coefficient = float("nan")

            if pressure_mode == "opening_choke":
                try:
                    thp_calc = wb.required_thp_for_target_bhp(
                        target_bhp_bar=bhp_opm,
                        q_sm3_day=q_actual,
                        wellhead_temperature_C=float(args.t_head_C),
                        wellhead_enthalpy_J_kg=upstream_enthalpy_J_kg,
                        thp_bounds_bar=(float(args.thp_min_bar), float(args.thp_max_bar)),
                        tol_bar=float(args.p_tolerance_bar),
                        max_iter=int(args.inner_solver_max_iter),
                    )

                    if thp_calc >= float(args.upstream_pressure_bar) * 0.999:
                        kv_required = float("inf")
                        valve_resid = float("inf")
                        choke_capacity_kg_s = 0.0
                        choke_capacity_residual_kg_s = float("inf")
                        state = {
                            "downstream_pressure_bar": float(thp_calc),
                            "downstream_temperature_C": float("nan"),
                            "quality_mass": float("nan"),
                            "phase_label": "infeasible_thp_above_upstream",
                        }
                    else:
                        state = choke.downstream_state_at_pressure(
                            upstream_pressure_bar=float(args.upstream_pressure_bar),
                            upstream_temperature_C=float(args.upstream_temperature_C),
                            downstream_pressure_bar=float(thp_calc),
                            mass_rate_kg_s=mass_rate_kg_s,
                        )

                        if choke_model_name == "hem":
                            choke_capacity_kg_s = float(state.get("mass_flow_capacity_kg_s", float("nan")))
                            choke_capacity_residual_kg_s = float(mass_rate_kg_s) - float(choke_capacity_kg_s)

                            # In HEM mode the pressure residual is physical mass-flow residual.
                            # Do not use equivalent Kv for convergence or rate bracketing.
                            kv_required = float("nan")
                            valve_resid = float(choke_capacity_residual_kg_s)

                            choke_flow_regime = str(state.get("flow_regime", ""))
                            choke_selected_throat_pressure_bar = float(state.get("selected_throat_pressure_bar", float("nan")))
                            choke_selected_throat_temperature_C = float(state.get("selected_throat_temperature_C", float("nan")))
                            choke_selected_throat_quality_mass = float(state.get("selected_throat_quality_mass", float("nan")))
                            choke_selected_mass_flux_kg_m2_s = float(state.get("selected_mass_flux_kg_m2_s", float("nan")))
                            choke_effective_area_m2 = float(state.get("effective_area_m2", float("nan")))
                            choke_discharge_coefficient = float(state.get("discharge_coefficient", float("nan")))
                        else:
                            kv_required = choke.required_kv_liquid_like(
                                upstream_pressure_bar=float(args.upstream_pressure_bar),
                                downstream_pressure_bar=float(thp_calc),
                                upstream_temperature_C=float(args.upstream_temperature_C),
                                mass_rate_kg_s=mass_rate_kg_s,
                            )

                            valve_resid = kv_required - choke.config.effective_kv_m3_h

                    profile = wb.profile_from_thp_and_rate(
                        thp_bar=float(thp_calc),
                        q_sm3_day=q_actual,
                        wellhead_temperature_C=float(args.t_head_C),
                        wellhead_enthalpy_J_kg=upstream_enthalpy_J_kg,
                    )

                    inversion_status = "ok"

                except Exception as exc:
                    # Robust nonlinear-solver behavior:
                    # Do not abort on an unsafe high-rate trial. Treat it as
                    # "rate too high", shrink the upper bracket, and continue.
                    #
                    # This typically happens when the wellbore inversion tries
                    # a low-THP CO2 P-H flash outside the stable property region.
                    current_q = float(q_target)
                    unsafe_high_trial_seen = True
                    q_hi = min(q_hi, current_q)
                    q_hi_resid = None

                    if q_hi <= q_lo + float(args.q_tolerance_sm3_day):
                        q_target = 0.5 * (q_lo + q_hi)
                    else:
                        q_target = 0.5 * (q_lo + q_hi)

                    append_status(
                        "opening_choke_unsafe_trial_shrink",
                        step=k,
                        iteration=it,
                        root=root,
                        q_actual=float(q_actual),
                        q_target_old=float(current_q),
                        q_lo=float(q_lo),
                        q_hi=float(q_hi),
                        q_target_next=float(q_target),
                        wbhp_opm_bar=float(bhp_opm),
                        error=str(exc)[:1000],
                    )

                    print()
                    print("=" * 96)
                    print("OPENING-CHOKE UNSAFE TRIAL: SHRINK RATE BRACKET")
                    print("-" * 96)
                    print(f"Step / Iteration     : {k} / {it}")
                    print(f"Opening              : {opening_percent:.3f} %")
                    print(f"Trial q              : {current_q:.3f} sm3/day")
                    print(f"OPM actual q         : {q_actual:.3f} sm3/day")
                    print(f"OPM WBHP             : {bhp_opm:.3f} bar")
                    print("Reason               : wellbore P-H inversion failed for this trial rate")
                    print("Action               : treat trial as too high and reduce upper q bracket")
                    print(f"New q bracket         : [{q_lo:.3f}, {q_hi:.3f}] sm3/day")
                    print(f"Next q trial          : {q_target:.3f} sm3/day")
                    print(f"WTEMP kept            : {wtemp_guess:.3f} C")
                    print("-" * 96)
                    print(f"Original error        : {str(exc)[:500]}")
                    print("=" * 96)
                    print()

                    continue

            else:
                # Original modes: fixed_rate and thp_limit.
                try:
                    thp_calc = wb.required_thp_for_target_bhp(
                        target_bhp_bar=bhp_opm,
                        q_sm3_day=q_actual,
                        wellhead_temperature_C=float(args.t_head_C),
                        thp_bounds_bar=(float(args.thp_min_bar), float(args.thp_max_bar)),
                        tol_bar=float(args.p_tolerance_bar),
                        max_iter=int(args.inner_solver_max_iter),
                        **wb_boundary_kwargs,
                    )
                    inversion_status = "ok"

                    profile = wb.profile_from_thp_and_rate(
                        thp_bar=float(thp_calc),
                        q_sm3_day=q_actual,
                        wellhead_temperature_C=float(args.t_head_C),
                        **wb_boundary_kwargs,
                    )

                except Exception as exc:
                    raise RuntimeError(
                        f"Wellbore P-H flash/inversion failed at step={k}, iteration={it}, "
                        f"q={q_actual:.6g} sm3/day, WBHP={bhp_opm:.6g} bar. "
                        "Do not use reference THP fallback for science runs. "
                        "If the inlet is near saturation, provide --wellhead-enthalpy-J-kg "
                        "or --wellhead-quality-mass."
                    ) from exc

                kv_required = float("nan")
                valve_resid = float("nan")
                state = {
                    "downstream_pressure_bar": float("nan"),
                    "downstream_temperature_C": float("nan"),
                    "quality_mass": float("nan"),
                    "phase_label": "",
                }

            bhp_calc = float(profile["pressure_bar"].iloc[-1])
            bht_calc = float(profile["temperature_C"].iloc[-1])
            inversion_resid = bhp_calc - bhp_opm
            temp_resid = bht_calc - float(wtemp_guess)
            rate_resid = q_actual - float(q_target)

            phase_split_fields: Dict[str, object] = {}
            if pressure_mode == "opening_choke":
                phase_split_fields.update(
                    co2_phase_split_from_ph(
                        wb.props,
                        pressure_bar=float(args.upstream_pressure_bar),
                        enthalpy_J_kg=float(upstream_enthalpy_J_kg),
                        mass_rate_kg_s=float(mass_rate_kg_s),
                        prefix="choke_upstream",
                    )
                )
                phase_split_fields.update(
                    co2_phase_split_from_ph(
                        wb.props,
                        pressure_bar=float(thp_calc),
                        enthalpy_J_kg=float(upstream_enthalpy_J_kg),
                        mass_rate_kg_s=float(mass_rate_kg_s),
                        prefix="choke_downstream",
                    )
                )

            thp_target_bar = float(args.thp_target_bar)
            thp_limit_resid = thp_calc - thp_target_bar
            thp_margin_bar = thp_target_bar - thp_calc

            # Decide whether the pressure/surface boundary is satisfied.
            if pressure_mode == "fixed_rate":
                pressure_resid_for_convergence = inversion_resid
                pressure_converged = abs(inversion_resid) <= float(args.p_tolerance_bar)
            elif pressure_mode == "thp_limit":
                pressure_resid_for_convergence = thp_limit_resid
                demand_feasible = bool(
                    use_rate_schedule_as_cap
                    and abs(float(q_target) - q_cap) <= float(args.q_tolerance_sm3_day)
                    and thp_limit_resid <= 0.0
                )
                pressure_converged = demand_feasible or (
                    thp_limit_resid <= 0.0
                    and abs(thp_limit_resid) <= float(args.thp_tolerance_bar)
                )

            elif pressure_mode == "opening_choke":
                if choke_model_name == "hem":
                    pressure_resid_for_convergence = choke_capacity_residual_kg_s
                    pressure_converged = (
                        math.isfinite(float(choke_capacity_residual_kg_s))
                        and abs(float(choke_capacity_residual_kg_s)) <= float(args.hem_flow_tolerance_kg_s)
                    )
                else:
                    pressure_resid_for_convergence = valve_resid
                    pressure_converged = abs(valve_resid) <= float(args.kv_tolerance_m3_h)

            temperature_converged = abs(temp_resid) <= float(args.t_tolerance_C)
            rate_converged = abs(rate_resid) <= float(args.q_tolerance_sm3_day)

            append_status(
                "wellbore_calculated",
                step=k,
                iteration=it,
                root=root,
                mode=pressure_mode,
                q_target=float(q_target),
                q_actual=float(q_actual),
                wbhp_opm_bar=float(bhp_opm),
                thp_required_bar=float(thp_calc),
                opening_percent=float(opening_percent),
                choke_kv_effective_m3_h=float(choke.config.effective_kv_m3_h),
                choke_kv_required_m3_h=float(kv_required) if choke_model_name != "hem" and math.isfinite(float(kv_required)) else "",
                choke_kv_residual_m3_h=float(valve_resid) if choke_model_name != "hem" and math.isfinite(float(valve_resid)) else "",
                choke_mass_rate_actual_kg_s=float(mass_rate_kg_s),
                choke_mass_flow_capacity_kg_s=float(choke_capacity_kg_s) if math.isfinite(float(choke_capacity_kg_s)) else "",
                choke_capacity_residual_kg_s=float(choke_capacity_residual_kg_s) if math.isfinite(float(choke_capacity_residual_kg_s)) else "",
                choke_flow_regime=str(choke_flow_regime),
                choke_selected_throat_pressure_bar=float(choke_selected_throat_pressure_bar) if math.isfinite(float(choke_selected_throat_pressure_bar)) else "",
                choke_selected_throat_temperature_C=float(choke_selected_throat_temperature_C) if math.isfinite(float(choke_selected_throat_temperature_C)) else "",
                choke_selected_throat_quality_mass=float(choke_selected_throat_quality_mass) if math.isfinite(float(choke_selected_throat_quality_mass)) else "",
                choke_selected_mass_flux_kg_m2_s=float(choke_selected_mass_flux_kg_m2_s) if math.isfinite(float(choke_selected_mass_flux_kg_m2_s)) else "",
                choke_effective_area_m2=float(choke_effective_area_m2) if math.isfinite(float(choke_effective_area_m2)) else "",
                choke_discharge_coefficient=float(choke_discharge_coefficient) if math.isfinite(float(choke_discharge_coefficient)) else "",
                choke_downstream_temperature_C=float(state["downstream_temperature_C"]) if isinstance(state, dict) and math.isfinite(float(state["downstream_temperature_C"])) else "",
                choke_quality_mass=float(state["quality_mass"]) if isinstance(state, dict) and math.isfinite(float(state["quality_mass"])) else "",
                choke_phase_label=str(state["phase_label"]) if isinstance(state, dict) else "",
                thp_target_bar=float(thp_target_bar),
                thp_limit_resid_bar=float(thp_limit_resid),
                bhp_calc_bar=float(bhp_calc),
                bht_calc_C=float(bht_calc),
                temp_resid_C=float(temp_resid),
                inversion_resid_bar=float(inversion_resid),
                rate_resid_sm3_day=float(rate_resid),
                inversion_status=inversion_status,
            )
            t2_report = (
                float(state["downstream_temperature_C"])
                if isinstance(state, dict) and math.isfinite(float(state["downstream_temperature_C"]))
                else float("nan")
            )
            quality_report = (
                float(state["quality_mass"])
                if isinstance(state, dict) and math.isfinite(float(state["quality_mass"]))
                else float("nan")
            )
            phase_report = str(state["phase_label"]) if isinstance(state, dict) else ""

            print_solver_iteration_block(
                step=k,
                max_steps=int(args.n_steps),
                iteration=it,
                max_iter=int(args.max_coupling_iter),
                mode=pressure_mode,
                root=root,
                flow_ok=bool(flow_ok),
                flow_seconds=float(flow_seconds),
                reservoir_source=str(res.source),
                q_target=float(q_target),
                q_actual=float(q_actual),
                q_lo=float(q_lo),
                q_hi=float(q_hi),
                opening_percent=float(opening_percent),
                kv_eff=float(choke.config.effective_kv_m3_h),
                kv_required=float(kv_required),
                valve_resid=float(valve_resid),
                kv_tol=float(args.kv_tolerance_m3_h),
                bhp_opm=float(bhp_opm),
                thp_calc=float(thp_calc),
                bhp_calc=float(bhp_calc),
                inversion_resid=float(inversion_resid),
                p_tol=float(args.p_tolerance_bar),
                wtemp_used=float(wtemp_guess),
                bht_calc=float(bht_calc),
                temp_resid=float(temp_resid),
                t_tol=float(args.t_tolerance_C),
                rate_resid=float(rate_resid),
                q_tol=float(args.q_tolerance_sm3_day),
                t2=t2_report,
                quality=quality_report,
                phase=phase_report,
                pressure_converged=bool(pressure_converged),
                temperature_converged=bool(temperature_converged),
                rate_converged=bool(rate_converged),
                choke_model=str(getattr(args, "choke_model", "legacy_kv")),
                choke_capacity_kg_s=float(choke_capacity_kg_s),
                choke_actual_mass_rate_kg_s=float(mass_rate_kg_s),
                choke_capacity_residual_kg_s=float(choke_capacity_residual_kg_s),
                hem_flow_tolerance_kg_s=float(args.hem_flow_tolerance_kg_s),
                flow_regime=str(choke_flow_regime),
            )

            wtemp_next, thermal_update_info = propose_next_wtemp(
                state=thermal_update_state,
                current_wtemp_C=float(wtemp_guess),
                residual_C=float(temp_resid),
                relaxation=float(args.temperature_relaxation),
                max_step_C=float(args.wtemp_max_step_C),
                min_wtemp_C=float(args.wtemp_min_C),
                max_wtemp_C=float(args.wtemp_max_C),
            )

            row: Dict[str, object] = {
                "step": k,
                "iteration": it,
                "accepted": False,
                "accepted_reason": "",
                "pressure_control_mode": pressure_mode,
                "choke_model": str(getattr(args, "choke_model", "legacy_kv")),
                "segment_root": root,
                "segment_dir": str(seg_dir),
                "restart_from_root": restart.root or "",
                "restart_from_report_step": restart.report_step if restart.report_step is not None else "",
                "restart_files_copied": ";".join(copied),
                "q_search_lower_sm3_day": float(q_lo),
                "q_search_upper_sm3_day": float(q_hi),
                "q_demand_sm3_day": float(q_demand),
                "q_cap_sm3_day": float(q_cap),
                "thp_target_bar": float(thp_target_bar),
                "opening_percent": float(opening_percent),
                "opening_fraction": float(opening_fraction),
                "choke_kv_full_m3_h": float(args.kv_full_m3_h),
                "choke_kv_effective_m3_h": float(choke.config.effective_kv_m3_h),
                "choke_kv_required_m3_h": float(kv_required) if choke_model_name != "hem" and math.isfinite(float(kv_required)) else "",
                "choke_kv_residual_m3_h": float(valve_resid) if choke_model_name != "hem" and math.isfinite(float(valve_resid)) else "",
                "choke_mass_rate_actual_kg_s": float(mass_rate_kg_s),
                "choke_mass_flow_capacity_kg_s": float(choke_capacity_kg_s) if math.isfinite(float(choke_capacity_kg_s)) else "",
                "choke_capacity_residual_kg_s": float(choke_capacity_residual_kg_s) if math.isfinite(float(choke_capacity_residual_kg_s)) else "",
                "choke_flow_regime": str(choke_flow_regime),
                "choke_selected_throat_pressure_bar": float(choke_selected_throat_pressure_bar) if math.isfinite(float(choke_selected_throat_pressure_bar)) else "",
                "choke_selected_throat_temperature_C": float(choke_selected_throat_temperature_C) if math.isfinite(float(choke_selected_throat_temperature_C)) else "",
                "choke_selected_throat_quality_mass": float(choke_selected_throat_quality_mass) if math.isfinite(float(choke_selected_throat_quality_mass)) else "",
                "choke_selected_mass_flux_kg_m2_s": float(choke_selected_mass_flux_kg_m2_s) if math.isfinite(float(choke_selected_mass_flux_kg_m2_s)) else "",
                "choke_effective_area_m2": float(choke_effective_area_m2) if math.isfinite(float(choke_effective_area_m2)) else "",
                "choke_discharge_coefficient": float(choke_discharge_coefficient) if math.isfinite(float(choke_discharge_coefficient)) else "",
                "choke_upstream_pressure_bar": float(args.upstream_pressure_bar),
                "choke_upstream_temperature_C": float(args.upstream_temperature_C),
                "choke_downstream_pressure_bar": float(thp_calc),
                "choke_downstream_temperature_C": float(state["downstream_temperature_C"]) if isinstance(state, dict) and math.isfinite(float(state["downstream_temperature_C"])) else "",
                "choke_quality_mass": float(state["quality_mass"]) if isinstance(state, dict) and math.isfinite(float(state["quality_mass"])) else "",
                "choke_phase_label": str(state["phase_label"]) if isinstance(state, dict) else "",
                **phase_split_fields,
                **{f"control_{key}": val for key, val in asdict(control).items()},
                **{f"reservoir_{key}": val for key, val in asdict(res).items()},
                "wellbore_required_thp_bar": float(thp_calc),
                "wellbore_bhp_calc_bar": float(bhp_calc),
                "wellbore_bht_calc_C": float(bht_calc),
                "wellbore_bottom_density_kg_m3": float(profile["density_kg_m3"].iloc[-1]),
                "wellbore_bottom_phase_label": str(profile["phase_label"].iloc[-1]),
                "wellbore_inversion_residual_bar": float(inversion_resid),
                "residual_pressure_calc_minus_opm_bar": float(inversion_resid),
                "thp_required_minus_target_bar": float(thp_limit_resid),
                "thp_margin_bar": float(thp_margin_bar),
                "thp_limit_feasible": bool(thp_limit_resid <= 0.0),
                "pressure_search_width_sm3_day": float(abs(q_hi - q_lo)),
                "pressure_residual_for_convergence": float(pressure_resid_for_convergence) if math.isfinite(float(pressure_resid_for_convergence)) else "",
                "residual_temperature_bht_minus_wtemp_C": float(temp_resid),
                "thermal_update_method": "",
                "thermal_wtemp_next_C": "",
                "thermal_wtemp_delta_C": "",
                **thermal_update_info,
                "residual_rate_actual_minus_target_sm3_day": float(rate_resid),
                "pressure_converged": bool(pressure_converged),
                "temperature_converged": bool(temperature_converged),
                "rate_converged": bool(rate_converged),
                "inversion_status": inversion_status,
                "flow_ok": bool(flow_ok),
                "flow_reason": flow_reason,
                "flow_seconds": float(flow_seconds),
            }

            # Store the latest feasible pressure-controlled row as a fallback candidate.
            if (
                pressure_mode == "thp_limit"
                and inversion_status == "ok"
                and thp_limit_resid <= 0.0
                and rate_converged
            ):
                last_feasible_q = float(q_target)
                last_feasible_row = dict(row)
                last_feasible_profile = profile.copy()
                last_feasible_state = RestartPointer(segment_dir=str(seg_dir), root=root, report_step=k)

            converged = bool(pressure_converged and temperature_converged and rate_converged)

            capacity_surplus_accept = bool(
                pressure_mode == "opening_choke"
                and bool(getattr(args, "accept_capacity_surplus", True))
                and math.isfinite(float(pressure_resid_for_convergence))
                and float(pressure_resid_for_convergence) < 0.0
                and bool(temperature_converged)
                and bool(rate_converged)
                and (
                    bool(unsafe_high_trial_seen)
                    or abs(float(q_hi) - float(q_lo)) <= float(getattr(args, "capacity_surplus_q_width_sm3_day", 5000.0))
                )
            )

            row["capacity_surplus_accept"] = bool(capacity_surplus_accept)
            row["unsafe_high_trial_seen"] = bool(unsafe_high_trial_seen)
            row["active_constraint"] = (
                "choke_capacity"
                if converged
                else ("wellbore_or_property_limit" if capacity_surplus_accept else "")
            )

            force_accept = bool(args.allow_force_accept and it == int(args.max_coupling_iter))

            if converged or capacity_surplus_accept or force_accept:
                row["accepted"] = True
                if converged:
                    row["accepted_reason"] = "CONVERGED"
                elif capacity_surplus_accept:
                    row["accepted_reason"] = "ACCEPTED_CAPACITY_SURPLUS"
                else:
                    row["accepted_reason"] = "ACCEPTED_MAX_ITER"
                accepted_this_step = True
                prev_cum = float(res.cum_gas_sm3) if math.isfinite(float(res.cum_gas_sm3)) else prev_cum
                last_accepted_bht = bht_calc
                accepted_state = RestartPointer(segment_dir=str(seg_dir), root=root, report_step=k)
                profile.insert(0, "step", k)
                profile.insert(1, "accepted_iteration", it)
                profile.insert(2, "time_days", float(res.time_days))
                accepted_profiles.append(profile)
                accepted_rows.append(row)
                iteration_rows.append(row)

                pd.DataFrame(iteration_rows).to_csv(iterations_live_csv, index=False)
                pd.DataFrame(accepted_rows).to_csv(accepted_live_csv, index=False)
                (seg_dir / f"exchange_step_{k:04d}_it{it:02d}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                print(
                    f"[ACCEPT {k:02d}/{int(args.n_steps):02d}] "
                    f"it={it:02d} {row['accepted_reason']} | "
                    f"open={opening_percent:.1f}% "
                    f"q={q_actual:.0f} sm3/d | "
                    f"WBHP={bhp_opm:.1f} "
                    f"THP={thp_calc:.2f} | "
                    f"WTEMP={wtemp_guess:.2f}C "
                    f"BHT={bht_calc:.2f}C",
                    flush=True,
                )
                break

            iteration_rows.append(row)
            pd.DataFrame(iteration_rows).to_csv(iterations_live_csv, index=False)
            (seg_dir / f"exchange_step_{k:04d}_it{it:02d}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

            # Update pressure-search bracket for the next restart evaluation.
            if pressure_mode == "fixed_rate":
                q_target = q_demand
            elif pressure_mode == "thp_limit":
                current_q = float(q_target)
                if thp_limit_resid > 0.0:
                    q_hi = min(q_hi, current_q)
                    q_hi_resid = float(thp_limit_resid)
                else:
                    q_lo = max(q_lo, current_q)
                    q_lo_resid = float(thp_limit_resid)

                # If demand is already feasible, keep the demanded rate and only iterate temperature.
                if use_rate_schedule_as_cap and abs(current_q - q_cap) <= float(args.q_tolerance_sm3_day) and thp_limit_resid <= 0.0:
                    q_target = current_q
                else:
                    # Prefer a false-position/secant step once both endpoint residuals are known.
                    if (
                        q_lo_resid is not None
                        and q_hi_resid is not None
                        and q_lo_resid * q_hi_resid <= 0.0
                        and abs(q_hi_resid - q_lo_resid) > 1.0e-12
                    ):
                        q_next = q_lo + (0.0 - q_lo_resid) * (q_hi - q_lo) / (q_hi_resid - q_lo_resid)
                        # Keep the next trial inside the bracket; avoid landing exactly on an endpoint.
                        width = max(q_hi - q_lo, 0.0)
                        if width > float(args.q_tolerance_sm3_day):
                            eps = 0.05 * width
                            q_target = min(max(q_next, q_lo + eps), q_hi - eps)
                        else:
                            q_target = 0.5 * (q_lo + q_hi)
                    else:
                        q_target = 0.5 * (q_lo + q_hi)

            elif pressure_mode == "opening_choke":
                current_q = float(q_target)

                current_pressure_resid = float(pressure_resid_for_convergence)

                if current_pressure_resid > 0.0:
                    # HEM: actual mass rate exceeds capacity.
                    # Legacy: required Kv is larger than available Kv.
                    # In both cases current rate is too high for this opening.
                    q_hi = min(q_hi, current_q)
                    q_hi_resid = float(current_pressure_resid)
                else:
                    # HEM: choke can pass more flow.
                    # Legacy: Kv_required < Kv_effective.
                    q_lo = max(q_lo, current_q)
                    q_lo_resid = float(current_pressure_resid)

                if (
                    q_lo_resid is not None
                    and q_hi_resid is not None
                    and q_lo_resid * q_hi_resid <= 0.0
                    and abs(q_hi_resid - q_lo_resid) > 1.0e-12
                ):
                    q_next = q_lo + (0.0 - q_lo_resid) * (q_hi - q_lo) / (q_hi_resid - q_lo_resid)
                    width = max(q_hi - q_lo, 0.0)
                    if width > float(args.q_tolerance_sm3_day):
                        eps = 0.05 * width
                        q_target = min(max(q_next, q_lo + eps), q_hi - eps)
                    else:
                        q_target = 0.5 * (q_lo + q_hi)
                else:
                    q_target = 0.5 * (q_lo + q_hi)
            wtemp_guess = float(wtemp_next)

        # Optional fallback: if max iterations were reached and force-accept is disabled,
        # accept the last feasible pressure-controlled row only if explicitly allowed.
        if not accepted_this_step and bool(args.accept_last_feasible) and last_feasible_row and last_feasible_profile is not None and last_feasible_state is not None:
            last_feasible_temp_resid = abs(float(last_feasible_row.get("residual_temperature_bht_minus_wtemp_C", 1.0e99)))
            if last_feasible_temp_resid > float(args.t_tolerance_C):
                raise RuntimeError(
                    f"Step {k} found a pressure-feasible point, but WTEMP/BHT did not converge: "
                    f"|dT|={last_feasible_temp_resid:.6g} C > tolerance {float(args.t_tolerance_C):.6g} C. "
                    "Increase --max-coupling-iter or relax --t-tolerance-C."
                )
            last_feasible_row["accepted"] = True
            if pressure_mode == "thp_limit":
                last_feasible_row["accepted_reason"] = "ACCEPTED_LAST_FEASIBLE_THP_LIMIT"
            else:
                last_feasible_row["accepted_reason"] = "ACCEPTED_LAST_FEASIBLE_PRESSURE"
            accepted_this_step = True
            accepted_state = last_feasible_state
            prev_cum = float(last_feasible_row.get("reservoir_cum_gas_sm3", prev_cum) or prev_cum)
            last_accepted_bht = float(last_feasible_row.get("wellbore_bht_calc_C", last_accepted_bht))
            last_feasible_profile.insert(0, "step", k)
            last_feasible_profile.insert(1, "accepted_iteration", int(last_feasible_row.get("iteration", -1)))
            last_feasible_profile.insert(2, "time_days", float(last_feasible_row.get("reservoir_time_days", cumulative_days)))
            accepted_profiles.append(last_feasible_profile)
            accepted_rows.append(last_feasible_row)
            pd.DataFrame(accepted_rows).to_csv(accepted_live_csv, index=False)
            pd.DataFrame(iteration_rows).to_csv(iterations_live_csv, index=False)
            print(
                f"[{last_feasible_row['accepted_reason']}] step={k} "
                f"q={float(last_feasible_q):.3f} sm3/d "
                f"THP_req={float(last_feasible_row.get('wellbore_required_thp_bar', float('nan'))):.3f} bar",
                flush=True,
            )

        if not accepted_this_step or accepted_state is None:
            raise RuntimeError(f"Step {k} was not accepted. Increase --max-coupling-iter or relax tolerances.")
        restart = accepted_state

    accepted_df = pd.DataFrame(accepted_rows)
    iterations_df = pd.DataFrame(iteration_rows)
    profiles_df = pd.concat(accepted_profiles, ignore_index=True) if accepted_profiles else pd.DataFrame()

    accepted_csv = case_dir / "coupled_exchange_accepted.csv"
    iterations_csv = case_dir / "coupled_exchange_iterations.csv"
    profiles_csv = case_dir / "wellbore_profiles_accepted.csv"
    meta_json = case_dir / "coupling_metadata.json"
    accepted_df.to_csv(accepted_csv, index=False)
    iterations_df.to_csv(iterations_csv, index=False)
    profiles_df.to_csv(profiles_csv, index=False)

    figures_dir = Path(args.figures_dir).expanduser()
    if not figures_dir.is_absolute():
        figures_dir = (Path.cwd() / figures_dir).resolve()
    figure_paths = generate_coupling_figures(
        accepted_df=accepted_df,
        iterations_df=iterations_df,
        profiles_df=profiles_df,
        figures_dir=figures_dir,
        case_id=str(args.case_id),
    )

    print(f"[OK] Figures directory: {figures_dir}")
    for fig_path in figure_paths:
        print(f"[FIGURE] {fig_path}")

    meta = {
        "base_dir": str(base_dir),
        "deck": str(deck_path),
        "case_dir": str(case_dir),
        "algorithm": "OPM RATE-restart residual evaluations + CoolProp/ManyWells external wellbore + THP/valve pressure-control loop",
        "pressure_control_mode": pressure_mode,
        "well_name": str(args.well_name),
        "n_steps": int(args.n_steps),
        "max_coupling_iter": int(args.max_coupling_iter),
        "accepted_csv": str(accepted_csv),
        "iterations_csv": str(iterations_csv),
        "profiles_csv": str(profiles_csv),
        "figures_dir": str(figures_dir),
        "figures": [str(p) for p in figure_paths],
        "wellbore_settings": wb.config_dict(),
        "control_settings": {
            "thp_target_bar": float(args.thp_target_bar),
            "q_min_sm3_day": float(args.q_min_sm3_day),
            "q_max_sm3_day": float(args.q_max_sm3_day),
            "rate_schedule_is_cap": bool(args.rate_schedule_is_cap),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = case_dir / "README_RUN.txt"
    readme.write_text(
        "CO2 restart-coupled CoolProp/ManyWells run folder\n"
        "=================================================\n\n"
        f"Base deck: {deck_path}\n"
        f"Case directory: {case_dir}\n"
        f"Pressure-control mode: {pressure_mode}\n\n"
        "Important outputs:\n"
        f"- Accepted coupling exchange: {accepted_csv}\n"
        f"- All coupling iterations: {iterations_csv}\n"
        f"- Accepted wellbore profiles: {profiles_csv}\n\n"
        "Meaning:\n"
        "- reservoir_bhp_bar is OPM WBHP from a RATE segment.\n"
        "- wellbore_required_thp_bar is the downstream wellhead THP required to reproduce OPM WBHP at q.\n"
        "- wellbore_inversion_residual_bar is only the inversion residual, not validation.\n"
        "- thp_required_minus_target_bar is the real THP-limit residual in thp_limit mode.\n"
        "- residual_temperature_bht_minus_wtemp_C shows thermal WTEMP consistency.\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Coupled case directory: {case_dir}")
    print(f"[OK] Accepted exchange CSV: {accepted_csv}")
    print(f"[OK] Iterations CSV: {iterations_csv}")
    print(f"[OK] Wellbore profiles CSV: {profiles_csv}")
    return case_dir