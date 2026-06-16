from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from co2_wellbore.constants import BAR_TO_PA

from co2_wellbore import (
    CO2WellboreCalculator,
    WellboreGeometry,
    ThermalConfig,
    DriftFluxConfig,
    SolverConfig,
    CO2ChokeValve,
    ChokeConfig,
    CO2HEMChokeValve,
    HEMChokeConfig,
)

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

from .opening_choke import co2_phase_split_from_ph

from .plotting import generate_coupling_figures

from .reporting import (
    _safe_float,
    print_solver_iteration_block,
)

@dataclass
class ReservoirResult:
    time_days: float
    rate_sm3_day: float
    bhp_bar: float
    thp_bar: float
    cum_gas_sm3: float
    reservoir_pressure_bar: float
    source: str


def build_advanced_wellbore(args: argparse.Namespace) -> CO2WellboreCalculator:
    """Build the package CO2 wellbore calculator."""

    geometry = WellboreGeometry(
        tvd_m=float(args.tvd_m),
        diameter_m=float(args.diameter_m),
        roughness_m=float(args.roughness_m),
        n_segments=int(args.n_segments),
    )

    thermal = ThermalConfig(
        enabled=bool(args.heat_transfer),
        overall_U_W_m2K=float(args.overall_U_W_m2K),
        heat_transfer_diameter_m=float(args.diameter_m),
        surface_temperature_C=float(args.surface_temperature_C),
        geothermal_gradient_C_per_m=float(args.geothermal_gradient_C_per_m),
        max_abs_dT_per_segment_C=float(args.max_abs_dT_per_segment_C),
        model=str(args.thermal_model),
        tubing_outer_diameter_m=float(args.tubing_outer_diameter_m),
        cement_outer_diameter_m=float(args.cement_outer_diameter_m),
        inner_heat_transfer_coefficient_W_m2K=float(args.inner_heat_transfer_coefficient_W_m2K),
        tubing_k_W_mK=float(args.tubing_k_W_mK),
        cement_k_W_mK=float(args.cement_k_W_mK),
        formation_k_W_mK=float(args.formation_k_W_mK),
        formation_heat_capacity_J_m3K=float(args.formation_heat_capacity_J_m3K),
        transient_formation=bool(args.transient_formation_thermal),
        min_thermal_time_days=float(args.min_thermal_time_days),
        max_formation_radius_m=float(args.max_formation_radius_m),
    )

    drift_flux = DriftFluxConfig(
        enabled=bool(args.two_phase),
        C0=float(args.drift_flux_C0),
        drift_velocity_coeff=float(args.drift_velocity_coeff),
        surface_tension_N_m=float(args.surface_tension_N_m),
        friction_density=str(args.friction_density),
    )

    return CO2WellboreCalculator(
        geometry=geometry,
        thermal=thermal,
        drift_flux=drift_flux,
    )


def run_flow(
    flow_exe: str,
    case_root: str,
    cwd: Path,
    timeout_sec: float,
    mpi_nproc: int,
    flow_args: Optional[List[str]] = None,
) -> Tuple[bool, str, float]:
    """Run OPM Flow with optional extra command-line arguments.

    flow_args are passed BETWEEN the flow executable and the case root, e.g.
      flow --enable-vtk-output=true --vtk-write-pressures=true CASE.DATA
    """
    extra = list(flow_args or [])
    if int(mpi_nproc) > 1:
        cmd = ["mpirun", "--bind-to", "core", "-np", str(int(mpi_nproc)), flow_exe] + extra + [case_root]
    else:
        cmd = [flow_exe] + extra + [case_root]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    with (cwd / "FLOW_STDOUT.log").open("w", encoding="utf-8") as f:
        f.write("FLOW COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=f, text=True, env=env)
        try:
            rc = proc.wait(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "timeout", time.time() - t0
    dt = time.time() - t0
    return (rc == 0), ("ok" if rc == 0 else f"returncode_{rc}"), dt


def _read_summary_with_opm_io(case_base: Path, well_name: str) -> Optional[ReservoirResult]:
    """Read OPM summary using opm.io if available.

    This may fail inside a user-created .venv even when OPM Flow itself is installed.
    In that case the orchestrator falls back to parsing the text PRT report.
    """
    try:
        from opm.io.ecl import ESmry  # type: ignore
    except Exception:
        return None

    try:
        smry = ESmry(str(case_base))
    except Exception:
        return None

    keys: List[str] = []
    for attr in ["keys", "keywordList", "names"]:
        if hasattr(smry, attr):
            try:
                v = getattr(smry, attr)
                keys = list(v() if callable(v) else v)
                break
            except Exception:
                pass
    key_map = {str(k).upper(): str(k) for k in keys}

    def get(candidates: Sequence[str]) -> Optional[np.ndarray]:
        for cand in candidates:
            if cand.upper() in key_map:
                try:
                    return np.asarray(smry[key_map[cand.upper()]], dtype=float)
                except Exception:
                    pass
        for k in keys:
            ku = str(k).upper()
            if any(c.upper() in ku for c in candidates):
                try:
                    return np.asarray(smry[k], dtype=float)
                except Exception:
                    pass
        return None

    def last(arr: Optional[np.ndarray], default: float = math.nan) -> float:
        if arr is None or arr.size == 0:
            return float(default)
        x = float(arr[-1])
        return x if math.isfinite(x) else float(default)

    time_days = last(get(["TIME", "DAYS"]), 0.0)
    well = well_name.upper()
    rate = last(get([f"WGMIR:{well}", f"WGIR:{well}", f"WGMIR_{well}", "WGMIR"]), math.nan)
    cum = last(get([f"WGMIT:{well}", f"WGIT:{well}", f"WGMIT_{well}", "WGMIT"]), math.nan)
    bhp = last(get([f"WBHP:{well}", f"WBHP_{well}", "WBHP"]), math.nan)
    thp = last(get([f"WTHP:{well}", f"WTHP_{well}", "WTHP"]), math.nan)
    fpr = last(get(["FPR"]), math.nan)
    if not math.isfinite(rate) or not math.isfinite(bhp):
        return None
    return ReservoirResult(
        time_days=time_days,
        rate_sm3_day=rate,
        bhp_bar=bhp,
        thp_bar=thp,
        cum_gas_sm3=cum,
        reservoir_pressure_bar=fpr,
        source="opm_summary",
    )


def _parse_last_well_line_from_prt(prt_text: str, well_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return gas injection rate [sm3/day], BHP [bar], THP [bar] from the last PRT injection row.

    Expected OPM PRT line example:
    :INJ1    : 25, 41    :      :      :  GRAT: ... :   100000.0:      204.8:   161.6:     0.0:

    The parser is intentionally simple and only extracts the final numeric columns from lines
    containing the injector name in production/injection report tables.
    """
    well = well_name.upper()
    rate = bhp = thp = None

    # Use only PRODUCTION/INJECTION report blocks, not cumulative tables.
    # There can be multiple report steps; the last block is the accepted result.
    blocks = []
    for m in re.finditer(r"INJECTION REPORT", prt_text, flags=re.IGNORECASE):
        start = m.start()
        stop_candidates = [
            x for x in [
                prt_text.find("CUMULATIVE", start),
                prt_text.find("Restart file written", start),
                prt_text.find("Report step", start + 20),
            ] if x != -1
        ]
        stop = min(stop_candidates) if stop_candidates else len(prt_text)
        blocks.append(prt_text[start:stop])

    for block in blocks:
        for line in block.splitlines():
            if not line.lstrip().startswith(":"):
                continue
            if f":{well}" not in line.upper().replace(" ", ""):
                continue
            fields = [x.strip() for x in line.split(":")]
            nums = []
            for f in fields:
                # Accept normal decimal/scientific values. Ignore location "25, 41" and text fields.
                if re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?", f):
                    try:
                        nums.append(float(f.replace("D", "E").replace("d", "E")))
                    except Exception:
                        pass
            # Injection report row has: oil rate, water rate, gas rate, fluid resvol, BHP, THP.
            if len(nums) >= 6:
                rate = nums[-4]
                bhp = nums[-2]
                thp = nums[-1]
    return rate, bhp, thp


def _read_summary_or_prt(case_base: Path, well_name: str, control: StepControl, prev_cum: float, cumulative_days: float) -> Optional[ReservoirResult]:
    """Read OPM result through opm.io, and if that is unavailable parse the PRT report.

    This makes the script independent from Python opm.io bindings, which are often not
    visible inside a local .venv even when the `flow` executable runs successfully.
    """
    res = _read_summary_with_opm_io(case_base, well_name)
    if res is not None:
        if not math.isfinite(res.cum_gas_sm3):
            res.cum_gas_sm3 = prev_cum + res.rate_sm3_day * control.step_days
        if not math.isfinite(res.time_days):
            res.time_days = cumulative_days
        return res

    prt_path = case_base.with_suffix(".PRT")
    if not prt_path.exists():
        return None
    prt_text = prt_path.read_text(encoding="utf-8", errors="ignore")
    rate, bhp, thp = _parse_last_well_line_from_prt(prt_text, well_name)
    if rate is None or bhp is None:
        return None
    if thp is None:
        thp = math.nan
    return ReservoirResult(
        time_days=float(cumulative_days),
        rate_sm3_day=float(rate),
        bhp_bar=float(bhp),
        thp_bar=float(thp),
        cum_gas_sm3=float(prev_cum + float(rate) * control.step_days),
        reservoir_pressure_bar=float("nan"),
        source="prt_report_fallback",
    )


def dry_run_reservoir_result(
    wb: CO2CoolPropManyWellsCalculator,
    control: StepControl,
    t_head_C: float,
    thp_reference_bar: float,
    prev_cum: float,
    cumulative_days: float,
    wellhead_enthalpy_J_kg: Optional[float] = None,
    wellhead_quality_mass: Optional[float] = None,
) -> ReservoirResult:
    # Synthetic reservoir for debugging only: BHP comes from external wellbore at reference THP.
    bhp = wb.bhp_from_thp_and_rate(
        thp_reference_bar,
        q_sm3_day=control.rate_target_sm3_day,
        wellhead_temperature_C=t_head_C,
        wellhead_enthalpy_J_kg=wellhead_enthalpy_J_kg,
        wellhead_quality_mass=wellhead_quality_mass,
    )
    return ReservoirResult(
        time_days=float(cumulative_days),
        rate_sm3_day=float(control.rate_target_sm3_day),
        bhp_bar=min(float(bhp), float(control.bhp_limit_bar)),
        thp_bar=float("nan"),
        cum_gas_sm3=prev_cum + control.rate_target_sm3_day * control.step_days,
        reservoir_pressure_bar=float("nan"),
        source="dry_run_synthetic",
    )


def run_orchestrator(args: argparse.Namespace) -> Path:
    base_dir = Path(args.base_dir).expanduser().resolve()
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
    print(f"[INIT] Coupling case directory: {case_dir}", flush=True)
    print(f"[INIT] pressure_control_mode={pressure_mode}", flush=True)

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
            print(
                f"[DECK] step={k} it={it} root={root} mode={pressure_mode} "
                f"q={q_target:.3f} sm3/day WTEMP={wtemp_guess:.3f} C bracket=[{q_lo:.3f}, {q_hi:.3f}]",
                flush=True,
            )

            flow_ok = True
            flow_reason = "dry_run"
            flow_seconds = 0.0
            if args.run_flow and not args.dry_run:
                append_status("flow_start", step=k, iteration=it, root=root, seg_dir=str(seg_dir), log_file=str(seg_dir / "FLOW_STDOUT.log"))
                print(f"[FLOW START] step={k} it={it} root={root}. Log: {seg_dir / 'FLOW_STDOUT.log'}", flush=True)
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
                print(f"[FLOW END] step={k} it={it} root={root} ok={flow_ok} reason={flow_reason} seconds={flow_seconds:.2f}", flush=True)
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
                print(f"[READ] step={k} it={it} source={res.source} q={res.rate_sm3_day:.3f} WBHP={res.bhp_bar:.3f} bar", flush=True)
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
                    f"[{row['accepted_reason']}] step={k} it={it} q={q_actual:.3g} sm3/d "
                    f"WBHP_OPM={bhp_opm:.3f} bar THP_req={thp_calc:.3f} bar "
                    f"WTEMP_used={wtemp_guess:.3f} C BHT_calc={bht_calc:.3f} C",
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

            # Relax WTEMP for the next coupling iteration of the SAME reservoir time step.
            wtemp_guess = float(wtemp_guess + float(args.temperature_relaxation) * temp_resid)
            print(
                f"[ITER] step={k} it={it} next q={q_target:.3f} sm3/d, "
                f"next WTEMP={wtemp_guess:.3f} C",
                flush=True,
            )

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OPM Flow restart coupling with CoolProp/ManyWells CO2 wellbore and THP/valve pressure control"
    )

    p.add_argument("--base-dir", default="/home/utente/Scrivania/Project_Surr_CO2/BaseCase", help="Folder containing Base.DATA and calculator scripts")
    p.add_argument("--deck", default="Base.DATA", help="Deck filename or absolute path")
    p.add_argument("--output-root", default="runs_restart_coupled_coolprop_manywells")
    p.add_argument("--figures-dir", default="figures", help="Directory where informative 07 figures will be saved")
    p.add_argument("--case-id", default="Base_COOLPROP_MANYWELLS_THPVALVE")
    p.add_argument("--well-name", default="INJ1")

    p.add_argument("--run-flow", action="store_true", help="Actually run OPM Flow")
    p.add_argument("--dry-run", action="store_true", help="Do not run OPM; use synthetic reservoir values for checking script logic")
    p.add_argument("--flow-exe", default="flow")
    p.add_argument("--mpi-nproc", type=int, default=1)
    p.add_argument("--flow-timeout-sec", type=float, default=1800.0)

    # VTK/PVTU output options for ML dataset collection.
    # These are passed directly to OPM Flow as command-line switches.
    p.add_argument("--enable-vtk-output", action="store_true",
                   help="Pass --enable-vtk-output=true and VTK array switches to OPM Flow")
    p.add_argument("--vtk-write-temperature", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request temperature output if supported by Flow")
    p.add_argument("--vtk-write-pressures", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request pressure output")
    p.add_argument("--vtk-write-saturations", action=argparse.BooleanOptionalAction, default=True,
                   help="When VTK is enabled, request saturation output")

    p.add_argument("--n-steps", type=int, default=36)
    p.add_argument("--step-days", type=float, default=30.0)
    p.add_argument("--rate-sm3-day", type=float, default=100000.0)
    p.add_argument("--rate-schedule", type=float, nargs="*", default=None, help="Demand/cap q(t) list, sm3/day, one value per step")
    p.add_argument("--rate-schedule-is-cap", action=argparse.BooleanOptionalAction, default=True,
                   help="In pressure-control modes, treat rate schedule as an upper demand/cap rather than an exact target")
    p.add_argument("--bhp-limit-bar", type=float, default=240.0)

    p.add_argument(
        "--pressure-control-mode",
        choices=["fixed_rate", "thp_limit", "opening_choke"],
        default="thp_limit",
        help="fixed_rate = old robust baseline; thp_limit = reduce q until THP limit is satisfied",
    )

    p.add_argument(
        "--opening-schedule-percent",
        type=float,
        nargs="*",
        default=None,
        help="Choke opening schedule in percent, one value per coupling step.",
    )

    p.add_argument("--upstream-pressure-bar", type=float, default=98.0)
    p.add_argument("--upstream-temperature-C", type=float, default=4.0)
    p.add_argument("--kv-full-m3-h", type=float, default=118.0)
    p.add_argument("--choke-model", choices=["legacy_kv", "hem"], default="hem",
                   help="legacy_kv = fluids/IEC liquid-like Kv model; hem = real-fluid HEM choke backend")
    p.add_argument("--valve-FL", type=float, default=0.90)
    p.add_argument("--valve-Fd", type=float, default=1.00)
    p.add_argument("--valve-diameter-m", type=float, default=0.16,
                   help="For HEM: physical full-open choke diameter. For legacy_kv: valve diameter for IEC sizing.")
    p.add_argument("--hem-discharge-coefficient", type=float, default=0.84)
    p.add_argument("--hem-area-exponent", type=float, default=1.0,
                   help="HEM effective area law: A_eff = A_full * opening_fraction ** exponent")
    p.add_argument("--hem-min-pressure-bar", type=float, default=6.0,
                   help="Minimum pressure for HEM throat search; keep above CO2 triple-region problems")
    p.add_argument("--hem-pressure-samples", type=int, default=220)
    p.add_argument("--hem-flow-tolerance-kg-s", type=float, default=0.05,
                   help="HEM choke residual tolerance: |m_actual - m_capacity| in kg/s")
    p.add_argument("--kv-tolerance-m3-h", type=float, default=0.05,
                   help="Legacy Kv tolerance. Ignored by HEM residual; HEM uses --hem-flow-tolerance-kg-s.")
    p.add_argument("--thp-target-bar", type=float, default=80.0,
                   help="Target/maximum downstream THP for thp_limit mode")
    p.add_argument("--thp-tolerance-bar", type=float, default=0.25)
    p.add_argument("--q-min-sm3-day", type=float, default=0.0)
    p.add_argument("--q-max-sm3-day", type=float, default=300000.0)

    p.add_argument("--initial-wtemp-C", type=float, default=40.0, help="First WTEMP guess if not reading from deck")
    p.add_argument("--use-deck-wtemp-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--carry-wtemp-between-steps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-coupling-iter", type=int, default=12,
                   help="Total restart evaluations per step for both pressure and temperature convergence")
    p.add_argument("--temperature-relaxation", type=float, default=0.7)
    p.add_argument("--t-tolerance-C", type=float, default=0.5)
    p.add_argument("--p-tolerance-bar", type=float, default=0.05,
                   help="Wellbore inversion tolerance; not the THP-control tolerance")
    p.add_argument("--q-tolerance-sm3-day", type=float, default=100.0)
    p.add_argument("--allow-force-accept", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, accept the last iteration at max iterations; default False for strict science runs")
    p.add_argument("--accept-last-feasible", action=argparse.BooleanOptionalAction, default=True,
                   help="If strict root convergence fails, accept the last pressure-feasible row once temperature is converged")
    p.add_argument("--accept-capacity-surplus", action=argparse.BooleanOptionalAction, default=True,
                   help="In opening_choke mode, accept a feasible point when Kv_required < Kv_effective and the next higher rate is unsafe")
    p.add_argument("--capacity-surplus-q-width-sm3-day", type=float, default=5000.0,
                   help="If the q bracket is narrower than this and Kv_required < Kv_effective, accept as non-binding choke capacity")

    p.add_argument("--reference-thp-bar", type=float, default=100.0, help="Only for dry-run fallback and inversion fallback")
    p.add_argument("--thp-min-bar", type=float, default=1.0)
    p.add_argument("--thp-max-bar", type=float, default=350.0)
    p.add_argument("--t-head-C", type=float, default=5.0, help="External wellhead/tubing-head temperature for the calculator")

    p.add_argument(
        "--wellhead-enthalpy-J-kg",
        type=float,
        default=None,
        help="Post-choke inlet enthalpy for wellbore P-H flash. Best option for isenthalpic choke boundary.",
    )
    p.add_argument(
        "--wellhead-quality-mass",
        type=float,
        default=None,
        help="Post-choke saturated CO2 mass quality at wellhead. Use only if you know x after choke.",
    )

    # CoolProp/ManyWells wellbore parameters matching the Base.DATA-like case.
    p.add_argument("--tvd-m", type=float, default=1600.0)
    p.add_argument("--diameter-m", type=float, default=0.10)
    p.add_argument("--roughness-m", type=float, default=1.5e-5)
    p.add_argument("--n-segments", type=int, default=120)
    p.add_argument("--std-pressure-bar", type=float, default=1.01325)
    p.add_argument("--std-temperature-C", type=float, default=15.0)
    p.add_argument("--surface-temperature-C", type=float, default=32.0)
    p.add_argument("--geothermal-gradient-C-per-m", type=float, default=0.03)
    p.add_argument("--relaxation-length-m", type=float, default=700.0)

    p.add_argument("--heat-transfer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overall-U-W-m2K", type=float, default=4.0)
    p.add_argument("--thermal-model", choices=["overall_U", "layered"], default="overall_U",
                   help="overall_U = old fixed U; layered = tubing/cement/formation resistance model")
    p.add_argument("--tubing-outer-diameter-m", type=float, default=0.1143)
    p.add_argument("--cement-outer-diameter-m", type=float, default=0.2160)
    p.add_argument("--inner-heat-transfer-coefficient-W-m2K", type=float, default=1000.0)
    p.add_argument("--tubing-k-W-mK", type=float, default=45.0)
    p.add_argument("--cement-k-W-mK", type=float, default=1.0)
    p.add_argument("--formation-k-W-mK", type=float, default=2.5)
    p.add_argument("--formation-heat-capacity-J-m3K", type=float, default=2.2e6)
    p.add_argument("--transient-formation-thermal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-thermal-time-days", type=float, default=0.01)
    p.add_argument("--max-formation-radius-m", type=float, default=30.0)
    p.add_argument("--max-abs-dT-per-segment-C", type=float, default=10.0)
    p.add_argument("--max-abs-jt-K-per-Pa", type=float, default=2.0e-5)
    p.add_argument("--include-joule-thomson-in-well", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--two-phase", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable CoolProp-saturation-triggered drift-flux closure; usually inactive for dense/supercritical storage states")
    p.add_argument("--drift-flux-C0", type=float, default=1.20)
    p.add_argument("--drift-velocity-coeff", type=float, default=0.35)
    p.add_argument("--surface-tension-N-m", type=float, default=0.01)
    p.add_argument("--friction-density", choices=["homogeneous", "hydrostatic"], default="homogeneous")

    p.add_argument("--inner-solver-max-iter", type=int, default=80)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.run_flow:
        args.dry_run = True
    if args.dry_run:
        print("[INFO] dry-run mode: OPM Flow will not be executed; this only checks deck/script logic.")
    run_orchestrator(args)


if __name__ == "__main__":
    main()
