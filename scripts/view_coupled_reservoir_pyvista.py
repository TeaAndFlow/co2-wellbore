#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResInsight-style full-field PyVista viewer for coupled OPM reservoir VTK outputs.

Important:
- plume means FULL FIELD saturation_gas, not thresholded plume only
- pressure is converted Pa -> bar
- temperature is converted K -> degC
- default view shows accepted coupling steps only
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VARIABLE_CANDIDATES: Dict[str, List[str]] = {
    "pressure": ["pressure_gas", "pressure_oil", "pressure_water", "PRESSURE", "pressure"],
    "temp": ["temperature", "TEMPERATURE", "TEMP", "temp"],
    "sgas": ["saturation_gas", "SGAS", "sgas", "gas_saturation"],
    "plume": ["saturation_gas", "SGAS", "sgas", "gas_saturation"],
}


def natural_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", str(path))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def segment_root_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if re.fullmatch(r"SEG_\d{4}_IT\d{2}", part):
            return part
    return None


def discover_all_vtk_files(case_dir: Path) -> List[Path]:
    files = sorted(list(case_dir.rglob("*.pvtu")) + list(case_dir.rglob("*.vtu")), key=natural_key)
    return [p for p in files if p.is_file()]


def discover_accepted_csv(case_dir: Path) -> Optional[Path]:
    for name in ["coupled_exchange_accepted.csv", "coupled_exchange_accepted_LIVE.csv"]:
        p = case_dir / name
        if p.exists():
            return p
    return None


def infer_accepted_roots(case_dir: Path, vtk_files: Sequence[Path]) -> Tuple[List[str], Dict[str, Dict]]:
    csv = discover_accepted_csv(case_dir)
    info_by_root: Dict[str, Dict] = {}

    if csv is None:
        return [], info_by_root

    df = pd.read_csv(csv)
    print(f"[accepted] {csv}")

    root_col = None
    for c in ["root", "segment_root", "accepted_root", "flow_root", "restart_root", "run_root", "opm_root"]:
        if c in df.columns:
            root_col = c
            break

    roots: List[str] = []

    if root_col is not None:
        for _, row in df.iterrows():
            root = str(row[root_col])
            roots.append(root)
            info_by_root[root] = dict(row)
        return roots, info_by_root

    step_col = "step" if "step" in df.columns else None

    iter_col = None
    for c in ["iteration", "accepted_iteration", "iter", "it"]:
        if c in df.columns:
            iter_col = c
            break

    if step_col and iter_col:
        for _, row in df.iterrows():
            step = int(row[step_col])
            it = int(row[iter_col])
            root = f"SEG_{step:04d}_IT{it:02d}"
            roots.append(root)
            info_by_root[root] = dict(row)
        return roots, info_by_root

    # Fallback: if accepted CSV has only step, choose latest IT folder for each step.
    if step_col:
        available_roots = sorted(
            {r for r in (segment_root_from_path(p) for p in vtk_files) if r},
            key=natural_key,
        )
        for _, row in df.iterrows():
            step = int(row[step_col])
            prefix = f"SEG_{step:04d}_IT"
            candidates = [r for r in available_roots if r.startswith(prefix)]
            if candidates:
                root = sorted(candidates, key=natural_key)[-1]
                roots.append(root)
                info_by_root[root] = dict(row)

    return roots, info_by_root


def filter_accepted_vtk_files(case_dir: Path, vtk_files: Sequence[Path], accepted_only: bool) -> Tuple[List[Path], Dict[str, Dict]]:
    if not accepted_only:
        return list(vtk_files), {}

    roots, info_by_root = infer_accepted_roots(case_dir, vtk_files)
    if not roots:
        print("[accepted] could not infer accepted roots; using all VTK files")
        return list(vtk_files), info_by_root

    selected: List[Path] = []
    for root in roots:
        hits = [p for p in vtk_files if segment_root_from_path(p) == root]
        if hits:
            selected.append(sorted(hits, key=natural_key)[-1])

    if not selected:
        print("[accepted] no VTK files matched accepted roots; using all VTK files")
        return list(vtk_files), info_by_root

    return selected, info_by_root


def available_arrays(mesh) -> List[str]:
    names = list(mesh.cell_data.keys()) + list(mesh.point_data.keys())
    return sorted(set(map(str, names)))


def find_array_name(mesh, variable: str) -> Tuple[str, str]:
    candidates = VARIABLE_CANDIDATES.get(variable, [variable])
    cell = {str(k).lower(): str(k) for k in mesh.cell_data.keys()}
    point = {str(k).lower(): str(k) for k in mesh.point_data.keys()}

    for cand in candidates:
        key = cand.lower()
        if key in cell:
            return cell[key], "cell"
        if key in point:
            return point[key], "point"

    for cand in candidates:
        key = cand.lower()
        for name in mesh.cell_data.keys():
            n = str(name).lower()
            if key in n or n in key:
                return str(name), "cell"
        for name in mesh.point_data.keys():
            n = str(name).lower()
            if key in n or n in key:
                return str(name), "point"

    raise KeyError(
        f"Could not find variable={variable}. Available arrays:\n  "
        + "\n  ".join(available_arrays(mesh))
    )


def get_array(mesh, name: str, preference: str) -> np.ndarray:
    if preference == "cell":
        return np.asarray(mesh.cell_data[name], dtype=float)
    return np.asarray(mesh.point_data[name], dtype=float)


def set_array(mesh, name: str, preference: str, values: np.ndarray) -> None:
    if preference == "cell":
        mesh.cell_data[name] = np.asarray(values, dtype=np.float32)
    else:
        mesh.point_data[name] = np.asarray(values, dtype=np.float32)


def resinsight_cmap():
    # Blue low values, green/yellow/orange/red high values.
    return [
        "#0000ff",
        "#0066ff",
        "#00a6ff",
        "#00d6d6",
        "#00cc66",
        "#66cc00",
        "#ffff00",
        "#ff9900",
        "#ff3300",
        "#990000",
    ]


def converted_display_mesh(mesh, array_name: str, preference: str, variable: str):
    """Copy mesh and add display scalar with engineering units."""
    m = mesh.copy(deep=True)

    raw = get_array(m, array_name, preference)
    finite = raw[np.isfinite(raw)]

    if variable == "temp":
        # OPM VTK usually writes SI Kelvin.
        if finite.size and float(np.nanmedian(finite)) > 150.0:
            vals = raw - 273.15
            display = f"{array_name}_C"
        else:
            vals = raw
            display = array_name
        unit = "degC"
        set_array(m, display, preference, vals)

    elif variable == "pressure":
        # OPM VTK usually writes SI Pa.
        if finite.size and float(np.nanmax(np.abs(finite))) > 1.0e4:
            vals = raw / 1.0e5
            display = f"{array_name}_bar"
        else:
            vals = raw
            display = array_name
        unit = "bar"
        set_array(m, display, preference, vals)

    else:
        # plume/sgas: full-field gas saturation, dimensionless.
        display = array_name
        unit = "fraction"

    return m, display, unit


def finite_clim(meshes: Sequence, array_name: str, preference: str, variable: str, percentile: float) -> Tuple[float, float]:
    values = []

    for mesh in meshes:
        dm, display, _ = converted_display_mesh(mesh, array_name, preference, variable)
        arr = get_array(dm, display, preference)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)

    if not values:
        return 0.0, 1.0

    v = np.concatenate(values)

    if variable in {"sgas", "plume"}:
        vmax = float(np.nanmax(v))
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        return 0.0, min(1.0, vmax)

    if percentile < 100.0:
        lo = float(np.percentile(v, 100.0 - percentile))
        hi = float(np.percentile(v, percentile))
    else:
        lo = float(v.min())
        hi = float(v.max())

    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def short_label(path: Path, case_dir: Path) -> str:
    try:
        return str(path.relative_to(case_dir))
    except Exception:
        return str(path)


def accepted_text(info: Dict) -> str:
    if not info:
        return ""

    fields = [
        ("step", "step", "{:.0f}"),
        ("opening", "opening_percent", "{:.3g}%"),
        ("q", "reservoir_rate_sm3_day", "{:.6g} sm3/day"),
        ("BHP", "reservoir_bhp_bar", "{:.4g} bar"),
        ("THP", "wellbore_required_thp_bar", "{:.4g} bar"),
        ("reason", "accepted_reason", "{}"),
        ("constraint", "active_constraint", "{}"),
    ]

    parts = []
    for label, col, fmt in fields:
        if col not in info:
            continue
        val = info[col]
        try:
            text = fmt.format(float(val))
        except Exception:
            text = fmt.format(str(val))
        parts.append(f"{label}: {text}")

    return "\n".join(parts)


def scalar_bar_args(title: str) -> Dict:
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.035,
        "position_y": 0.16,
        "width": 0.08,
        "height": 0.58,
        "n_labels": 8,
        "fmt": "%.4g",
        "label_font_size": 10,
        "title_font_size": 11,
        "color": "black",
        "shadow": False,
    }


def add_well_marker(
    plotter,
    mesh,
    *,
    well_i: int = 24,
    well_j: int = 45,
    grid_nx: int = 64,
    grid_ny: int = 64,
    index_base: int = 1,
    marker_frac: float = 0.004,
    show_label: bool = True,
):
    """Add injector marker at Eclipse-style I,J cell coordinates.

    Default assumes 1-based Eclipse coordinates:
        INJ1 I=24, J=45
    """
    import pyvista as pv

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # Convert Eclipse-style cell index to cell-center coordinate.
    ii = float(well_i - index_base) + 0.5
    jj = float(well_j - index_base) + 0.5

    ii = float(np.clip(ii, 0.0, max(float(grid_nx), 1.0)))
    jj = float(np.clip(jj, 0.0, max(float(grid_ny), 1.0)))

    x = xmin + (ii / float(grid_nx)) * (xmax - xmin)
    y = ymin + (jj / float(grid_ny)) * (ymax - ymin)
    z = zmin

    xy_span = max(abs(xmax - xmin), abs(ymax - ymin), 1.0)
    radius = float(marker_frac) * xy_span

    actors = []
    # Black vertical picket/pin marker, visible over red plume.
    picket_height = max((zmax - zmin) * 1.8, radius * 8.0)
    picket_center_z = z + 0.5 * picket_height

    actors.append(
        plotter.add_mesh(
            pv.Cylinder(
                center=(x, y, picket_center_z),
                direction=(0.0, 0.0, 1.0),
                radius=radius * 0.28,
                height=picket_height,
                resolution=20,
            ),
            color="black",
            reset_camera=False,
            lighting=True,
        )
    )

    actors.append(
        plotter.add_mesh(
            pv.Sphere(
                radius=radius * 0.75,
                center=(x, y, z + picket_height),
                theta_resolution=20,
                phi_resolution=20,
            ),
            color="black",
            reset_camera=False,
            lighting=True,
        )
    )

    if show_label:
        actors.append(
            plotter.add_point_labels(
                np.asarray([(x, y, z + 2.1 * radius)]),
                [f"INJ1 ({well_i},{well_j})"],
                font_size=8,
                point_size=0,
                text_color="white",
                shape_color="black",
                shape_opacity=0.55,
                always_visible=True,
                reset_camera=False,
            )
        )

    return actors


def main() -> None:
    ap = argparse.ArgumentParser(description="ResInsight-style full-field viewer for coupled OPM VTK outputs.")
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--variable", choices=["pressure", "temp", "sgas", "plume"], default="plume")
    ap.add_argument("--accepted-only", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--show-edges", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--edge-color", type=str, default="white")
    ap.add_argument("--edge-width", type=float, default=0.15)

    ap.add_argument("--threshold-plume", action="store_true", help="Optional: hide cells below plume threshold. Default is full field.")
    ap.add_argument("--plume-threshold", type=float, default=0.01)

    ap.add_argument("--show-well", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--well-i", type=int, default=24, help="Injector I index/cell coordinate.")
    ap.add_argument("--well-j", type=int, default=45, help="Injector J index/cell coordinate.")
    ap.add_argument("--well-index-base", type=int, default=1, choices=[0, 1], help="0 for Python indexing, 1 for Eclipse-style indexing.")
    ap.add_argument("--grid-nx", type=int, default=64)
    ap.add_argument("--grid-ny", type=int, default=64)
    ap.add_argument("--well-marker-frac", type=float, default=0.004, help="Marker radius as fraction of field size.")
    ap.add_argument("--show-grid", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--view", choices=["top", "iso"], default="top")
    ap.add_argument("--zoom", type=float, default=1.05)

    ap.add_argument("--window-width", type=int, default=1450)
    ap.add_argument("--window-height", type=int, default=900)

    args = ap.parse_args()

    import pyvista as pv

    case_dir = args.case_dir.expanduser().resolve()
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    all_files = discover_all_vtk_files(case_dir)
    if not all_files:
        raise FileNotFoundError(f"No .vtu/.pvtu files found under {case_dir}")

    files, info_by_root = filter_accepted_vtk_files(case_dir, all_files, bool(args.accepted_only))

    print(f"[case] {case_dir}")
    print(f"[vtk files] showing {len(files)} of {len(all_files)} total")
    print(f"[accepted-only] {args.accepted_only}")

    cache: Dict[int, object] = {}

    def load_mesh(i: int):
        i = int(np.clip(i, 0, len(files) - 1))
        if i not in cache:
            cache[i] = pv.read(str(files[i]))
        return cache[i]

    first_mesh = load_mesh(0)
    array_name, preference = find_array_name(first_mesh, args.variable)

    print(f"[variable] {args.variable} -> raw array={array_name}, preference={preference}")
    print("[available arrays]")
    for name in available_arrays(first_mesh):
        print(" ", name)

    sample_ids = sorted(set([0, len(files) // 2, len(files) - 1]))
    clim = finite_clim(
        [load_mesh(i) for i in sample_ids],
        array_name,
        preference,
        args.variable,
        float(args.percentile),
    )

    plotter = pv.Plotter(window_size=(args.window_width, args.window_height))
    plotter.set_background("#d8e6f3")

    actors: List[object] = []
    first = True

    def clear():
        nonlocal actors
        for a in actors:
            try:
                plotter.remove_actor(a, reset_camera=False)
            except Exception:
                pass
        actors = []

    def draw(idx_float):
        nonlocal first

        idx = int(round(float(idx_float)))
        idx = int(np.clip(idx, 0, len(files) - 1))

        clear()

        raw_mesh = load_mesh(idx)
        mesh, display_array, unit = converted_display_mesh(raw_mesh, array_name, preference, args.variable)

        shown_mesh = mesh
        shown_array = display_array
        local_clim = clim
        note = ""

        if args.variable == "plume" and args.threshold_plume:
            try:
                thresholded = raw_mesh.threshold(
                    value=float(args.plume_threshold),
                    scalars=array_name,
                    preference=preference,
                )
                if getattr(thresholded, "n_points", 0) > 0 and getattr(thresholded, "n_cells", 0) > 0:
                    shown_mesh = thresholded
                    shown_array = array_name
                    local_clim = (float(args.plume_threshold), max(float(args.plume_threshold) + 1e-6, clim[1]))
                    note = f"thresholded plume > {args.plume_threshold:g}"
                else:
                    note = f"threshold empty; showing full field"
            except Exception as e:
                note = f"threshold failed; showing full field: {e}"

        actors.append(
            plotter.add_mesh(
                shown_mesh,
                scalars=shown_array,
                preference=preference,
                cmap=resinsight_cmap(),
                clim=local_clim,
                opacity=float(args.opacity),
                show_edges=bool(args.show_edges),
                edge_color=str(args.edge_color),
                line_width=float(args.edge_width),
                scalar_bar_args=scalar_bar_args(f"Cell Results:\n{shown_array} [{unit}]"),
                reset_camera=False,
                lighting=True,
                ambient=0.48,
                diffuse=0.72,
                specular=0.08,
            )
        )

        try:
            actors.append(plotter.add_mesh(shown_mesh.outline(), color="black", line_width=1.2, reset_camera=False))
        except Exception:
            pass

        if args.show_well:
            actors.extend(
                add_well_marker(
                    plotter,
                    shown_mesh,
                    well_i=int(args.well_i),
                    well_j=int(args.well_j),
                    grid_nx=int(args.grid_nx),
                    grid_ny=int(args.grid_ny),
                    index_base=int(args.well_index_base),
                    marker_frac=float(args.well_marker_frac),
                    show_label=True,
                )
            )

        root = segment_root_from_path(files[idx])
        info = info_by_root.get(root or "", {})

        text = (
            f"Reservoir viewer\n"
            f"accepted frame {idx + 1}/{len(files)} | total VTK files={len(all_files)}\n"
            f"{short_label(files[idx], case_dir)}\n"
            f"variable={args.variable} | shown={shown_array} [{unit}]"
        )

        acc = accepted_text(info)
        if acc:
            text += "\n\n" + acc

        if args.variable in {"plume", "sgas"}:
            vals = get_array(raw_mesh, array_name, preference)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                text += f"\n\nfull-field Sg: min={float(vals.min()):.4g}, max={float(vals.max()):.4g}"
            if note:
                text += "\n" + note

        actors.append(plotter.add_text(text, position="upper_left", font_size=10, color="black"))

        if args.show_grid:
            plotter.show_bounds(
                grid="front",
                location="outer",
                xtitle="X/i",
                ytitle="Y/j",
                ztitle="Z/k",
                color="black",
                font_size=9,
            )

        if first:
            if args.view == "top":
                plotter.view_xy()
            else:
                plotter.view_isometric()
            plotter.camera.zoom(float(args.zoom))
            first = False

        plotter.render()

    draw(0)

    if len(files) > 1:
        plotter.add_slider_widget(
            callback=draw,
            rng=[0, len(files) - 1],
            value=0,
            title="accepted coupling step" if args.accepted_only else "VTK file / nonlinear iteration",
            pointa=(0.24, 0.04),
            pointb=(0.76, 0.04),
            style="modern",
            fmt="%.0f",
        )

    plotter.add_axes(line_width=2)
    print("[viewer] left=rotate, wheel=zoom, right=pan, slider=accepted step/file.")
    plotter.show()


if __name__ == "__main__":
    main()
