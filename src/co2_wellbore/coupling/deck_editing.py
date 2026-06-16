from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class StepControl:
    step_index: int
    iteration: int
    step_days: float
    rate_target_sm3_day: float
    wtemp_used_C: float
    bhp_limit_bar: float
    # For restart decks with SKIPREST, OPM requires report steps in the new deck
    # that reproduce the calendar/report-step time of the restart point.
    # These TSTEPs are written before the new controls and are skipped by SKIPREST.
    restart_alignment_tsteps: Optional[List[float]] = None


@dataclass
class RestartPointer:
    segment_dir: Optional[str] = None
    root: Optional[str] = None
    report_step: Optional[int] = None


SECTION_NAMES = [
    "RUNSPEC",
    "GRID",
    "EDIT",
    "PROPS",
    "REGIONS",
    "SOLUTION",
    "SUMMARY",
    "SCHEDULE",
    "END",
]

KNOWN_SCHEDULE_KEYWORDS = [
    "DATES",
    "TIME",
    "TSTEP",
    "RPTRST",
    "RPTSCHED",
    "SKIPREST",
    "INCLUDE",
    "WELLSTRE",
    "WELSPECS",
    "COMPDAT",
    "WINJGAS",
    "WCONINJE",
    "WCONPROD",
    "WTEMP",
    "WELOPEN",
    "WELTARG",
    "WELCNTL",
    "VFPINJ",
    "END",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def section_positions(deck_text: str) -> Dict[str, int]:
    pos: Dict[str, int] = {}
    for name in SECTION_NAMES:
        m = re.search(rf"(?im)^\s*{name}\s*$", deck_text)
        if m:
            pos[name] = m.start()
    return pos


def split_deck_sections(deck_text: str) -> Tuple[str, str, str, str]:
    pos = section_positions(deck_text)
    required = ["SOLUTION", "SUMMARY", "SCHEDULE"]
    missing = [x for x in required if x not in pos]
    if missing:
        raise ValueError(f"Deck must contain {required}; missing {missing}")
    before_solution = deck_text[:pos["SOLUTION"]]
    solution = deck_text[pos["SOLUTION"]:pos["SUMMARY"]]
    summary = deck_text[pos["SUMMARY"]:pos["SCHEDULE"]]
    schedule = deck_text[pos["SCHEDULE"]:]
    return before_solution, solution, summary, schedule


def ensure_runspec_keyword(before_solution: str, keyword: str, block: str) -> str:
    if re.search(rf"(?im)^\s*{re.escape(keyword)}\b", before_solution):
        return before_solution
    lines = before_solution.splitlines()
    insert_at = None
    # Put after DIMENS data record if possible; otherwise after RUNSPEC.
    for i, line in enumerate(lines):
        if re.match(r"(?i)^\s*DIMENS\b", line):
            insert_at = min(i + 3, len(lines))
            break
    if insert_at is None:
        for i, line in enumerate(lines):
            if re.match(r"(?i)^\s*RUNSPEC\b", line):
                insert_at = i + 1
                break
    if insert_at is None:
        insert_at = 0
    new_lines = lines[:insert_at] + ["", block.strip(), ""] + lines[insert_at:]
    return "\n".join(new_lines) + "\n"


def ensure_restart_runspec(before_solution: str) -> str:
    out = before_solution
    out = ensure_runspec_keyword(out, "UNIFIN", "UNIFIN")
    out = ensure_runspec_keyword(out, "UNIFOUT", "UNIFOUT")
    return out


def _keyword_positions(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for key in KNOWN_SCHEDULE_KEYWORDS:
        for m in re.finditer(rf"(?im)^\s*{re.escape(key)}\b", text):
            out.append((m.start(), key.upper()))
    return sorted(out, key=lambda x: x[0])


def extract_schedule_block(schedule_text: str, keyword: str) -> str:
    positions = _keyword_positions(schedule_text)
    key = keyword.upper()
    for i, (pos, found) in enumerate(positions):
        if found != key:
            continue
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(schedule_text)
        block = schedule_text[pos:next_pos].strip()
        block = re.split(r"(?im)^\s*END\s*$", block)[0].strip()
        return block + "\n" if block else ""
    return ""


def extract_static_well_setup(schedule_text: str) -> str:
    # WCONPROD is intentionally not here; we write it every segment as an operational control.
    keys = ["WELLSTRE", "WELSPECS", "COMPDAT", "WINJGAS"]
    chunks = [extract_schedule_block(schedule_text, k).strip() for k in keys]
    chunks = [c for c in chunks if c]
    return "\n\n".join(chunks) + "\n" if chunks else ""


def extract_base_producer_control(schedule_text: str) -> str:
    return extract_schedule_block(schedule_text, "WCONPROD")


def parse_first_wconinje_rate(deck_text: str, well_name: str, default: float) -> float:
    text = strip_comments(deck_text)
    for m in re.finditer(r"(?is)^\s*WCONINJE\b(.*?)/\s*", text, flags=re.MULTILINE):
        block = m.group(1)
        for rec in block.split("/"):
            parts = rec.replace("'", " ").split()
            if len(parts) >= 5 and parts[0].upper() == well_name.upper():
                try:
                    return float(parts[4].replace("D", "E"))
                except Exception:
                    pass
    return float(default)


def parse_first_wtemp(deck_text: str, well_name: str, default_C: float) -> float:
    text = strip_comments(deck_text)
    for m in re.finditer(r"(?is)^\s*WTEMP\b(.*?)/", text, flags=re.MULTILINE):
        block = m.group(1)
        for rec in block.split("/"):
            parts = rec.replace("'", " ").split()
            if len(parts) >= 2 and parts[0].upper() == well_name.upper():
                try:
                    return float(parts[1].replace("D", "E"))
                except Exception:
                    pass
    return float(default_C)


def parse_tstep_list(deck_text: str, fallback_step_days: float, n_steps: int) -> List[float]:
    text = strip_comments(deck_text)
    m = re.search(r"(?is)^\s*TSTEP\b(.*?)/", text, flags=re.MULTILINE)
    if not m:
        return [float(fallback_step_days)] * int(n_steps)
    tokens = m.group(1).replace("\n", " ").split()
    values: List[float] = []
    for tok in tokens:
        if "*" in tok:
            n, val = tok.split("*", 1)
            try:
                values.extend([float(val.replace("D", "E"))] * int(n))
            except Exception:
                continue
        else:
            try:
                values.append(float(tok.replace("D", "E")))
            except Exception:
                continue
    return values if values else [float(fallback_step_days)] * int(n_steps)


def segment_root(step_index: int, iteration: int) -> str:
    return f"SEG_{step_index:04d}_IT{iteration:02d}"

def extract_single_keyword_block(section_text: str, keyword: str) -> str:
    """Extract one slash-terminated keyword block from a deck section."""
    m = re.search(rf"(?im)^\s*{re.escape(keyword)}\b", section_text)
    if not m:
        return ""

    lines = section_text[m.start():].splitlines()
    out: List[str] = []

    for line in lines:
        out.append(line)
        if line.strip().endswith("/"):
            break

    block = "\n".join(out).strip()
    return block + "\n\n" if block else ""

def build_solution_section(original_solution: str, restart: RestartPointer) -> str:
    if restart.root is None:
        return original_solution.strip() + "\n\n"

    if restart.report_step is None:
        raise ValueError("Restart root exists but report_step is None")

    # Keep the fixed geothermal temperature profile in restart decks.
    # This does NOT change during coupling. WTEMP is written separately in SCHEDULE.
    tempvd_block = extract_single_keyword_block(original_solution, "TEMPVD")

    lines: List[str] = []
    lines.append("SOLUTION")

    if tempvd_block.strip():
        lines.append("-- Fixed geothermal temperature profile copied from base deck.")
        lines.append(tempvd_block.strip())
        lines.append("")

    lines.append("-- Restart from previous accepted segment.")
    lines.append("RESTART")
    lines.append(f" '{restart.root}' {int(restart.report_step)} 1* 1* /")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n\n"

def build_schedule_section(
    control: StepControl,
    well_name: str,
    static_well_setup: str,
    producer_control: str,
    is_restart: bool,
) -> str:
    """Build SCHEDULE for one restart-coupling segment.

    TEMPVD is not touched here.
    WTEMP is the current coupling temperature imposed for the injection well.

    For restart segments, old TSTEPs are needed only as SKIPREST alignment
    to reach the previous accepted report step. Current WTEMP/WCONINJE must
    come after this alignment block.
    """
    lines: List[str] = []

    lines.append("SCHEDULE")

    if is_restart:
        lines.append("-- Restart segment: align schedule to previous accepted restart report step.")
        lines.append("SKIPREST")
        lines.append("")

        if control.restart_alignment_tsteps:
            lines.append("-- Alignment report steps up to the RESTART time; skipped by SKIPREST")
            lines.append("TSTEP")
            lines.append(
                " "
                + " ".join(f"{float(x):.8g}" for x in control.restart_alignment_tsteps)
                + " /"
            )
            lines.append("/")
            lines.append("")
    else:
        if static_well_setup.strip():
            lines.append("-- Static well setup copied from the base deck.")
            lines.append(static_well_setup.strip())
            lines.append("")

    if producer_control.strip():
        lines.append("-- Producer controls copied from the base deck.")
        lines.append(producer_control.strip())
        lines.append("")

    lines.append("RPTRST")
    lines.append(" 'BASIC=2' 'ALLPROPS' /")
    lines.append("")

    lines.append("RPTSCHED")
    lines.append(" 'PRES' 'SGAS' 'TEMP' 'WELLS' /")
    lines.append("")

    lines.append("-- Bottomhole injection/source temperature imposed in OPM for this coupling iteration")
    lines.append("WTEMP")
    lines.append(f" {well_name} {float(control.wtemp_used_C):.8g} /")
    lines.append("/")
    lines.append("")

    lines.append("-- Robust control: OPM gets q(t), not direct THP. THP is calculated externally.")
    lines.append("WCONINJE")
    lines.append("-- WELL TYPE STATUS CONTROL RATE RESV BHP")
    lines.append(
        f" {well_name} GAS OPEN RATE "
        f"{float(control.rate_target_sm3_day):.8g} 1* "
        f"{float(control.bhp_limit_bar):.8g} /"
    )
    lines.append("/")
    lines.append("")

    lines.append("TSTEP")
    lines.append(f" {float(control.step_days):.8g} /")
    lines.append("/")
    lines.append("")

    lines.append("END")

    return "\n".join(lines).rstrip() + "\n"

def build_segment_deck(
    base_deck_text: str,
    control: StepControl,
    well_name: str,
    static_well_setup: str,
    producer_control: str,
    restart: RestartPointer,
) -> str:
    before_solution, solution, summary, schedule = split_deck_sections(base_deck_text)
    before_solution = ensure_restart_runspec(before_solution)
    solution_new = build_solution_section(solution, restart)
    schedule_new = build_schedule_section(
        control=control,
        well_name=well_name,
        static_well_setup=static_well_setup,
        producer_control=producer_control,
        is_restart=restart.root is not None,
    )
    return before_solution.rstrip() + "\n\n" + solution_new + summary.strip() + "\n\n" + schedule_new


def copy_restart_artifacts(prev_dir: Path, new_dir: Path, prev_root: str) -> List[str]:
    copied: List[str] = []
    patterns = [
        f"{prev_root}.UNRST", f"{prev_root}.X*", f"{prev_root}.RSSPEC",
        f"{prev_root}.EGRID", f"{prev_root}.GRID", f"{prev_root}.INIT",
        f"{prev_root}.SMSPEC", f"{prev_root}.UNSMRY", f"{prev_root}.S*",
    ]
    for pat in patterns:
        for src in prev_dir.glob(pat):
            if src.is_file():
                dst = new_dir / src.name
                shutil.copy2(src, dst)
                copied.append(src.name)
    for pat in ["*.INC", "*.inc"]:
        for src in prev_dir.glob(pat):
            if src.is_file() and not (new_dir / src.name).exists():
                shutil.copy2(src, new_dir / src.name)
                copied.append(src.name)
    return sorted(set(copied))


def copy_base_input_files(base_dir: Path, seg_dir: Path) -> List[str]:
    """Copy static input files needed by INCLUDE statements into a segment folder."""
    copied: List[str] = []

    patterns = [
        "*.INC", "*.inc",
        "*.GRDECL", "*.grdecl",
        "*.DATA", "*.data",
    ]

    for pattern in patterns:
        for src in base_dir.glob(pattern):
            if not src.is_file():
                continue
            if src.name.upper().endswith(".DATA"):
                # Do not copy Base.DATA over SEG_xxxx.DATA
                continue
            dst = seg_dir / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)

    for dirname in ["INCLUDE", "include", "Includes", "includes"]:
        src_dir = base_dir / dirname
        if src_dir.is_dir():
            dst_dir = seg_dir / dirname
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            copied.append(dirname + "/")

    return sorted(set(copied))
