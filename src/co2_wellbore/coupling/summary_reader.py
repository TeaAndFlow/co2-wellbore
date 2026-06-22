from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .deck_editing import StepControl
from .models import ReservoirResult

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
