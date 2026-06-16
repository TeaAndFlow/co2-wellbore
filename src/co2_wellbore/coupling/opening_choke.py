from __future__ import annotations

import math
from typing import Any, Dict

from co2_wellbore.constants import BAR_TO_PA

from .reporting import _safe_float

def co2_phase_split_from_ph(
    props: Any,
    *,
    pressure_bar: float,
    enthalpy_J_kg: float,
    mass_rate_kg_s: float,
    prefix: str,
) -> Dict[str, object]:
    """Return vapor/liquid-or-dense mass split from a P-H CO2 flash.

    quality_mass is interpreted as vapor mass fraction when 0 <= Q <= 1.
    If the state is single phase, the split is assigned by phase label:
      - gas/vapor/superheated -> vapor fraction = 1
      - dense/compressed/liquid/supercritical_dense -> vapor fraction = 0
    """
    out: Dict[str, object] = {
        f"{prefix}_pressure_bar": float(pressure_bar),
        f"{prefix}_temperature_C": float("nan"),
        f"{prefix}_phase_label": "",
        f"{prefix}_quality_mass": float("nan"),
        f"{prefix}_vapor_mass_fraction": float("nan"),
        f"{prefix}_liquid_or_dense_mass_fraction": float("nan"),
        f"{prefix}_vapor_mass_rate_kg_s": float("nan"),
        f"{prefix}_liquid_or_dense_mass_rate_kg_s": float("nan"),
    }

    try:
        flash = props.props_ph(float(pressure_bar) * BAR_TO_PA, float(enthalpy_J_kg))
        phase = str(flash.get("phase_label", ""))
        q_raw = _safe_float(flash.get("quality_mass", float("nan")))
        temp_C = _safe_float(flash.get("temperature_K", float("nan"))) - 273.15

        if math.isfinite(q_raw) and 0.0 <= q_raw <= 1.0:
            vapor_fraction = min(max(q_raw, 0.0), 1.0)
            quality = vapor_fraction
        else:
            phase_low = phase.lower()
            if ("vapor" in phase_low or "gas" in phase_low) and "dense" not in phase_low:
                vapor_fraction = 1.0
            else:
                vapor_fraction = 0.0
            quality = float("nan")

        liquid_fraction = 1.0 - vapor_fraction

        out.update(
            {
                f"{prefix}_temperature_C": float(temp_C),
                f"{prefix}_phase_label": phase,
                f"{prefix}_quality_mass": float(quality),
                f"{prefix}_vapor_mass_fraction": float(vapor_fraction),
                f"{prefix}_liquid_or_dense_mass_fraction": float(liquid_fraction),
                f"{prefix}_vapor_mass_rate_kg_s": float(vapor_fraction * mass_rate_kg_s),
                f"{prefix}_liquid_or_dense_mass_rate_kg_s": float(liquid_fraction * mass_rate_kg_s),
            }
        )
    except Exception as exc:
        out[f"{prefix}_phase_label"] = f"flash_failed: {exc}"

    return out
