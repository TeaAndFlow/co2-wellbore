"""Hydraulic correlations and numerical utilities."""

from __future__ import annotations

import math


def darcy_friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    """Darcy friction factor using laminar 64/Re or Haaland turbulent approximation."""
    re = max(float(reynolds), 1.0e-12)

    if re < 2300.0:
        return 64.0 / re

    eps_d = max(float(roughness_m) / max(float(diameter_m), 1.0e-12), 0.0)
    return float((-1.8 * math.log10((eps_d / 3.7) ** 1.11 + 6.9 / re)) ** -2)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return float(min(max(float(x), float(lo)), float(hi)))
