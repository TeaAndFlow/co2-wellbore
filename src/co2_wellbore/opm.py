"""OPM/ECL export helpers."""

from __future__ import annotations

from typing import Optional, Sequence


def make_vfpinj_table(
    calculator,
    table_id: int,
    rates_sm3_day: Sequence[float],
    thp_values_bar: Sequence[float],
    wellhead_temperature_C: float = 40.0,
    reference_depth_m: Optional[float] = None,
    flo_phase: str = "GAS",
    decimals: int = 4,
) -> str:
    """Generate an ECL/OPM-style VFPINJ table from a wellbore calculator."""
    ref = calculator.geometry.tvd_m if reference_depth_m is None else float(reference_depth_m)

    rates = [float(x) for x in rates_sm3_day]
    thps = [float(x) for x in thp_values_bar]

    if any(rate <= 0.0 for rate in rates):
        raise ValueError("rates_sm3_day must contain positive values.")
    if any(thp <= 0.0 for thp in thps):
        raise ValueError("thp_values_bar must contain positive values.")
    if rates != sorted(rates):
        raise ValueError("rates_sm3_day must be monotonically increasing.")
    if thps != sorted(thps):
        raise ValueError("thp_values_bar must be monotonically increasing.")

    fmt = f"{{:.{decimals}f}}"

    lines: list[str] = []
    lines.append("VFPINJ")
    lines.append(f"  {int(table_id)}  {fmt.format(ref)}  {flo_phase.upper()}  THP  1*  BHP /")
    lines.append("  " + "  ".join(fmt.format(rate) for rate in rates) + " /")
    lines.append("  " + "  ".join(fmt.format(thp) for thp in thps) + " /")

    for thp in thps:
        bhps = [
            calculator.bhp_from_thp_and_rate(
                thp,
                q_sm3_day=rate,
                wellhead_temperature_C=wellhead_temperature_C,
            )
            for rate in rates
        ]
        lines.append("  " + "  ".join(fmt.format(bhp) for bhp in bhps) + " /")

    lines.append("/")
    return "\n".join(lines)
