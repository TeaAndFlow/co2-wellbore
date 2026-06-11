"""Diagnostic helpers for computed wellbore profiles."""

from __future__ import annotations


def min_temperature_C(profile) -> float:
    """Return minimum temperature in a profile."""
    if hasattr(profile, "__getitem__") and "temperature_C" in profile:
        return float(profile["temperature_C"].min())

    return min(float(row["temperature_C"]) for row in profile)


def has_low_temperature_risk(profile, threshold_C: float = 0.0) -> bool:
    """Return True if the profile goes below a temperature threshold."""
    return min_temperature_C(profile) < threshold_C


def summarize_profile(profile) -> dict:
    """Return a compact summary of a wellbore profile."""
    if hasattr(profile, "iloc"):
        return {
            "min_temperature_C": float(profile["temperature_C"].min()),
            "max_temperature_C": float(profile["temperature_C"].max()),
            "wellhead_pressure_bar": float(profile["pressure_bar"].iloc[0]),
            "bottomhole_pressure_bar": float(profile["pressure_bar"].iloc[-1]),
            "bottomhole_temperature_C": float(profile["temperature_C"].iloc[-1]),
        }

    return {
        "min_temperature_C": min(float(row["temperature_C"]) for row in profile),
        "max_temperature_C": max(float(row["temperature_C"]) for row in profile),
        "wellhead_pressure_bar": float(profile[0]["pressure_bar"]),
        "bottomhole_pressure_bar": float(profile[-1]["pressure_bar"]),
        "bottomhole_temperature_C": float(profile[-1]["temperature_C"]),
    }
