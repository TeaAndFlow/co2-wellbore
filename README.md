# CO2 Wellbore

A transparent Python calculator for CO2 injection wellbore profiles, CCS/CCUS studies, CoolProp properties, and OPM Flow coupling.

## Status

Experimental research prototype.

This package is intended for research workflows, quick sensitivity studies, and coupling experiments with reservoir simulators such as OPM Flow.

## What this is

`co2-wellbore` computes 1D CO2 injection wellbore pressure and temperature profiles using CoolProp thermophysical properties and a P-H flash formulation.

Current capabilities:

- CO2 pressure and temperature profile from THP and injection rate
- BHP and BHT calculation
- Required THP for target BHP
- Injectivity-controlled rate solver
- Simplified saturated two-phase drift-flux closure
- OPM/ECL-style VFPINJ table generation
- Basic profile diagnostics

## What this is not

This is not OLGA, LedaFlow, PIPESIM, or a fully validated industrial transient multiphase simulator.

The current model is a transparent research calculator. It uses simplified wellbore physics and should be validated before any engineering decision-making.

## Installation

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"