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

## Basic example

```cmd
python examples\01_basic_profile.py
```
Expected output includes a pressure/temperature profile and a compact summary with BHP and BHT.

## VFPINJ example

python examples\03_vfpinj_table.py

```cmd
python examples\03_vfpinj_table.py
```

This generates an OPM/ECL-style VFPINJ table from the same wellbore model.

## Run test

```cmd
python pytest
```
Current smoke tests check:

* configuration validation
* Darcy friction factor
* rate conversion
* basic profile calculation
* VFPINJ table generation

## Package structure

```cmd
src/co2_wellbore/
    constants.py      physical constants and unit conversions
    config.py         configuration dataclasses
    properties.py     CoolProp CO2 property backend
    correlations.py   hydraulic correlations
    drift_flux.py     simplified drift-flux closure
    calculator.py     main wellbore calculator
    opm.py            OPM/ECL export helpers
    diagnostics.py    profile diagnostic helpers
```

## Physical assumptions

Current assumptions:

Pure CO2 property backend through CoolProp
1D vertical wellbore
Positive injection rate means downward flow
P-H flash formulation for wellbore integration
Simplified steady heat exchange with geothermal surroundings
Darcy-Weisbach friction factor with Haaland turbulent approximation
Optional simplified drift-flux closure for saturated two-phase CO2 states
No full transient PDE solution yet
No hydrate model yet
No deviated well geometry yet
No explicit surface valve/choke model yet


## Roadmap

v0.1.0: clean Python package
v0.2.0: THP valve / choke model
v0.3.0: hydrate and ice diagnostics
v0.4.0: deviated well geometry
v0.5.0: OPM restart coupling example
v0.6.0: validation against analytical and published benchmark cases

## License

MIT License.
