# Validation and Verification Scope for v0.2.8

This validation layer covers the external reduced-order CO₂ wellbore and choke components.

The OPM Flow reservoir simulator is treated as an external simulator and is not revalidated here. The coupling workflow uses OPM outputs such as WBHP, rates and restart states, but the present validation focuses only on:

- CO₂ thermodynamic property sanity checks,
- 1D wellbore pressure limiting cases,
- reduced thermal-model limiting cases,
- HEM choke physical checks,
- HEM critical-pressure numerical convergence.

## Scientific claim

The tests in this repository verify that the reduced-order wellbore/choke model behaves consistently with thermodynamic, hydraulic and numerical limiting cases.

They do **not** claim equivalence to OLGA, PROSPER, a field-calibrated commercial wellbore simulator, or a full transient multiphase wellbore PDE simulator.

## Validation ladder

| Component | Test type | Evidence |
|---|---|---|
| CO₂ properties | sanity checks | finite density, enthalpy, viscosity, phase labels, P-H flash consistency |
| CO₂ two-phase flash | thermodynamic consistency | saturated P-H state gives `0 <= quality <= 1` |
| Wellbore pressure | analytical limiting case | tiny-flow pressure trend is close to hydrostatic estimate |
| Wellbore pressure | monotonicity | increasing THP increases BHP |
| Wellbore thermal model | limiting cases | no heat transfer, low-U, high-U behavior |
| HEM choke capacity | physical positivity | finite positive mass capacity and critical pressure |
| HEM choke capacity | monotonicity | larger opening and larger upstream pressure increase capacity |
| HEM critical flow | choked-flow behavior | capacity is insensitive to downstream pressure below critical pressure |
| HEM critical pressure | numerical convergence | grid-refined critical pressure/capacity compared against high-resolution reference |

## HEM critical-pressure search

Before v0.2.8, the HEM critical pressure was selected directly from a finite log-spaced pressure grid.

That approach is robust, but grid-dependent: the true maximum mass flux can lie between two grid points.

In v0.2.8, the algorithm uses:

1. a coarse log-spaced pressure grid to locate the approximate maximum,
2. a bounded scalar optimizer in log-pressure space to refine the maximum,
3. the original grid result as a fallback if the optimizer fails or enters invalid CoolProp states.

This makes the HEM critical pressure less dependent on arbitrary grid resolution while preserving robustness.

## Running the validation layer

From the repository root:

```bash
pip install -e ".[dev]"
pytest -q tests/validation
python scripts/validation/run_hem_grid_convergence.py
```

Or run everything:

```bash
./scripts/validation/run_validation_suite.sh
```


