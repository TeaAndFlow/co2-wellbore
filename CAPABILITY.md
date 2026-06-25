# Capability Envelope and Stress-Test Scope for v0.2.9

Version `v0.2.9` is a capability-envelope release.

It does not add salt precipitation or hydrate prediction. Instead, it maps where the current external CO2 wellbore/choke model behaves robustly, where it fails, and where the results should be treated with caution.

## Purpose

The goal is to stress-test the existing components before adding new physics:

- CoolProp CO2 property backend;
- P-H flash behavior;
- reduced 1D CO2 wellbore pressure/thermal model;
- drift-flux closure for saturated two-phase CO2 states;
- real-fluid HEM choke capacity model;
- restart-based OPM Flow coupling diagnostics.

## Main output files

After running:

```bash
./scripts/validation/run_validation_suite.sh
```

the capability sweep writes:
