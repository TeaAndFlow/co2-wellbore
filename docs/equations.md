# Equations

The current implementation uses a segment-by-segment pressure and enthalpy update.

Pressure increment for downward injection:

```text
dP_down = dP_hydro - dP_friction + dP_acceleration
```

Heat exchange updates enthalpy rather than directly imposing a temperature increment.
Joule-Thomson cooling/heating appears through the P-H flash.
