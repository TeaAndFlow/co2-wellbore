# v0.2.9 full capability sweep summary

Total component-level cases: **44761**

## Status counts

| component | status | count |
|---|---:|---:|
| hem_choke | ok | 3120 |
| properties | ok | 169 |
| wellbore | fail | 132 |
| wellbore | ok | 36316 |
| wellbore | warning | 5024 |

## Failure / warning reasons

| component | status | reason | count |
|---|---|---|---:|
| wellbore | warning | VERY_LOW_WELLBORE_PRESSURE | 4942 |
| wellbore | fail | COOLPROP_INVALID_STATE | 132 |
| wellbore | warning | HIGH_WELLBORE_VELOCITY | 82 |

## Runtime by component

| component | count | mean s | median s | max s | sum s |
|---|---:|---:|---:|---:|---:|
| hem_choke | 3120 | 0.2928 | 0.2967 | 0.4279 | 913.5 |
| properties | 169 | 0.001153 | 0.001119 | 0.001954 | 0.1948 |
| wellbore | 41472 | 0.4257 | 0.3398 | 1.788 | 1.765e+04 |

## Interpretation

- HEM choke behavior is stable and physically monotonic over the tested grid.
- CO2 P/T property calls are stable over the tested grid.
- The reduced wellbore model identifies a substantial warning region dominated by VERY_LOW_WELLBORE_PRESSURE.
- This release documents capability boundaries before adding salt precipitation or hydrate prediction.
- Post-choke P-H stress maps and hydrate-risk proxy maps are future work.