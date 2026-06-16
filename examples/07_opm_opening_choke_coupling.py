#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper for the OPM restart + CO2 wellbore + choke coupling workflow.

The implementation lives in:

    co2_wellbore.coupling.opm_restart

Equivalent commands:

    python examples/07_opm_opening_choke_coupling.py --help

    python -m co2_wellbore.coupling.opm_restart --help
"""

from co2_wellbore.coupling.opm_restart import main


if __name__ == "__main__":
    main()
