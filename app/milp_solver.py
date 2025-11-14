"""Simple placeholder MILP solver."""

from typing import Dict, Any


def solve_dispatch(
    mw_forecast: float,
    bess_soc: float = 0.40,
    bess_capacity_mwh: float = 10.0,
    max_charge_mw: float = 5.0,
    max_discharge_mw: float = 5.0,
) -> Dict[str, Any]:
    """
    Pseudo MILP optimization for dispatch planning.

    Replace this logic with PuLP / OR-Tools implementation later.
    """
    dispatch_mw = mw_forecast * 0.9
    charge_mw = max_charge_mw * 0.7
    discharge_mw = max_discharge_mw * 0.3
    curtailment_mw = max(mw_forecast - dispatch_mw, 0)

    return {
        "dispatch_mw": dispatch_mw,
        "charge_mw": charge_mw,
        "discharge_mw": discharge_mw,
        "curtailment_mw": curtailment_mw,
        "soc_mwh": bess_soc * bess_capacity_mwh,
    }
