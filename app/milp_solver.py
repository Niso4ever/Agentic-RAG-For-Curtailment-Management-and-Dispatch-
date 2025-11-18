"""MILP-based (single-period) BESS dispatch with curtailment minimization."""

from typing import Dict, Any, Optional

try:
    from ortools.linear_solver import pywraplp
except Exception as exc:  # pragma: no cover
    pywraplp = None  # type: ignore
    _ORTOOLS_IMPORT_ERROR = exc
else:
    _ORTOOLS_IMPORT_ERROR = None


def solve_dispatch(
    mw_forecast: float,
    bess_soc: float = 0.35,
    bess_capacity_mwh: float = 50.0,
    max_charge_mw: float = 5.0,
    max_discharge_mw: float = 5.0,
    grid_limit_mw: Optional[float] = None,
    curtailment_weight: float = 1000.0,
    cycle_penalty: float = 1.0,
) -> Dict[str, Any]:
    """
    Optimize single-period dispatch to minimize curtailment while respecting BESS limits.

    Assumptions:
    - One-hour interval (power == energy for this step).
    - Charging/discharging efficiencies are 100% (can be extended later).
    - Grid export is capped (default 90% of forecast if not provided) to model clipping.
    """
    if pywraplp is None:
        raise ImportError(f"Missing ortools dependency: {_ORTOOLS_IMPORT_ERROR}")

    mw_forecast = max(float(mw_forecast), 0.0)
    bess_soc = max(min(float(bess_soc), 1.0), 0.0)
    bess_capacity_mwh = max(float(bess_capacity_mwh), 0.0)
    max_charge_mw = max(float(max_charge_mw), 0.0)
    max_discharge_mw = max(float(max_discharge_mw), 0.0)

    # Default export limit: mimic prior heuristic (10% headroom, i.e., 90% deliverable)
    if grid_limit_mw is None:
        grid_limit_mw = 0.9 * mw_forecast
    grid_limit_mw = max(float(grid_limit_mw), 0.0)

    soc_start = bess_soc * bess_capacity_mwh

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:  # pragma: no cover
        raise RuntimeError("Failed to create CBC solver")

    # Decision variables
    charge_mw = solver.NumVar(0.0, max_charge_mw, "charge_mw")
    discharge_mw = solver.NumVar(0.0, max_discharge_mw, "discharge_mw")
    export_mw = solver.NumVar(0.0, grid_limit_mw, "export_mw")
    curtailment_mw = solver.NumVar(0.0, solver.infinity(), "curtailment_mw")
    soc_end = solver.NumVar(0.0, bess_capacity_mwh, "soc_end")

    # Prevent simultaneous charge and discharge via binary switches.
    y_charge = solver.BoolVar("y_charge")
    y_discharge = solver.BoolVar("y_discharge")
    solver.Add(y_charge + y_discharge <= 1)
    solver.Add(charge_mw <= max_charge_mw * y_charge)
    solver.Add(discharge_mw <= max_discharge_mw * y_discharge)

    # Power balance: forecast + discharge = export + charge + curtailment
    solver.Add(export_mw + charge_mw + curtailment_mw == mw_forecast + discharge_mw)

    # State-of-charge update (1-hour interval, unity efficiency)
    solver.Add(soc_end == soc_start + charge_mw - discharge_mw)

    # Objective: prioritize minimizing curtailment, then mild penalty on cycling.
    objective = solver.Objective()
    objective.SetCoefficient(curtailment_mw, curtailment_weight)
    objective.SetCoefficient(charge_mw, cycle_penalty)
    objective.SetCoefficient(discharge_mw, cycle_penalty)
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP solve failed with status {status}")

    return {
        "dispatch_mw": export_mw.solution_value(),
        "charge_mw": charge_mw.solution_value(),
        "discharge_mw": discharge_mw.solution_value(),
        "curtailment_mw": curtailment_mw.solution_value(),
        "soc_mwh": soc_end.solution_value(),
    }
