"""Multi-interval MILP-based BESS dispatch with curtailment minimization."""

from typing import Dict, Any, Optional, List

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
    dispatch_intervals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Multi-interval MILP:

    - If dispatch_intervals is provided (recommended), it defines the horizon.
      Each interval dict may contain:

        {
          "label": str,
          "mw_forecast": float,
          "grid_limit_mw": float,        # optional (default 0.9 * mw_forecast)
          "curtailment_weight": float,   # optional (default curtailment_weight)
          "cycle_penalty": float,        # optional (default cycle_penalty)
          "irradiance_factor": float,    # optional (pass-through)
          "forecast_confidence": float,  # optional (pass-through)
          ...
        }

    - If dispatch_intervals is None, we fall back to a single-interval model
      using mw_forecast and grid_limit_mw.

    The objective minimizes the sum over intervals of:

        curtailment[i] * curtailment_weight[i]
      + (charge[i] + discharge[i]) * cycle_penalty[i]

    Subject to:
      - energy balance for each interval and SoC evolution across time
      - mutual exclusivity of charge/discharge
      - grid export limits per interval
    """
    if pywraplp is None:
        raise ImportError(f"Missing ortools dependency: {_ORTOOLS_IMPORT_ERROR}")

    # Validate and normalize inputs
    mw_forecast = max(float(mw_forecast), 0.0)
    bess_soc = max(min(float(bess_soc), 1.0), 0.0)
    bess_capacity_mwh = max(float(bess_capacity_mwh), 0.0)
    max_charge_mw = max(float(max_charge_mw), 0.0)
    max_discharge_mw = max(float(max_discharge_mw), 0.0)

    # Build interval list
    intervals: List[Dict[str, Any]] = []

    if dispatch_intervals and isinstance(dispatch_intervals, list):
        for idx, raw in enumerate(dispatch_intervals):
            if not isinstance(raw, dict):
                continue
            interval = dict(raw)
            interval.setdefault("label", f"t{idx}")
            interval_mw = float(interval.get("mw_forecast", mw_forecast) or 0.0)
            interval["mw_forecast"] = max(interval_mw, 0.0)

            # Dynamic grid limit: default to 0.9 * mw_forecast if not provided
            gl = interval.get("grid_limit_mw")
            if gl is None:
                gl = 0.9 * interval["mw_forecast"]
            interval["grid_limit_mw"] = max(float(gl), 0.0)

            cw = float(interval.get("curtailment_weight", curtailment_weight))
            cp = float(interval.get("cycle_penalty", cycle_penalty))
            interval["curtailment_weight"] = cw
            interval["cycle_penalty"] = cp

            intervals.append(interval)

    # Fallback: single interval based on top-level mw_forecast.
    if not intervals:
        label = "t0"
        interval_mw = mw_forecast
        if grid_limit_mw is None:
            grid_limit_mw = 0.9 * interval_mw
        interval = {
            "label": label,
            "mw_forecast": max(interval_mw, 0.0),
            "grid_limit_mw": max(float(grid_limit_mw), 0.0),
            "curtailment_weight": float(curtailment_weight),
            "cycle_penalty": float(cycle_penalty),
            "irradiance_factor": None,
            "forecast_confidence": None,
        }
        intervals.append(interval)

    horizon = len(intervals)

    soc_start = bess_soc * bess_capacity_mwh

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:  # pragma: no cover
        raise RuntimeError("Failed to create CBC solver")

    # Decision variables
    charge_vars: List[Any] = []
    discharge_vars: List[Any] = []
    dispatch_vars: List[Any] = []
    curtailment_vars: List[Any] = []
    soc_vars: List[Any] = []
    y_charge_vars: List[Any] = []
    y_discharge_vars: List[Any] = []

    for t in range(horizon):
        interval = intervals[t]
        glimit = float(interval["grid_limit_mw"])

        charge = solver.NumVar(0.0, max_charge_mw, f"charge_mw_{t}")
        discharge = solver.NumVar(0.0, max_discharge_mw, f"discharge_mw_{t}")
        dispatch = solver.NumVar(0.0, glimit, f"dispatch_mw_{t}")
        curtail = solver.NumVar(0.0, solver.infinity(), f"curtailment_mw_{t}")
        soc = solver.NumVar(0.0, bess_capacity_mwh, f"soc_mwh_end_{t}")

        y_charge = solver.BoolVar(f"y_charge_{t}")
        y_discharge = solver.BoolVar(f"y_discharge_{t}")

        charge_vars.append(charge)
        discharge_vars.append(discharge)
        dispatch_vars.append(dispatch)
        curtailment_vars.append(curtail)
        soc_vars.append(soc)
        y_charge_vars.append(y_charge)
        y_discharge_vars.append(y_discharge)

    # Constraints: mutual exclusivity and operating bounds
    for t in range(horizon):
        charge = charge_vars[t]
        discharge = discharge_vars[t]
        y_charge = y_charge_vars[t]
        y_discharge = y_discharge_vars[t]

        # Only charge or discharge in a given interval
        solver.Add(y_charge + y_discharge <= 1.0)
        solver.Add(charge <= max_charge_mw * y_charge)
        solver.Add(discharge <= max_discharge_mw * y_discharge)

    # Energy and flow constraints
    for t in range(horizon):
        interval = intervals[t]
        mw_t = float(interval["mw_forecast"])
        dispatch = dispatch_vars[t]
        charge = charge_vars[t]
        discharge = discharge_vars[t]
        curtail = curtailment_vars[t]
        soc = soc_vars[t]

        # Power balance: generation + discharge = dispatch + charge + curtailment
        solver.Add(dispatch + charge + curtail == mw_t + discharge)

        # SoC evolution
        if t == 0:
            solver.Add(soc == soc_start + charge - discharge)
        else:
            solver.Add(soc == soc_vars[t - 1] + charge - discharge)

    # Objective: minimize total curtailment and cycling penalties over the horizon
    objective = solver.Objective()
    for t in range(horizon):
        interval = intervals[t]
        cw = float(interval["curtailment_weight"])
        cp = float(interval["cycle_penalty"])

        objective.SetCoefficient(curtailment_vars[t], cw)
        objective.SetCoefficient(charge_vars[t], cp)
        objective.SetCoefficient(discharge_vars[t], cp)

    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP solve failed with status {status}")

    # Collect interval-wise results
    interval_results: List[Dict[str, Any]] = []
    total_curtailment = 0.0

    for t in range(horizon):
        interval = intervals[t]
        dispatch = dispatch_vars[t].solution_value()
        charge = charge_vars[t].solution_value()
        discharge = discharge_vars[t].solution_value()
        curtail = curtailment_vars[t].solution_value()
        soc_end = soc_vars[t].solution_value()

        total_curtailment += curtail

        interval_results.append(
            {
                "interval": t,
                "label": interval.get("label", f"t{t}"),
                "mw_forecast": float(interval["mw_forecast"]),
                "grid_limit_mw": float(interval["grid_limit_mw"]),
                "curtailment_weight": float(interval["curtailment_weight"]),
                "cycle_penalty": float(interval["cycle_penalty"]),
                "irradiance_factor": interval.get("irradiance_factor"),
                "forecast_confidence": interval.get("forecast_confidence"),
                "dispatch_mw": dispatch,
                "charge_mw": charge,
                "discharge_mw": discharge,
                "curtailment_mw": curtail,
                "soc_mwh_end": soc_end,
            }
        )

    # For backward compatibility, expose the first-interval decisions at top level.
    first = interval_results[0]

    return {
        "dispatch_mw": first["dispatch_mw"],
        "charge_mw": first["charge_mw"],
        "discharge_mw": first["discharge_mw"],
        "curtailment_mw": first["curtailment_mw"],
        "soc_mwh": interval_results[-1]["soc_mwh_end"],
        "total_curtailment_mw": total_curtailment,
        "intervals": interval_results,
    }
