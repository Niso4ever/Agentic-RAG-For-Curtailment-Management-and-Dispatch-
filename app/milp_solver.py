"""MILP-based (multi-period) BESS dispatch with curtailment minimization."""

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
    Optimize dispatch over one or more intervals to minimize curtailment while respecting BESS limits.

    Assumptions:
    - Each interval represents one hour (power == energy for this step).
    - Charging/discharging efficiencies are 100% (can be extended later).
    - Grid export is capped per interval (default 90% of forecast if not provided) to model clipping.
    """
    if pywraplp is None:
        raise ImportError(f"Missing ortools dependency: {_ORTOOLS_IMPORT_ERROR}")

    mw_forecast = max(float(mw_forecast), 0.0)
    bess_soc = max(min(float(bess_soc), 1.0), 0.0)
    bess_capacity_mwh = max(float(bess_capacity_mwh), 0.0)
    max_charge_mw = max(float(max_charge_mw), 0.0)
    max_discharge_mw = max(float(max_discharge_mw), 0.0)

    default_grid_limit = max(float(grid_limit_mw) if grid_limit_mw is not None else 0.9 * mw_forecast, 0.0)

    horizon: List[Dict[str, Any]] = []
    if dispatch_intervals:
        for idx, interval in enumerate(dispatch_intervals):
            mw_val = max(float(interval.get("mw_forecast", mw_forecast)), 0.0)
            grid_cap = interval.get("grid_limit_mw")
            if grid_cap is None:
                grid_cap = 0.9 * mw_val
            grid_cap = max(float(grid_cap), 0.0)
            horizon.append(
                {
                    "mw_forecast": mw_val,
                    "grid_limit_mw": grid_cap,
                    "curtailment_weight": float(interval.get("curtailment_weight", curtailment_weight)),
                    "cycle_penalty": float(interval.get("cycle_penalty", cycle_penalty)),
                    "label": interval.get("label", f"t{idx}"),
                    "irradiance_factor": interval.get("irradiance_factor"),
                    "forecast_confidence": interval.get("forecast_confidence"),
                }
            )
    else:
        horizon = [
            {
                "mw_forecast": mw_forecast,
                "grid_limit_mw": default_grid_limit,
                "curtailment_weight": curtailment_weight,
                "cycle_penalty": cycle_penalty,
                "label": "t0",
            }
        ]

    num_steps = len(horizon)
    if num_steps == 0:
        raise ValueError("dispatch_intervals must contain at least one step.")

    soc_start = bess_soc * bess_capacity_mwh

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:  # pragma: no cover
        raise RuntimeError("Failed to create CBC solver")

    # Decision variables per interval
    charge_vars = [
        solver.NumVar(0.0, max_charge_mw, f"charge_mw_{idx}") for idx in range(num_steps)
    ]
    discharge_vars = [
        solver.NumVar(0.0, max_discharge_mw, f"discharge_mw_{idx}") for idx in range(num_steps)
    ]
    export_vars = [
        solver.NumVar(0.0, horizon[idx]["grid_limit_mw"], f"export_mw_{idx}")
        for idx in range(num_steps)
    ]
    curtailment_vars = [
        solver.NumVar(0.0, solver.infinity(), f"curtailment_mw_{idx}")
        for idx in range(num_steps)
    ]
    soc_vars = [
        solver.NumVar(0.0, bess_capacity_mwh, f"soc_mwh_{idx}") for idx in range(num_steps + 1)
    ]

    # Prevent simultaneous charge/discharge per interval.
    for idx in range(num_steps):
        y_charge = solver.BoolVar(f"y_charge_{idx}")
        y_discharge = solver.BoolVar(f"y_discharge_{idx}")
        solver.Add(y_charge + y_discharge <= 1)
        solver.Add(charge_vars[idx] <= max_charge_mw * y_charge)
        solver.Add(discharge_vars[idx] <= max_discharge_mw * y_discharge)

    # Initial SOC
    solver.Add(soc_vars[0] == soc_start)

    for idx in range(num_steps):
        interval = horizon[idx]
        solver.Add(
            export_vars[idx] + charge_vars[idx] + curtailment_vars[idx]
            == interval["mw_forecast"] + discharge_vars[idx]
        )
        solver.Add(
            soc_vars[idx + 1] == soc_vars[idx] + charge_vars[idx] - discharge_vars[idx]
        )

    # Objective: prioritize minimizing curtailment, then penalize cycling.
    objective = solver.Objective()
    for idx, interval in enumerate(horizon):
        objective.SetCoefficient(curtailment_vars[idx], interval["curtailment_weight"])
        objective.SetCoefficient(charge_vars[idx], interval["cycle_penalty"])
        objective.SetCoefficient(discharge_vars[idx], interval["cycle_penalty"])
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP solve failed with status {status}")

    interval_results: List[Dict[str, Any]] = []
    for idx in range(num_steps):
        interval = horizon[idx]
        interval_results.append(
            {
                "interval": idx,
                "label": interval.get("label", f"t{idx}"),
                "mw_forecast": interval["mw_forecast"],
                "grid_limit_mw": interval["grid_limit_mw"],
                "curtailment_weight": interval["curtailment_weight"],
                "cycle_penalty": interval["cycle_penalty"],
                "irradiance_factor": interval.get("irradiance_factor"),
                "forecast_confidence": interval.get("forecast_confidence"),
                "dispatch_mw": export_vars[idx].solution_value(),
                "charge_mw": charge_vars[idx].solution_value(),
                "discharge_mw": discharge_vars[idx].solution_value(),
                "curtailment_mw": curtailment_vars[idx].solution_value(),
                "soc_mwh_end": soc_vars[idx + 1].solution_value(),
            }
        )

    summary = interval_results[0] if interval_results else {}

    return {
        "dispatch_mw": summary.get("dispatch_mw", 0.0),
        "charge_mw": summary.get("charge_mw", 0.0),
        "discharge_mw": summary.get("discharge_mw", 0.0),
        "curtailment_mw": summary.get("curtailment_mw", 0.0),
        "soc_mwh": soc_vars[-1].solution_value(),
        "intervals": interval_results,
    }
