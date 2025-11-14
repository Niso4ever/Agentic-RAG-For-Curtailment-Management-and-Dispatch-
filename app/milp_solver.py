def solve_dispatch(forecast_mw: float):
    """
    Placeholder MILP optimization.
    Later we replace this with PuLP / OR-Tools real optimization.
    """
    optimal_dispatch = max(0, forecast_mw - 5)  # fake logic
    return {
        "charge_mw": 5,
        "dispatch_mw": optimal_dispatch
    }
