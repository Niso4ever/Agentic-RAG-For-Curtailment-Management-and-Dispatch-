from app.forecasting import forecast_solar
from app.rag_engine import retrieve_grounded_knowledge
from app.milp_solver import solve_dispatch


def run_agent(query: str):
    # Step 1 — ML Forecast
    forecast = forecast_solar()

    # Step 2 — RAG Retrieval
    grounded = retrieve_grounded_knowledge(query)

    # Step 3 — Optimization (MILP)
    dispatch = solve_dispatch(forecast["mw"])

    # Step 4 — Final combined answer
    final_answer = f"""
=== AGENTIC DISPATCH ANALYSIS ===

User Query:
{query}

Solar Forecast:
 - MW: {forecast['mw']}
 - Confidence: {forecast['confidence']}

Grounded RAG Insights:
{grounded}

Optimized Dispatch (MILP):
 - Charge MW: {dispatch['charge_mw']}
 - Dispatch MW: {dispatch['dispatch_mw']}

Final Recommendation:
Combine real forecast + RAG engineering insights + MILP to minimize curtailment and improve BESS performance.
"""

    return final_answer

