# app/agent_tools.py

"""
Central registry of the tools that the Agentic AI system calls.

Each tool exposes a Python function that:
- takes structured arguments from the LLM
- calls the real subsystem (forecasting / RAG / MILP)
- returns a JSON-serializable Python dict

These functions are mapped inside agentic_dispatch_agent.py.
"""

from typing import Dict, Any

from app.forecasting import forecast_solar          # Real or Vertex AI version
from app.rag_engine import retrieve_grounded_knowledge
from app.milp_solver import solve_dispatch          # OR-Tools MILP version


# =====================================================================
# SOLAR FORECAST TOOL
# =====================================================================

def get_solar_forecast_stub() -> Dict[str, Any]:
    """
    Wrapper tool — returns dictionary so LLM can consume it.
    If Vertex AI is configured, forecast_solar() uses real prediction.
    """
    try:
        return forecast_solar()
    except Exception as e:
        return {
            "error": f"Forecasting error: {str(e)}",
            "mw": 0.0,
            "confidence": 0.0
        }


def prepare_milp_payload(forecast_output: Dict[str, Any], rag_output: Dict[str, Any], plant_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the MILP input payload from trusted sources (ignore any LLM-provided numbers).
    """
    return {
        "mw_forecast": forecast_output.get("mw", 0.0) or 0.0,
        "bess_soc": plant_meta.get("soc", 0.35),
        "bess_capacity_mwh": plant_meta.get("capacity_mwh", 50.0),
        "max_charge_mw": plant_meta.get("max_charge_mw", 5.0),
        "max_discharge_mw": plant_meta.get("max_discharge_mw", 5.0),
    }


# =====================================================================
# RAG TOOL — real FAISS vector search
# =====================================================================

def get_rag_insights_stub(query: str) -> Dict[str, Any]:
    """
    Wrapper for FAISS RAG engine: returns retrieved chunks + metadata.
    """
    try:
        return retrieve_grounded_knowledge(query=query)
    except Exception as e:
        return {
            "error": f"RAG retrieval error: {str(e)}",
            "query": query,
            "results": []
        }


# =====================================================================
# MILP TOOL — OR-Tools optimization engine
# =====================================================================

def solve_milp_dispatch_stub(
    mw_forecast: float,
    bess_soc: float = 0.35,
    bess_capacity_mwh: float = 50.0,
    max_charge_mw: float = 5.0,
    max_discharge_mw: float = 5.0,
) -> Dict[str, Any]:
    """
    Wrapper for OR-Tools MILP solver — fully realistic optimization.
    The LLM passes parameters through tool call JSON.
    """
    try:
        result = solve_dispatch(
            mw_forecast=mw_forecast,
            bess_soc=bess_soc,
            bess_capacity_mwh=bess_capacity_mwh,
            max_charge_mw=max_charge_mw,
            max_discharge_mw=max_discharge_mw,
        )
        return result

    except Exception as e:
        return {
            "error": f"MILP solver error: {str(e)}",
            "mw_forecast": mw_forecast,
            "bess_soc": bess_soc,
            "bess_capacity_mwh": bess_capacity_mwh,
            "max_charge_mw": max_charge_mw,
            "max_discharge_mw": max_discharge_mw
        }
