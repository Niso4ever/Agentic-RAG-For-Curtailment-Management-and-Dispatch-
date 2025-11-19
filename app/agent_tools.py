# app/agent_tools.py

"""
Central registry of the tools that the Agentic AI system calls.

Each tool exposes a Python function that:
- takes structured arguments from the LLM
- calls the real subsystem (forecasting / RAG / MILP)
- returns a JSON-serializable Python dict

These functions are mapped inside agentic_dispatch_agent.py.
"""

from typing import Dict, Any, Optional, List

from app.forecasting import (
    forecast_solar,  # Real or Vertex AI version
    get_latest_weather_features,
)
from app.rag_engine import retrieve_grounded_knowledge
from app.milp_solver import solve_dispatch  # OR-Tools MILP version

BASE_CURTAILMENT_WEIGHT = 1000.0
BASE_CYCLE_PENALTY = 1.0


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


def get_live_weather_features() -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest weather snapshot directly from the OpenWeather API.
    Returns None when the API key/location are not configured or if the call fails.
    """
    try:
        return get_latest_weather_features()
    except Exception:
        return None


def prepare_milp_payload(
    forecast_output: Dict[str, Any],
    rag_output: Dict[str, Any],
    plant_meta: Dict[str, Any],
    weather_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the MILP input payload from trusted sources (ignore any LLM-provided numbers).
    Dynamically derives grid limits and penalty weights based on forecast confidence
    and near-real-time weather inputs.
    """
    mw_forecast = float(forecast_output.get("mw", 0.0) or 0.0)
    features_used = forecast_output.get("features_used") or {}
    confidence = float(forecast_output.get("confidence") or 0.5)
    confidence = max(0.0, min(confidence, 1.0))

    plant_rating = float(
        plant_meta.get("plant_rating_mw")
        or features_used.get("plant_rating_mw")
        or max(mw_forecast, plant_meta.get("capacity_mwh", 50.0))
    )
    interconnect_limit = float(
        plant_meta.get("interconnect_limit_mw") or plant_rating
    )

    interval_sources: List[Dict[str, Any]] = []

    # Allow callers to supply an explicit interval list; otherwise, fall back to a single step.
    forecast_intervals = forecast_output.get("dispatch_intervals") or []
    if isinstance(forecast_intervals, list) and forecast_intervals:
        interval_sources.extend(forecast_intervals)

    if not interval_sources:
        merged_weather = {**features_used, **(weather_features or {})}
        interval_sources.append(
            {
                "label": "t0",
                "mw_forecast": mw_forecast,
                "features": merged_weather,
                "confidence": confidence,
            }
        )

    dispatch_intervals: List[Dict[str, Any]] = []
    for idx, interval in enumerate(interval_sources):
        interval_mw = float(interval.get("mw_forecast", mw_forecast) or 0.0)
        interval_conf = float(interval.get("confidence", confidence) or confidence)
        interval_conf = max(0.0, min(interval_conf, 1.0))
        interval_features = interval.get("features") or interval.get("features_used") or features_used

        irradiance_factor = _estimate_irradiance_factor(
            interval_features or {},
            fallback_mw=interval_mw or mw_forecast,
            plant_rating=plant_rating,
        )
        dynamic_grid_limit = min(interconnect_limit, irradiance_factor * plant_rating)

        dispatch_intervals.append(
            {
                "label": interval.get("label", f"t{idx}"),
                "mw_forecast": interval_mw,
                "grid_limit_mw": max(dynamic_grid_limit, 0.0),
                "curtailment_weight": max(
                    10.0, interval_conf * BASE_CURTAILMENT_WEIGHT
                ),
                "cycle_penalty": max(
                    0.05, (1.0 - interval_conf) * BASE_CYCLE_PENALTY
                ),
                "irradiance_factor": irradiance_factor,
                "forecast_confidence": interval_conf,
            }
        )

    payload = {
        "mw_forecast": dispatch_intervals[0]["mw_forecast"],
        "bess_soc": plant_meta.get("soc", 0.35),
        "bess_capacity_mwh": plant_meta.get("capacity_mwh", 50.0),
        "max_charge_mw": plant_meta.get("max_charge_mw", 5.0),
        "max_discharge_mw": plant_meta.get("max_discharge_mw", 5.0),
        "dispatch_intervals": dispatch_intervals,
        "interconnect_limit_mw": interconnect_limit,
        "plant_rating_mw": plant_rating,
        "forecast_confidence": confidence,
    }

    return payload


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
    dispatch_intervals: Optional[List[Dict[str, Any]]] = None,
    **_extra: Any,
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
            dispatch_intervals=dispatch_intervals,
        )
        return result

    except Exception as e:
        return {
            "error": f"MILP solver error: {str(e)}",
            "mw_forecast": mw_forecast,
            "bess_soc": bess_soc,
            "bess_capacity_mwh": bess_capacity_mwh,
            "max_charge_mw": max_charge_mw,
            "max_discharge_mw": max_discharge_mw,
            "dispatch_intervals": dispatch_intervals,
        }


def _estimate_irradiance_factor(
    weather_features: Dict[str, Any], fallback_mw: float, plant_rating: float
) -> float:
    """
    Approximate an irradiance multiplier (0-1) from available weather signals.
    Uses temperature, wind speed, and cloud cover as heuristics.
    """
    temp = float(weather_features.get("mean_temperature") or weather_features.get("temperature") or 25.0)
    wind = float(weather_features.get("mean_wind_speed") or weather_features.get("wind_speed") or 3.0)

    cloud_cover = weather_features.get("cloud_cover")
    if cloud_cover is None:
        cloud_cover = weather_features.get("clouds")
    if cloud_cover is None:
        debug_source = weather_features.get("debug_weather_source")
        if isinstance(debug_source, dict):
            clouds = debug_source.get("clouds")
            if isinstance(clouds, dict):
                cloud_cover = clouds.get("all")
            elif isinstance(clouds, (int, float)):
                cloud_cover = clouds

    temp_factor = max(0.0, min((temp + 5.0) / 45.0, 1.0))
    wind_factor = max(0.5, min(1.0, 1.0 - 0.03 * wind))

    cloud_factor = 1.0
    if cloud_cover is not None:
        cover = max(0.0, min(float(cloud_cover), 100.0))
        cloud_factor = 1.0 - cover / 100.0

    derived = temp_factor * wind_factor * cloud_factor
    baseline_ratio = 0.0
    if plant_rating > 0:
        baseline_ratio = max(0.0, min(fallback_mw / plant_rating, 1.0))

    irradiance_factor = max(baseline_ratio, derived)
    return max(0.0, min(irradiance_factor, 1.0))
