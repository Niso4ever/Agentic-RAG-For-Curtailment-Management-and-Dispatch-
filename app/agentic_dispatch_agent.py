# app/agentic_dispatch_agent.py

import json
import os
from statistics import mean
from typing import List, Dict, Any

from app.llm_client import client, MODEL_NAME
from app import agent_tools


# ============================================================
# TOOL DEFINITIONS (Responses API — correct structure)
# ============================================================

def _get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_solar_forecast_stub",
            "type": "function",
            "description": "Retrieve short-term solar forecast.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_rag_insights_stub",
            "type": "function",
            "description": "Retrieve grounded engineering knowledge using the local RAG engine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "solve_milp_dispatch_stub",
            "type": "function",
            "description": "Run MILP-based dispatch optimization using OR-Tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mw_forecast": {"type": "number"},
                    "bess_soc": {"type": "number"},
                    "bess_capacity_mwh": {"type": "number"},
                    "max_charge_mw": {"type": "number"},
                    "max_discharge_mw": {"type": "number"}
                },
                "required": [
                    "mw_forecast",
                    "bess_soc",
                    "bess_capacity_mwh",
                    "max_charge_mw",
                    "max_discharge_mw"
                ]
            }
        },
    ]


# ============================================================
# TOOL ROUTER (LLM → Python)
# ============================================================

def _run_client_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:

    if name == "get_solar_forecast_stub":
        return agent_tools.get_solar_forecast_stub()

    if name == "get_rag_insights_stub":
        return agent_tools.get_rag_insights_stub(arguments.get("query", ""))

    if name == "solve_milp_dispatch_stub":
        return agent_tools.solve_milp_dispatch_stub(
            mw_forecast=float(arguments["mw_forecast"]),
            bess_soc=float(arguments["bess_soc"]),
            bess_capacity_mwh=float(arguments["bess_capacity_mwh"]),
            max_charge_mw=float(arguments["max_charge_mw"]),
            max_discharge_mw=float(arguments["max_discharge_mw"]),
        )

    return {"error": f"Unknown tool '{name}'"}


# ============================================================
# LOCAL FALLBACK (when no LLM)
# ============================================================

def _run_local_stub_answer(user_query: str) -> str:
    forecast = agent_tools.get_solar_forecast_stub()
    rag = agent_tools.get_rag_insights_stub(user_query)
    dispatch = agent_tools.solve_milp_dispatch_stub(
        mw_forecast=float(forecast.get("mw", 0) or 0.0)
    )

    return f"""
LLM client not configured. Showing offline pipeline output instead.

=== OFFLINE DISPATCH ANALYSIS ===
Query: {user_query}

Solar Forecast → {forecast}
RAG Insights → {rag}
MILP Dispatch → {dispatch}

Set OPENAI_API_KEY and install the OpenAI SDK to enable real model usage.
"""


# ============================================================
# SOLAR FORECAST PREDICTION (historical + future rows)
# ============================================================

def _naive_projection(
    historical_data: List[Dict[str, Any]],
    future_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Lightweight, dependency-free forecast: extrapolate using the average of recent deltas.
    Falls back to a flat line when only one or zero historical points exist.
    """
    outputs = [
        float(row["target_solar_output"])
        for row in historical_data
        if row.get("target_solar_output") is not None
    ]

    if not outputs:
        last_point = 0.0
        slope = 0.0
    elif len(outputs) == 1:
        last_point = outputs[-1]
        slope = 0.0
    else:
        last_point = outputs[-1]
        deltas = [b - a for a, b in zip(outputs[-3:-1], outputs[-2:])]
        slope = mean(deltas) if deltas else outputs[-1] - outputs[-2]

    projections: List[Dict[str, Any]] = []
    for idx, row in enumerate(future_data):
        next_val = last_point + slope * (idx + 1)
        projections.append(
            {
                **row,
                "target_solar_output": float(next_val),
            }
        )
    return projections


def _vertex_timeseries_prediction(
    historical_data: List[Dict[str, Any]],
    future_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Send combined historical+future rows to a Vertex AI endpoint.
    Assumes a time-series/AutoML model that can handle missing targets for the prediction window.
    """
    project = "pristine-valve-477208i1"
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID") or os.getenv("VERTEX_FORECAST_ENDPOINT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not project or not endpoint_id:
        raise ValueError("Missing VERTEX_PROJECT_ID or VERTEX_ENDPOINT_ID for Vertex inference")

    try:
        from google.cloud import aiplatform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("google-cloud-aiplatform is not installed") from exc

    aiplatform.init(project=project, location=location)

    endpoint_path = (
        endpoint_id
        if endpoint_id.startswith("projects/")
        else f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )

    endpoint = aiplatform.Endpoint(endpoint_path)

    # Ensure future rows include the target column with None so the model knows to predict it.
    instances: List[Dict[str, Any]] = []
    for row in historical_data:
        instances.append(dict(row))
    for row in future_data:
        instance = dict(row)
        instance.setdefault("target_solar_output", None)
        instances.append(instance)

    prediction = endpoint.predict(instances=instances)
    raw_predictions = prediction.predictions or []

    # Grab as many predictions as we have future rows (take the tail if extra values are returned).
    future_count = len(future_data)
    trimmed = raw_predictions[-future_count:] if len(raw_predictions) >= future_count else raw_predictions

    normalized: List[Dict[str, Any]] = []
    for base_row, raw in zip(future_data, trimmed):
        if isinstance(raw, dict):
            value = raw.get("value") or raw.get("predicted_value") or next(iter(raw.values()), 0.0)
        else:
            value = raw
        normalized.append(
            {
                **base_row,
                "target_solar_output": float(value),
            }
        )

    # If the endpoint returned fewer predictions than requested, pad with naive estimates.
    if len(normalized) < future_count:
        missing_future = future_data[len(normalized):]
        normalized.extend(_naive_projection(historical_data, missing_future))

    return normalized


def get_solar_forecast_prediction(
    historical_data: List[Dict[str, Any]],
    future_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Predict solar output for the provided future horizon.
    Uses Vertex AI when configured; otherwise falls back to a simple local extrapolation.
    """
    provider = os.getenv("FORECAST_PROVIDER", "stub").lower()

    if provider == "vertex":
        try:
            return _vertex_timeseries_prediction(historical_data, future_data)
        except Exception as exc:
            print(f"Vertex prediction failed ({exc}); falling back to naive projection.")

    return _naive_projection(historical_data, future_data)


# ============================================================
# MAIN AGENT LOOP (LLM → tool calls → final answer)
# ============================================================

def run_agentic_dispatch(user_query: str) -> str:
    if client is None:
        return _run_local_stub_answer(user_query)

    tools = _get_tool_definitions()

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an Agentic AI system specializing in curtailment and BESS dispatch optimization.\n"
                    "Follow this strict sequence for EVERY request:\n"
                    "  1. Call get_solar_forecast_stub (forecast)\n"
                    "  2. Call get_rag_insights_stub with the user query\n"
                    "  3. Call solve_milp_dispatch_stub using the BESS parameters from the user (or reasonable defaults)\n"
                    "Only after all THREE tool outputs are available may you craft a final answer.\n"
                    "Your answer must explicitly reference the forecast, retrieved insights, and MILP dispatch recommendation."
                )
            },
            {"role": "user", "content": user_query},
        ],
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    while True:
        _debug_dump("Model output", response.output)
        tool_outputs = []

        for item in response.output:
            if getattr(item, "type", "") in ("function_call", "custom_tool_call"):
                name = item.name
                call_id = item.call_id
                raw_args = item.arguments

                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                else:
                    args = raw_args or {}

                result = _run_client_tool(name, args)
                print(f"Tool {name} returned: {json.dumps(result, indent=2)}")

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    }
                )

        if tool_outputs:
            response = client.responses.create(
                model=MODEL_NAME,
                input=tool_outputs,
                previous_response_id=response.id,
                tools=tools,
            )
            continue

        final_chunks: List[str] = []
        for item in response.output:
            if getattr(item, "type", "") == "message":
                for part in item.content:
                    if getattr(part, "type", "") == "output_text":
                        final_chunks.append(part.text)

        print(f"Final agent response: {final_chunks}")
        return "\n".join(final_chunks) if final_chunks else "Agent finished with no text output."
def _debug_dump(label: str, payload: List[Any]) -> None:
    """Best-effort JSON debug logging for OpenAI SDK objects."""
    serializable = []
    for item in payload:
        if hasattr(item, "model_dump"):
            serializable.append(item.model_dump())
        elif hasattr(item, "to_dict"):
            serializable.append(item.to_dict())
        else:
            serializable.append(str(item))

    print(f"{label}: {json.dumps(serializable, indent=2)}")
