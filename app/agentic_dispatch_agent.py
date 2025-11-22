import json
import os
from statistics import mean
from typing import List, Dict, Any, Optional

from app.llm_client import client, MODEL_NAME

try:
    from openai import AuthenticationError
except ImportError:  # pragma: no cover - optional dependency
    AuthenticationError = Exception  # type: ignore

from app import agent_tools


# ============================================================
# TOOL DEFINITIONS (Responses API — function calling schema)
# ============================================================


def _get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Tools exposed to the LLM. Note: names keep the *_stub suffix for
    backward compatibility with your existing agent_tools module, but
    they now route to the live forecast / RAG / MILP logic.
    """
    return [
        {
            "name": "get_solar_forecast_stub",
            "type": "function",
            "description": (
                "Retrieve short-term solar forecast (MW and confidence) over a 6-interval horizon "
                "using the live forecasting pipeline when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "get_rag_insights_stub",
            "type": "function",
            "description": (
                "Retrieve grounded engineering knowledge using the local RAG engine. "
                "Use this for curtailment, BESS, and grid-code reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "solve_milp_dispatch_stub",
            "type": "function",
            "description": (
                "Run multi-interval MILP-based dispatch optimization using OR-Tools. "
                "Takes the solar forecast horizon, BESS SoC/capacity, and charge/discharge limits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mw_forecast": {"type": "number"},
                    "bess_soc": {"type": "number"},
                    "bess_capacity_mwh": {"type": "number"},
                    "max_charge_mw": {"type": "number"},
                    "max_discharge_mw": {"type": "number"},
                    # The agent loop will override this with a trusted payload,
                    # but we expose it here so the schema is explicit.
                    "dispatch_intervals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "mw_forecast": {"type": "number"},
                                "grid_limit_mw": {"type": "number"},
                                "curtailment_weight": {"type": "number"},
                                "cycle_penalty": {"type": "number"},
                            },
                        },
                    },
                },
                "required": [
                    "mw_forecast",
                    "bess_soc",
                    "bess_capacity_mwh",
                    "max_charge_mw",
                    "max_discharge_mw",
                ],
            },
        },
    ]


# ============================================================
# TOOL ROUTER (generic fallback; main tools are handled inline)
# ============================================================


def _run_client_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic tool router used only for tools that are NOT handled
    explicitly inside run_agentic_dispatch.
    """
    if name == "get_solar_forecast_stub":
        return agent_tools.get_solar_forecast_stub()

    if name == "get_rag_insights_stub":
        return agent_tools.get_rag_insights_stub(arguments.get("query", ""))

    # We intentionally do NOT route solve_milp_dispatch_stub here,
    # because in the main loop we override the LLM arguments and
    # construct a clean MILP payload ourselves.
    return {"error": f"Unknown tool '{name}'"}


# ============================================================
# LOCAL FALLBACK (when no LLM / no API key)
# ============================================================


def _run_local_stub_answer(user_query: str) -> str:
    """
    Offline pipeline used when the OpenAI client is not configured.
    Uses the same forecast → RAG → MILP logic as the main agent,
    so behaviour is consistent.
    """
    plant_meta = {
        "soc": 0.35,
        "capacity_mwh": 50.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
    }

    forecast = agent_tools.get_solar_forecast_stub()
    weather = agent_tools.get_live_weather_features()
    rag = agent_tools.get_rag_insights_stub(user_query)

    # Build a MILP payload from forecast + rag + plant meta
    milp_payload = agent_tools.prepare_milp_payload(
        forecast_output=forecast,
        rag_output=rag,
        plant_meta=plant_meta,
        weather_features=weather,
    )

    # Hard override to guarantee we never pass zeros by mistake
    mw_forecast = float(forecast.get("mw", 0.0) or 0.0)
    milp_payload["mw_forecast"] = mw_forecast
    milp_payload["bess_capacity_mwh"] = float(plant_meta.get("capacity_mwh", 50.0))
    milp_payload["bess_soc"] = float(plant_meta.get("soc", 0.35))
    milp_payload["max_charge_mw"] = float(plant_meta.get("max_charge_mw", 50.0))
    milp_payload["max_discharge_mw"] = float(plant_meta.get("max_discharge_mw", 50.0))

    dispatch = agent_tools.solve_milp_dispatch_stub(**milp_payload)

    return f"""
LLM client not configured. Showing offline pipeline output instead.

=== OFFLINE DISPATCH ANALYSIS ===
Query: {user_query}

Solar Forecast → {json.dumps(forecast, indent=2)}
RAG Insights   → {json.dumps(rag, indent=2)}
MILP Payload   → {json.dumps(milp_payload, indent=2)}
MILP Dispatch  → {json.dumps(dispatch, indent=2)}

Set OPENAI_API_KEY and install the OpenAI SDK to enable real model usage.
"""


# ============================================================
# SOLAR FORECAST PREDICTION (legacy helpers — kept for compatibility)
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
        projections.append({**row, "target_solar_output": float(next_val)})
    return projections


def _vertex_timeseries_prediction(
    historical_data: List[Dict[str, Any]],
    future_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Send combined historical+future rows to a Vertex AI endpoint.
    Assumes a time-series/AutoML model that can handle missing targets
    for the prediction window.
    """
    project = os.getenv("VERTEX_PROJECT_ID", "pristine-valve-477208i1")
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID") or os.getenv(
        "VERTEX_FORECAST_ENDPOINT_ID"
    )
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not project or not endpoint_id:
        raise ValueError("Missing VERTEX_PROJECT_ID or VERTEX_ENDPOINT_ID for Vertex inference")

    try:
        import vertexai
        from google.cloud import aiplatform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("google-cloud-aiplatform is not installed") from exc

    vertexai.init(project=project, location=location)

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
    trimmed = (
        raw_predictions[-future_count:]
        if len(raw_predictions) >= future_count
        else raw_predictions
    )

    normalized: List[Dict[str, Any]] = []
    for base_row, raw in zip(future_data, trimmed):
        if isinstance(raw, dict):
            value = (
                raw.get("value")
                or raw.get("predicted_value")
                or next(iter(raw.values()), 0.0)
            )
        else:
            value = raw
        normalized.append({**base_row, "target_solar_output": float(value)})

    # If the endpoint returned fewer predictions than requested, pad with naive estimates.
    if len(normalized) < future_count:
        missing_future = future_data[len(normalized) :]
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
    provider = os.getenv("FORECAST_PROVIDER", "vertex").lower()

    if provider == "vertex":
        try:
            return _vertex_timeseries_prediction(historical_data, future_data)
        except Exception as exc:
            print(f"Vertex prediction failed ({exc}); falling back to naive projection.")

    return _naive_projection(historical_data, future_data)


# ============================================================
# MAIN AGENT LOOP (LLM → tools → MILP → final answer)
# ============================================================


def run_agentic_dispatch(user_query: str, plant_meta: Dict[str, Any] = None) -> str:
    """
    Main entrypoint for the Agentic Dispatch analysis.

    CRITICAL GUARANTEES:
    - The MW forecast horizon used by the MILP solver is ALWAYS taken from the
      latest forecast tool output (agent_tools.get_solar_forecast_stub),
      never from LLM-generated arguments.
    - Multi-interval MILP is used (6 intervals when available).
    """
    plant_meta = plant_meta or {
        "soc": 0.35,
        "capacity_mwh": 50.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
    }

    if client is None:
        return _run_local_stub_answer(user_query)

    tools = _get_tool_definitions()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an Agentic AI system specializing in curtailment and BESS dispatch optimization.\n"
                        "Follow this strict sequence for EVERY request:\n"
                        "  1. Call get_solar_forecast_stub (obtain a 6-interval solar forecast horizon)\n"
                        "  2. Call get_rag_insights_stub with the user query\n"
                        "  3. Call solve_milp_dispatch_stub using ONLY the BESS parameters from the user "
                        "(default to 50 MWh capacity and 35% SoC when unspecified) and the horizon prepared "
                        "by the tools.\n"
                        "Only after all THREE tool outputs are available may you craft a final answer.\n"
                        "Your answer must explicitly reference the forecast horizon, retrieved insights, "
                        "and the multi-interval MILP dispatch recommendation (including curtailment behaviour "
                        "and SoC evolution)."
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    except AuthenticationError as exc:
        print(f"OpenAI authentication failed ({exc}); using local stub pipeline instead.")
        return _run_local_stub_answer(user_query)

    last_forecast: Dict[str, Any] = {}
    last_rag: Dict[str, Any] = {}
    last_weather: Optional[Dict[str, Any]] = None
    milp_payload: Optional[Dict[str, Any]] = None

    while True:
        _debug_dump("Model output", response.output)
        tool_outputs: List[Dict[str, Any]] = []

        # Extract tool calls from the response
        tool_calls = [
            item
            for item in response.output
            if getattr(item, "type", "") in ("function_call", "custom_tool_call")
        ]

        # Enforce deterministic ordering: forecast → RAG → MILP
        def get_sort_key(tool_call):
            if tool_call.name == "get_solar_forecast_stub":
                return 0
            if tool_call.name == "get_rag_insights_stub":
                return 1
            if tool_call.name == "solve_milp_dispatch_stub":
                return 2
            return 3

        sorted_tool_calls = sorted(tool_calls, key=get_sort_key)

        for item in sorted_tool_calls:
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

            # --------------------------------------------------------
            # TOOL EXECUTION WITH HARD OVERRIDES FOR MILP PAYLOAD
            # --------------------------------------------------------
            if name == "get_solar_forecast_stub":
                result = agent_tools.get_solar_forecast_stub()
                last_forecast = result

            elif name == "get_rag_insights_stub":
                result = agent_tools.get_rag_insights_stub(args.get("query", ""))
                last_rag = result

            elif name == "solve_milp_dispatch_stub":
                # Ensure we have a forecast even if MILP is called first.
                if not last_forecast:
                    last_forecast = agent_tools.get_solar_forecast_stub()
                if not last_rag:
                    last_rag = {}
                # Always refresh weather immediately before optimization to capture live changes.
                last_weather = agent_tools.get_live_weather_features()

                # Build a MILP payload from forecast + rag + plant meta
                milp_payload = agent_tools.prepare_milp_payload(
                    forecast_output=last_forecast,
                    rag_output=last_rag,
                    plant_meta=plant_meta,
                    weather_features=last_weather,
                )

                # CRITICAL: Override any LLM / helper nonsense.
                mw_forecast_top = float(last_forecast.get("mw", 0.0) or 0.0)
                milp_payload["mw_forecast"] = mw_forecast_top
                milp_payload["bess_capacity_mwh"] = float(
                    plant_meta.get("capacity_mwh", 50.0)
                )
                milp_payload["bess_soc"] = float(plant_meta.get("soc", 0.35))
                milp_payload["max_charge_mw"] = float(
                    plant_meta.get("max_charge_mw", 50.0)
                )
                milp_payload["max_discharge_mw"] = float(
                    plant_meta.get("max_discharge_mw", 50.0)
                )

                result = agent_tools.solve_milp_dispatch_stub(**milp_payload)

            else:
                # Any other tool: go through the generic router
                result = _run_client_tool(name, args)

            print(f"Tool {name} returned: {json.dumps(result, indent=2)}")

            # For MILP we also return the payload so the LLM can reason
            output_payload: Any = result
            if name == "solve_milp_dispatch_stub":
                output_payload = {"payload": milp_payload, "result": result}

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output_payload),
                }
            )

        # If there are tool outputs, send them back to the model and continue the loop.
        if tool_outputs:
            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=tool_outputs,
                    previous_response_id=response.id,
                    tools=tools,
                )
            except AuthenticationError as exc:
                print(f"OpenAI authentication failed ({exc}); using local stub pipeline instead.")
                return _run_local_stub_answer(user_query)
            continue

        # Otherwise, collect the final message text and return it.
        final_chunks: List[str] = []
        for item in response.output:
            if getattr(item, "type", "") == "message":
                for part in item.content:
                    if getattr(part, "type", "") == "output_text":
                        final_chunks.append(part.text)

        print(f"Final agent response: {final_chunks}")
        return (
            "\n".join(final_chunks)
            if final_chunks
            else "Agent finished with no text output."
        )


# ============================================================
# DEBUG HELPER
# ============================================================


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
