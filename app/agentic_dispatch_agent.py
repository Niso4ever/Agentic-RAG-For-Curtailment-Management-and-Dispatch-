# app/agentic_dispatch_agent.py

import json
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
