from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys
import os
import uvicorn

# Add the project root to sys.path so we can import from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agentic_dispatch_agent import run_agentic_dispatch

app = FastAPI(
    title="Agentic Dispatch API",
    description="API for Solar Forecasting, RAG Insights, and MILP Optimization",
    version="1.0.0"
)

class PlantMeta(BaseModel):
    soc: float = Field(0.35, description="Current State of Charge (0.0 to 1.0)")
    capacity_mwh: float = Field(50.0, description="Total energy capacity in MWh")
    max_charge_mw: float = Field(50.0, description="Max charge rate in MW")
    max_discharge_mw: float = Field(50.0, description="Max discharge rate in MW")

class DispatchRequest(BaseModel):
    query: str = Field(..., description="Operator query about dispatch strategy")
    plant_meta: Optional[PlantMeta] = None

class DispatchResponse(BaseModel):
    result: str

@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(request: DispatchRequest):
    try:
        plant_meta_dict = request.plant_meta.model_dump() if request.plant_meta else None
        
        # Run the agent
        response = run_agentic_dispatch(request.query, plant_meta=plant_meta_dict)
        
        return DispatchResponse(result=response)
    except Exception as e:
        # Log the exception for easier debugging
        print(f"ERROR in /dispatch endpoint: {e}")
        # Consider more specific logging or error handling here
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Agentic Dispatch API is running. Visit /docs for Swagger UI."}

if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # Uvicorn is used to run the server
    uvicorn.run(app, host="0.0.0.0", port=port)