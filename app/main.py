# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # Used to define the structure of incoming requests
import uvicorn

# Assuming your agentic_dispatch_agent.py is in a subdirectory named 'app'
# Make sure 'app' folder is at the same level as main.py and requirements.txt
from app.agentic_dispatch_agent import run_agentic_dispatch

app = FastAPI()

# Define the structure of the incoming request body
from typing import Optional, Dict, Any

# Define the structure of the incoming request body
class QueryRequest(BaseModel):
    query: str
    plant_meta: Optional[Dict[str, Any]] = None

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount the 'public' directory to serve static files (CSS, JS, etc.)
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
async def read_index():
    """
    Serve the index.html file from the public directory.
    """
    return FileResponse("public/index.html")

@app.post("/dispatch")
async def dispatch_analysis(request: QueryRequest):
    """
    Endpoint to receive user queries for agentic dispatch analysis.
    """
    try:
        print(f"Received user query: {request.query}") # Logs to Cloud Logging
        answer = run_agentic_dispatch(request.query, plant_meta=request.plant_meta)
        print(f"Agent finished, sending response.")
        # Return 'result' to match frontend expectation (index.html: data.result)
        return {"query": request.query, "result": answer}
    except Exception as e:
        print(f"Error during dispatch analysis: {e}") # Logs to Cloud Logging
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Cloud Run sets the PORT environment variable
    # Your app must listen on 0.0.0.0 for external access
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting FastAPI app on 0.0.0.0:{port}") # Logs to Cloud Logging
    uvicorn.run(app, host="0.0.0.0", port=port)

