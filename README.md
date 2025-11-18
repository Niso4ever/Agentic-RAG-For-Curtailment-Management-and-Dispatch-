# Agentic RAG for Curtailment Management and Dispatch

An end-to-end prototype for an **agentic workflow** that reduces renewable curtailment by combining three disciplines:

- Short-term solar forecasting (ML placeholder for now)
- Retrieval-Augmented Generation for grounded engineering insights
- Mixed Integer Linear Programming to optimize battery dispatch

The current code is intentionally lightweight so you can iterate quickly while plugging in real models, retrieval stacks, and optimizers.

## Architecture

```text
┌──────────┐   ┌────────────────────┐   ┌──────────────────────┐   ┌───────────┐
│ Forecast │ → │ RAG knowledge base │ → │ MILP dispatch solver │ → │ Response │
└──────────┘   └────────────────────┘   └──────────────────────┘   └───────────┘
```

Each component lives in `app/`:

| File | Responsibility |
| --- | --- |
| `forecasting.py` | Replace the stubbed `forecast_solar` with Prophet, Vertex AI, or your preferred model. |
| `rag_engine.py` | Swap in FAISS / Chroma + embeddings + PDF ingestion for domain knowledge. |
| `milp_solver.py` | Upgrade the sample logic with PuLP, OR-Tools, or a custom Gurobi formulation. |
| `agent_loop.py` | Orchestrates the three steps and prepares the final operator-facing answer. |
| `config.py` | Loads secrets such as `OPENAI_API_KEY` from `.env`. |

## Quickstart

1. **Clone**
   ```bash
   git clone https://github.com/Niso4ever/Agentic-RAG-For-Curtailment-Management-and-Dispatch-.git
   cd Agentic-RAG-For-Curtailment-Management-and-Dispatch-
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies** (expand `requirements.txt` as you add libraries)
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure secrets** – create a `.env` file with:
   ```bash
   OPENAI_API_KEY=sk-...
   OPENWEATHER_API_KEY=ow-...
   OPENWEATHER_LOCATION=Abu Dhabi  # optional, defaults to Abu Dhabi
   ```
5. **Run the prototype**
   ```bash
   python -m app.main
   ```
   You will receive a stitched response that includes the forecast, grounded context, and MILP recommendation.

## Developing the Agent Further

- **Upgrade forecasting**: hook `forecast_solar()` to live telemetry, weather APIs, or ML pipelines.
- **Build the retrieval stack**: ingest operator playbooks, interconnection agreements, or congestion reports into vector storage for better answers.
- **Solve real dispatch**: model BESS SOC limits, efficiency, and cost curves in a proper MILP.
- **Expose an API/UI**: wrap `run_agent` in FastAPI or Streamlit so operators can query curtailment strategies.

Contributions and ideas are welcome—open an issue or PR with your approach. Once you add real models, remember to update `requirements.txt` and document additional environment variables. Let's curb curtailment together. 💡⚡
