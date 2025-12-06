# Agentic RAG for Curtailment Management and Dispatch

An end-to-end **Agentic AI Application** that optimizes renewable energy dispatch by combining three advanced disciplines:

1.  **Live Solar Forecasting**: Integrates with Google Cloud Vertex AI to fetch real-time solar generation predictions.
2.  **Retrieval-Augmented Generation (RAG)**: Uses a vector database to provide grounded engineering insights for curtailment and grid stability.
3.  **MILP Optimization**: Uses Operations Research (OR-Tools) to mathematically solve for the optimal battery (BESS) dispatch schedule.

---

## 🚀 Live Demo
**Public Endpoint:** [https://rag-app-406792367928.us-central1.run.app](https://rag-app-406792367928.us-central1.run.app)

---

## 🏗️ Architecture

```mermaid
graph LR
    User[Operator Query] --> Guardrail{Relevance Check}
    Guardrail -- Irrelevant --> Reject[Return Error]
    Guardrail -- Relevant --> Agent[Agentic Engine]
    
    subgraph "Google Cloud Platform"
        Agent --> Forecast[Vertex AI Forecast]
        Agent --> RAG[Vertex Vector Search]
        Agent --> Solver[MILP Solver (Cloud Run)]
    end
    
    Forecast --> Optimizer[Optimization Logic]
    RAG --> Optimizer
    Solver --> Optimizer
    
    Optimizer --> Response[Final Dispatch Plan]
```

## ✨ Key Features

-   **Relevance Guardrail**: Automatically detects and rejects irrelevant queries (e.g., "Who won the World Cup?") using a lightweight LLM check.
-   **Dynamic Dispatch**: The MILP solver adapts to real-time grid constraints. If the grid is congested, it automatically charges the battery to prevent curtailment.
-   **Live Data Integration**: Connects to `Rag_Solar Forecast` endpoint on Vertex AI for production-grade telemetry.

## 🛠️ Deployment

This application is containerized and deployed on **Google Cloud Run**.

### Prerequisites
-   Google Cloud Project (with Vertex AI enabled)
-   `gcloud` CLI installed
-   Docker

### Deploy to Cloud Run
We use a source-based deployment strategy for simplicity and reliability.

1.  **Configure Environment**:
    Ensure your `.env` file is populated with your keys:
    ```bash
    OPENAI_API_KEY=sk-...
    VERTEX_PROJECT_ID=...
    VERTEX_ENDPOINT_ID=...
    ```

2.  **Run Deployment Script**:
    ```bash
    ./deploy.sh
    ```
    This script will:
    -   Convert your `.env` file to a secure `env.yaml`.
    -   Build the Docker container in Google Cloud Build.
    -   Deploy the new revision to Cloud Run.

## 💻 Local Development

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Niso4ever/Agentic-RAG-For-Curtailment-Management-and-Dispatch-.git
    cd Agentic-RAG-For-Curtailment-Management-and-Dispatch-
    ```

2.  **Install Dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload
    ```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
