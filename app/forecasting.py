import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google.cloud import bigquery

# Ensure .env variables are loaded even when this module is used standalone
load_dotenv()


def _stub_forecast() -> Dict[str, float]:
    """Fallback forecast if no ML endpoint is available."""
    return {"mw": 42.5, "confidence": 0.87}


def _load_features_from_bigquery() -> Optional[Dict[str, Any]]:
    """Load the latest feature row from the BigQuery table."""
    # Use env vars when provided; fall back to sensible defaults if empty/undefined.
    dataset_id = os.getenv("VERTEX_DATASET_ID") or "solar_forcast_data"
    table_id = os.getenv("VERTEX_TABLE_ID") or "daily_solar_output"

    client = bigquery.Client(project="pristine-valve-477208i1")
    query = f"""
        SELECT *
        FROM `pristine-valve-477208-i1.{dataset_id}.{table_id}`
        ORDER BY date DESC
        LIMIT 1
    """
    try:
        query_job = client.query(query)
        rows = list(query_job.result())
        if not rows:
            return None
        # Convert row to a dictionary, converting bytes to strings if necessary
        row_dict = {key: (value.decode('utf-8') if isinstance(value, bytes) else value) for key, value in rows[0].items()}
        return row_dict
    except Exception as e:
        print(f"BigQuery query failed: {e}")
        return None


def _predict_with_vertex(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Send features to a deployed Vertex AI Tabular endpoint and return MW forecast.
    Expects environment variables:
      - VERTEX_PROJECT_ID
      - VERTEX_ENDPOINT_ID (either full path or bare ID)
      - VERTEX_LOCATION (defaults to us-central1)
    """
    project = "pristine-valve-477208"
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID") or "7273285910412656640"
    location = os.getenv("VERTEX_LOCATION") or "us-central1"

    if not project or not endpoint_id:
        raise ValueError("Missing VERTEX_PROJECT_ID or VERTEX_ENDPOINT_ID")

    endpoint_path = (
        endpoint_id
        if endpoint_id.startswith("projects/")
        else f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )

    try:
        from google.cloud import aiplatform
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("google-cloud-aiplatform is not installed") from exc

    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(endpoint_path)
    prediction = endpoint.predict(instances=[features])

    if not prediction.predictions:
        raise ValueError("Vertex endpoint returned no predictions")

    raw = prediction.predictions[0]
    confidence = 0.0

    if isinstance(raw, dict):
        mw = raw.get("value") or raw.get("predicted_value") or next(iter(raw.values()))
        confidence = float(raw.get("confidence", confidence))
    else:
        mw = raw

    return {"mw": float(mw), "confidence": float(confidence)}


def forecast_solar(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Forecast solar generation for the agentic dispatch pipeline.

    Set FORECAST_PROVIDER=vertex to use a deployed Vertex AI Tabular endpoint.
    Features will be loaded from BigQuery. Falls back to stub values on error.
    """
    provider = os.getenv("FORECAST_PROVIDER", "stub").lower()

    if provider != "vertex":
        return _stub_forecast()

    features = features or _load_features_from_bigquery()

    if not features:
        return {**_stub_forecast(), "note": "Vertex requested but no features supplied from BigQuery"}

    try:
        return _predict_with_vertex(features)
    except Exception as exc:
        return {
            **_stub_forecast(),
            "error": f"Vertex forecast failed ({exc}); using stub",
        }
