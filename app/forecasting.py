import json
import os
from typing import Any, Dict, Optional
from datetime import datetime, date, timezone

from dotenv import load_dotenv
from google.cloud import bigquery
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# -------------------------------------------------------------------
# Load .env file
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# CONSTANTS — Your model's true feature schema
# -------------------------------------------------------------------
ALLOWED_FEATURES = [
    "forecast_timestamp",
    "mean_temperature",
    "mean_wind_speed",
    "series_id",
]

# Always use this value (most common in training dataset)
DEFAULT_SERIES_ID = "725300"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _stub_forecast() -> Dict[str, float]:
    return {"mw": 13.23, "confidence": 0.0, "source": "stub"}


def _coerce_json_safe(value: Any) -> Any:
    """Convert types (datetime, date, bytes) → JSON safe values."""
    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


# -------------------------------------------------------------------
# Auth helpers
# -------------------------------------------------------------------
def _get_adc_credentials():
    """
    Prefer Application Default Credentials (gcloud auth application-default login).
    Falls back to a service-account key file if provided. Ignores missing files.
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if key_path and os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(key_path)
        project_id = credentials.project_id
    else:
        if key_path and not os.path.exists(key_path):
            print(f"[forecasting][WARN] GOOGLE_APPLICATION_CREDENTIALS points to missing file: {key_path}. Ignoring and using ADC instead.")

        # Temporarily clear the env var so google.auth.default() doesn't try to open the missing file again
        original = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        try:
            credentials, project_id = google.auth.default()
        finally:
            if original is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original

    if not credentials.valid:
        credentials.refresh(Request())
    return credentials, project_id


# -------------------------------------------------------------------
# BigQuery loader
# -------------------------------------------------------------------
def _load_features_from_bigquery() -> Optional[Dict[str, Any]]:
    dataset_id = os.getenv("VERTEX_DATASET_ID", "solar_forcast_data")
    table_id = os.getenv("VERTEX_TABLE_ID", "daily_solar_output")

    credentials, adc_project = _get_adc_credentials()
    project_id = (
        os.getenv("BIGQUERY_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or adc_project
    )

    client = bigquery.Client(project=project_id, credentials=credentials)

    print(f"[forecasting] Loading features from BQ: {project_id}.{dataset_id}.{table_id}")

    query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        ORDER BY forecast_timestamp DESC
        LIMIT 1
    """
    try:
        rows = list(client.query(query).result())
        if not rows:
            print("[forecasting][WARN] No rows returned from BigQuery.")
            return None

        # Convert entire row to JSON-safe values
        row = {k: _coerce_json_safe(v) for k, v in rows[0].items()}
        return row

    except Exception as e:
        print(f"[forecasting][ERROR] BigQuery query failed: {e}")
        return None


# -------------------------------------------------------------------
# Vertex AI predictor
# -------------------------------------------------------------------
def _predict_with_vertex(features: Dict[str, Any]) -> Dict[str, float]:
    from google.cloud import aiplatform

    credentials, adc_project = _get_adc_credentials()

    project = os.getenv("VERTEX_PROJECT_ID") or adc_project
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID", "7273285910412656640")

    endpoint_path = (
        endpoint_id
        if endpoint_id.startswith("projects/")
        else f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )

    aiplatform.init(project=project, location=location, credentials=credentials)
    endpoint = aiplatform.Endpoint(endpoint_path)

    print("[forecasting] Sending instance to Vertex AI:")
    print(json.dumps(features, indent=2))

    prediction = endpoint.predict(instances=[features])

    if not prediction.predictions:
        raise ValueError("Vertex returned empty predictions")

    raw = prediction.predictions[0]

    # AutoML Tabular regression → no confidence scores → assign default
    confidence = 0.90

    if isinstance(raw, dict):
        mw = (
            raw.get("value")
            or raw.get("predicted_value")
            or next(iter(raw.values()))
        )
    else:
        mw = raw

    return {
        "mw": float(mw),
        "confidence": confidence,
        "source": "vertex",
    }


# -------------------------------------------------------------------
# Main forecasting function
# -------------------------------------------------------------------
def forecast_solar(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    provider = os.getenv("FORECAST_PROVIDER", "vertex").lower()
    print(f"[forecasting] Using provider: {provider}")

    if provider == "stub":
        return _stub_forecast()

    # Load latest row from BigQuery
    row = features or _load_features_from_bigquery()
    if not row:
        return {**_stub_forecast(), "note": "No BigQuery features"}

    # -------------------------------------------------------------------
    # Construct the EXACT feature set model expects
    # -------------------------------------------------------------------
    instance = {}

    # Required datetime feature
    instance["forecast_timestamp"] = row.get("forecast_timestamp")

    # Weather features
    instance["mean_temperature"] = float(row.get("mean_temperature", 0))
    instance["mean_wind_speed"] = float(row.get("mean_wind_speed", 0))

    # FIXED categorical feature (to match training distribution)
    instance["series_id"] = DEFAULT_SERIES_ID

    print("[forecasting] Final instance for Vertex AI:")
    print(json.dumps(instance, indent=2))

    try:
        return _predict_with_vertex(instance)
    except Exception as e:
        print(f"[forecasting][ERROR] Vertex prediction failed: {e}")
        return {**_stub_forecast(), "error": str(e)}


# -------------------------------------------------------------------
# CLI entrypoint for quick manual runs
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Forecasting module test ===")
    result = forecast_solar()
    print("\nForecast result:")
    print(json.dumps(result, indent=2))
