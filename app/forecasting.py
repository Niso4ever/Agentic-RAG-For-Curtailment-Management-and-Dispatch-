import json
import os
from typing import Any, Dict, Optional, List
from datetime import datetime, date, timezone, timedelta

import requests
from dotenv import load_dotenv
from google.cloud import bigquery
import google.auth
from google.auth.exceptions import DefaultCredentialsError

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
    "target_solar_output",
]

# Always use this value (most common in training dataset)
DEFAULT_SERIES_ID = "725300"
DEFAULT_LOCATION = "Abu Dhabi"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
DEFAULT_GCP_PROJECT = "pristine-valve-477208-i1"
DEFAULT_VERTEX_LOCATION = "us-central1"

# Multi-interval horizon length
HORIZON_INTERVALS = 6


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _stub_forecast_single() -> Dict[str, float]:
    """Single-point stub forecast used when Vertex is unavailable."""
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


def _normalize_date(value: Any) -> str:
    """
    Ensure forecast_timestamp is a date string (YYYY-MM-DD) as required by the
    AutoML tabular schema. Accepts datetime/date/str.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        # Trust caller; assume already ISO formatted
        return value
    # Fallback to "today" to avoid missing required field
    return datetime.now(timezone.utc).date().isoformat()


def _build_vertex_instance(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the exact instance the Vertex AutoML tabular model expects.
    Drops any extra keys and fills required fields with safe defaults.
    """
    instance: Dict[str, Any] = {}
    instance["forecast_timestamp"] = _normalize_date(row.get("forecast_timestamp"))
    instance["mean_temperature"] = float(row.get("mean_temperature", 0.0))
    instance["mean_wind_speed"] = float(row.get("mean_wind_speed", 0.0))
    instance["series_id"] = str(row.get("series_id") or DEFAULT_SERIES_ID)

    # Allow null for inference targets; ensure key exists for schema consistency
    instance["target_solar_output"] = (
        float(row["target_solar_output"])
        if row.get("target_solar_output") not in (None, "")
        else None
    )

    return instance


def _resolve_project_id(*env_vars: str, fallback: str = DEFAULT_GCP_PROJECT) -> str:
    """
    Determine the Google Cloud project to use, preferring env vars and falling
    back to Application Default Credentials. A hard-coded fallback maintains a
    sane default for local/manual runs.
    """
    for var in env_vars:
        value = os.getenv(var)
        if value:
            return value

    try:
        _, project_id = google.auth.default()
        if project_id:
            return project_id
    except DefaultCredentialsError as exc:
        print(f"[forecasting][WARN] Unable to resolve default GCP project via ADC: {exc}")

    return fallback


# -------------------------------------------------------------------
# BigQuery loader
# -------------------------------------------------------------------
def _load_features_from_bigquery() -> Optional[Dict[str, Any]]:
    dataset_id = os.getenv("VERTEX_DATASET_ID", "solar_forcast_data")
    table_id = os.getenv("VERTEX_TABLE_ID", "daily_solar_output")

    project_id = _resolve_project_id(
        "BIGQUERY_PROJECT_ID", "VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"
    )

    client = bigquery.Client(project=project_id)

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
# OpenWeather loader
# -------------------------------------------------------------------
def _fetch_weather_features(location: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current weather for a location from OpenWeather and map to the
    model's tabular feature names.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("[forecasting][WARN] OPENWEATHER_API_KEY is not set; skipping live weather fetch.")
        return None

    params = {"q": location, "appid": api_key, "units": "metric"}
    try:
        resp = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[forecasting][WARN] OpenWeather call failed: {exc}")
        return None

    main = data.get("main", {})
    wind = data.get("wind", {})

    ts = data.get("dt")
    if ts:
        ts = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    else:
        ts = datetime.now(timezone.utc).date().isoformat()

    return {
        "forecast_timestamp": ts,
        "mean_temperature": float(main.get("temp", 0.0)),
        "mean_wind_speed": float(wind.get("speed", 0.0)),
        "series_id": DEFAULT_SERIES_ID,
        "target_solar_output": None,
        "debug_weather_source": data,  # preserved for debugging; filtered before Vertex call
    }


def get_latest_weather_features(location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Public helper used by the agent loop to refresh real-time weather inputs
    before each MILP solve. Defaults to OPENWEATHER_LOCATION when not provided.
    """
    resolved_location = location or os.getenv("OPENWEATHER_LOCATION", DEFAULT_LOCATION)
    return _fetch_weather_features(resolved_location)


# -------------------------------------------------------------------
# Vertex AI predictor (single-point)
# -------------------------------------------------------------------
def _predict_with_vertex(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single-point AutoML Tabular regression call. The multi-interval horizon
    is constructed in forecast_solar() using this base prediction.
    """
    import vertexai
    from google.cloud import aiplatform

    project = _resolve_project_id("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID", "7273285910412656640")

    endpoint_path = (
        endpoint_id
        if endpoint_id.startswith("projects/")
        else f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )

    vertexai.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(endpoint_path)

    payload = {k: features.get(k) for k in ALLOWED_FEATURES}

    print("[forecasting] Sending instance to Vertex AI:")
    print(json.dumps(payload, indent=2))

    prediction = endpoint.predict(instances=[payload])

    if not prediction.predictions:
        raise ValueError("Vertex returned empty predictions")

    raw = prediction.predictions[0]

    # AutoML Tabular regression → no native confidence scores → assign default
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
        "features_used": payload,
    }


# -------------------------------------------------------------------
# Multi-interval helper
# -------------------------------------------------------------------
def _build_horizon_from_single(
    base_mw: float,
    confidence: float,
    features_used: Dict[str, Any],
    horizon: int = HORIZON_INTERVALS,
) -> List[Dict[str, Any]]:
    """
    Construct a simple 6-interval horizon from a single-point forecast.

    For now we keep all 6 intervals at the same MW value. This makes the MILP
    genuinely multi-interval (SoC evolves over time) while the forecast stays
    consistent with the underlying single-point AutoML model.
    """
    base_label = features_used.get("forecast_timestamp") or datetime.now(timezone.utc).date().isoformat()
    intervals: List[Dict[str, Any]] = []

    # Use hourly steps for labels (even though AutoML is daily) just to
    # provide distinct identifiers per interval.
    base_dt = datetime.strptime(str(base_label), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for i in range(horizon):
        label_dt = base_dt + timedelta(hours=i)
        label_str = label_dt.isoformat(timespec="minutes").replace("+00:00", "Z")
        intervals.append(
            {
                "label": label_str,
                "mw_forecast": float(base_mw),
                "features": dict(features_used),
                "confidence": confidence,
            }
        )
    return intervals


# -------------------------------------------------------------------
# Main forecasting function (multi-interval)
# -------------------------------------------------------------------
def forecast_solar(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a 6-interval solar forecast horizon.

    Structure:
    {
      "mw": <float>,                     # representative forecast (first interval)
      "confidence": <0-1>,
      "source": "vertex" | "stub",
      "features_used": {...},            # base features passed to Vertex
      "dispatch_intervals": [
          {
             "label": "...",
             "mw_forecast": <float>,
             "features": {...},
             "confidence": <0-1>
          },
          ... (6 intervals total)
      ]
    }
    """
    provider = os.getenv("FORECAST_PROVIDER", "vertex").lower()
    print(f"[forecasting] Using provider: {provider}")

    location = os.getenv("OPENWEATHER_LOCATION", DEFAULT_LOCATION)

    if provider == "stub":
        # Stub provider produces a simple flat 6-interval horizon.
        base = _stub_forecast_single()
        base_mw = float(base.get("mw", 0.0) or 0.0)
        confidence = float(base.get("confidence", 0.0) or 0.0)
        features_used: Dict[str, Any] = {
            "forecast_timestamp": datetime.now(timezone.utc).date().isoformat(),
            "mean_temperature": 25.0,
            "mean_wind_speed": 2.0,
            "series_id": DEFAULT_SERIES_ID,
            "target_solar_output": None,
        }
        intervals = _build_horizon_from_single(
            base_mw=base_mw,
            confidence=confidence,
            features_used=features_used,
            horizon=HORIZON_INTERVALS,
        )
        return {
            "mw": base_mw,
            "confidence": confidence,
            "source": "stub",
            "features_used": features_used,
            "dispatch_intervals": intervals,
        }

    # Real provider → prefer live weather → explicit overrides → BigQuery fallback.
    row = (
        _fetch_weather_features(location)
        or features
        or _load_features_from_bigquery()
    )

    if not row:
        # No inputs at all → fall back to stub horizon
        base = _stub_forecast_single()
        base_mw = float(base.get("mw", 0.0) or 0.0)
        confidence = float(base.get("confidence", 0.0) or 0.0)
        features_used = {
            "forecast_timestamp": datetime.now(timezone.utc).date().isoformat(),
            "mean_temperature": 25.0,
            "mean_wind_speed": 2.0,
            "series_id": DEFAULT_SERIES_ID,
            "target_solar_output": None,
        }
        intervals = _build_horizon_from_single(
            base_mw=base_mw,
            confidence=confidence,
            features_used=features_used,
            horizon=HORIZON_INTERVALS,
        )
        return {
            "mw": base_mw,
            "confidence": confidence,
            "source": "stub-no-inputs",
            "features_used": features_used,
            "dispatch_intervals": intervals,
            "note": "No weather/BQ features; using stub horizon.",
        }

    instance = _build_vertex_instance(row)

    print("[forecasting] Final instance for Vertex AI (tabular):")
    print(json.dumps({k: v for k, v in instance.items() if k in ALLOWED_FEATURES}, indent=2))

    try:
        base_pred = _predict_with_vertex(instance)
        base_mw = float(base_pred.get("mw", 0.0) or 0.0)
        confidence = float(base_pred.get("confidence", 0.9) or 0.9)
        confidence = max(0.0, min(confidence, 1.0))
        features_used = dict(base_pred.get("features_used") or instance)
        intervals = _build_horizon_from_single(
            base_mw=base_mw,
            confidence=confidence,
            features_used=features_used,
            horizon=HORIZON_INTERVALS,
        )
        return {
            "mw": base_mw,
            "confidence": confidence,
            "source": base_pred.get("source", "vertex"),
            "features_used": features_used,
            "dispatch_intervals": intervals,
        }
    except Exception as e:
        print(f"[forecasting][ERROR] Vertex prediction failed: {e}")
        base = _stub_forecast_single()
        base_mw = float(base.get("mw", 0.0) or 0.0)
        confidence = float(base.get("confidence", 0.0) or 0.0)
        features_used = instance
        intervals = _build_horizon_from_single(
            base_mw=base_mw,
            confidence=confidence,
            features_used=features_used,
            horizon=HORIZON_INTERVALS,
        )
        return {
            "mw": base_mw,
            "confidence": confidence,
            "source": "stub-on-vertex-error",
            "features_used": features_used,
            "dispatch_intervals": intervals,
            "error": str(e),
        }


# -------------------------------------------------------------------
# CLI entrypoint for quick manual runs
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Forecasting module test (6-interval horizon) ===")
    result = forecast_solar()
    print("\nForecast result:")
    print(json.dumps(result, indent=2))
