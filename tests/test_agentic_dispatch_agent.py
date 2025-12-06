import os
from unittest.mock import patch, MagicMock

import pytest

from app.agentic_dispatch_agent import (
    get_solar_forecast_prediction,
    _naive_projection,
    _vertex_timeseries_prediction,
)

DUMMY_HISTORICAL_DATA = [
    {
        "forecast_timestamp": "2023-10-15T10:00:00Z",
        "mean_temperature": 24.0,
        "mean_wind_speed": 4.5,
        "target_solar_output": 32.0,
    },
    {
        "forecast_timestamp": "2023-10-15T11:00:00Z",
        "mean_temperature": 24.2,
        "mean_wind_speed": 4.3,
        "target_solar_output": 38.5,
    },
    {
        "forecast_timestamp": "2023-10-15T12:00:00Z",
        "mean_temperature": 24.5,
        "mean_wind_speed": 4.0,
        "target_solar_output": 41.0,
    },
]

DUMMY_FUTURE_DATA = [
    {
        "forecast_timestamp": "2023-10-15T13:00:00Z",
        "mean_temperature": 25.0,
        "mean_wind_speed": 4.2,
    },
    {
        "forecast_timestamp": "2023-10-15T14:00:00Z",
        "mean_temperature": 25.5,
        "mean_wind_speed": 4.4,
    },
]


def test_naive_projection():
    predictions = _naive_projection(DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA)
    assert len(predictions) == 2
    # Slope is calculated from the last two points: (41.0 - 38.5) = 2.5
    # First prediction: 41.0 + 2.5 = 43.5
    # Second prediction: 43.5 + 2.5 = 46.0
    assert predictions[0]["target_solar_output"] == 45.5
    assert predictions[1]["target_solar_output"] == 50.0


def test_get_solar_forecast_prediction_stub_provider():
    with patch.dict(os.environ, {"FORECAST_PROVIDER": "stub"}):
        predictions = get_solar_forecast_prediction(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )
        assert len(predictions) == 2
        assert predictions[0]["target_solar_output"] == 45.5
        assert predictions[1]["target_solar_output"] == 50.0


@patch("app.agentic_dispatch_agent._vertex_timeseries_prediction")
def test_get_solar_forecast_prediction_vertex_provider(mock_vertex_prediction):
    mock_vertex_prediction.return_value = [
        {"target_solar_output": 50.0},
        {"target_solar_output": 52.0},
    ]
    with patch.dict(os.environ, {"FORECAST_PROVIDER": "vertex"}):
        predictions = get_solar_forecast_prediction(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )
        mock_vertex_prediction.assert_called_once_with(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )
        assert len(predictions) == 2
        assert predictions[0]["target_solar_output"] == 50.0
        assert predictions[1]["target_solar_output"] == 52.0


@patch("app.agentic_dispatch_agent._vertex_timeseries_prediction")
def test_get_solar_forecast_prediction_vertex_provider_fallback(mock_vertex_prediction):
    mock_vertex_prediction.side_effect = Exception("Vertex AI error")
    with patch.dict(os.environ, {"FORECAST_PROVIDER": "vertex"}):
        predictions = get_solar_forecast_prediction(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )
        mock_vertex_prediction.assert_called_once_with(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )
        # Should fallback to naive projection
        assert len(predictions) == 2
        assert predictions[0]["target_solar_output"] == 45.5
        assert predictions[1]["target_solar_output"] == 50.0


@patch("google.cloud.aiplatform.Endpoint")
@patch("google.cloud.aiplatform.init")
def test_vertex_timeseries_prediction(mock_aiplatform_init, mock_endpoint):
    # Mock the Vertex AI endpoint prediction result
    mock_prediction_result = MagicMock()
    mock_prediction_result.predictions = [
        {"value": 50.0},
        {"value": 52.0},
    ]
    mock_endpoint.return_value.predict.return_value = mock_prediction_result

    with patch.dict(os.environ, {"VERTEX_ENDPOINT_ID": "dummy_endpoint"}):
        predictions = _vertex_timeseries_prediction(
            DUMMY_HISTORICAL_DATA, DUMMY_FUTURE_DATA
        )

    mock_aiplatform_init.assert_called_once_with(
        project="pristine-valve-477208-i1", location="us-central1"
    )
    assert len(predictions) == 2
    assert predictions[0]["target_solar_output"] == 50.0
    assert predictions[1]["target_solar_output"] == 52.0
