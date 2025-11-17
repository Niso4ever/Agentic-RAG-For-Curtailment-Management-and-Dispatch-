"""
Quick harness to exercise get_solar_forecast_prediction with sample data.
Populate the historical portion with rows that include target_solar_output
and a short future horizon with target_solar_output=None.
"""

from app.agentic_dispatch_agent import get_solar_forecast_prediction

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


def main() -> None:
    predictions = get_solar_forecast_prediction(
        historical_data=DUMMY_HISTORICAL_DATA,
        future_data=DUMMY_FUTURE_DATA,
    )
    print("Predicted Solar Output:", predictions)


if __name__ == "__main__":
    main()
