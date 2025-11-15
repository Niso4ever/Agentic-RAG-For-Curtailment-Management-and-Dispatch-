import google.cloud.aiplatform as aiplatform

PROJECT_ID = "pristine-valve-477208-i1"
REGION = "us-central1"
DATASET_DISPLAY_NAME = "Rag_AutoML"
BIGQUERY_SOURCE_URI = "bq://pristine-valve-477208-i1.solar_forcast_data.daily_solar_output"
DATASET_RESOURCE_NAME = "projects/3281379919/locations/us-central1/datasets/2549896657428807680" # Replace with your actual dataset resource name

MODEL_DISPLAY_NAME = "solar_forecast_automl_model"
TARGET_COLUMN = "target_solar_output"
TRAINING_BUDGET_MILLI_NODE_HOURS = 8000 # 8 hours

def train_automl_tabular_model(
    project: str,
    location: str,
    dataset_resource_name: str,
    model_display_name: str,
    target_column: str,
    training_budget_milli_node_hours: int,
):
    aiplatform.init(project=project, location=location)

    # Load the existing dataset
    dataset = aiplatform.TabularDataset(dataset_resource_name)

    # Create and run the AutoML Tabular training job
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"{model_display_name}_training_job",
        optimization_prediction_type="regression",
        column_transformations=[
            {"auto": {"column_name": "forecast_timestamp"}},
            {"auto": {"column_name": "series_id"}},
            {"auto": {"column_name": "mean_temperature"}},
            {"auto": {"column_name": "mean_wind_speed"}},
        ],
    )

    model = job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=training_budget_milli_node_hours,
        model_display_name=model_display_name,
        sync=True,
    )

    print(f"Model training completed. Model resource name: {model.resource_name}")
    print(f"Model ID: {model.name}")
    return model

if __name__ == "__main__":
    train_automl_tabular_model(
        project=PROJECT_ID,
        location=REGION,
        dataset_resource_name=DATASET_RESOURCE_NAME,
        model_display_name=MODEL_DISPLAY_NAME,
        target_column=TARGET_COLUMN,
        training_budget_milli_node_hours=TRAINING_BUDGET_MILLI_NODE_HOURS,
    )
