import google.cloud.aiplatform as aiplatform

PROJECT_ID = "pristine-valve-477208"
REGION = "us-central1"
DATASET_DISPLAY_NAME = "solar_forcast_data"
BIGQUERY_SOURCE_URI = "bq://pristine-valve-477208i1.solar_forcast_data.daily_solar_output"

def create_automl_tabular_dataset(
    project: str,
    location: str,
    display_name: str,
    bigquery_source: str,
):
    aiplatform.init(project=project, location=location)

    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        bq_source=bigquery_source,
        sync=True,
    )

    print(f"Dataset created. Resource name: {dataset.resource_name}")
    print(f"Dataset ID: {dataset.name}")
    return dataset

if __name__ == "__main__":
    create_automl_tabular_dataset(
        project=PROJECT_ID,
        location=REGION,
        display_name=DATASET_DISPLAY_NAME,
        bigquery_source=BIGQUERY_SOURCE_URI,
    )
