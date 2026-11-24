import joblib
import numpy as np
import importlib.util

from prefect import task

from config import (
    OUT,
    SEED,
    NUM,
    MODEL_NAME
)

if importlib.util.find_spec("mlflow"):

    import mlflow
    import mlflow.sklearn

    from mlflow.models import infer_signature


@task(name="Register Final Model", log_prints=True)
def register_model(
    final_pipeline,
    best_name,
    models,
    test_metrics,
    X_dev,
    tracking_available
):

    # Save a local model copy
    joblib.dump(
        final_pipeline,
        OUT / "final_pipeline.joblib"
    )

    print("Local model saved.")

    if not tracking_available:

        print("MLflow registry not completed.")

        return None

    # Log final selected model
    with mlflow.start_run(
        run_name="final_selected_model"
    ) as run:

        # Log model parameters
        mlflow.log_params({
            "selected_model": best_name,
            "seed": SEED,
            **models[best_name].get_params()
        })

        # Log test metrics
        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        # Log artifacts
        mlflow.log_artifact(
            str(OUT / "schema.json")
        )

        mlflow.log_artifact(
            str(OUT / "validation_metrics.csv")
        )

        # Prepare input schema
        mlflow_X_dev = X_dev.astype({
            column: "float64"
            for column in NUM
        })

        # Register the complete fitted pipeline
        mlflow.sklearn.log_model(
            final_pipeline,
            name="pipeline",
            registered_model_name=MODEL_NAME,
            input_example=mlflow_X_dev.head(5),
            signature=infer_signature(
                mlflow_X_dev,
                final_pipeline.predict(mlflow_X_dev)
            ),
            serialization_format="cloudpickle"
        )

    # Retrieve registered model version
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(
        f"name = '{MODEL_NAME}'"
    )

    version = max(
        int(v.version)
        for v in versions
        if v.run_id == run.info.run_id
    )

    model_uri = f"models:/{MODEL_NAME}/{version}"

    print("Registered model:", model_uri)

    return model_uri



@task(name="Verify Registered Model", log_prints=True)
def verify_model(
    model_uri,
    final_pipeline,
    splits
):

    X_test, y_test = splits["test"]

    # Select sample observations
    sample = X_test.head(10)

    # Original model predictions
    expected = final_pipeline.predict(sample)

    # Verify MLflow model
    if model_uri is not None:

        loaded = mlflow.sklearn.load_model(
            model_uri
        )

        actual = loaded.predict(sample)

        np.testing.assert_array_equal(
            actual,
            expected
        )

        print("MLflow model verified:", model_uri)

    # Verify local joblib model
    local_model = joblib.load(
        OUT / "final_pipeline.joblib"
    )

    local_predictions = local_model.predict(
        sample
    )

    np.testing.assert_array_equal(
        local_predictions,
        expected
    )

    print("Local model reload verified.")

    return True
