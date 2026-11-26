
import os

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

load_dotenv()


def load_registered_model():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001"
    )

    model_name = os.environ["MODEL_NAME"]

    model_version = os.getenv("MODEL_VERSION", "").strip()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    # Automatically select the latest ready version
    # when MODEL_VERSION is not specified.
    if not model_version:

        client = mlflow.MlflowClient()

        versions = client.search_model_versions(
            f"name = '{model_name}'"
        )

        ready_versions = [
            int(v.version)
            for v in versions
            if v.status == "READY"
        ]

        if not ready_versions:
            raise RuntimeError(
                f"No ready model found: {model_name}"
            )

        model_version = str(max(ready_versions))

    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri)

    print(f"Model loaded: {model_uri}")

    return model, model_name, model_version
