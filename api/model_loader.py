
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

    model_version = os.environ["MODEL_VERSION"]

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri)

    print(f"Model loaded: {model_uri}")

    return model, model_name, model_version
