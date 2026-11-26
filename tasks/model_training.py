import importlib.util
from contextlib import nullcontext

import pandas as pd

from prefect import task

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import (
    SEED,
    OUT,
    NUM,
    TRACKING_URI,
    EXPERIMENT_NAME
)

from tasks.model_evaluation import evaluate

TRACKING_AVAILABLE = (
    importlib.util.find_spec("mlflow") is not None
)

if TRACKING_AVAILABLE:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

@task(name="Setup MLflow", log_prints=True)
def setup_mlflow():

    if TRACKING_AVAILABLE:

        mlflow.set_tracking_uri(TRACKING_URI)

        mlflow.set_registry_uri(TRACKING_URI)

        mlflow.set_experiment(EXPERIMENT_NAME)

        print("MLflow tracking initialized.")

    else:

        print("MLflow is not installed.")

    return TRACKING_AVAILABLE


@task(name="Build Candidate Models", log_prints=True)
def build_models(splits):

    X_train, y_train = splits["train"]

    models = {
        "baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "logistic_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED
        ),

        "rf_depth_5": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=2
        ),

        "rf_depth_12": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=2
        )
    }

    if importlib.util.find_spec("xgboost"):

        from xgboost import XGBClassifier

        positive_weight = (
            (y_train == 0).sum()
            / (y_train == 1).sum()
        )

        weights = {
            "xgboost": positive_weight,
            "xgboost_weight_15": 15,
            "xgboost_weight_5": 5,
            "xgboost_weight_3": 3,
            "xgboost_weight_2": 2
        }

        for name, weight in weights.items():

            models[name] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=weight,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=2
            )

    print("Candidate models:", list(models.keys()))

    return models




@task(name="Train Candidate Models", log_prints=True)
def train_models(models, preprocessor, splits,
                 tracking_available):

    # Extract training and validation data
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]

    results = []
    fitted = {}

    for name, model in models.items():

        print(f"Training model: {name}")

        with (
            mlflow.start_run(run_name=name)
            if tracking_available
            else nullcontext()
        ) as run:

            # Build complete model pipeline
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model))
            ])

            # Fit the pipeline
            pipeline.fit(X_train, y_train)

            # Evaluate on validation data
            metrics = evaluate(
                pipeline,
                X_val,
                y_val
            )

            # MLflow experiment tracking
            if tracking_available:

                mlflow.log_params({
                    "model_name": name,
                    "seed": SEED,
                    **model.get_params()
                })

                mlflow.log_metrics({
                    f"val_{key}": value
                    for key, value in metrics.items()
                })

                mlflow_X_train = X_train.astype({
                    column: "float64"
                    for column in NUM
                })

                mlflow.sklearn.log_model(
                    pipeline,
                    name=f"{name}_pipeline", 
                    input_example=mlflow_X_train.head(5),
                    signature=infer_signature(
                        mlflow_X_train,
                        pipeline.predict(mlflow_X_train)
                    ),
                    serialization_format="cloudpickle"
                )

            # Store fitted model
            fitted[name] = pipeline

            # Store validation metrics
            results.append({
                "model": name,
                **metrics
            })

            print(f"{name} completed.")
            print(metrics)

    # Rank models using validation average precision
    results = (
        pd.DataFrame(results)
        .sort_values(
            "average_precision",
            ascending=False
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        OUT / "validation_metrics.csv",
        index=False
    )

    print(results.round(4))

    return results, fitted
