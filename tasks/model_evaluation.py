import json
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from prefect import task
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from config import OUT


def evaluate(pipeline, features, labels):

    scores = pipeline.predict_proba(features)[:, 1]

    predictions = pipeline.predict(features)

    return {
        "accuracy": accuracy_score(
            labels, predictions
        ),

        "precision": precision_score(
            labels, predictions, zero_division=0
        ),

        "recall": recall_score(
            labels, predictions, zero_division=0
        ),

        "f1": f1_score(
            labels, predictions, zero_division=0
        ),

        "roc_auc": roc_auc_score(
            labels, scores
        ),

        "average_precision": average_precision_score(
            labels, scores
        )
    }



@task(name="Select Best Model", log_prints=True)
def select_best_model(results):

    results = results.sort_values(
        "f1",
        ascending=False
    )

    best_name = results.iloc[0]["model"]

    print("Selected model:", best_name)

    # Generate model comparison chart
    results.set_index("model")[
        ["average_precision", "f1", "roc_auc"]
    ].plot.bar(
        figsize=(9, 4),
        ylim=(0, 1),
        rot=25
    )

    plt.ylabel("Validation score")
    plt.title("Candidate model comparison")
    plt.tight_layout()

    plt.savefig(
        OUT / "model_comparison.png",
        dpi=150
    )

    plt.close()

    return best_name




@task(name="Evaluate Final Model", log_prints=True)
def evaluate_final_model(
    best_name,
    models,
    preprocessor,
    splits
):

    # Extract datasets
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    # Combine train and validation datasets
    X_dev = pd.concat([X_train, X_val])

    y_dev = pd.concat([y_train, y_val])

    # Create final pipeline
    final_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(models[best_name]))
    ])

    # Refit using development data
    final_pipeline.fit(X_dev, y_dev)

    # Final evaluation
    test_metrics = evaluate(
        final_pipeline,
        X_test,
        y_test
    )

    print("Final model:", best_name)

    print(
        pd.Series(test_metrics).round(4)
    )

    # Generate confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        final_pipeline,
        X_test,
        y_test,
        display_labels=["Normal", "Failure"]
    )

    plt.title("Final model - held-out test set")
    plt.tight_layout()

    plt.savefig(
        OUT / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # Save final test metrics
    (OUT / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    return final_pipeline, test_metrics, X_dev
