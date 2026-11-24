
from prefect import flow

# Part 2: Data Preparation
from tasks.data_preparation import (
    load_data,
    clean_data,
    split_data,
    save_datasets
)

from tasks.preprocessing import (
    build_preprocessor
)

# Part 3: Model Training
from tasks.model_training import (
    setup_mlflow,
    build_models,
    train_models
)

# Part 3: Model Evaluation
from tasks.model_evaluation import (
    select_best_model,
    evaluate_final_model
)

# Part 3: Model Registry
from tasks.model_registry import (
    register_model,
    verify_model
)


@flow(name="AI4I Training Flow", log_prints=True)
def training_flow():

    # Step 1: Initialize MLflow
    tracking_available = setup_mlflow()

    # Step 2: Load raw data
    raw = load_data()

    # Step 3: Clean data
    X, y = clean_data(raw)

    # Step 4: Split dataset
    splits = split_data(X, y)

    # Step 5: Save datasets and schema
    save_datasets(splits)

    # Step 6: Build preprocessor
    preprocessor = build_preprocessor()

    # Step 7: Build candidate models
    models = build_models(splits)

    # Step 8: Train candidate models
    results, fitted = train_models(
        models,
        preprocessor,
        splits,
        tracking_available
    )

    # Step 9: Select best model
    best_name = select_best_model(results)

    # Step 10: Train and evaluate final model
    final_pipeline, test_metrics, X_dev = (
        evaluate_final_model(
            best_name,
            models,
            preprocessor,
            splits
        )
    )

    # Step 11: Register final model
    model_uri = register_model(
        final_pipeline,
        best_name,
        models,
        test_metrics,
        X_dev,
        tracking_available
    )

    # Step 12: Verify saved model
    verified = verify_model(
        model_uri,
        final_pipeline,
        splits
    )

    print("Training pipeline completed!")
    print("Best model:", best_name)
    print("Model URI:", model_uri)
    print("Verified:", verified)

    return {
        "best_model": best_name,
        "test_metrics": test_metrics,
        "model_uri": model_uri,
        "verified": verified
    }


if __name__ == "__main__":
    training_flow()
