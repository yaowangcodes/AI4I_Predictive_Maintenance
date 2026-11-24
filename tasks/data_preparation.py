import json
import numpy as np
import pandas as pd

from prefect import task
from sklearn.model_selection import train_test_split

from config import (
    DATA,
    OUT,
    SEED,
    TARGET,
    CAT,
    NUM,
    FEATURES
)

@task(name="Load Raw Data", log_prints=True)
def load_data(data_path=DATA):

    raw = pd.read_csv(data_path)

    print("Shape:", raw.shape)

    print("Missing values:")
    print(raw.isna().sum())

    print("Exact duplicates:", raw.duplicated().sum())

    print("Duplicate UDI:", raw["UDI"].duplicated().sum())

    return raw


@task(name="Clean and Validate Data", log_prints=True)
def clean_data(raw):

    # Validate required columns
    assert set(
        FEATURES + [TARGET, "UDI"]
    ).issubset(raw.columns)

    # Remove exact duplicates
    clean = raw.drop_duplicates().copy()

    assert clean["UDI"].is_unique

    # Validate numerical features
    for col in NUM:
        clean[col] = pd.to_numeric(
            clean[col],
            errors="raise"
        )

        assert np.isfinite(
            clean[col].dropna()
        ).all()

    # Validate categorical features
    clean["Type"] = (
        clean["Type"].str.strip().str.upper()
    )

    assert set(
        clean["Type"].dropna()
    ).issubset({"L", "M", "H"})

    # Validate target
    assert clean[TARGET].notna().all()

    assert set(clean[TARGET].unique()) == {0, 1}

    # Validate numerical ranges
    for col in NUM[:3]:
        assert (clean[col].dropna() > 0).all()

    for col in NUM[3:]:
        assert (clean[col].dropna() >= 0).all()

    # Separate features and target
    X = clean[FEATURES].copy()

    y = clean[TARGET].astype(int)

    print("Validated rows:", len(X))
    print(y.value_counts())

    return X, y



@task(name="Split Dataset", log_prints=True)
def split_data(X, y):

    X_train, X_hold, y_train, y_hold = (
        train_test_split(
            X,
            y,
            test_size=0.4,
            stratify=y,
            random_state=SEED
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_hold,
            y_hold,
            test_size=0.5,
            stratify=y_hold,
            random_state=SEED
        )
    )

    # Validate no overlapping observations
    assert not (
        set(X_train.index) & set(X_val.index)
        or set(X_train.index) & set(X_test.index)
        or set(X_val.index) & set(X_test.index)
    )

    assert (
        len(X_train) + len(X_val) + len(X_test)
    ) == len(X)

    # Package all datasets into a dictionary
    splits = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test)
    }

    for name, (features, labels) in splits.items():

        print(
            name,
            "Rows:", len(features),
            "Failures:", int(labels.sum())
        )

    return splits



@task(name="Save Datasets and Schema", log_prints=True)
def save_datasets(splits):

    X_train, y_train = splits["train"]

    # Save train, validation and test datasets
    for name, (features, labels) in splits.items():

        dataset = features.assign(
            **{TARGET: labels}
        )

        dataset.to_csv(
            OUT / f"{name}.csv",
            index=False
        )

    # Save reference dataset for drift detection
    X_train.to_csv(
        OUT / "reference.csv",
        index=False
    )

    # Define input schema
    schema = {
        "features": [
            {
                "name": col,
                "dtype": str(X_train[col].dtype),
                "kind": (
                    "categorical"
                    if col in CAT
                    else "numeric"
                )
            }
            for col in FEATURES
        ],

        "target": {
            "name": TARGET,
            "values": [0, 1]
        },

        "allowed_categories": {
            "Type": ["L", "M", "H"]
        },

        "missing_policy":
            "median for numeric, most frequent for categorical",

        "feature_order": FEATURES
    }

    (OUT / "schema.json").write_text(
        json.dumps(schema, indent=2)
    )

    assert list(
        pd.read_csv(OUT / "reference.csv").columns
    ) == FEATURES

    print("All datasets and schema saved.")
