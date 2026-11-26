"""Batch monitoring calculations, kept independent of Prefect and database startup."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import Accuracy, F1Score, Precision, Recall, RocAuc
from evidently.presets import DataDriftPreset

from config import CAT, FEATURES, NUM, OUT

EVIDENTLY_VERSION = "0.7.23"
PERFORMANCE_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LOG_COLUMNS = ["id", "prediction", "probability", "actual_label", "model_name",
               "model_version", "predicted_at", "labeled_at"]


def validate_features(frame, allow_empty=False):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing required raw feature columns")
    result = frame[FEATURES].copy()
    if result.empty and not allow_empty:
        raise ValueError("Reference/current feature dataset is empty")
    if not result["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must contain only L, M, or H without missing values")
    for name in NUM:
        if (not pd.api.types.is_numeric_dtype(result[name])
                or pd.api.types.is_bool_dtype(result[name])):
            raise ValueError(f"Feature must be numerical: {name}")
        if not np.isfinite(result[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Feature contains missing or nonfinite values: {name}")
        minimum = 0
        if (result[name] < minimum).any() or (name in NUM[:2] and (result[name] == 0).any()):
            raise ValueError(f"Feature is outside the API input range: {name}")
    for name in ["Rotational speed [rpm]", "Tool wear [min]"]:
        if (result[name] % 1 != 0).any():
            raise ValueError(f"Feature must contain whole numbers: {name}")
        result[name] = result[name].astype("int64")
    result["Type"] = result["Type"].astype(str)
    return result


def load_reference(path=OUT / "reference.csv", training_path=OUT / "train.csv"):
    """Verify the reference is exactly the saved training subset, excluding labels."""
    path, training_path = Path(path), Path(training_path)
    reference = validate_features(pd.read_csv(path))
    training = validate_features(pd.read_csv(training_path))
    if not reference.equals(training):
        raise ValueError("Reference does not match the saved original training features")
    return reference, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "rows": len(reference),
        "scope": "original_training_subset; final model also fitted on validation",
        "model_association": "operator-confirmed; historical registry has no reference checksum",
    }


def load_batch_manifest(path):
    manifest = json.loads(Path(path).read_text())
    UUID(manifest["batch_id"])
    if manifest.get("mode") not in {"normal", "drift"}:
        raise ValueError("Manifest mode must be normal or drift")
    if manifest.get("status") != "completed":
        raise ValueError("Use a completed simulation manifest; interrupted batches need reconciliation")
    ids = manifest.get("prediction_ids")
    if not isinstance(ids, list) or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError("Manifest prediction IDs must be positive integers")
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate prediction IDs")
    successful = [r for r in manifest.get("records", []) if r.get("prediction_status") == "success"]
    if len(successful) != len(ids) or {r["prediction_id"] for r in successful} != set(ids):
        raise ValueError("Manifest records and prediction IDs are misaligned")
    for name in ["started_at", "finished_at"]:
        timestamp = pd.Timestamp(manifest[name])
        if timestamp.tzinfo is None:
            raise ValueError("Batch timestamps must include a timezone")
    if pd.Timestamp(manifest["finished_at"]) < pd.Timestamp(manifest["started_at"]):
        raise ValueError("Batch end precedes its start")
    return manifest


def reconstruct_predictions(records):
    """Build features and metadata together, preserving one row per prediction ID."""
    rows = []
    for record in records:
        features = record["features"]
        if isinstance(features, str):
            features = json.loads(features)
        if not isinstance(features, dict) or not set(FEATURES).issubset(features):
            raise ValueError("Prediction log contains invalid or incomplete feature JSON")
        rows.append({**{name: features[name] for name in FEATURES},
                     **{name: record[name] for name in LOG_COLUMNS}})
    if not rows:
        return pd.DataFrame(columns=FEATURES + LOG_COLUMNS)
    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs in database results")
    frame[FEATURES] = validate_features(frame)
    if not frame["prediction"].isin([0, 1]).all():
        raise ValueError("Prediction labels must be binary")
    probabilities = pd.to_numeric(frame["probability"], errors="raise")
    if not probabilities.between(0, 1).all():
        raise ValueError("Failure probabilities must be finite and between zero and one")
    frame["probability"] = probabilities
    for name in ["model_name", "model_version"]:
        if frame[name].isna().any() or not frame[name].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("Missing model identity in prediction logs")
    frame["predicted_at"] = pd.to_datetime(frame["predicted_at"], utc=True, errors="raise")
    frame["labeled_at"] = pd.to_datetime(frame["labeled_at"], utc=True, errors="raise")
    if frame["predicted_at"].isna().any():
        raise ValueError("Missing prediction timestamp")
    return frame.sort_values("id").reset_index(drop=True)


def load_current_batch(session, manifest, model_name=None, model_version=None):
    from sqlalchemy import select
    from database.models import PredictionLog

    statement = select(PredictionLog.__table__).where(PredictionLog.id.in_(manifest["prediction_ids"]))
    records = session.execute(statement).mappings().all()
    frame = reconstruct_predictions(records)
    if set(frame["id"]) != set(manifest["prediction_ids"]):
        raise ValueError("Some manifest prediction IDs are absent from this database")
    # Detect a manifest accidentally applied to a different database/model deployment.
    expected = {r["prediction_id"]: r for r in manifest["records"] if r["prediction_status"] == "success"}
    for _, row in frame.iterrows():
        for key in ["model_name", "model_version"]:
            if row[key] != expected[row["id"]][key]:
                raise ValueError("Database model identity does not match the batch manifest")
    if model_name is not None:
        frame = frame.loc[frame["model_name"] == model_name]
    if model_version is not None:
        frame = frame.loc[frame["model_version"] == model_version]
    if len(frame[["model_name", "model_version"]].drop_duplicates()) > 1:
        raise ValueError("Batch contains multiple models/versions; select --model-name and --model-version")
    return frame.reset_index(drop=True)


def check_parameters(drift_share, min_samples, min_labeled, numerical_threshold, categorical_threshold):
    if not math.isfinite(drift_share) or not 0 < drift_share <= 1:
        raise ValueError("Drift share must be in (0, 1]")
    if min_samples < 2 or min_labeled < 1:
        raise ValueError("Minimum drift samples must be >= 2 and labeled samples >= 1")
    if any(not math.isfinite(x) or x <= 0 for x in [numerical_threshold, categorical_threshold]):
        raise ValueError("Feature distance thresholds must be finite and positive")


def calculate_drift(reference, current, drift_share=0.5, min_samples=100,
                    numerical_threshold=0.1, categorical_threshold=0.1):
    check_parameters(drift_share, min_samples, 1, numerical_threshold, categorical_threshold)
    reference = validate_features(reference)
    result = {"status": "insufficient_data", "sample_size": len(current),
              "reference_size": len(reference), "number_of_features": len(FEATURES),
              "number_of_drifted_features": None, "share_of_drifted_features": None,
              "dataset_drift": None, "threshold": drift_share, "features": []}
    if current.empty or min(len(reference), len(current)) < min_samples:
        result["reason"] = f"Both reference and current require at least {min_samples} rows"
        return result, None
    current = validate_features(current)
    definition = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
    snapshot = Report([DataDriftPreset(
        columns=FEATURES, drift_share=drift_share,
        num_method="wasserstein", cat_method="jensenshannon",
        num_threshold=numerical_threshold, cat_threshold=categorical_threshold,
    )], include_tests=True).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    raw = snapshot.dict()
    tests = {t["metric_config"]["metric_id"]: t for t in raw["tests"]}
    for metric in raw["metrics"]:
        config = metric["config"]
        kind = config["type"].split(":")[-1]
        if kind == "ValueDrift":
            score = float(metric["value"])
            if not math.isfinite(score):
                raise ValueError("Evidently returned a nonfinite drift score")
            result["features"].append({
                "feature": config["column"], "score": score,
                "method": config["method"], "threshold": config["threshold"],
                "score_kind": "distance", "drift_detected": tests[metric["id"]]["status"] == "FAIL",
                "interpretation": "larger distances indicate more distribution shift",
            })
        elif kind == "DriftedColumnsCount":
            result["number_of_drifted_features"] = int(metric["value"]["count"])
            result["share_of_drifted_features"] = float(metric["value"]["share"])
            result["dataset_drift"] = tests[metric["id"]]["status"] == "FAIL"
    if len(result["features"]) != len(FEATURES) or result["dataset_drift"] is None:
        raise ValueError("Unexpected Evidently drift report structure")
    result["status"] = "drift" if result["dataset_drift"] else "normal"
    return result, snapshot


def calculate_performance(current, mode="normal", min_labeled=30):
    if min_labeled < 1:
        raise ValueError("Minimum labeled sample count must be positive")
    valid = current["actual_label"].isin([0, 1])
    labeled = current.loc[valid].copy() if mode != "drift" else current.iloc[:0].copy()
    result = {
        "status": "unavailable", "labeled_predictions": len(labeled),
        "stored_valid_labels": int(valid.sum()),
        "invalid_labels": int((current["actual_label"].notna() & ~valid).sum()),
        "label_coverage": len(labeled) / len(current) if len(current) else None,
        "positive_class": 1, "metrics": {name: None for name in PERFORMANCE_NAMES},
        "unavailable_reasons": {}, "confusion_matrix": None,
    }
    if mode == "drift":
        result["reason"] = "Synthetic feature modification: no valid labeling mechanism established"
        return result, None
    if len(labeled) < min_labeled:
        result["reason"] = f"Need at least {min_labeled} valid labels; found {len(labeled)}"
        return result, None
    # Keep logged prediction labels. Never reconstruct labels from a threshold.
    target = labeled["actual_label"].astype(int)
    prediction = labeled["prediction"].astype(int)
    positives = int((target == 1).sum())
    predicted_positives = int((prediction == 1).sum())
    unavailable = result["unavailable_reasons"]
    if not predicted_positives:
        unavailable["precision"] = "No positive predictions"
    if not positives:
        unavailable["recall"] = "No actual positives"
    if not positives and not predicted_positives:
        unavailable["f1"] = "No actual or predicted positives"
    if target.nunique() < 2:
        unavailable["roc_auc"] = "ROC-AUC requires both target classes"
    metric_types = {"accuracy": Accuracy, "precision": Precision, "recall": Recall,
                    "f1": F1Score, "roc_auc": RocAuc}
    metrics = [cls(classification_name="probabilities" if name == "roc_auc" else "labels")
               for name, cls in metric_types.items() if name not in unavailable]
    # String labels avoid implicit label coercion in the binary data definition.
    labeled["actual_label"] = target.astype(str)
    labeled["prediction"] = prediction.astype(str)
    classifications = [BinaryClassification(
        name="labels", target="actual_label", prediction_labels="prediction", pos_label="1")]
    if "roc_auc" not in unavailable:
        classifications.append(BinaryClassification(
            name="probabilities", target="actual_label", prediction_probas="probability", pos_label="1"))
    definition = DataDefinition(classification=classifications)
    snapshot = Report(metrics).run(Dataset.from_pandas(
        labeled[["actual_label", "prediction", "probability"]], data_definition=definition))
    names = {cls.__name__: name for name, cls in metric_types.items()}
    for metric in snapshot.dict()["metrics"]:
        name = names[metric["config"]["type"].split(":")[-1]]
        value = float(metric["value"])
        if math.isfinite(value):
            result["metrics"][name] = value
        else:
            unavailable[name] = "Evidently returned an undefined value"
    # Counts preserve an explicit 2x2 layout even when one class is absent.
    result["confusion_matrix"] = {
        "labels": [0, 1], "rows": "actual", "columns": "predicted",
        "values": [[int(((target == a) & (prediction == p)).sum()) for p in [0, 1]] for a in [0, 1]],
    }
    result["status"] = "partial" if unavailable else "available"
    return result, snapshot


def build_metric_rows(summary):
    """One scalar per row, with null values and explicit reasons when unavailable."""
    context = summary["context"]
    drift, performance = summary["drift"], summary["performance"]
    common = {
        "run_id": summary["run_id"], "batch_id": context["batch_id"],
        "simulation_mode": context["mode"], "model_name": context["model_name"],
        "model_version": context["model_version"],
        "window_start": pd.Timestamp(context["window_start"]).to_pydatetime(),
        "window_end": pd.Timestamp(context["window_end"]).to_pydatetime(),
        "calculated_at": pd.Timestamp(summary["calculated_at"]).to_pydatetime(),
    }
    rows = []

    def add(name, value, status, sample_size, feature=None, threshold=None, method=None, reason=None):
        details = {"input_fingerprint": summary["input_fingerprint"],
                   "reference": summary["reference"], "parameters": summary["parameters"],
                   "evidently_version": EVIDENTLY_VERSION}
        if reason:
            details["reason"] = reason
        rows.append({**common, "metric_name": name, "metric_key": f"{name}:{feature or ''}",
                     "metric_value": float(value) if value is not None else None,
                     "feature_name": feature, "threshold": threshold, "statistical_method": method,
                     "status": status, "sample_size": sample_size, "details": details})

    for feature in drift["features"]:
        status = "drift" if feature["drift_detected"] else "normal"
        add("feature_drift_score", feature["score"], status, drift["sample_size"],
            feature["feature"], feature["threshold"], feature["method"])
        rows[-1]["details"]["score_interpretation"] = feature["interpretation"]
        add("feature_drift_detected", feature["drift_detected"], status, drift["sample_size"], feature["feature"])
    for key in ["number_of_features", "number_of_drifted_features", "share_of_drifted_features", "dataset_drift"]:
        add(key, drift[key], drift["status"], drift["sample_size"],
            threshold=drift["threshold"] if key == "share_of_drifted_features" else None,
            reason=drift.get("reason"))
    for key in ["labeled_predictions", "label_coverage", "invalid_labels"]:
        add(key, performance[key], "available" if performance[key] is not None else "unavailable",
            drift["sample_size"], reason=performance.get("reason"))
    for name, value in performance["metrics"].items():
        add(name, value, "available" if value is not None else "unavailable",
            performance["labeled_predictions"], method="evidently_classification",
            reason=performance["unavailable_reasons"].get(name, performance.get("reason")))
    if performance["confusion_matrix"] is not None:
        matrix = performance["confusion_matrix"]["values"]
        for name, value in zip(["true_negatives", "false_positives", "false_negatives", "true_positives"],
                               [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]):
            add(name, value, "available", performance["labeled_predictions"], method="observed_label_counts")
    return rows


def persist_metrics(engine, rows):
    """Atomically write one run; identical retries are no-ops, changed retries fail."""
    from sqlalchemy import insert, select, text
    from database.models import MonitoringMetric

    if not rows:
        raise ValueError("No monitoring metrics to persist")
    run_id = rows[0]["run_id"]
    UUID(run_id)
    if any(row["run_id"] != run_id for row in rows):
        raise ValueError("Cannot persist multiple runs in one transaction")
    if len({r["metric_key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate metric keys within a run")
    digest = metric_fingerprint(rows)
    values = [{**row, "details": {**row["details"], "result_fingerprint": digest}} for row in rows]
    table = MonitoringMetric.__table__
    with engine.begin() as connection:
        # Serialize concurrent retries for the same run; unique index is a second guard.
        connection.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:run_id, 0))"), {"run_id": run_id})
        existing = connection.execute(select(table).where(table.c.run_id == run_id)).mappings().all()
        if existing:
            if metric_fingerprint(existing) != digest:
                raise ValueError("Run ID already exists with different inputs/results; use a new run ID")
            return 0
        connection.execute(insert(table), values)
    return len(values)


def metric_fingerprint(rows):
    # Category summation order can change distance scores by ~1e-17 across Python
    # processes. Compare at 12 decimal places, retaining full precision in storage.
    # Input fingerprints and statuses are still exact, so this cannot hide changed
    # inputs/configuration or a changed drift decision.
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items() if k != "result_fingerprint"}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, float):
            return round(value, 12)
        return value

    content = [normalize({k: v for k, v in row.items() if k not in {"id", "calculated_at"}}) for row in rows]
    content.sort(key=lambda x: x["metric_key"])
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def save_reports(directory, summary, drift_snapshot, performance_snapshot):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, snapshot in [("drift", drift_snapshot), ("performance", performance_snapshot)]:
        if snapshot is not None:
            snapshot.save_json(str(directory / f"{name}.json"))
            snapshot.save_html(str(directory / f"{name}.html"))
    temporary = directory / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "summary.json")


def prepare_summary(manifest, current, reference_info, reference_model_name,
                    reference_model_version, run_id, parameters, drift, performance):
    UUID(run_id)
    identities = current[["model_name", "model_version"]].drop_duplicates()
    expected = (reference_model_name, reference_model_version)
    if not reference_model_name or not reference_model_version:
        raise ValueError("Explicit reference model name/version confirmation is required")
    if len(identities) > 1 or (len(identities) == 1 and tuple(identities.iloc[0]) != expected):
        raise ValueError("Reference model identity does not match the selected prediction model")
    reference_info = {**reference_info, "model_name": reference_model_name, "model_version": reference_model_version}
    context = {"batch_id": manifest["batch_id"], "mode": manifest["mode"],
               "model_name": reference_model_name, "model_version": reference_model_version,
               "prediction_ids": current["id"].tolist(), "batch_predictions": len(manifest["prediction_ids"]),
               "window_start": current["predicted_at"].min().isoformat() if len(current) else manifest["started_at"],
               "window_end": current["predicted_at"].max().isoformat() if len(current) else manifest["finished_at"]}
    inputs = {"context": context, "reference": reference_info, "parameters": parameters,
              "evidently_version": EVIDENTLY_VERSION,
              "prediction_snapshot": current.to_json(orient="records", date_format="iso", double_precision=15)}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"run_id": run_id, "calculated_at": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint, "context": context, "reference": reference_info,
            "parameters": parameters, "evidently_version": EVIDENTLY_VERSION,
            "drift": drift, "performance": performance, "alert": drift["dataset_drift"] is True}
