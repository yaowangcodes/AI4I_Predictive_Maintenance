"""Deterministic monitoring tests; no live PostgreSQL, MLflow, or Prefect required."""
import copy
from datetime import datetime, timezone
import json
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from config import FEATURES, NUM
from tasks import monitoring as m


@pytest.fixture
def features():
    rng = np.random.default_rng(684)
    n = 300
    return pd.DataFrame({
        "Type": np.tile(["L", "M", "H"], n // 3),
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 2, n),
        "Rotational speed [rpm]": rng.integers(1200, 2000, n),
        "Torque [Nm]": rng.uniform(25, 55, n),
        "Tool wear [min]": rng.integers(0, 240, n),
    })


@pytest.fixture
def records(features):
    now = datetime.now(timezone.utc)
    return [{"id": i + 1, "features": row.to_dict(), "prediction": i % 2,
             "probability": 0.1 if i % 2 == 0 else 0.8, "actual_label": i % 2,
             "model_name": "model", "model_version": "7", "predicted_at": now,
             "labeled_at": now} for i, (_, row) in enumerate(features.iloc[:4].iterrows())]


def manifest_for(records):
    now = datetime.now(timezone.utc).isoformat()
    return {"batch_id": str(uuid4()), "mode": "normal", "status": "completed",
            "prediction_ids": [r["id"] for r in records], "started_at": now, "finished_at": now,
            "records": [{"prediction_id": r["id"], "prediction_status": "success",
                         "model_name": r["model_name"], "model_version": r["model_version"]} for r in records]}


@pytest.fixture
def offline_database(monkeypatch):
    # Constructing the engine does not connect. These placeholders also work in CI without .env.
    for key, value in {"POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
                       "POSTGRES_DB": "test", "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432"}.items():
        monkeypatch.setenv(key, value)
    from database.models import MonitoringMetric
    return MonitoringMetric


def test_reference_excludes_leakage_and_matches_training(tmp_path, features):
    ref, train = tmp_path / "ref.csv", tmp_path / "train.csv"
    features.assign(UDI=range(len(features))).to_csv(ref, index=False)
    features.assign(**{"Machine failure": 0}).to_csv(train, index=False)
    loaded, metadata = m.load_reference(ref, train)
    assert list(loaded.columns) == FEATURES
    assert len(metadata["sha256"]) == 64
    assert metadata["rows"] == len(features)
    changed = features.copy()
    changed.loc[0, NUM[0]] += 1
    changed.to_csv(ref, index=False)
    with pytest.raises(ValueError, match="does not match"):
        m.load_reference(ref, train)


@pytest.mark.parametrize("problem", ["missing", "empty", "bad_type", "numeric_string", "null", "inf", "fractional"])
def test_invalid_features(features, problem):
    if problem == "missing":
        features = features.drop(columns="Type")
    elif problem == "empty":
        features = features.iloc[:0]
    elif problem == "bad_type":
        features.loc[0, "Type"] = "unknown"
    elif problem == "numeric_string":
        features[NUM[0]] = features[NUM[0]].astype(str)
    elif problem in {"null", "inf"}:
        features.loc[0, NUM[0]] = np.nan if problem == "null" else np.inf
    else:
        features["Tool wear [min]"] = features["Tool wear [min]"].astype(float) + 0.1
    with pytest.raises(ValueError):
        m.validate_features(features)


def test_missing_reference(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.load_reference(tmp_path / "missing.csv")


def test_reconstruction_alignment_json_and_jsonb(records):
    records[1]["features"] = json.dumps(records[1]["features"])
    frame = m.reconstruct_predictions(list(reversed(records)))
    assert frame["id"].tolist() == [1, 2, 3, 4]
    for row in frame.to_dict("records"):
        source = records[row["id"] - 1]
        features = json.loads(source["features"]) if isinstance(source["features"], str) else source["features"]
        assert row["actual_label"] == source["actual_label"]
        assert row["Torque [Nm]"] == features["Torque [Nm]"]
    assert str(frame["predicted_at"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("problem", ["duplicate", "missing_feature", "bad_probability", "bad_prediction"])
def test_invalid_production_records(records, problem):
    if problem == "duplicate":
        records.append(records[0])
    elif problem == "missing_feature":
        del records[0]["features"]["Type"]
    elif problem == "bad_probability":
        records[0]["probability"] = np.inf
    else:
        records[0]["prediction"] = 2
    with pytest.raises(ValueError):
        m.reconstruct_predictions(records)


def test_batch_ids_and_version_filter(records, offline_database):
    records[3]["model_version"] = "8"
    manifest = manifest_for(records)
    session = MagicMock()
    session.execute.return_value.mappings.return_value.all.return_value = records
    with pytest.raises(ValueError, match="multiple"):
        m.load_current_batch(session, manifest)
    result = m.load_current_batch(session, manifest, model_name="model", model_version="7")
    assert result["id"].tolist() == [1, 2, 3]
    statement = session.execute.call_args.args[0]
    # The database query can only return explicit batch members, not a broad time window.
    assert list(statement.compile().params.values()) == [[1, 2, 3, 4]]
    assert m.load_current_batch(session, manifest, model_version="missing").empty


def test_missing_database_ids_and_identity_mismatch(records, offline_database):
    manifest = manifest_for(records)
    session = MagicMock()
    session.execute.return_value.mappings.return_value.all.return_value = records[:2]
    with pytest.raises(ValueError, match="absent"):
        m.load_current_batch(session, manifest)
    records[0]["model_version"] = "other"
    session.execute.return_value.mappings.return_value.all.return_value = records
    with pytest.raises(ValueError, match="identity"):
        m.load_current_batch(session, manifest)


@pytest.mark.parametrize("problem", [None, "duplicate", "interrupted", "misalignment"])
def test_manifest_validation(tmp_path, records, problem):
    manifest = manifest_for(records)
    if problem == "duplicate":
        manifest["prediction_ids"].append(1)
    elif problem == "interrupted":
        manifest["status"] = "interrupted"
    elif problem == "misalignment":
        manifest["records"] = []
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(manifest))
    if problem:
        with pytest.raises(ValueError):
            m.load_batch_manifest(path)
    else:
        assert m.load_batch_manifest(path) == manifest


def test_identical_distributions(features):
    result, report = m.calculate_drift(features, features)
    assert result["dataset_drift"] is False
    assert result["number_of_drifted_features"] == 0
    assert all(f["score"] == pytest.approx(0) for f in result["features"])
    categorical = next(f for f in result["features"] if f["feature"] == "Type")
    assert categorical["method"] == "jensenshannon"
    assert report is not None


def test_strong_shift_and_threshold_boundary(features):
    shifted = features.copy()
    for name in NUM[:3]:
        shifted[name] += 10000
    result, _ = m.calculate_drift(features, shifted, drift_share=0.5)
    assert result["number_of_drifted_features"] == 3
    assert result["share_of_drifted_features"] == 0.5
    assert result["dataset_drift"] is True  # at least half, not strictly greater
    higher, _ = m.calculate_drift(features, shifted, drift_share=0.51)
    assert higher["dataset_drift"] is False


def test_categorical_shift(features):
    shifted = features.copy()
    shifted["Type"] = "H"
    result, _ = m.calculate_drift(features, shifted)
    assert [f["feature"] for f in result["features"] if f["drift_detected"]] == ["Type"]


def test_insufficient_drift_and_empty_production(features):
    result, report = m.calculate_drift(features, features.iloc[:5])
    assert result["status"] == "insufficient_data"
    assert result["dataset_drift"] is None and report is None
    empty = m.reconstruct_predictions([])
    result, report = m.calculate_drift(features, empty)
    performance, _ = m.calculate_performance(empty)
    assert result["sample_size"] == 0
    assert performance["label_coverage"] is None


def test_known_classification_and_positive_probability():
    current = pd.DataFrame({"actual_label": [0, 0, 0, 1], "prediction": [0, 0, 1, 1],
                            "probability": [0.1, 0.2, 0.6, 0.8]})
    result, _ = m.calculate_performance(current, min_labeled=1)
    assert result["metrics"] == pytest.approx({"accuracy": .75, "precision": .5,
                                               "recall": 1, "f1": 2 / 3, "roc_auc": 1})
    assert result["confusion_matrix"]["values"] == [[2, 1], [0, 1]]
    current["probability"] = 1 - current["probability"]
    inverted, _ = m.calculate_performance(current, min_labeled=1)
    assert inverted["metrics"]["roc_auc"] == 0
    for name in ["accuracy", "precision", "recall", "f1"]:
        assert inverted["metrics"][name] == result["metrics"][name]


def test_logged_predictions_are_not_rethresholded():
    current = pd.DataFrame({"actual_label": [0, 0, 1, 1], "prediction": [0, 0, 0, 0],
                            "probability": [.1, .2, .6, .8]})
    result, _ = m.calculate_performance(current, min_labeled=1)
    assert result["metrics"] == {"accuracy": .5, "precision": None, "recall": 0, "f1": 0, "roc_auc": 1}


@pytest.mark.parametrize("targets,predictions,undefined", [
    ([0]*4, [0]*4, {"precision", "recall", "f1", "roc_auc"}),
    ([0]*4, [0, 0, 1, 1], {"recall", "roc_auc"}),
    ([1]*4, [1]*4, {"roc_auc"}),
])
def test_single_class_metrics(targets, predictions, undefined):
    frame = pd.DataFrame({"actual_label": targets, "prediction": predictions, "probability": .5})
    result, _ = m.calculate_performance(frame, min_labeled=1)
    assert {key for key, val in result["metrics"].items() if val is None} == undefined
    assert sum(map(sum, result["confusion_matrix"]["values"])) == 4
    assert result["status"] == "partial"


def test_labels_missing_invalid_or_disallowed(records):
    frame = m.reconstruct_predictions(records)
    frame["actual_label"] = [None, None, 2, 1]
    result, _ = m.calculate_performance(frame, min_labeled=2)
    assert result["labeled_predictions"] == 1
    assert result["label_coverage"] == .25
    assert result["invalid_labels"] == 1
    assert result["status"] == "unavailable"
    assert all(v is None for v in result["metrics"].values())
    drift, report = m.calculate_performance(frame, mode="drift", min_labeled=1)
    assert drift["labeled_predictions"] == 0 and report is None
    assert "Synthetic" in drift["reason"]


@pytest.fixture
def summary(features, records):
    current = m.reconstruct_predictions(records)
    drift, _ = m.calculate_drift(features, current)
    performance, _ = m.calculate_performance(current)
    return m.prepare_summary(manifest_for(records), current, {"sha256": "abc"},
                             "model", "7", str(uuid4()), {}, drift, performance)


def test_reference_model_mismatch(features, records):
    current = m.reconstruct_predictions(records)
    with pytest.raises(ValueError, match="identity"):
        m.prepare_summary(manifest_for(records), current, {}, "other", "7", str(uuid4()), {}, {}, {})


def test_metric_transformation_preserves_nulls_and_context(summary):
    rows = m.build_metric_rows(summary)
    assert len({r["metric_key"] for r in rows}) == len(rows)
    for row in rows:
        assert row["run_id"] == summary["run_id"]
        assert row["model_version"] == "7"
        assert row["batch_id"] == summary["context"]["batch_id"]
    auc = next(r for r in rows if r["metric_name"] == "roc_auc")
    assert auc["metric_value"] is None and auc["status"] == "unavailable"
    assert "reason" in auc["details"]


def test_transactional_persistence_idempotency_and_conflict(summary, offline_database):
    rows = m.build_metric_rows(summary)
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value.all.return_value = []
    assert m.persist_metrics(engine, rows) == len(rows)
    inserted = connection.execute.call_args.args[1]
    assert all("result_fingerprint" in r["details"] for r in inserted)
    connection.execute.return_value.mappings.return_value.all.return_value = inserted
    assert m.persist_metrics(engine, rows) == 0
    changed = copy.deepcopy(rows)
    changed[0]["metric_value"] = 999
    with pytest.raises(ValueError, match="different inputs"):
        m.persist_metrics(engine, changed)
    assert engine.begin.return_value.__exit__.call_args.args[0] is ValueError


def test_database_failure_rolls_back(summary, offline_database):
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    result = MagicMock()
    result.mappings.return_value.all.return_value = []
    connection.execute.side_effect = [MagicMock(), result, RuntimeError("database unavailable")]
    with pytest.raises(RuntimeError):
        m.persist_metrics(engine, m.build_metric_rows(summary))
    assert engine.begin.return_value.__exit__.call_args.args[0] is RuntimeError


def test_report_files(tmp_path, features, summary):
    _, report = m.calculate_drift(features, features)
    m.save_reports(tmp_path, summary, report, None)
    assert json.loads((tmp_path / "summary.json").read_text())["run_id"] == summary["run_id"]
    assert "metrics" in json.loads((tmp_path / "drift.json").read_text())
    assert "<html" in (tmp_path / "drift.html").read_text().lower()
    assert not (tmp_path / "performance.json").exists()


@pytest.mark.parametrize("option,value", [("--drift-share", "0"), ("--drift-share", "1.1"),
    ("--drift-share", "nan"), ("--min-samples", "1"), ("--min-labeled", "0"),
    ("--numerical-threshold", "-1"), ("--categorical-threshold", "inf"), ("--run-id", "bad")])
def test_cli_validation(option, value):
    from monitoring_flow import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--batch-id", str(uuid4()), "--reference-model-name", "model",
                    "--reference-model-version", "7", option, value])


def test_model_metrics_json_and_filters(offline_database):
    from fastapi.testclient import TestClient
    from api.main import app, get_db
    session = MagicMock()
    session.scalars.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: session
    try:
        # No context manager: avoid starting the MLflow-dependent lifespan.
        client = TestClient(app)
        for endpoint in ["/model-metrics"]:
            response = client.get(endpoint, params={"run_id": "run", "batch_id": "batch", "limit": 10})
            assert response.status_code == 200
            assert response.json() == {"count": 0, "metrics": []}
            parameters = session.scalars.call_args.args[0].compile().params
            assert {"run", "batch", 10}.issubset(set(parameters.values()))
        assert client.get("/model-metrics?limit=1001").status_code == 422
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_drift_metric_rows_preserve_methods(features, records):
    current = m.reconstruct_predictions(records)
    drift, _ = m.calculate_drift(features, features)
    performance, _ = m.calculate_performance(current, min_labeled=1)
    summary = m.prepare_summary(manifest_for(records), current, {}, "model", "7", str(uuid4()), {}, drift, performance)
    rows = m.build_metric_rows(summary)
    scores = [r for r in rows if r["metric_name"] == "feature_drift_score"]
    assert len(scores) == 6
    assert {r["statistical_method"] for r in scores} == {"wasserstein", "jensenshannon"}
    assert all(r["threshold"] == .1 for r in scores)
    assert next(r["metric_value"] for r in rows if r["metric_name"] == "true_positives") == 2


def test_migration_uses_transaction_and_additive_ddl():
    from database.migrate_monitoring import migrate
    engine = MagicMock()
    migrate(engine)
    connection = engine.begin.return_value.__enter__.return_value
    sql = [str(c.args[0]) for c in connection.execute.call_args_list]
    assert any("CREATE UNIQUE INDEX IF NOT EXISTS" in command for command in sql)
    assert any("metric_value DROP NOT NULL" in command for command in sql)
    assert not any("DROP TABLE" in command or "DELETE" in command for command in sql)
    assert engine.begin.return_value.__exit__.call_args.args == (None, None, None)


def test_retry_fingerprint_tolerates_only_roundoff(summary):
    rows = m.build_metric_rows(summary)
    rows[0]["metric_value"] = 0.0205556864669264
    stored = copy.deepcopy(rows)
    stored[0]["metric_value"] = 0.02055568646692641
    stored[0]["id"] = 12
    stored[0]["calculated_at"] = datetime.now(timezone.utc)
    stored[0]["details"]["result_fingerprint"] = "old-algorithm-hash"
    assert m.metric_fingerprint(rows) == m.metric_fingerprint(stored)
    stored[0]["metric_value"] += .00001
    assert m.metric_fingerprint(rows) != m.metric_fingerprint(stored)
    stored = copy.deepcopy(rows)
    stored[0]["details"]["input_fingerprint"] = "changed"
    assert m.metric_fingerprint(rows) != m.metric_fingerprint(stored)
