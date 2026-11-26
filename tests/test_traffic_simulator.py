import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from config import TARGET
from scripts import simulate_traffic as sim


@pytest.fixture
def data():
    return pd.DataFrame([
        {"Type": "M", "Air temperature [K]": 299.9,
         "Process temperature [K]": 309.2, "Rotational speed [rpm]": 1539,
         "Torque [Nm]": 43.2, "Tool wear [min]": 132, TARGET: 0},
        {"Type": "H", "Air temperature [K]": 301.0,
         "Process temperature [K]": 311.0, "Rotational speed [rpm]": 1700,
         "Torque [Nm]": 60.0, "Tool wear [min]": 200, TARGET: 1},
    ])


def response(body, status=200):
    return Mock(status_code=status, json=Mock(return_value=body))


def prediction(number=1):
    return {"prediction_id": number, "prediction": 0, "failure_probability": 0.1,
            "model_name": "AI4I_Predictive_Maintenance", "model_version": "7"}


def arguments(tmp_path, data, mode="normal", n=5):
    path = tmp_path / "test.csv"
    data.to_csv(path, index=False)
    return sim.parse_args(["--dataset", str(path), "--manifest-dir", str(tmp_path / "batches"),
                           "--mode", mode, "--n", str(n)])


def test_payload_preserves_values(data):
    payload = sim.build_payload(data.iloc[0])
    assert payload == {"Type": "M", "air_temperature": 299.9,
                       "process_temperature": 309.2, "rotational_speed": 1539,
                       "torque": 43.2, "tool_wear": 132}
    assert TARGET not in payload


@pytest.mark.parametrize("strength", [0, 1, 2, 0.13])
def test_drift_and_no_mutation(data, strength):
    original = data.copy(deep=True)
    normal = sim.build_payload(data.iloc[0])
    drift = sim.build_payload(data.iloc[0], "drift", strength)
    for key in normal:
        shift = sim.DRIFT.get(key, 0) * strength
        if key == "tool_wear":
            shift = round(shift)
        assert drift[key] == (normal[key] + shift if key != "Type" else normal[key])
    pd.testing.assert_frame_equal(data, original)


def test_sampling_alignment_and_reproducibility(data):
    first = sim.sample_rows(data, 50, 42)
    pd.testing.assert_frame_equal(first, sim.sample_rows(data, 50, 42))
    assert set(first.index) == {0, 1}
    for index, row in first.iterrows():
        pd.testing.assert_series_equal(row, data.loc[index])


@pytest.mark.parametrize("mode", ["normal", "drift"])
def test_run_feedback_and_manifest(tmp_path, data, mode):
    args = arguments(tmp_path, data, mode)
    session = Mock()
    counter = 0

    def post(url, json, **kwargs):
        nonlocal counter
        assert kwargs == {"timeout": 10.0, "allow_redirects": False}
        if url.endswith("/predict"):
            counter += 1
            return response(prediction(counter))
        return response({"status": "success", **json})

    session.post.side_effect = post
    result = sim.run_simulation(args, session)
    assert result["summary"]["successful_predictions"] == 5
    assert result["summary"]["successful_feedback"] == (5 if mode == "normal" else 0)
    assert result["prediction_ids"] == [1, 2, 3, 4, 5]
    assert result["status"] == "completed"
    for record in result["records"]:
        if mode == "normal":
            assert record["actual_label"] == data.loc[record["source_row"], TARGET]
        else:
            assert "actual_label" not in record
    calls = session.post.call_args_list
    feedback_calls = [c for c in calls if c.args[0].endswith("/feedback")]
    assert len(feedback_calls) == (5 if mode == "normal" else 0)
    saved = json.loads((args.manifest_dir / f"{result['batch_id']}.json").read_text())
    assert saved == result
    second = sim.run_simulation(args, session)
    assert second["batch_id"] != result["batch_id"]
    assert [r["source_row"] for r in result["records"]] == [r["source_row"] for r in second["records"]]


@pytest.mark.parametrize("failure", [requests.ConnectionError("secret"), requests.Timeout("secret"),
                                    response({}, 422), response({}, 500), response({}, 302),
                                    response({}), response({**prediction(), "failure_probability": 2}),
                                    response({**prediction(), "prediction_id": -1}),
                                    response({**prediction(), "failure_probability": float("nan")})])
def test_prediction_failures(tmp_path, data, failure, capsys):
    args = arguments(tmp_path, data, n=1)
    session = Mock()
    if isinstance(failure, Exception):
        session.post.side_effect = failure
    else:
        session.post.return_value = failure
    result = sim.run_simulation(args, session)
    assert result["summary"]["failed_predictions"] == 1
    assert result["summary"]["successful_feedback"] == 0
    assert session.post.call_count == 1
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", [requests.Timeout(), response({}, 404), response({}), response(None)])
def test_failed_feedback_keeps_prediction_id(tmp_path, data, failure):
    args = arguments(tmp_path, data, n=1)
    session = Mock()
    session.post.side_effect = [response(prediction()), failure]
    result = sim.run_simulation(args, session)
    assert result["prediction_ids"] == [1]
    assert result["summary"]["successful_predictions"] == 1
    assert result["summary"]["failed_feedback"] == 1


@pytest.mark.parametrize("option,value", [("--n", "0"), ("--n", "-1"), ("--n", "1.2"),
    ("--delay", "-1"), ("--delay", "nan"), ("--drift-strength", "inf"),
    ("--drift-strength", "-1"), ("--mode", "bad"), ("--seed", "-1"),
    ("--seed", str(2**32)), ("--timeout", "0"), ("--api-url", "ftp://host"),
    ("--api-url", "http://user:password@host"), ("--api-url", "http://host?token=secret")])
def test_cli_rejects_invalid_arguments(option, value):
    with pytest.raises(SystemExit):
        sim.parse_args([option, value])


def test_api_url_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8000/")
    assert sim.parse_args([]).api_url == "http://api:8000"
    assert sim.parse_args(["--api-url", "http://another:8000"]).api_url == "http://another:8000"


@pytest.mark.parametrize("problem", ["missing_column", "missing_label", "invalid_label", "invalid_feature", "empty"])
def test_bad_dataset(tmp_path, data, problem):
    if problem == "missing_column":
        data = data.drop(columns="Type")
    elif problem == "missing_label":
        data.loc[1, TARGET] = float("nan")
    elif problem == "invalid_label":
        data.loc[1, TARGET] = 2
    elif problem == "invalid_feature":
        data.loc[1, "Tool wear [min]"] = -1
    else:
        data = data.iloc[:0]
    path = tmp_path / "bad.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        sim.load_dataset(path)


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        sim.load_dataset(tmp_path / "missing.csv")


def test_interruption_persists_manifest(tmp_path, data):
    args = arguments(tmp_path, data, n=1)
    session = Mock()
    session.post.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        sim.run_simulation(args, session)
    saved = json.loads(next(args.manifest_dir.glob("*.json")).read_text())
    assert saved["status"] == "interrupted"
    assert saved["records"][0]["prediction_status"] == "pending"


def test_continues_after_failure_and_bad_json(tmp_path, data):
    args = arguments(tmp_path, data, mode="drift", n=3)
    bad_json = Mock(status_code=200)
    bad_json.json.side_effect = ValueError("private response body")
    session = Mock()
    session.post.side_effect = [requests.Timeout(), bad_json, response(prediction(3))]
    result = sim.run_simulation(args, session)
    assert result["summary"]["attempted_predictions"] == 3
    assert result["summary"]["failed_predictions"] == 2
    assert result["prediction_ids"] == [3]


def test_main_missing_dataset(tmp_path, capsys):
    assert sim.main(["--dataset", str(tmp_path / "missing.csv")]) == 1
    assert "not found" in capsys.readouterr().err
