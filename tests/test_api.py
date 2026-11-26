
import pytest

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:

        yield test_client


VALID_PAYLOAD = {
    "Type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 108
}


# Test 1: Root endpoint
def test_root(client):

    response = client.get("/")

    assert response.status_code == 200


# Test 2: Health endpoint
def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["database_connected"] is True


# Test 3: Valid prediction
def test_predict(client):

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert 0 <= data["failure_probability"] <= 1

    assert data["prediction_id"] > 0

    assert "model_version" in data


# Test 4: Invalid product type
def test_invalid_type(client):

    payload = VALID_PAYLOAD.copy()

    payload["Type"] = "A"

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 5: Missing feature
def test_missing_feature(client):

    payload = VALID_PAYLOAD.copy()

    del payload["torque"]

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 6: Negative tool wear
def test_negative_tool_wear(client):

    payload = VALID_PAYLOAD.copy()

    payload["tool_wear"] = -10

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 7: Feedback endpoint
def test_feedback(client):

    prediction_response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert prediction_response.status_code == 200

    prediction_id = prediction_response.json()["prediction_id"]

    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "actual_label": 0
        }
    )

    assert response.status_code == 200

    assert response.json()["actual_label"] == 0


# Test 8: Invalid feedback label
def test_invalid_feedback(client):

    response = client.post(
        "/feedback",
        json={
            "prediction_id": 1,
            "actual_label": 2
        }
    )

    assert response.status_code == 422


# Test 9: Metrics endpoint
def test_metrics(client):

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "count" in data

    assert isinstance(data["metrics"], list)

    assert data["count"] == len(data["metrics"])
