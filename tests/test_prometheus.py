from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from prometheus_client.parser import text_string_to_metric_families
import pytest

from api.instrumentation import instrument_app


@pytest.fixture
def client():
    app = FastAPI()
    instrument_app(app)

    class Input(BaseModel):
        value: int

    @app.post("/predict")
    def predict(body: Input):
        if body.value < 0:
            raise RuntimeError("simulated failure")
        return {"ok": True}

    @app.get("/items/{item_id}")
    def item(item_id: int):
        return {"id": item_id}

    return TestClient(app, raise_server_exceptions=False)


def samples(client, name):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return [sample for family in text_string_to_metric_families(response.text)
            for sample in family.samples if sample.name == name]


def test_request_counts_errors_and_prediction_outcomes(client):
    assert client.post("/predict", json={"value": 1}).status_code == 200
    assert client.post("/predict", json={"value": -1}).status_code == 500
    assert client.post("/predict", json={}).status_code == 422
    counter = samples(client, "ai4i_http_requests_total")
    assert {(s.labels["status"], s.value) for s in counter} == {("200", 1), ("500", 1), ("422", 1)}
    assert all(s.labels["endpoint"] == "/predict" for s in counter)
    errors = samples(client, "ai4i_http_errors_total")
    assert {s.labels["status"] for s in errors} == {"500", "422"}
    predictions = samples(client, "ai4i_prediction_requests_total")
    assert sum(s.value for s in predictions) == 3
    assert all(set(s.labels) == {"status"} for s in predictions)


def test_duration_records_success_and_failure(client):
    client.post("/predict", json={"value": 1})
    client.post("/predict", json={"value": -1})
    counts = samples(client, "ai4i_http_request_duration_seconds_count")
    assert {s.labels["status"] for s in counts} == {"200", "500"}
    assert all(s.value == 1 for s in counts)
    assert all(s.value > 0 for s in samples(client, "ai4i_http_request_duration_seconds_sum"))
    assert any(s.labels["le"] == "+Inf" for s in samples(client, "ai4i_http_request_duration_seconds_bucket"))


def test_routes_are_normalized_and_scrapes_excluded(client):
    for number in range(5):
        client.get(f"/items/{number}")
        client.get(f"/unknown/{number}")
        client.get("/metrics")
    data = samples(client, "ai4i_http_requests_total")
    assert {(s.labels["endpoint"], s.value) for s in data} == {("/items/{item_id}", 5), ("unmatched", 5)}
    assert samples(client, "ai4i_prediction_requests_total") == []


def test_instrumentation_registration_is_app_scoped_and_idempotent():
    first, second = FastAPI(), FastAPI()
    metrics = instrument_app(first)
    assert instrument_app(first) is metrics
    assert instrument_app(second).registry is not metrics.registry
    assert len([r for r in first.routes if r.path == "/metrics"]) == 1
    TestClient(first).get("/docs")
    assert not samples(TestClient(second), "ai4i_http_requests_total")
