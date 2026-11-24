from datetime import datetime, timezone, timedelta
from sqlalchemy import select

from database.connection import SessionLocal
from database.models import PredictionLog, MonitoringMetric


# Test 1: Prediction Logs
def test_prediction_log():

    with SessionLocal() as session:

        log = PredictionLog(
            features={
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 120
            },
            prediction=1,
            probability=0.87,
            actual_label=None,
            model_name="demo_model",
            model_version="test"
        )

        session.add(log)
        session.flush()

        result = session.execute(
            select(PredictionLog).where(
                PredictionLog.id == log.id
            )
        ).scalar_one()

        assert result.prediction == 1
        assert result.actual_label is None

        # Update actual label
        result.actual_label = 1
        result.labeled_at = datetime.now(timezone.utc)

        session.flush()
        session.refresh(result)

        assert result.actual_label == 1

        session.rollback()

        print("Prediction logs test passed!")


# Test 2: Monitoring Metrics
def test_monitoring_metric():

    with SessionLocal() as session:

        now = datetime.now(timezone.utc)

        metric = MonitoringMetric(
            metric_name="f1_score",
            metric_value=0.85,
            feature_name=None,
            threshold=0.70,
            status="normal",
            sample_size=100,
            model_name="demo_model",
            model_version="test",
            window_start=now - timedelta(hours=1),
            window_end=now
        )

        session.add(metric)
        session.flush()

        result = session.execute(
            select(MonitoringMetric).where(
                MonitoringMetric.id == metric.id
            )
        ).scalar_one()

        assert result.metric_name == "f1_score"
        assert result.metric_value == 0.85

        session.rollback()

        print("Monitoring metrics test passed!")


# Run all tests
if __name__ == "__main__":

    test_prediction_log()
    test_monitoring_metric()

    print("All database tests passed!")