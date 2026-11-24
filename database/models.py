from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    CheckConstraint,
    func
)

from sqlalchemy.dialects.postgresql import JSONB

from database.connection import Base


class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Input features
    features = Column(
        JSONB,
        nullable=False
    )

    # Model prediction
    prediction = Column(
        Integer,
        nullable=False
    )

    probability = Column(
        Float,
        nullable=False
    )

    # Actual outcome
    actual_label = Column(
        Integer,
        nullable=True
    )

    # Model information
    model_name = Column(
        String(100),
        nullable=False
    )

    model_version = Column(
        String(50),
        nullable=False
    )

    # Timestamps
    predicted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )

    labeled_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    __table_args__ = (
        CheckConstraint("prediction IN (0, 1)"),
        CheckConstraint("actual_label IN (0, 1)"),
        CheckConstraint("probability BETWEEN 0 AND 1"),
    )

class MonitoringMetric(Base):

    __tablename__ = "monitoring_metrics"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Monitoring information
    metric_name = Column(
        String(100),
        nullable=False
    )

    metric_value = Column(
        Float,
        nullable=False
    )

    feature_name = Column(
        String(100),
        nullable=True
    )

    threshold = Column(
        Float,
        nullable=True
    )

    status = Column(
        String(20),
        nullable=False
    )

    # Evaluation context
    sample_size = Column(
        Integer,
        nullable=False
    )

    model_name = Column(
        String(100),
        nullable=False
    )

    model_version = Column(
        String(50),
        nullable=False
    )

    window_start = Column(
        DateTime(timezone=True),
        nullable=False
    )

    window_end = Column(
        DateTime(timezone=True),
        nullable=False
    )

    calculated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )


