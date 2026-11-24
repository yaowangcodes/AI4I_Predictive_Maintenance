from database.connection import engine, Base

# Import table definitions
from database.models import (
    PredictionLog,
    MonitoringMetric
)


def init_database():

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")


if __name__ == "__main__":
    init_database()