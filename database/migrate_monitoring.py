"""Add monitoring context without dropping or rewriting existing records.

Run explicitly before starting the updated API: python -m database.migrate_monitoring
"""
from sqlalchemy import text

STATEMENTS = [
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS run_id VARCHAR(36)",
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS batch_id VARCHAR(36)",
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS simulation_mode VARCHAR(20)",
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS metric_key VARCHAR(200)",
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS statistical_method VARCHAR(100)",
    "ALTER TABLE monitoring_metrics ADD COLUMN IF NOT EXISTS details JSONB",
    "ALTER TABLE monitoring_metrics ALTER COLUMN metric_value DROP NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_monitoring_metrics_run_id ON monitoring_metrics (run_id)",
    "CREATE INDEX IF NOT EXISTS ix_monitoring_metrics_batch_id ON monitoring_metrics (batch_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_monitoring_run_metric "
    "ON monitoring_metrics (run_id, metric_key)",
]


def migrate(engine):
    # PostgreSQL DDL is transactional. Fail instead of waiting indefinitely on an API query.
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL lock_timeout = '5s'"))
        for statement in STATEMENTS:
            connection.execute(text(statement))


if __name__ == "__main__":
    from database.connection import engine
    migrate(engine)
    print("Monitoring migration completed; existing records preserved.")
