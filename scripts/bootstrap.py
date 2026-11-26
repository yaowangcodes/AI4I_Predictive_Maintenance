
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import psycopg

from psycopg import sql

from config import OUT

from database.connection import engine
from database.init_db import init_database
from database.migrate_monitoring import migrate

from training_flow import training_flow


def setup_grafana_database():

    role = os.environ["GRAFANA_DB_USER"]
    password = os.environ["GRAFANA_DB_PASSWORD"]

    with psycopg.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"]
    ) as connection:

        exists = connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s",
            (role,)
        ).fetchone()

        action = "ALTER" if exists else "CREATE"

        connection.execute(
            sql.SQL(
                action
                + " ROLE {} LOGIN NOSUPERUSER "
                "NOCREATEDB NOCREATEROLE PASSWORD {}"
            ).format(
                sql.Identifier(role),
                sql.Literal(password)
            )
        )

        connection.execute(
            sql.SQL(
                "GRANT CONNECT ON DATABASE {} TO {}"
            ).format(
                sql.Identifier(os.environ["POSTGRES_DB"]),
                sql.Identifier(role)
            )
        )

        connection.execute(
            sql.SQL(
                "GRANT USAGE ON SCHEMA public TO {}"
            ).format(sql.Identifier(role))
        )

        connection.execute(
            sql.SQL(
                "GRANT SELECT ON public.monitoring_metrics TO {}"
            ).format(sql.Identifier(role))
        )

        connection.execute(
            sql.SQL(
                "ALTER ROLE {} SET default_transaction_read_only = on"
            ).format(sql.Identifier(role))
        )

    print("Grafana database access configured.")


def get_ready_versions(client, model_name):

    versions = client.search_model_versions(
        f"name = '{model_name}'"
    )

    return sorted(
        int(v.version)
        for v in versions
        if v.status == "READY"
    )


def ensure_model():

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MODEL_NAME"]
    requested_version = os.getenv("MODEL_VERSION", "").strip()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = mlflow.MlflowClient()

    versions = get_ready_versions(client, model_name)

    if not versions:

        if requested_version:
            raise RuntimeError(
                "Registry is empty. Set MODEL_VERSION="
                " for automatic initialization."
            )

        print("No registered model found.")
        print("Starting training pipeline...")

        result = training_flow()

        if not result.get("verified") or not result.get("model_uri"):
            raise RuntimeError("Model training or registration failed.")

        versions = get_ready_versions(client, model_name)

        if not versions:
            raise RuntimeError("No ready model after training.")

    if requested_version:
        version = int(requested_version)

        if version not in versions:
            raise RuntimeError(
                f"Requested model version {version} not available."
            )
    else:
        version = max(versions)

    model_uri = f"models:/{model_name}/{version}"

    # Verify that the actual model artifact is loadable.
    mlflow.sklearn.load_model(model_uri)

    required_files = [
        OUT / "test.csv",
        OUT / "train.csv"
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Registered model verified: {model_uri}")


def bootstrap():

    print("Starting system initialization...")

    # Step 1: Create database tables
    init_database()

    # Step 2: Apply monitoring migrations
    migrate(engine)

    # Step 3: Configure Grafana read-only database user
    setup_grafana_database()

    # Step 4: Verify or initialize the MLflow model
    ensure_model()

    print("SYSTEM INITIALIZATION COMPLETED!")


if __name__ == "__main__":
    bootstrap()
