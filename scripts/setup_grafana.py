"""Create local monitoring secrets and grant Grafana SELECT on monitoring_metrics."""
import os
from pathlib import Path
import secrets
import sys

import psycopg
from psycopg import sql
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def setup():
    load_dotenv(ROOT / ".env")
    path = ROOT / ".env.monitoring"
    if not path.exists():
        # Exclusive create and mode 0600; credentials are never printed.
        contents = ("GRAFANA_ADMIN_USER=admin\n"
                    f"GRAFANA_ADMIN_PASSWORD={secrets.token_urlsafe(32)}\n"
                    "GRAFANA_DB_USER=ai4i_grafana\n"
                    f"GRAFANA_DB_PASSWORD={secrets.token_urlsafe(32)}\n")
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(contents)
    settings = dotenv_values(path)
    role = settings["GRAFANA_DB_USER"]
    if role != "ai4i_grafana":
        raise ValueError("This setup script only manages the dedicated ai4i_grafana role")
    with psycopg.connect(user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
                         dbname=os.environ["POSTGRES_DB"], host=os.environ["POSTGRES_HOST"],
                         port=os.environ["POSTGRES_PORT"]) as connection:
        exists = connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        action = "ALTER" if exists else "CREATE"
        connection.execute(sql.SQL(action + " ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}")
                           .format(sql.Identifier(role), sql.Literal(settings["GRAFANA_DB_PASSWORD"])))
        connection.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}")
                           .format(sql.Identifier(os.environ["POSTGRES_DB"]), sql.Identifier(role)))
        connection.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        connection.execute(sql.SQL("GRANT SELECT ON public.monitoring_metrics TO {}").format(sql.Identifier(role)))
        connection.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(role)))
    print("Grafana read-only access configured. Secrets are in ignored .env.monitoring (not printed).")


if __name__ == "__main__":
    setup()
