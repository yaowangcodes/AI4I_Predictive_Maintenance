
#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Starting AI4I MLOps deployment..."

# Check Docker
docker info > /dev/null

# Generate passwords
generate_password() {
    openssl rand -hex 24
}

# Create environment variables if missing
if [ ! -f .env ]; then

    umask 077

    cat > .env <<EOF
POSTGRES_USER=ai4i_user
POSTGRES_PASSWORD=$(generate_password)
POSTGRES_DB=ai4i_db
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432

MODEL_NAME=AI4I_Predictive_Maintenance
MODEL_VERSION=
MLFLOW_TRACKING_URI=http://127.0.0.1:5001

GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(generate_password)
GRAFANA_DB_USER=ai4i_grafana
GRAFANA_DB_PASSWORD=$(generate_password)
EOF

    echo "Environment configuration created."

fi

# Create persistent directories
mkdir -p mlflow_storage artifacts

# Deploy using published images
docker compose \
    -f docker-compose.yml \
    -f docker-compose.prod.yml \
    up -d --no-build

echo ""
echo "Deployment completed."
echo ""
echo "Streamlit:  http://localhost:8501"
echo "FastAPI:    http://localhost:8000/docs"
echo "MLflow:     http://localhost:5001"
echo "Grafana:    http://localhost:3000"
echo "Prometheus: http://localhost:9090"
echo ""
echo "Grafana credentials are stored in .env"

docker compose \
    -f docker-compose.yml \
    -f docker-compose.prod.yml \
    ps -a
