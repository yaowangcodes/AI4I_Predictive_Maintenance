
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from api.model_loader import load_registered_model

import pandas as pd
import logging

from fastapi import HTTPException
from api.schemas import PredictionRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load model from MLflow
    model, model_name, model_version = load_registered_model()

    # Store model in application memory
    app.state.model = model
    app.state.model_name = model_name
    app.state.model_version = model_version

    print("API startup completed!")

    yield

    print("API shutdown completed!")


app = FastAPI(
    title="AI4I Predictive Maintenance API",
    description="Machine failure prediction service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():

    return {
        "message": "AI4I API is running"
    }


@app.get("/health")
def health(request: Request):

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": request.app.state.model_name,
        "model_version": request.app.state.model_version
    }



@app.post("/predict")
def predict(
    payload: PredictionRequest,
    request: Request
):

    try:

        # Step 1: Convert user input into model features
        features = payload.to_features()

        # Step 2: Create a DataFrame
        X = pd.DataFrame([features])

        # Step 3: Retrieve the loaded model
        model = request.app.state.model

        # Step 4: Generate prediction
        prediction = int(model.predict(X)[0])

        # Step 5: Calculate failure probability
        failure_index = list(model.classes_).index(1)

        probability = float(
            model.predict_proba(X)[0][failure_index]
        )

        # Step 6: Return prediction results
        return {
            "prediction": prediction,
            "failure_probability": probability,
            "model_name": request.app.state.model_name,
            "model_version": request.app.state.model_version
        }

    except Exception:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

