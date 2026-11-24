
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from api.model_loader import load_registered_model

from api.schemas import FeedbackRequest
from datetime import datetime, timezone

import pandas as pd
import logging

from fastapi import HTTPException
from api.schemas import PredictionRequest

from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import SessionLocal
from database.models import PredictionLog

from api.schemas import PredictionResponse

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

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()


@app.get("/")
def root():

    return {
        "message": "AI4I API is running"
    }



@app.get("/health")
def health(
    request: Request,
    db: Session = Depends(get_db)
):

    try:

        # Check database connection
        db.execute(text("SELECT 1"))

        # Check model
        model_loaded = (
            getattr(request.app.state, "model", None)
            is not None
        )

        if not model_loaded:

            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        return {
            "status": "healthy",
            "model_loaded": True,
            "database_connected": True,
            "model_name": request.app.state.model_name,
            "model_version": request.app.state.model_version
        }

    except HTTPException:

        raise

    except Exception:

        logger.exception("Health check failed")

        raise HTTPException(
            status_code=503,
            detail="Database connection failed"
        )





@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    db: Session = Depends(get_db)
):

    try:

        # Step 1: Convert input into model features
        features = payload.to_features()

        X = pd.DataFrame([features])

        # Step 2: Retrieve model information
        model = request.app.state.model

        model_name = request.app.state.model_name

        model_version = request.app.state.model_version

        # Step 3: Generate prediction
        prediction = int(model.predict(X)[0])

        failure_index = list(model.classes_).index(1)

        probability = float(
            model.predict_proba(X)[0][failure_index]
        )

        # Step 4: Create prediction log
        log = PredictionLog(
            features=features,
            prediction=prediction,
            probability=probability,
            model_name=model_name,
            model_version=model_version
        )

        # Step 5: Save prediction into PostgreSQL
        db.add(log)

        db.commit()

        db.refresh(log)

        # Step 6: Return response
        return {
            "prediction_id": log.id,
            "prediction": prediction,
            "failure_probability": probability,
            "model_name": model_name,
            "model_version": model_version
        }

    except Exception:

        db.rollback()

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction service failed"
        )


@app.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db)
):

    try:

        # Step 1: Find the prediction record
        log = db.get(
            PredictionLog,
            payload.prediction_id
        )

        # Step 2: Check whether the record exists
        if log is None:

            raise HTTPException(
                status_code=404,
                detail="Prediction ID not found"
            )

        # Step 3: Update actual label
        log.actual_label = payload.actual_label

        # Step 4: Record feedback time
        log.labeled_at = datetime.now(timezone.utc)

        # Step 5: Save changes
        db.commit()

        db.refresh(log)

        # Step 6: Return feedback result
        return {
            "status": "success",
            "prediction_id": log.id,
            "prediction": log.prediction,
            "actual_label": log.actual_label,
            "correct": log.prediction == log.actual_label
        }

    except HTTPException:

        db.rollback()

        raise

    except Exception:

        db.rollback()

        logger.exception("Feedback failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to save feedback"
        )
