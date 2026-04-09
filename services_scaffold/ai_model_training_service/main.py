from fastapi import FastAPI, Depends, HTTPException, status, Path
from ai_model_training_service import schemas
from ai_model_training_service.celery_worker import train_personal_model_task
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Model Training Service",
    description="Manages the training and continuous fine-tuning of personal models.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/train/personal-model/{athlete_id}", response_model=schemas.PersonalModelTrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_personal_model_training(
    athlete_id: int = Path(..., description="ID of the athlete whose personal model is to be trained."),
    request: schemas.PersonalModelTrainRequest = schemas.PersonalModelTrainRequest() # Allow default values
) -> schemas.PersonalModelTrainResponse:
    """
    Triggers the asynchronous training/fine-tuning process for an athlete's personal model.
    """
    logger.info(f"Received request to train personal model for athlete_id: {athlete_id}. Dispatching Celery task.")

    # Dispatch Celery task
    task_info = train_personal_model_task.delay(
        athlete_id=athlete_id,
        retrain_epochs=request.retrain_epochs,
        learning_rate=request.learning_rate
    )

    return schemas.PersonalModelTrainResponse(
        status="training_initiated",
        message=f"Personal model training initiated for athlete {athlete_id}. Task ID: {task_info.id}",
        athlete_id=athlete_id,
        model_version=None, # Model version will be available after training completes
        model_artifact_s3_path=None # S3 path will be available after training completes
    )
