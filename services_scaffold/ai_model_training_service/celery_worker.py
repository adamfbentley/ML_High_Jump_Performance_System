from celery import Celery
import httpx
import logging
from typing import Optional, Dict, Any
from ai_model_training_service.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, USER_PROFILE_SERVICE_URL, SESSION_BVH_DATA_SERVICE_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
from ai_model_training_service.schemas import UserProfileResponseMinimal, SessionResponseMinimal, AttemptResponseMinimal
from datetime import datetime
import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

celery_app = Celery(
    'ai_model_training_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['ai_model_training_service.celery_worker']
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True
)

def upload_model_artifact_to_s3(athlete_id: int, model_version: str, model_data: bytes) -> str:
    """Simulates uploading a model artifact to S3."""
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
        logger.warning("S3 credentials or bucket name not fully configured. Skipping actual S3 upload.")
        return f"s3://{S3_BUCKET_NAME}/simulated/models/athlete-{athlete_id}/{model_version}.pth"

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3_key = f"models/athlete-{athlete_id}/{model_version}.pth"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=model_data)
        logger.info(f"Simulated S3 upload successful for model {model_version} to s3://{S3_BUCKET_NAME}/{s3_key}")
        return f"s3://{S3_BUCKET_NAME}/{s3_key}"
    except Exception as e:
        logger.error(f"Failed to simulate S3 upload for model {model_version}: {e}")
        return f"s3://{S3_BUCKET_NAME}/simulated/models/athlete-{athlete_id}/{model_version}.pth" # Return simulated path on failure


@celery_app.task(name="train_personal_model_task")
def train_personal_model_task(athlete_id: int, retrain_epochs: int, learning_rate: float) -> Dict[str, Any]:
    """
    Celery task to simulate the training/fine-tuning of an athlete's personal model.
    """
    logger.info(f"Celery task: Initiating personal model training for athlete_id: {athlete_id} with epochs={retrain_epochs}, lr={learning_rate}")

    try:
        with httpx.Client() as client:
            # 1. Fetch athlete's profile data (anthropometrics) from User & Profile Service (BE-02)
            try:
                profile_response = client.get(f"{USER_PROFILE_SERVICE_URL}/profiles/{athlete_id}", timeout=10.0)
                profile_response.raise_for_status()
                user_profile = UserProfileResponseMinimal(**profile_response.json())
                logger.info(f"Fetched user profile for athlete {athlete_id}: {user_profile.dict()}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to fetch user profile for athlete {athlete_id}: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Failed to fetch user profile: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Network error fetching user profile for athlete {athlete_id}: {e}")
                raise Exception(f"Could not connect to user profile service: {e}")

            # 2. Fetch historical jump attempts and BVH data from Session & BVH Data Service (BE-04)
            # This data would be used as input for the optimization engine.
            try:
                sessions_response = client.get(f"{SESSION_BVH_DATA_SERVICE_URL}/users/{athlete_id}/sessions", timeout=10.0)
                sessions_response.raise_for_status()
                sessions_data = [SessionResponseMinimal(**s) for s in sessions_response.json()]
                logger.info(f"Fetched {len(sessions_data)} sessions for athlete {athlete_id}.")

                all_attempts_data = []
                for session in sessions_data:
                    attempts_response = client.get(f"{SESSION_BVH_DATA_SERVICE_URL}/sessions/{session.id}/attempts?user_id={athlete_id}", timeout=10.0)
                    attempts_response.raise_for_status()
                    attempts_for_session = [AttemptResponseMinimal(**a) for a in attempts_response.json()]
                    all_attempts_data.extend(attempts_for_session)
                logger.info(f"Fetched {len(all_attempts_data)} attempts for athlete {athlete_id}.")

                logger.info(f"Simulating data preparation from {len(all_attempts_data)} attempts and user profile.")

            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to fetch session/attempt data for athlete {athlete_id}: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Failed to fetch session/attempt data: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Network error fetching session/attempt data for athlete {athlete_id}: {e}")
                raise Exception(f"Could not connect to session BVH data service: {e}")

            # 3. Simulate LoRA model training/fine-tuning
            logger.info(f"Simulating LoRA model training for athlete {athlete_id} for {retrain_epochs} epochs.")
            import time
            time.sleep(5) # Simulate work

            # 4. Simulate saving model artifacts (e.g., to S3) and updating a model registry
            model_version = f"v1.0.0-athlete-{athlete_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Simulate model artifact data (e.g., a dummy byte string)
            simulated_model_data = f"dummy_model_weights_for_athlete_{athlete_id}_version_{model_version}".encode('utf-8')
            s3_path = upload_model_artifact_to_s3(athlete_id, model_version, simulated_model_data)
            
            logger.info(f"Simulated model training complete. Model version: {model_version}. Artifact stored at: {s3_path}")
            logger.info(f"Simulating storing model metadata in a database for athlete {athlete_id}, model_version {model_version}, s3_path {s3_path}.")

            return {
                "status": "success",
                "message": f"Personal model for athlete {athlete_id} trained successfully.",
                "athlete_id": athlete_id,
                "model_version": model_version,
                "model_artifact_s3_path": s3_path
            }

    except Exception as e:
        logger.error(f"An error occurred during personal model training for athlete {athlete_id}: {e}")
        return {
            "status": "failed",
            "message": str(e),
            "athlete_id": athlete_id,
            "model_version": None,
            "model_artifact_s3_path": None
        }
