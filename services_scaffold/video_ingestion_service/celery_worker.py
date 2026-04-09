from celery import Celery
import httpx
from video_ingestion_service.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, PROCESSING_ORCHESTRATION_SERVICE_URL
import logging
from typing import Optional
from session_bvh_data_service.models import SessionTypeEnum # Import enum for type hinting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

celery_app = Celery(
    'video_ingestion_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['video_ingestion_service.celery_worker']
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True
)

@celery_app.task(name="trigger_orchestration_service_task")
def trigger_orchestration_service_task(
    video_id: str, user_id: int, s3_key: str, session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING,
    resolution: Optional[str] = None, frame_rate: Optional[float] = None, duration_ms: Optional[int] = None
):
    """
    Celery task to notify the Processing Orchestration Service that a video is ready for processing.
    This is for historical uploads where a new session needs to be created.
    """
    logger.info(f"Celery task: Triggering orchestration for video_id: {video_id}, user_id: {user_id}, s3_key: {s3_key}, session_type: {session_type}, resolution: {resolution}, frame_rate: {frame_rate}, duration_ms: {duration_ms}")
    try:
        # The Processing Orchestration Service expects a POST to /internal/process-video/{video_id}
        # with a body containing user_id, raw_video_s3_key, session_type, and video metadata.
        payload = {
            "user_id": user_id,
            "raw_video_s3_key": s3_key,
            "session_type": session_type.value if session_type else None,
            "resolution": resolution,
            "frame_rate": frame_rate,
            "duration_ms": duration_ms
        }
        response = httpx.post(
            f"{PROCESSING_ORCHESTRATION_SERVICE_URL}/process-video/{video_id}",
            json=payload,
            timeout=30.0 # Set a reasonable timeout
        )
        response.raise_for_status()
        logger.info(f"Successfully triggered Processing Orchestration Service for video_id: {video_id}. Response: {response.json()}")
        return {"status": "success", "video_id": video_id, "orchestration_response": response.json()}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error triggering Processing Orchestration Service for video_id {video_id}: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error triggering Processing Orchestration Service for video_id {video_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in trigger_orchestration_service_task for video_id {video_id}: {e}")
        raise

@celery_app.task(name="trigger_processing_for_live_session_task")
def trigger_processing_for_live_session_task(
    session_id: int, user_id: int, raw_video_s3_key: str,
    resolution: Optional[str] = None, frame_rate: Optional[float] = None, duration_ms: Optional[int] = None
):
    """
    Celery task to notify the Processing Orchestration Service that a live session's video is ready for processing.
    """
    logger.info(f"Celery task: Triggering orchestration for live session_id: {session_id}, user_id: {user_id}, raw_video_s3_key: {raw_video_s3_key}, resolution: {resolution}, frame_rate: {frame_rate}, duration_ms: {duration_ms}")
    try:
        payload = {
            "user_id": user_id,
            "raw_video_s3_key": raw_video_s3_key,
            "resolution": resolution,
            "frame_rate": frame_rate,
            "duration_ms": duration_ms
        }
        response = httpx.post(
            f"{PROCESSING_ORCHESTRATION_SERVICE_URL}/process-existing-session-video/{session_id}",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        logger.info(f"Successfully triggered Processing Orchestration Service for live session_id: {session_id}. Response: {response.json()}")
        return {"status": "success", "session_id": session_id, "orchestration_response": response.json()}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error triggering Processing Orchestration Service for live session_id {session_id}: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error triggering Processing Orchestration Service for live session_id {session_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in trigger_processing_for_live_session_task for session_id {session_id}: {e}")
        raise
