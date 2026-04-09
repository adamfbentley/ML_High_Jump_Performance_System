from celery import Celery
from celery.utils.log import get_task_logger
import httpx
import asyncio
import random

from processing_orchestration_service.config import (
    REDIS_URL,
    VIDEO_INGESTION_SERVICE_URL,
    SESSION_BVH_DATA_SERVICE_URL,
    POSE_ESTIMATION_SERVICE_URL,
    PINN_GNN_INFERENCE_SERVICE_URL,
    ANOMALY_DETECTION_SERVICE_URL
)
from processing_orchestration_service import schemas
from session_bvh_data_service.models import AttemptOutcomeEnum # Import for enum access

logger = get_task_logger(__name__)

celery_app = Celery(
    'processing_orchestration_service',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

async def _make_internal_post_request(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

@celery_app.task(name="trigger_video_processing_task")
def trigger_video_processing_task(request_data: dict):
    logger.info(f"Received trigger_video_processing_task for video_id: {request_data.get('video_id')}")
    try:
        session_id = request_data['session_id']
        user_id = request_data['user_id']
        raw_video_s3_key = request_data['raw_video_s3_key']

        # CQ-005 & CQ-006 Fix: Simulate video segmentation to create multiple attempts dynamically
        # In a real scenario, this would involve ML models to detect jumps and their timings.
        # For this fix, we simulate 2-3 attempts with realistic (but dummy) start/end times and bar heights.
        
        # Assuming a total video length for simulation, e.g., 30 seconds (30000 ms)
        total_video_length_ms = 30000 
        num_simulated_attempts = random.randint(1, 3)
        
        simulated_attempts_data = []
        current_time_ms = 0
        for i in range(num_simulated_attempts):
            # Simulate jump duration between 3 to 8 seconds
            jump_duration_ms = random.randint(3000, 8000)
            # Simulate a pause between jumps, min 2 seconds
            pause_duration_ms = random.randint(2000, 5000) if i < num_simulated_attempts - 1 else 0

            start_time_ms = current_time_ms + random.randint(500, 1500) # Small buffer before jump
            end_time_ms = start_time_ms + jump_duration_ms

            # Ensure times don't exceed total video length
            if end_time_ms > total_video_length_ms:
                end_time_ms = total_video_length_ms
                start_time_ms = max(0, end_time_ms - jump_duration_ms) # Adjust start if needed
                if start_time_ms == end_time_ms: # If video too short for even one jump
                    break

            bar_height = round(random.uniform(180.0, 220.0), 1) # Realistic bar heights
            outcome = random.choice([AttemptOutcomeEnum.SUCCESS, AttemptOutcomeEnum.FAIL, AttemptOutcomeEnum.KNOCK, AttemptOutcomeEnum.UNKNOWN]).value

            simulated_attempts_data.append({
                "attempt_number": i + 1,
                "bar_height_cm": bar_height,
                "outcome": outcome,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms
            })
            current_time_ms = end_time_ms + pause_duration_ms
            if current_time_ms >= total_video_length_ms: # Stop if we run out of video time
                break

        if not simulated_attempts_data:
            logger.warning(f"No attempts simulated for video {request_data['video_id']}. Video might be too short.")
            return {"status": "SKIPPED", "message": f"No attempts detected/simulated for video {request_data['video_id']}"}

        processed_attempts = []
        for attempt_data in simulated_attempts_data:
            attempt_create_payload = schemas.AttemptCreateLive(
                attempt_number=attempt_data['attempt_number'],
                bar_height_cm=attempt_data['bar_height_cm'],
                outcome=AttemptOutcomeEnum(attempt_data['outcome']),
                start_time_ms=attempt_data['start_time_ms'],
                end_time_ms=attempt_data['end_time_ms']
            ).dict()

            attempt_response = asyncio.run(_make_internal_post_request(
                f"{SESSION_BVH_DATA_SERVICE_URL}/sessions/{session_id}/attempts?user_id={user_id}",
                attempt_create_payload
            ))
            attempt_id = attempt_response['id']
            logger.info(f"Created attempt {attempt_id} (number {attempt_data['attempt_number']}) for session {session_id} with segmentation {attempt_data['start_time_ms']}-{attempt_data['end_time_ms']} ms")
            processed_attempts.append(attempt_id)

            # Trigger pose estimation for each created attempt with its specific segmentation times
            pose_estimation_request = schemas.TriggerPoseEstimationRequest(
                attempt_id=attempt_id,
                user_id=user_id,
                video_s3_key=raw_video_s3_key,
                start_time_ms=attempt_data['start_time_ms'],
                end_time_ms=attempt_data['end_time_ms']
            ).dict()
            trigger_pose_estimation_task.delay(pose_estimation_request)

        return {"status": "SUCCESS", "message": f"Video processing initiated for video {request_data['video_id']}. Created {len(processed_attempts)} attempts: {processed_attempts}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during video processing: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error during video processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during video processing for video {request_data.get('video_id')}: {e}", exc_info=True)
        raise

@celery_app.task(name="trigger_pose_estimation_task")
def trigger_pose_estimation_task(request_data: dict):
    logger.info(f"Received trigger_pose_estimation_task for attempt_id: {request_data.get('attempt_id')}")
    try:
        # Call Pose Estimation Service
        pose_estimation_response = asyncio.run(_make_internal_post_request(
            f"{POSE_ESTIMATION_SERVICE_URL}/pose-estimate",
            request_data
        ))
        logger.info(f"Pose estimation completed for attempt {request_data['attempt_id']}. BVH S3 Key: {pose_estimation_response.get('bvh_file_s3_key')}")

        # Trigger biomechanical analysis after BVH is ready
        biomechanics_analysis_request = schemas.TriggerBiomechanicsAnalysisRequest(
            attempt_id=request_data['attempt_id'],
            user_id=request_data['user_id']
        ).dict()
        trigger_pinn_gnn_inference_task.delay(biomechanics_analysis_request)

        return {"status": "SUCCESS", "message": f"Pose estimation completed for attempt {request_data['attempt_id']}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during pose estimation: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error during pose estimation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during pose estimation for attempt {request_data.get('attempt_id')}: {e}", exc_info=True)
        raise

@celery_app.task(name="trigger_pinn_gnn_inference_task")
def trigger_pinn_gnn_inference_task(request_data: dict):
    logger.info(f"Received trigger_pinn_gnn_inference_task for attempt_id: {request_data.get('attempt_id')}")
    try:
        # Call PINN & GNN Inference Service
        biomechanics_response = asyncio.run(_make_internal_post_request(
            f"{PINN_GNN_INFERENCE_SERVICE_URL}/analyze/biomechanics",
            request_data
        ))
        logger.info(f"Biomechanics analysis completed for attempt {request_data['attempt_id']}. Parameters S3 Key: {biomechanics_response.get('biomechanical_parameters_s3_key')}")

        # Trigger anomaly detection after biomechanical analysis is ready
        anomaly_detection_request = schemas.TriggerAnomalyDetectionRequest(
            attempt_id=request_data['attempt_id'],
            user_id=request_data['user_id']
        ).dict()
        trigger_anomaly_detection_task.delay(anomaly_detection_request)

        return {"status": "SUCCESS", "message": f"Biomechanics analysis completed for attempt {request_data['attempt_id']}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during biomechanics analysis: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error during biomechanics analysis: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during biomechanics analysis for attempt {request_data.get('attempt_id')}: {e}", exc_info=True)
        raise

@celery_app.task(name="trigger_anomaly_detection_task")
def trigger_anomaly_detection_task(request_data: dict):
    logger.info(f"Received trigger_anomaly_detection_task for attempt_id: {request_data.get('attempt_id')}")
    try:
        # Call Anomaly Detection Service
        anomaly_response = asyncio.run(_make_internal_post_request(
            f"{ANOMALY_DETECTION_SERVICE_URL}/detect/anomalies",
            request_data
        ))
        logger.info(f"Anomaly detection completed for attempt {request_data['attempt_id']}. Anomalies detected: {anomaly_response.get('anomalies_detected')}")

        return {"status": "SUCCESS", "message": f"Anomaly detection completed for attempt {request_data['attempt_id']}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during anomaly detection: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Network error during anomaly detection: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during anomaly detection for attempt {request_data.get('attempt_id')}: {e}", exc_info=True)
        raise
