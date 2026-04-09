from fastapi import FastAPI, Depends, HTTPException, status
from processing_orchestration_service import schemas
from processing_orchestration_service.celery_worker import process_video_pipeline_task, segment_attempt_task, trigger_processing_for_live_session_task
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Processing Orchestration Service",
    description="Orchestrates the asynchronous processing pipeline for videos.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/process-video/{video_id}", response_model=schemas.ProcessVideoResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_video_processing(
    video_id: str,
    request: schemas.ProcessVideoRequest
):
    """
    Triggers the full video processing pipeline for a given video (historical upload).
    This endpoint is expected to be called by internal services (e.g., Video Ingestion Service via Celery task).
    """
    logger.info(f"Received request to process video_id: {video_id} for user_id: {request.user_id} from S3 key: {request.raw_video_s3_key}. Dispatching Celery task.")

    # Dispatch Celery task to create session/attempt records and initiate further processing
    process_video_pipeline_task.delay(
        video_id=video_id,
        user_id=request.user_id,
        raw_video_s3_key=request.raw_video_s3_key,
        session_type=request.session_type.value if request.session_type else None,
        resolution=request.resolution,
        frame_rate=request.frame_rate,
        duration_ms=request.duration_ms
    )

    return schemas.ProcessVideoResponse(
        status="processing_initiated",
        message=f"Processing pipeline initiated for video {video_id}. Session and initial attempt creation dispatched.",
        video_id=video_id
    )

@app.post("/internal/process-existing-session-video/{session_id}", response_model=schemas.ProcessVideoResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_processing_for_live_session_endpoint(
    session_id: int,
    request: schemas.ProcessExistingSessionVideoRequest
):
    """
    Triggers the full video processing pipeline for an existing live session's video.
    This endpoint is expected to be called by internal services (e.g., Video Ingestion Service via Celery task).
    """
    logger.info(f"Received request to process existing live session_id: {session_id} for user_id: {request.user_id} from S3 key: {request.raw_video_s3_key}. Dispatching Celery task.")

    trigger_processing_for_live_session_task.delay(
        session_id=session_id,
        user_id=request.user_id,
        raw_video_s3_key=request.raw_video_s3_key,
        resolution=request.resolution,
        frame_rate=request.frame_rate,
        duration_ms=request.duration_ms
    )

    return schemas.ProcessVideoResponse(
        status="processing_initiated",
        message=f"Processing pipeline initiated for existing live session {session_id}.",
        video_id=str(session_id) # Use session_id as video_id for response consistency
    )

@app.post("/internal/segment-attempt/{attempt_id}", status_code=status.HTTP_202_ACCEPTED)
async def trigger_attempt_segmentation(attempt_id: int, request: schemas.SegmentAttemptRequest):
    """
    Triggers the segmentation update and subsequent processing for a specific jump attempt.
    This endpoint is expected to be called by internal services (e.g., a segmentation UI).
    """
    logger.info(f"Received request to segment attempt_id: {attempt_id} for user_id: {request.user_id} with start: {request.start_time_ms}ms, end: {request.end_time_ms}ms. Dispatching Celery task.")

    # Dispatch Celery task to update segmentation and trigger next steps (e.g., pose estimation)
    segment_attempt_task.delay(
        attempt_id=attempt_id,
        user_id=request.user_id,
        start_time_ms=request.start_time_ms,
        end_time_ms=request.end_time_ms
    )

    return {"message": f"Attempt {attempt_id} segmentation update acknowledged. Further processing initiated."}
