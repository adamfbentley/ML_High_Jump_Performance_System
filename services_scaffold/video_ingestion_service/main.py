from fastapi import FastAPI, Depends, HTTPException, status
from typing import Dict, Any
from uuid import uuid4
import os
import logging
import urllib.parse # Added for S3 key sanitization
import httpx # Added for internal service calls

from video_ingestion_service import schemas, s3_utils
from video_ingestion_service.config import S3_BUCKET_NAME, SESSION_BVH_DATA_SERVICE_URL
from video_ingestion_service.celery_worker import trigger_orchestration_service_task, trigger_processing_for_live_session_task
from session_bvh_data_service.models import SessionTypeEnum # Import enum for type hinting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Ingestion Service",
    description="Handles video uploads, manages video metadata, and triggers asynchronous processing tasks.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/videos/upload-request", response_model=schemas.VideoUploadResponse, status_code=status.HTTP_200_OK)
async def request_video_upload(request: schemas.VideoUploadRequest):
    """
    Requests a pre-signed S3 URL for direct video upload.
    The client should use the returned URL and form_fields to perform a direct S3 POST upload.
    """
    video_id = str(uuid4())
    # Construct S3 object key: user_id/videos/{video_id}/{original_file_name}
    # Sanitize file_name to prevent path traversal or other issues, and URL-encode for S3 compatibility
    base_file_name = os.path.basename(request.file_name)
    sanitized_file_name = urllib.parse.quote_plus(base_file_name)
    s3_object_key = f"{request.user_id}/videos/{video_id}/{sanitized_file_name}"

    # Define conditions for the presigned POST policy
    conditions = [
        {"acl": "private"}, # Videos are private by default
        {"bucket": S3_BUCKET_NAME},
        ["content-length-range", 1, 500 * 1024 * 1024], # Max 500 MB
        {"Content-Type": request.content_type}
    ]
    fields = {"acl": "private", "Content-Type": request.content_type}

    try:
        presigned_post = s3_utils.generate_presigned_post(
            object_name=s3_object_key,
            fields=fields,
            conditions=conditions,
            expiration=3600 # URL valid for 1 hour
        )
        return schemas.VideoUploadResponse(
            video_id=video_id,
            upload_url=presigned_post['url'],
            form_fields=presigned_post['fields']
        )
    except Exception as e:
        logger.error(f"Failed to generate pre-signed S3 URL for user {request.user_id}, file {request.file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate pre-signed S3 URL: {e}"
        )

@app.post("/internal/videos/{video_id}/uploaded", status_code=status.HTTP_202_ACCEPTED)
async def video_upload_complete_notification(video_id: str, notification: schemas.VideoUploadedNotification):
    """
    Receives notification that a video upload to S3 is complete and triggers asynchronous processing.
    This is typically for historical uploads where a new session needs to be created.
    """
    if video_id != notification.video_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video ID in path does not match body.")

    logger.info(f"Video {video_id} uploaded by user {notification.user_id} to S3 key {notification.s3_key}. Dispatching processing task.")
    # Dispatch Celery task to trigger the Processing Orchestration Service
    trigger_orchestration_service_task.delay(
        video_id=notification.video_id,
        user_id=notification.user_id,
        s3_key=notification.s3_key,
        session_type=notification.session_type, # Pass session_type
        resolution=notification.resolution,
        frame_rate=notification.frame_rate,
        duration_ms=notification.duration_ms
    )
    return {"message": f"Video {video_id} upload acknowledged. Processing initiated."}

@app.post("/internal/sessions/{session_id}/video-uploaded-for-live-session", status_code=status.HTTP_202_ACCEPTED)
async def live_session_video_upload_complete_endpoint(
    session_id: int,
    request: schemas.LiveSessionVideoUploadCompleteRequest
):
    """
    Receives notification that a live-captured video has been uploaded to S3 and triggers processing.
    This updates an existing session record with the raw video S3 key and metadata.
    """
    logger.info(f"Live session video upload complete for session_id: {session_id}, user_id: {request.user_id}, S3 key: {request.s3_key}")

    async with httpx.AsyncClient() as client:
        # 1. Update the existing Session record in Session & BVH Data Service
        update_payload = schemas.SessionUpdateRawVideoDetails(
            raw_video_s3_key=request.s3_key,
            resolution=request.resolution,
            frame_rate=request.frame_rate,
            duration_ms=request.duration_ms
        )
        try:
            response = await client.put(
                f"{SESSION_BVH_DATA_SERVICE_URL}/sessions/{session_id}/raw-video-details?user_id={request.user_id}",
                json=update_payload.dict(exclude_unset=True)
            )
            response.raise_for_status()
            logger.info(f"Successfully updated session {session_id} with raw video details.")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error updating session {session_id} raw video details: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error updating session {session_id} raw video details: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

        # 2. Dispatch Celery task to trigger processing for the existing live session
        trigger_processing_for_live_session_task.delay(
            session_id=session_id,
            user_id=request.user_id,
            raw_video_s3_key=request.s3_key,
            resolution=request.resolution,
            frame_rate=request.frame_rate,
            duration_ms=request.duration_ms
        )

    return {"message": f"Live session {session_id} video upload acknowledged. Processing initiated."}
