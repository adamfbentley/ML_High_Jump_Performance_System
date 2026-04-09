from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from session_bvh_data_service.models import SessionTypeEnum # Import enum

class VideoUploadRequest(BaseModel):
    file_name: str = Field(..., description="Original name of the video file.")
    content_type: str = Field(..., description="MIME type of the video file, e.g., 'video/mp4'.")
    user_id: int = Field(..., description="ID of the user uploading the video.")
    session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING # Added session_type
    resolution: Optional[str] = Field(None, description="Resolution of the raw video, e.g., '1920x1080'.") # NEW
    frame_rate: Optional[float] = Field(None, description="Frame rate of the raw video, e.g., 30.0.") # NEW
    duration_ms: Optional[int] = Field(None, description="Duration of the raw video in milliseconds.") # NEW

class VideoUploadResponse(BaseModel):
    video_id: str = Field(..., description="Unique ID for the uploaded video (UUID part).")
    upload_url: str = Field(..., description="Pre-signed URL for direct S3 upload.")
    form_fields: Dict[str, str] = Field(..., description="Form fields required for S3 POST upload.")

class VideoUploadedNotification(BaseModel):
    video_id: str = Field(..., description="Unique ID of the uploaded video (UUID part).")
    s3_key: str = Field(..., description="S3 object key (path) of the uploaded video (full S3 key).")
    user_id: int = Field(..., description="ID of the user who uploaded the video.")
    session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING # Added session_type
    resolution: Optional[str] = Field(None, description="Resolution of the raw video, e.g., '1920x1080'.") # NEW
    frame_rate: Optional[float] = Field(None, description="Frame rate of the raw video, e.g., 30.0.") # NEW
    duration_ms: Optional[int] = Field(None, description="Duration of the raw video in milliseconds.") # NEW

class LiveSessionVideoUploadCompleteRequest(BaseModel): # NEW for live session video upload completion
    user_id: int = Field(..., description="ID of the user who owns the session.")
    s3_key: str = Field(..., description="S3 object key (path) of the uploaded video.")
    resolution: Optional[str] = Field(None, description="Resolution of the raw video, e.g., '1920x1080'.")
    frame_rate: Optional[float] = Field(None, description="Frame rate of the raw video, e.g., 30.0.")
    duration_ms: Optional[int] = Field(None, description="Duration of the raw video in milliseconds.")

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "video_ingestion_service"
