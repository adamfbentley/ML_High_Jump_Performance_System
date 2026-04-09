from pydantic import BaseModel, Field
from typing import Optional

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "processing_orchestration_service"

class TriggerVideoProcessingRequest(BaseModel):
    video_id: int = Field(..., description="ID of the video to process.")
    user_id: int = Field(..., description="ID of the user who owns the video.")
    session_id: int = Field(..., description="ID of the session associated with the video.")
    raw_video_s3_key: str = Field(..., description="S3 key of the raw video file.")

class TriggerAttemptSegmentationRequest(BaseModel):
    attempt_id: int = Field(..., description="ID of the jump attempt to segment.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")
    video_s3_key: str = Field(..., description="S3 key of the raw video file.")
    start_time_ms: int = Field(..., description="Start time of the jump segment in milliseconds.")
    end_time_ms: int = Field(..., description="End time of the jump segment in milliseconds.")

class TriggerPoseEstimationRequest(BaseModel):
    attempt_id: int = Field(..., description="ID of the jump attempt for pose estimation.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")
    video_s3_key: str = Field(..., description="S3 key of the raw video file.")
    start_time_ms: int = Field(..., description="Start time of the jump segment in milliseconds.")
    end_time_ms: int = Field(..., description="End time of the jump segment in milliseconds.")

class TriggerBiomechanicsAnalysisRequest(BaseModel): # NEW
    attempt_id: int = Field(..., description="ID of the jump attempt for biomechanical analysis.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")

class TriggerAnomalyDetectionRequest(BaseModel): # NEW
    attempt_id: int = Field(..., description="ID of the jump attempt for anomaly detection.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
