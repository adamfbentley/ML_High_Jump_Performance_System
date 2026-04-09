from pydantic import BaseModel, Field
from typing import Optional

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "pose_estimation_service"

class PoseEstimationRequest(BaseModel):
    attempt_id: int = Field(..., description="ID of the jump attempt to process.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")
    video_s3_key: str = Field(..., description="S3 key of the raw video file.")
    start_time_ms: int = Field(..., description="Start time of the jump segment in milliseconds.")
    end_time_ms: int = Field(..., description="End time of the jump segment in milliseconds.")

class PoseEstimationResponse(BaseModel):
    status: str = Field(..., description="Status of the pose estimation process.")
    message: str = Field(..., description="Details about the pose estimation result.")
    attempt_id: int = Field(..., description="ID of the processed attempt.")
    bvh_s3_key: Optional[str] = Field(None, description="S3 key of the generated BVH file.")

class UserProfileAnthropometrics(BaseModel):
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
