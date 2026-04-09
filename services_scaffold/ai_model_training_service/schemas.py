from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

# Re-use enums from session_bvh_data_service for consistency if needed, or define locally
class SessionTypeEnum(str, Enum):
    TRAINING = "TRAINING"
    COMPETITION = "COMPETITION"
    OTHER = "OTHER"

class AttemptOutcomeEnum(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    KNOCK = "KNOCK"
    DID_NOT_ATTEMPT = "DID_NOT_ATTEMPT"
    UNKNOWN = "UNKNOWN"

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "ai_model_training_service"

class PersonalModelTrainRequest(BaseModel):
    # athlete_id is expected in path, so not in body for internal request
    retrain_epochs: int = Field(100, description="Number of epochs for LoRA fine-tuning.")
    learning_rate: float = Field(0.001, description="Learning rate for the fine-tuning process.")

class PersonalModelTrainResponse(BaseModel):
    status: str = Field(..., description="Status of the personal model training process.")
    message: str = Field(..., description="Details about the training result.")
    athlete_id: int = Field(..., description="ID of the athlete whose model is being trained.")
    model_version: Optional[str] = Field(None, description="Version or identifier of the newly trained model.")
    model_artifact_s3_path: Optional[str] = Field(None, description="S3 path to the stored model artifact.")

# Minimal UserProfileResponse for fetching anthropometrics
class UserProfileResponseMinimal(BaseModel):
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    # Add other anthropometric data as needed for model training
    class Config:
        from_attributes = True

# Minimal SessionResponse for fetching historical data
class SessionResponseMinimal(BaseModel):
    id: int
    raw_video_s3_key: Optional[str]
    session_type: SessionTypeEnum
    created_at: datetime
    class Config:
        from_attributes = True

# Minimal AttemptResponse for fetching historical data
class AttemptResponseMinimal(BaseModel):
    id: int
    session_id: int
    attempt_number: int
    start_time_ms: Optional[int]
    end_time_ms: Optional[int]
    bar_height_cm: Optional[float]
    outcome: AttemptOutcomeEnum
    bvh_file_s3_key: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True
