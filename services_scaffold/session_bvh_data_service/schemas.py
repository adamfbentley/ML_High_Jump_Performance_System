from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from session_bvh_data_service.models import SessionTypeEnum, AttemptOutcomeEnum

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "session_bvh_data_service"

class SessionBase(BaseModel):
    raw_video_s3_key: Optional[str] = Field(None, description="S3 key of the raw video associated with this session. Optional for live sessions.") # Changed to Optional
    session_date: Optional[datetime] = None
    session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING
    notes: Optional[str] = None

class SessionCreate(SessionBase):
    user_id: int = Field(..., description="ID of the user who owns this session.")

class SessionResponseWithRawVideoS3Key(SessionBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class SessionResponse(SessionResponseWithRawVideoS3Key):
    # This schema is used for external API responses where raw_video_s3_key might be omitted or handled differently
    # For internal use, SessionResponseWithRawVideoS3Key is more explicit.
    pass

class AttemptBase(BaseModel):
    attempt_number: int = Field(..., description="The sequence number of the attempt within the session.")
    start_time_ms: Optional[int] = Field(None, description="Start time of the jump segment in milliseconds.")
    end_time_ms: Optional[int] = Field(None, description="End time of the jump segment in milliseconds.")
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = AttemptOutcomeEnum.UNKNOWN
    bvh_file_s3_key: Optional[str] = Field(None, description="S3 key for the processed BVH file.")

class AttemptCreate(AttemptBase):
    session_id: int = Field(..., description="ID of the session this attempt belongs to.")

class AttemptCreateLive(BaseModel): # New schema for external API to create attempts
    attempt_number: int = Field(..., description="The sequence number of the attempt within the session.")
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = AttemptOutcomeEnum.UNKNOWN

class AttemptUpdateMetadata(BaseModel):
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = None
    notes: Optional[str] = None # Adding notes for attempt metadata

class AttemptUpdateSegmentation(BaseModel):
    start_time_ms: int = Field(..., description="Start time of the jump segment in milliseconds.")
    end_time_ms: int = Field(..., description="End time of the jump segment in milliseconds.")

class AttemptResponse(AttemptBase):
    id: int
    session_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class SessionWithAttemptsResponse(SessionResponseWithRawVideoS3Key):
    attempts: List[AttemptResponse]

    class Config:
        from_attributes = True

class AttemptUpdateBvhKey(BaseModel):
    bvh_file_s3_key: str = Field(..., description="S3 key for the processed BVH file.")
