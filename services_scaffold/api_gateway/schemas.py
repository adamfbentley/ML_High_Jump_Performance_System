from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

# Redefined GenderEnum to remove tight coupling with user_profile_service
class GenderEnum(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    UNSPECIFIED = "UNSPECIFIED"

# NEW ENUMS FOR SESSION & ATTEMPT
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

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    class Config:
        from_attributes = True

class UserProfileBase(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[GenderEnum] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    primary_sport: Optional[str] = None

class UserProfileUpdate(UserProfileBase):
    pass

class UserProfileResponse(UserProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# SCHEMAS FOR VIDEO INGESTION (from Sprint 2)
class VideoUploadRequest(BaseModel):
    file_name: str = Field(..., description="Original name of the video file.")
    content_type: str = Field(..., description="MIME type of the video file, e.g., 'video/mp4'.")
    session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING

class VideoUploadResponse(BaseModel):
    video_id: str = Field(..., description="Unique ID for the uploaded video.")
    upload_url: str = Field(..., description="Pre-signed URL for direct S3 upload.")
    form_fields: Dict[str, str] = Field(..., description="Form fields required for S3 POST upload.")

# NEW SCHEMAS FOR SESSION & ATTEMPT
class SessionCreate(BaseModel):
    raw_video_s3_key: Optional[str] = Field(None, description="S3 key of the raw video associated with this session. Optional for live sessions.") # Changed to Optional
    session_date: Optional[datetime] = None
    session_type: Optional[SessionTypeEnum] = SessionTypeEnum.TRAINING
    notes: Optional[str] = None

class SessionResponse(BaseModel):
    id: int
    user_id: int
    raw_video_s3_key: Optional[str] # Changed to Optional
    session_date: datetime
    session_type: SessionTypeEnum
    notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AttemptCreate(BaseModel):
    session_id: int = Field(..., description="ID of the session this attempt belongs to.")
    attempt_number: int = Field(..., description="The sequence number of the attempt within the session.")
    start_time_ms: Optional[int] = Field(None, description="Start time of the jump segment in milliseconds.")
    end_time_ms: Optional[int] = Field(None, description="End time of the jump segment in milliseconds.")
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = AttemptOutcomeEnum.UNKNOWN

class AttemptCreateLiveRequest(BaseModel): # New schema for external API to create attempts
    attempt_number: int = Field(..., description="The sequence number of the attempt within the session.")
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = AttemptOutcomeEnum.UNKNOWN

class AttemptUpdateMetadata(BaseModel):
    bar_height_cm: Optional[float] = Field(None, description="Bar height for this attempt in centimeters.")
    outcome: Optional[AttemptOutcomeEnum] = None
    notes: Optional[str] = None

class AttemptResponse(BaseModel):
    id: int
    session_id: int
    attempt_number: int
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    bar_height_cm: Optional[float] = None
    outcome: AttemptOutcomeEnum
    bvh_file_s3_key: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class SessionWithAttemptsResponse(SessionResponse):
    attempts: List[AttemptResponse] = Field(..., description="List of jump attempts associated with this session.")

    class Config:
        from_attributes = True

# NEW SCHEMAS FOR FEEDBACK & REPORTING SERVICE (BE-11)
class SessionSummaryResponse(BaseModel):
    session_id: int
    session_date: datetime
    session_type: SessionTypeEnum
    total_attempts: int
    successful_attempts: int
    average_bar_height_cm: Optional[float] = None
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class AttemptFeedbackResponse(BaseModel):
    attempt_id: int
    bar_height_cm: Optional[float] = None
    outcome: AttemptOutcomeEnum
    feedback_score: Optional[float] = Field(None, description="Overall score for the attempt, e.g., 1-10.")
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    actionable_cues: List[str] = Field(default_factory=list)
    drill_recommendations: List[str] = Field(default_factory=list)

class AttemptVisualsResponse(BaseModel):
    attempt_id: int
    bvh_s3_key: Optional[str] = Field(None, description="S3 key for the processed BVH file for 3D visualization.")
    video_s3_key: Optional[str] = Field(None, description="S3 key for the original video segment.")
    overlay_image_urls: List[str] = Field(default_factory=list, description="URLs to images with biomechanical overlays.")
    comparison_data_urls: List[str] = Field(default_factory=list, description="URLs to comparison graphs/data.")

class ProgressDashboardResponse(BaseModel):
    athlete_id: int
    total_sessions: int
    total_attempts: int
    personal_best_height_cm: Optional[float] = None
    progress_chart_data: Dict[str, Any] = Field(default_factory=dict, description="Data for a time-series progress chart.")
    recent_sessions_summary: List[SessionSummaryResponse] = Field(default_factory=list)

class CoachReportResponse(BaseModel):
    coach_id: int
    athlete_id: int
    report_date: datetime
    athlete_summary: str
    performance_trends: Dict[str, Any] = Field(default_factory=dict)
    custom_notes: Optional[str] = None
    recommended_drills_for_athlete: List[str] = Field(default_factory=list)
