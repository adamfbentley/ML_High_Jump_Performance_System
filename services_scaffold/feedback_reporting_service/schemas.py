from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "feedback_reporting_service"

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

class SessionResponseWithRawVideoS3Key(BaseModel):
    id: int
    user_id: int
    raw_video_s3_key: Optional[str]
    session_date: datetime
    session_type: SessionTypeEnum
    notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

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
