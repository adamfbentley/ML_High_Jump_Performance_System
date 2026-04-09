from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "anomaly_detection_service"

class AnomalyDetectionRequest(BaseModel):
    attempt_id: int = Field(..., description="ID of the jump attempt to analyze.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")

class FaultLocalization(BaseModel):
    joint: str = Field(..., description="Joint where a fault is detected (e.g., 'left_knee', 'right_ankle').")
    deviation_score: float = Field(..., description="Magnitude of deviation from baseline (0-100).")
    deviation_type: str = Field(..., description="Type of deviation (e.g., 'excessive_flexion', 'insufficient_extension').")
    recommendation: str = Field(..., description="Actionable recommendation for the detected fault.")

class AnomalyDetectionResponse(BaseModel):
    attempt_id: int
    user_id: int
    anomalies_detected: bool = Field(..., description="True if any anomalies were detected.")
    overall_anomaly_score: float = Field(..., description="An aggregate score indicating the severity of anomalies (0-100).")
    fault_localization: List[FaultLocalization] = Field(..., description="Details of specific faults and their locations.")
    fatigue_score: float = Field(..., description="Indicates potential fatigue based on performance deviations (0-100).")
    injury_adaptation_status: str = Field(..., description="Describes how analysis adapted to known injuries (e.g., 'Adapted for knee injury', 'No known injuries').")
    analysis_timestamp: datetime

# Internal schemas for inter-service communication
class UserProfileInternal(BaseModel):
    user_id: int
    injury_status: Optional[str] = None
    injury_date: Optional[datetime] = None
    recovery_date: Optional[datetime] = None

    class Config:
        from_attributes = True

class BiomechanicalParametersInternal(BaseModel):
    s3_key: str
    parameters: Dict[str, Any]
