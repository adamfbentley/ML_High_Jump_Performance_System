import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Redefined GenderEnum for internal consistency and decoupling
class GenderEnum(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    UNSPECIFIED = "UNSPECIFIED"

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "population_model_service"

class UserProfileAnthropometrics(BaseModel):
    # Subset of UserProfileResponse from User & Profile Service, relevant for cohort matching
    date_of_birth: Optional[datetime] = None
    gender: Optional[GenderEnum] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    primary_sport: Optional[str] = None

class CohortMatchRequestInternal(BaseModel):
    user_id: int = Field(..., description="ID of the user for whom to retrieve cohort data.")
    target_bar_height_cm: Optional[float] = Field(None, description="Optional bar height for context-specific cohort matching.")

class CohortDataPoint(BaseModel):
    metric_name: str = Field(..., description="Name of the biomechanical metric.")
    user_value: Optional[float] = Field(None, description="Value of the metric for the current user.")
    cohort_average: Optional[float] = Field(None, description="Average value of the metric in the matched cohort.")
    cohort_std_dev: Optional[float] = Field(None, description="Standard deviation of the metric in the matched cohort.")
    percentile: Optional[float] = Field(None, description="User's percentile within the matched cohort for this metric.")

class CohortMatchResponse(BaseModel):
    matched_cohort_description: str = Field(..., description="Description of the population cohort matched to the user.")
    comparison_data: List[CohortDataPoint] = Field(..., description="List of biomechanical metrics comparing user to cohort.")
