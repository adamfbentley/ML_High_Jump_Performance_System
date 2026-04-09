from pydantic import BaseModel, Field
from typing import Optional, List

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "optimization_engine_service"

class UserProfileAnthropometrics(BaseModel):
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

class OptimalTechniqueParameters(BaseModel):
    takeoff_angle_degrees: float = Field(..., description="Optimal takeoff angle in degrees.")
    approach_speed_mps: float = Field(..., description="Optimal approach speed in meters per second.")
    plant_foot_position_cm: float = Field(..., description="Optimal plant foot position relative to bar in centimeters.")

class OptimizationRequest(BaseModel):
    target_bar_height_cm: float = Field(..., gt=0, description="The target bar height in centimeters for which to optimize technique.")

class OptimizationResponse(BaseModel):
    athlete_id: int = Field(..., description="ID of the athlete for whom optimization was run.")
    target_bar_height_cm: float = Field(..., description="The target bar height in centimeters.")
    optimal_parameters: OptimalTechniqueParameters = Field(..., description="The set of optimal technique parameters.")
    predicted_performance_cm: float = Field(..., description="Predicted maximum jump height in centimeters with optimal technique.")
    message: str = Field(..., description="A descriptive message about the optimization result.")

class SensitivityAnalysisRequest(BaseModel):
    bar_height_cm: float = Field(..., gt=0, description="The bar height in centimeters for which to analyze sensitivity.")

class SensitivityAnalysisResult(BaseModel):
    parameter: str = Field(..., description="Name of the technique parameter.")
    impact_score: float = Field(..., description="A score indicating the sensitivity of performance to this parameter (e.g., 0-1).")
    direction: str = Field(..., description="Indicates if increasing/decreasing parameter improves performance (e.g., 'positive', 'negative', 'neutral').")

class SensitivityAnalysisResponse(BaseModel):
    athlete_id: int = Field(..., description="ID of the athlete for whom sensitivity analysis was run.")
    bar_height_cm: float = Field(..., description="The bar height for which sensitivity was analyzed.")
    results: List[SensitivityAnalysisResult] = Field(..., description="List of sensitivity analysis results for various technique parameters.")
    message: str = Field(..., description="A descriptive message about the sensitivity analysis.")
