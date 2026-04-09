from pydantic import BaseModel, Field
from typing import Dict, Any

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    service: str = "pinn_gnn_inference_service"

class BiomechanicsAnalysisRequest(BaseModel):
    attempt_id: int = Field(..., description="ID of the jump attempt to analyze.")
    user_id: int = Field(..., description="ID of the user who owns the attempt.")

class BiomechanicsAnalysisResponse(BaseModel):
    attempt_id: int
    status: str
    biomechanical_parameters_s3_key: str = Field(..., description="S3 key where the detailed biomechanical parameters are stored.")
    biomechanical_parameters: Dict[str, Any] = Field(..., description="A sample of the biomechanical parameters (full data in S3).")
