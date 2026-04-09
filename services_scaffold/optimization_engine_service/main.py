from fastapi import FastAPI, Depends, HTTPException, status, Query
import httpx
import logging
from typing import List

from optimization_engine_service import schemas
from optimization_engine_service.config import USER_PROFILE_SERVICE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optimization Engine Service",
    description="Provides capabilities for personal optimal movement signature, sensitivity analysis, and height-specific targets.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

async def _fetch_user_profile_anthropometrics(client: httpx.AsyncClient, user_id: int) -> schemas.UserProfileAnthropometrics:
    try:
        profile_response = await client.get(f"{USER_PROFILE_SERVICE_URL}/profiles/{user_id}")
        profile_response.raise_for_status()
        profile_data = profile_response.json()
        return schemas.UserProfileAnthropometrics(height_cm=profile_data.get('height_cm'), weight_kg=profile_data.get('weight_kg'))
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch user profile for user {user_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching user profile for user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user profile service: {e}")

@app.post("/internal/optimize/{athlete_id}", response_model=schemas.OptimizationResponse, status_code=status.HTTP_200_OK)
async def run_personal_optimum_solver(
    athlete_id: int,
    request: schemas.OptimizationRequest,
):
    """
    Runs the personal optimum solver for an athlete to generate height-specific technique targets.
    """
    logger.info(f"Received optimization request for athlete_id: {athlete_id} with target_bar_height_cm: {request.target_bar_height_cm}")

    async with httpx.AsyncClient() as client:
        user_profile = await _fetch_user_profile_anthropometrics(client, athlete_id)

        logger.info(f"Simulating fetching personal model parameters for athlete {athlete_id}.")
        base_takeoff_angle = 45.0 # degrees
        base_approach_speed = 8.0 # m/s
        base_plant_foot_position = 100.0 # cm relative to bar

        logger.info(f"Simulating personal optimum solver for athlete {athlete_id} and target height {request.target_bar_height_cm}cm.")

        # Example: Adjust parameters based on target height and athlete's height
        height_factor = 1.0
        if user_profile.height_cm:
            height_factor = user_profile.height_cm / 180.0 # Normalize by average height

        # Simple linear scaling for demonstration
        optimal_takeoff_angle = base_takeoff_angle + (request.target_bar_height_cm - 200) * 0.1 * height_factor
        optimal_approach_speed = base_approach_speed + (request.target_bar_height_cm - 200) * 0.01 * height_factor
        optimal_plant_foot_position = base_plant_foot_position + (request.target_bar_height_cm - 200) * 0.5 * height_factor

        optimal_params = schemas.OptimalTechniqueParameters(
            takeoff_angle_degrees=round(max(30.0, min(60.0, optimal_takeoff_angle)), 2),
            approach_speed_mps=round(max(6.0, min(10.0, optimal_approach_speed)), 2),
            plant_foot_position_cm=round(max(50.0, min(150.0, optimal_plant_foot_position)), 2)
        )

        # Simulate predicted performance (e.g., target height + a small buffer)
        predicted_performance = request.target_bar_height_cm + 5.0 # cm

        message = f"Optimal technique parameters generated for {request.target_bar_height_cm}cm."

        return schemas.OptimizationResponse(
            athlete_id=athlete_id,
            target_bar_height_cm=request.target_bar_height_cm,
            optimal_parameters=optimal_params,
            predicted_performance_cm=predicted_performance,
            message=message
        )

@app.get("/internal/optimize/{athlete_id}/sensitivity", response_model=schemas.SensitivityAnalysisResponse, status_code=status.HTTP_200_OK)
async def get_sensitivity_analysis_results(
    athlete_id: int,
    bar_height_cm: float = Query(..., gt=0, description="The bar height for which to analyze sensitivity.")
):
    """
    Retrieves sensitivity analysis results for an athlete's technique at a specific bar height.
    """
    logger.info(f"Received sensitivity analysis request for athlete_id: {athlete_id} at bar_height_cm: {bar_height_cm}")

    async with httpx.AsyncClient() as client:
        user_profile = await _fetch_user_profile_anthropometrics(client, athlete_id)

        logger.info(f"Simulating fetching personal model parameters for athlete {athlete_id} for sensitivity analysis.")

        logger.info(f"Simulating sensitivity analysis for athlete {athlete_id} at {bar_height_cm}cm.")

        # Example: Generate dummy sensitivity results based on bar height and athlete's weight
        sensitivity_factor = 1.0
        if user_profile.weight_kg:
            sensitivity_factor = user_profile.weight_kg / 70.0 # Normalize by average weight

        results = [
            schemas.SensitivityAnalysisResult(
                parameter="takeoff_angle_degrees",
                impact_score=round(0.8 * sensitivity_factor, 2),
                direction="positive" if bar_height_cm > 200 else "neutral"
            ),
            schemas.SensitivityAnalysisResult(
                parameter="approach_speed_mps",
                impact_score=round(0.6 * sensitivity_factor, 2),
                direction="positive"
            ),
            schemas.SensitivityAnalysisResult(
                parameter="plant_foot_position_cm",
                impact_score=round(0.4 * sensitivity_factor, 2),
                direction="negative"
            ),
        ]

        message = f"Sensitivity analysis complete for {bar_height_cm}cm."

        return schemas.SensitivityAnalysisResponse(
            athlete_id=athlete_id,
            bar_height_cm=bar_height_cm,
            results=results,
            message=message
        )
