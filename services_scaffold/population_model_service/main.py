from fastapi import FastAPI, Depends, HTTPException, status, Query
import httpx
import logging
from datetime import datetime, timezone
from typing import List, Optional

from population_model_service import schemas
from population_model_service.config import USER_PROFILE_SERVICE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Population Model Service",
    description="Provides population model cohort matching and comparison data.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

# Base values for a generic athlete
BASE_METRICS = {
    "Max Vertical Jump (cm)": {"base_value": 60.0, "std_dev": 10.0},
    "Approach Speed (m/s)": {"base_value": 6.5, "std_dev": 0.7},
    "Take-off Angle (degrees)": {"base_value": 20.0, "std_dev": 2.0},
    "Bar Clearance Rate": {"base_value": 0.6, "std_dev": 0.1}, # For a generic bar height
}

# Adjustments based on cohort characteristics (can be positive or negative)
ADJUSTMENTS = {
    "gender": {
        schemas.GenderEnum.MALE: {"Max Vertical Jump (cm)": 10.0, "Approach Speed (m/s)": 0.5, "Take-off Angle (degrees)": 1.0, "Bar Clearance Rate": 0.05},
        schemas.GenderEnum.FEMALE: {"Max Vertical Jump (cm)": 0.0, "Approach Speed (m/s)": 0.0, "Take-off Angle (degrees)": 0.0, "Bar Clearance Rate": 0.0},
        schemas.GenderEnum.OTHER: {"Max Vertical Jump (cm)": 5.0, "Approach Speed (m/s)": 0.2, "Take-off Angle (degrees)": 0.5, "Bar Clearance Rate": 0.02},
        schemas.GenderEnum.UNSPECIFIED: {"Max Vertical Jump (cm)": 0.0, "Approach Speed (m/s)": 0.0, "Take-off Angle (degrees)": 0.0, "Bar Clearance Rate": 0.0},
    },
    "age_group": {
        "Junior": {"Max Vertical Jump (cm)": -5.0, "Approach Speed (m/s)": -0.5, "Take-off Angle (degrees)": -1.0, "Bar Clearance Rate": -0.05},
        "Young Adult": {"Max Vertical Jump (cm)": 5.0, "Approach Speed (m/s)": 0.5, "Take-off Angle (degrees)": 1.0, "Bar Clearance Rate": 0.05},
        "Adult": {"Max Vertical Jump (cm)": 0.0, "Approach Speed (m/s)": 0.0, "Take-off Angle (degrees)": 0.0, "Bar Clearance Rate": 0.0},
        "Master": {"Max Vertical Jump (cm)": -10.0, "Approach Speed (m/s)": -1.0, "Take-off Angle (degrees)": -2.0, "Bar Clearance Rate": -0.1},
    },
    "height_group": {
        "Taller": {"Max Vertical Jump (cm)": 5.0, "Approach Speed (m/s)": 0.3, "Take-off Angle (degrees)": 0.5, "Bar Clearance Rate": 0.03},
        "Shorter": {"Max Vertical Jump (cm)": -5.0, "Approach Speed (m/s)": -0.3, "Take-off Angle (degrees)": -0.5, "Bar Clearance Rate": -0.03},
        "Average Height": {"Max Vertical Jump (cm)": 0.0, "Approach Speed (m/s)": 0.0, "Take-off Angle (degrees)": 0.0, "Bar Clearance Rate": 0.0},
    },
    "weight_group": {
        "Heavier": {"Max Vertical Jump (cm)": -2.0, "Approach Speed (m/s)": -0.1, "Take-off Angle (degrees)": 0.2, "Bar Clearance Rate": -0.01},
        "Lighter": {"Max Vertical Jump (cm)": 2.0, "Approach Speed (m/s)": 0.1, "Take-off Angle (degrees)": -0.2, "Bar Clearance Rate": 0.01},
        "Average Weight": {"Max Vertical Jump (cm)": 0.0, "Approach Speed (m/s)": 0.0, "Take-off Angle (degrees)": 0.0, "Bar Clearance Rate": 0.0},
    }
}

def calculate_base_metric_for_characteristics(metric_key: str, gender: Optional[schemas.GenderEnum], age_group: Optional[str], height_group: Optional[str], weight_group: Optional[str]):
    """Calculates a metric's base value based on a set of characteristics, without bar height specific adjustments."""
    base_value = BASE_METRICS[metric_key]["base_value"]
    value = base_value

    if gender and gender in ADJUSTMENTS["gender"]:
        value += ADJUSTMENTS["gender"][gender].get(metric_key, 0.0)
    if age_group and age_group in ADJUSTMENTS["age_group"]:
        value += ADJUSTMENTS["age_group"][age_group].get(metric_key, 0.0)
    if height_group and height_group in ADJUSTMENTS["height_group"]:
        value += ADJUSTMENTS["height_group"][height_group].get(metric_key, 0.0)
    if weight_group and weight_group in ADJUSTMENTS["weight_group"]:
        value += ADJUSTMENTS["weight_group"][weight_group].get(metric_key, 0.0)
    return value

def calculate_bar_clearance_rate(base_clearance_rate: float, estimated_max_jump: float, target_bar_height_cm: float) -> float:
    """Calculates Bar Clearance Rate with specific logic for target bar height."""
    value = base_clearance_rate
    if target_bar_height_cm is not None:
        # Simple linear drop-off: for every 10cm above estimated max jump, reduce clearance rate by 0.02
        height_diff = target_bar_height_cm - estimated_max_jump
        value -= (height_diff / 10.0) * 0.02
        value = max(0.0, min(1.0, value)) # Clamp clearance rate between 0 and 1
    return value

def calculate_percentile(user_value: float, cohort_average: float, cohort_std_dev: float) -> float:
    """Calculates a percentile based on user value, cohort average, and standard deviation."""
    if cohort_std_dev == 0:
        return 50.0 # If no variation, everyone is average

    # Using a simplified approximation for percentile based on Z-score
    # This maps values within +/- 2 standard deviations to roughly 2.2% to 97.8%
    z_score = (user_value - cohort_average) / cohort_std_dev
    percentile = 50 + (z_score / 2.0) * 47.8 # Linear approximation
    return min(99.9, max(0.1, percentile))

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.get("/internal/models/population/cohorts", response_model=schemas.CohortMatchResponse)
async def get_population_cohort_data(
    user_id: int = Query(..., description="ID of the user for whom to retrieve cohort data."),
    target_bar_height_cm: Optional[float] = Query(None, description="Optional bar height for context-specific cohort matching.")
):
    logger.info(f"Received request for population cohort data for user_id: {user_id}, target_bar_height_cm: {target_bar_height_cm}")

    async with httpx.AsyncClient() as client:
        # 1. Fetch user profile for anthropometric data (BE-02)
        try:
            profile_response = await client.get(f"{USER_PROFILE_SERVICE_URL}/profiles/{user_id}")
            profile_response.raise_for_status()
            user_profile_data = profile_response.json()
            user_profile = schemas.UserProfileAnthropometrics(**user_profile_data)
            logger.info(f"Fetched user profile for user {user_id}: {user_profile.dict()}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch user profile for user {user_id}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error fetching user profile for user {user_id}: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user profile service: {e}")

        # 2. Determine user's characteristics for cohort matching and individual metric calculation
        user_gender = user_profile.gender
        user_age_group = None
        if user_profile.date_of_birth:
            today = datetime.now(timezone.utc)
            age = today.year - user_profile.date_of_birth.year - ( (today.month, today.day) < (user_profile.date_of_birth.month, user_profile.date_of_birth.day) )
            if age is not None:
                if age < 18: user_age_group = "Junior"
                elif 18 <= age <= 25: user_age_group = "Young Adult"
                elif 26 <= age <= 35: user_age_group = "Adult"
                else: user_age_group = "Master"

        user_height_group = None
        if user_profile.height_cm:
            if user_profile.height_cm > 185: user_height_group = "Taller"
            elif user_profile.height_cm < 170: user_height_group = "Shorter"
            else: user_height_group = "Average Height"

        user_weight_group = None
        if user_profile.weight_kg:
            if user_profile.weight_kg > 80: user_weight_group = "Heavier"
            elif user_profile.weight_kg < 60: user_weight_group = "Lighter"
            else: user_weight_group = "Average Weight"

        # Construct matched cohort description
        matched_cohort_description_parts = []
        if user_gender: matched_cohort_description_parts.append(f"{user_gender.value.capitalize()} Athletes")
        if user_age_group: matched_cohort_description_parts.append(user_age_group)
        if user_height_group: matched_cohort_description_parts.append(user_height_group)
        if user_weight_group: matched_cohort_description_parts.append(user_weight_group)

        matched_cohort_description = ", ".join(matched_cohort_description_parts) if matched_cohort_description_parts else "General Population"

        # 3. Generate comparison data based on the simulated model
        comparison_data: List[schemas.CohortDataPoint] = []

        metrics_to_calculate = [
            "Max Vertical Jump (cm)",
            "Approach Speed (m/s)",
            "Take-off Angle (degrees)",
        ]
        if target_bar_height_cm:
            metrics_to_calculate.append(f"Bar Clearance Rate @ {target_bar_height_cm}cm")

        for metric_name_template in metrics_to_calculate:
            actual_metric_key = "Bar Clearance Rate" if metric_name_template.startswith("Bar Clearance Rate") else metric_name_template
            cohort_std_dev = BASE_METRICS[actual_metric_key]["std_dev"]

            # Calculate the 'true' average for the matched cohort based on its defining characteristics
            cohort_average_base = calculate_base_metric_for_characteristics(
                actual_metric_key, user_gender, user_age_group, user_height_group, user_weight_group
            )

            # For Bar Clearance Rate, the cohort average also needs the estimated max jump for the cohort
            if actual_metric_key == "Bar Clearance Rate":
                cohort_estimated_max_jump = calculate_base_metric_for_characteristics(
                    "Max Vertical Jump (cm)", user_gender, user_age_group, user_height_group, user_weight_group
                )
                cohort_average = calculate_bar_clearance_rate(cohort_average_base, cohort_estimated_max_jump, target_bar_height_cm)
            else:
                cohort_average = cohort_average_base

            # Simulate user's value as slightly deviating from the cohort average
            # For a real model, user_value would come from actual performance data.
            # Here, we simulate it by adding a small, consistent offset to the cohort average.
            # This makes the comparison meaningful (user is not exactly average).
            user_value_offset = cohort_std_dev * 0.2 # User is 0.2 std dev above average, for example
            user_value = cohort_average + user_value_offset

            # Ensure values are non-negative where appropriate
            user_value = max(0.0, user_value)
            cohort_average = max(0.0, cohort_average)

            percentile = calculate_percentile(user_value, cohort_average, cohort_std_dev)

            comparison_data.append(schemas.CohortDataPoint(
                metric_name=metric_name_template,
                user_value=round(user_value, 1 if actual_metric_key != "Bar Clearance Rate" else 2),
                cohort_average=round(cohort_average, 1 if actual_metric_key != "Bar Clearance Rate" else 2),
                cohort_std_dev=round(cohort_std_dev, 1 if actual_metric_key != "Bar Clearance Rate" else 2),
                percentile=round(percentile, 1)
            ))

        return schemas.CohortMatchResponse(
            matched_cohort_description=matched_cohort_description,
            comparison_data=comparison_data
        )
