from fastapi import FastAPI, Depends, HTTPException, status
import httpx
import json
import random
from datetime import datetime, timezone
import logging

from anomaly_detection_service import schemas, config, s3_utils
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anomaly Detection Service",
    description="Establishes rolling baselines, calculates per-node deviations, performs fault localization, and detects fatigue patterns or adapts to injury states.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

async def get_user_profile_service_client():
    async with httpx.AsyncClient(base_url=config.USER_PROFILE_SERVICE_URL) as client:
        yield client

async def get_session_bvh_data_service_client():
    async with httpx.AsyncClient(base_url=config.SESSION_BVH_DATA_SERVICE_URL) as client:
        yield client

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/detect/anomalies", response_model=schemas.AnomalyDetectionResponse)
async def detect_anomalies_endpoint(
    request: schemas.AnomalyDetectionRequest,
    user_profile_client: httpx.AsyncClient = Depends(get_user_profile_service_client),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    try:
        # 1. Fetch user profile for injury status
        profile_response = await user_profile_client.get(f"/profiles/{request.user_id}")
        profile_response.raise_for_status()
        user_profile = schemas.UserProfileInternal(**profile_response.json())
        logger.info(f"Fetched user profile for user {request.user_id}. Injury status: {user_profile.injury_status}")

        injury_adaptation_status = "No known injuries."
        if user_profile.injury_status:
            injury_adaptation_status = f"Analysis adapted for: {user_profile.injury_status}"
            if user_profile.injury_date: 
                injury_adaptation_status += f" (since {user_profile.injury_date.strftime('%Y-%m-%d')})"

        # 2. Get biomechanical parameters S3 key from Session & BVH Data Service
        params_key_response = await session_bvh_client.get(f"/attempts/{request.attempt_id}/parameters?user_id={request.user_id}")
        params_key_response.raise_for_status()
        biomech_params_data = schemas.BiomechanicalParametersInternal(**params_key_response.json())
        biomechanical_parameters_s3_key = biomech_params_data.s3_key
        logger.info(f"Fetched biomechanical parameters S3 key for attempt {request.attempt_id}: {biomechanical_parameters_s3_key}")

        # Extract object name from S3 key
        params_object_name = biomechanical_parameters_s3_key.split(f"s3://{config.S3_BUCKET_NAME}/", 1)[-1]

        # 3. Download biomechanical parameters from S3
        biomech_params_content = await s3_utils.download_file_from_s3(params_object_name)
        biomech_params = json.loads(biomech_params_content.decode('utf-8'))
        logger.info(f"Downloaded and parsed biomechanical parameters for attempt {request.attempt_id}")

        # 4. Simulate Anomaly Detection Logic (CQ-004 Fix: Enhanced simulation using actual biomechanical parameters)
        # This is a placeholder for actual anomaly detection algorithms.
        # It would involve comparing current attempt parameters against baselines (personal bests, population models),
        # calculating deviations, identifying patterns of fatigue, and adjusting for injury status.
        
        anomalies_detected = False
        overall_anomaly_score = 0.0
        fatigue_score = 0.0
        fault_localization: List[schemas.FaultLocalization] = []

        # Example: Make anomaly detection somewhat dependent on biomech_params
        peak_velocity = biomech_params.get("peak_velocity_m_s", 0.0)
        landing_impact = biomech_params.get("landing_impact_g", 0.0)
        hip_flexion_avg = sum(biomech_params.get("joint_angles", {}).get("hip_flexion", [0])) / len(biomech_params.get("joint_angles", {}).get("hip_flexion", [1])) if biomech_params.get("joint_angles", {}).get("hip_flexion") else 0

        # Simple rules for simulated anomalies
        if peak_velocity < 7.0: # Arbitrary threshold
            anomalies_detected = True
            overall_anomaly_score += 30
            fault_localization.append(schemas.FaultLocalization(
                joint="overall_performance",
                deviation_score=random.uniform(30, 50),
                deviation_type="low_peak_velocity",
                recommendation="Focus on explosive power training."
            ))
        if landing_impact > 4.0: # Arbitrary threshold
            anomalies_detected = True
            overall_anomaly_score += 40
            fault_localization.append(schemas.FaultLocalization(
                joint="landing",
                deviation_score=random.uniform(40, 60),
                deviation_type="high_landing_impact",
                recommendation="Improve landing mechanics and shock absorption."
            ))
        if hip_flexion_avg > 0.4: # Arbitrary threshold for excessive flexion
            anomalies_detected = True
            overall_anomaly_score += 25
            fault_localization.append(schemas.FaultLocalization(
                joint="hip",
                deviation_score=random.uniform(25, 45),
                deviation_type="excessive_hip_flexion",
                recommendation="Strengthen hip extensors and glutes."
            ))

        # Simulate fatigue based on some parameter (e.g., if velocity is consistently lower over time, not implemented here)
        # For now, make fatigue score somewhat random but influenced by anomalies
        fatigue_score = random.uniform(0, 30) # Baseline low fatigue
        if anomalies_detected:
            fatigue_score += random.uniform(20, 70) # Higher fatigue if anomalies
        fatigue_score = min(fatigue_score, 100.0)

        # Adjust overall score based on number of faults
        if len(fault_localization) > 0:
            overall_anomaly_score = min(overall_anomaly_score + (len(fault_localization) * 15), 100.0)
            anomalies_detected = True
        else:
            overall_anomaly_score = min(overall_anomaly_score, 20.0) # Low score if no specific faults

        # If there's an injury, potentially flag related areas more often or adjust scores
        if user_profile.injury_status and "knee" in user_profile.injury_status.lower():
            # Example: Add a simulated knee fault if injury exists
            if random.random() < 0.5: # 50% chance to flag a knee issue if injured
                anomalies_detected = True
                overall_anomaly_score = min(overall_anomaly_score + 20, 100.0)
                fault_localization.append(schemas.FaultLocalization(
                    joint="left_knee" if random.random() < 0.5 else "right_knee",
                    deviation_score=random.uniform(60, 90),
                    deviation_type="injury_related_compensation",
                    recommendation="Consult with a physical therapist for injury management."
                ))
        
        overall_anomaly_score = round(overall_anomaly_score, 2)
        fatigue_score = round(fatigue_score, 2)

        logger.info(f"Anomaly detection simulated for attempt {request.attempt_id}. Anomalies: {anomalies_detected}, Score: {overall_anomaly_score}")

        return schemas.AnomalyDetectionResponse(
            attempt_id=request.attempt_id,
            user_id=request.user_id,
            anomalies_detected=anomalies_detected,
            overall_anomaly_score=overall_anomaly_score,
            fault_localization=fault_localization,
            fatigue_score=fatigue_score,
            injury_adaptation_status=injury_adaptation_status,
            analysis_timestamp=datetime.now(timezone.utc)
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from upstream service: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from upstream service: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error connecting to upstream service: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to upstream service: {e}")
    except ClientError as e:
        logger.error(f"S3 client error during anomaly detection: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 client error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding biomechanical parameters JSON from S3 for attempt {request.attempt_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error decoding biomechanical parameters JSON from S3: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during anomaly detection for attempt {request.attempt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during anomaly detection: {e}")
