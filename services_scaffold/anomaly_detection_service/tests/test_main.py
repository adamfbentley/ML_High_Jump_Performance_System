from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import pytest
import json
from datetime import datetime, timezone, timedelta

# Assuming conftest.py sets up mock_user_profile_client, mock_session_bvh_client, and mock_s3_utils

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "anomaly_detection_service"}

@pytest.mark.asyncio
async def test_detect_anomalies_success_no_injury(
    client: TestClient,
    mock_user_profile_client: AsyncMock,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 123
    user_id = 1
    params_s3_key = "s3://highjump-biomechanical-data/biomechanical_parameters/user1/attempt123_uuid.json"
    biomech_params_content = json.dumps({
        "joint_angles": {"hip_flexion": [0.1, 0.2, 0.3], "knee_flexion": [0.5, 0.4, 0.3]},
        "peak_velocity_m_s": 8.5,
        "landing_impact_g": 3.2,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat()
    }).encode('utf-8')

    # Mock User Profile Service
    mock_user_profile_client.get.return_value.json.return_value = {
        "user_id": user_id,
        "injury_status": None,
        "injury_date": None,
        "recovery_date": None
    }
    mock_user_profile_client.get.return_value.raise_for_status.return_value = None

    # Mock Session & BVH Data Service
    mock_session_bvh_client.get.return_value.json.return_value = {
        "s3_key": params_s3_key,
        "parameters": {"message": "Parameters would be fetched from S3 using this key"}
    }
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    # Mock S3 download
    mock_download.return_value = biomech_params_content

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/detect/anomalies", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["attempt_id"] == attempt_id
    assert data["user_id"] == user_id
    assert "anomalies_detected" in data
    assert "overall_anomaly_score" in data
    assert "fault_localization" in data
    assert "fatigue_score" in data
    assert data["injury_adaptation_status"] == "No known injuries."
    assert "analysis_timestamp" in data
    # Verify that the logic used biomech_params (e.g., no low velocity fault for 8.5 m/s)
    assert not any(f['deviation_type'] == 'low_peak_velocity' for f in data['fault_localization'])

    mock_user_profile_client.get.assert_called_with(f"/profiles/{user_id}")
    mock_session_bvh_client.get.assert_called_with(f"/attempts/{attempt_id}/parameters?user_id={user_id}")
    mock_download.assert_called_with(f"biomechanical_parameters/user1/attempt123_uuid.json".split(f"s3://highjump-biomechanical-data/", 1)[-1])

@pytest.mark.asyncio
async def test_detect_anomalies_with_injury(client: TestClient, mock_user_profile_client: AsyncMock, mock_session_bvh_client: AsyncMock, mock_s3_utils):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 124
    user_id = 2
    params_s3_key = "s3://highjump-biomechanical-data/biomechanical_parameters/user2/attempt124_uuid.json"
    biomech_params_content = json.dumps({
        "joint_angles": {"hip_flexion": [0.1, 0.2, 0.3], "knee_flexion": [0.5, 0.4, 0.3]},
        "peak_velocity_m_s": 7.8,
        "landing_impact_g": 3.5,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat()
    }).encode('utf-8')
    injury_date = datetime.now(timezone.utc) - timedelta(days=30)

    # Mock User Profile Service with injury
    mock_user_profile_client.get.return_value.json.return_value = {
        "user_id": user_id,
        "injury_status": "Left knee pain",
        "injury_date": injury_date.isoformat(),
        "recovery_date": None
    }
    mock_user_profile_client.get.return_value.raise_for_status.return_value = None

    # Mock Session & BVH Data Service
    mock_session_bvh_client.get.return_value.json.return_value = {
        "s3_key": params_s3_key,
        "parameters": {"message": "Parameters would be fetched from S3 using this key"}
    }
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    # Mock S3 download
    mock_download.return_value = biomech_params_content

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/detect/anomalies", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["injury_adaptation_status"] == f"Analysis adapted for: Left knee pain (since {injury_date.strftime('%Y-%m-%d')})"
    # Assert that a knee-related fault might be present due to injury simulation
    assert any("knee" in f['joint'] for f in data['fault_localization'])

@pytest.mark.asyncio
async def test_detect_anomalies_user_profile_service_error(
    client: TestClient,
    mock_user_profile_client: AsyncMock,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 125
    user_id = 3

    # Mock User Profile Service returning an error
    mock_user_profile_client.get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=httpx.Response(404, request=httpx.Request("GET", "http://test"), content=b'{"detail":"User not found"}')
    )

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/detect/anomalies", json=request_payload)

    assert response.status_code == 404
    assert "Error from upstream service: {\"detail\":\"User not found\"}" in response.json()["detail"]
    mock_download.assert_not_called()

@pytest.mark.asyncio
async def test_detect_anomalies_session_bvh_service_error(
    client: TestClient,
    mock_user_profile_client: AsyncMock,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 126
    user_id = 4
    injury_date = datetime.now(timezone.utc) - timedelta(days=10)

    # Mock User Profile Service (success)
    mock_user_profile_client.get.return_value.json.return_value = {
        "user_id": user_id,
        "injury_status": "Right shoulder strain",
        "injury_date": injury_date.isoformat(),
        "recovery_date": None
    }
    mock_user_profile_client.get.return_value.raise_for_status.return_value = None

    # Mock Session & BVH Data Service returning an error
    mock_session_bvh_client.get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=httpx.Response(404, request=httpx.Request("GET", "http://test"), content=b'{"detail":"Parameters not found"}')
    )

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/detect/anomalies", json=request_payload)

    assert response.status_code == 404
    assert "Error from upstream service: {\"detail\":\"Parameters not found\"}" in response.json()["detail"]
    mock_download.assert_not_called()

@pytest.mark.asyncio
async def test_detect_anomalies_s3_download_error(
    client: TestClient,
    mock_user_profile_client: AsyncMock,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 127
    user_id = 5
    params_s3_key = "s3://highjump-biomechanical-data/biomechanical_parameters/user5/attempt127_uuid.json"

    # Mock User Profile Service (success)
    mock_user_profile_client.get.return_value.json.return_value = {"user_id": user_id, "injury_status": None}
    mock_user_profile_client.get.return_value.raise_for_status.return_value = None

    # Mock Session & BVH Data Service (success)
    mock_session_bvh_client.get.return_value.json.return_value = {"s3_key": params_s3_key, "parameters": {}}
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    # Mock S3 download to raise an error
    from botocore.exceptions import ClientError
    mock_download.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/detect/anomalies", json=request_payload)

    assert response.status_code == 500
    assert "S3 client error" in response.json()["detail"]
