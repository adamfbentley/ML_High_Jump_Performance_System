from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import pytest
import json

# Assuming conftest.py sets up mock_session_bvh_client and mock_s3_utils

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "pinn_gnn_inference_service"}

@pytest.mark.asyncio
async def test_analyze_biomechanics_success(
    client: TestClient,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 123
    user_id = 1
    bvh_s3_key = "s3://highjump-biomechanical-data/bvh/user1/attempt123.bvh"
    params_s3_key = "s3://highjump-biomechanical-data/biomechanical_parameters/user1/attempt123_uuid.json"

    # Mock getting BVH S3 key from Session & BVH Data Service
    mock_session_bvh_client.get.return_value.json.return_value = {"bvh_file_s3_key": bvh_s3_key}
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    # Mock S3 download (now active)
    mock_download.return_value = b"dummy_bvh_content_12345"

    # Mock S3 upload for biomechanical parameters
    mock_upload.return_value = params_s3_key

    # Mock updating Session & BVH Data Service with new S3 key
    mock_session_bvh_client.post.return_value.raise_for_status.return_value = None

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/analyze/biomechanics", json=request_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["attempt_id"] == attempt_id
    assert data["status"] == "COMPLETED"
    assert data["biomechanical_parameters_s3_key"] == params_s3_key
    assert "biomechanical_parameters" in data
    assert "analysis_timestamp" in data["biomechanical_parameters"]
    assert data["biomechanical_parameters"]["simulated_bvh_content_length"] == len(b"dummy_bvh_content_12345")

    mock_session_bvh_client.get.assert_called_with(f"/attempts/{attempt_id}?user_id={user_id}")
    mock_download.assert_called_with("bvh/user1/attempt123.bvh") # This would be called if download was active
    mock_upload.assert_called_once()
    mock_session_bvh_client.post.assert_called_with(
        f"/attempts/{attempt_id}/parameters?user_id={user_id}",
        json={"biomechanical_parameters_s3_key": params_s3_key}
    )

@pytest.mark.asyncio
async def test_analyze_biomechanics_no_bvh_key(
    client: TestClient,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 124
    user_id = 1

    # Mock getting BVH S3 key, but return None
    mock_session_bvh_client.get.return_value.json.return_value = {"bvh_file_s3_key": None}
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/analyze/biomechanics", json=request_payload)

    assert response.status_code == 404
    assert "BVH file S3 key not found for this attempt." in response.json()["detail"]
    mock_download.assert_not_called()
    mock_upload.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_biomechanics_session_bvh_service_error(
    client: TestClient,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 125
    user_id = 1

    # Mock Session & BVH Data Service returning an error
    mock_session_bvh_client.get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("GET", "http://test"), response=httpx.Response(400, request=httpx.Request("GET", "http://test"), content=b'{"detail":"Error"}')
    )

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/analyze/biomechanics", json=request_payload)

    assert response.status_code == 400
    assert "Error from Session & BVH Data Service" in response.json()["detail"]
    mock_download.assert_not_called()
    mock_upload.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_biomechanics_s3_upload_error(
    client: TestClient,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 126
    user_id = 1
    bvh_s3_key = "s3://highjump-biomechanical-data/bvh/user1/attempt126.bvh"

    mock_session_bvh_client.get.return_value.json.return_value = {"bvh_file_s3_key": bvh_s3_key}
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    mock_download.return_value = b"dummy_bvh_content"

    # Mock S3 upload to raise an error
    from botocore.exceptions import ClientError
    mock_upload.side_effect = ClientError({"Error": {"Code": "500", "Message": "S3 Error"}}, "PutObject")

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/analyze/biomechanics", json=request_payload)

    assert response.status_code == 500
    assert "S3 client error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_analyze_biomechanics_s3_download_error(
    client: TestClient,
    mock_session_bvh_client: AsyncMock,
    mock_s3_utils
):
    mock_upload, mock_download = mock_s3_utils

    attempt_id = 127
    user_id = 1
    bvh_s3_key = "s3://highjump-biomechanical-data/bvh/user1/attempt127.bvh"

    mock_session_bvh_client.get.return_value.json.return_value = {"bvh_file_s3_key": bvh_s3_key}
    mock_session_bvh_client.get.return_value.raise_for_status.return_value = None

    # Mock S3 download to raise an error
    from botocore.exceptions import ClientError
    mock_download.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")

    request_payload = {"attempt_id": attempt_id, "user_id": user_id}
    response = client.post("/internal/analyze/biomechanics", json=request_payload)

    assert response.status_code == 500
    assert "S3 client error" in response.json()["detail"]
    mock_upload.assert_not_called()
