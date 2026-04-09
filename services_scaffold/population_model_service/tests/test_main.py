import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from datetime import datetime, timezone
import httpx

from population_model_service import schemas
from population_model_service.main import app

# Create a TestClient for the FastAPI app
client = TestClient(app)

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "population_model_service"}

@pytest.mark.asyncio
async def test_get_population_cohort_data_success(mock_httpx_client: AsyncMock):
    user_id = 1
    now = datetime.now(timezone.utc)
    dob_str = (now - timezone.timedelta(days=20*365)).isoformat()

    mock_user_profile_response = {
        "id": 10,
        "user_id": user_id,
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": dob_str,
        "gender": "MALE",
        "height_cm": 180.0,
        "weight_kg": 75.0,
        "primary_sport": "High Jump",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }

    mock_httpx_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_user_profile_response, raise_for_status=lambda: None
    )

    response = client.get(f"/internal/models/population/cohorts?user_id={user_id}")
    assert response.status_code == 200
    data = response.json()

    assert "matched_cohort_description" in data
    assert "Male Athletes, Young Adult, Average Height, Average Weight" in data["matched_cohort_description"]
    assert len(data["comparison_data"]) == 3 # Max Vertical Jump, Approach Speed, Take-off Angle

    mock_httpx_client.get.assert_called_once_with(f"http://user_profile_service:8001/internal/profiles/{user_id}")

@pytest.mark.asyncio
async def test_get_population_cohort_data_with_bar_height(mock_httpx_client: AsyncMock):
    user_id = 2
    target_bar_height = 200.0
    now = datetime.now(timezone.utc)
    dob_str = (now - timezone.timedelta(days=28*365)).isoformat()

    mock_user_profile_response = {
        "id": 11,
        "user_id": user_id,
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": dob_str,
        "gender": "FEMALE",
        "height_cm": 170.0,
        "weight_kg": 60.0,
        "primary_sport": "High Jump",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }

    mock_httpx_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_user_profile_response, raise_for_status=lambda: None
    )

    response = client.get(f"/internal/models/population/cohorts?user_id={user_id}&target_bar_height_cm={target_bar_height}")
    assert response.status_code == 200
    data = response.json()

    assert "Female Athletes, Adult, Average Height, Average Weight" in data["matched_cohort_description"]
    assert len(data["comparison_data"]) == 4 # Includes Bar Clearance Rate
    assert any(item["metric_name"] == f"Bar Clearance Rate @ {target_bar_height}cm" for item in data["comparison_data"])

@pytest.mark.asyncio
async def test_get_population_cohort_data_user_profile_service_error(mock_httpx_client: AsyncMock):
    user_id = 3
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.return_value = {"detail": "Internal Server Error"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_httpx_client.get.return_value = mock_response

    response = client.get(f"/internal/models/population/cohorts?user_id={user_id}")
    assert response.status_code == 500
    assert "Failed to fetch user profile" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_population_cohort_data_user_profile_not_found(mock_httpx_client: AsyncMock):
    user_id = 4
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "User profile not found"
    mock_response.json.return_value = {"detail": "User profile not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_httpx_client.get.return_value = mock_response

    response = client.get(f"/internal/models/population/cohorts?user_id={user_id}")
    assert response.status_code == 500 # Internal service error is 500, not 404
    assert "Failed to fetch user profile" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_population_cohort_data_network_error(mock_httpx_client: AsyncMock):
    user_id = 5
    mock_httpx_client.get.side_effect = httpx.RequestError("Connection refused", request=httpx.Request("GET", "http://test"))

    response = client.get(f"/internal/models/population/cohorts?user_id={user_id}")
    assert response.status_code == 503
    assert "Could not connect to user profile service" in response.json()["detail"]
