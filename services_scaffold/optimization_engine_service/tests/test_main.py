import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import httpx
from optimization_engine_service import schemas
from optimization_engine_service.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "optimization_engine_service"}

@pytest.mark.asyncio
async def test_run_personal_optimum_solver_success(mock_httpx_client: AsyncMock):
    athlete_id = 1
    
    # Mock user profile service response
    mock_httpx_client.get.return_value = AsyncMock(
        status_code=200,
        json=lambda: {"id": 1, "user_id": athlete_id, "first_name": "John", "height_cm": 180.0, "weight_kg": 75.0},
        raise_for_status=lambda: None
    )
    
    response = client.post(f"/internal/optimize/{athlete_id}", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["athlete_id"] == athlete_id
    assert "optimal_parameters" in data
    assert data["optimal_parameters"]["approach_velocity_mps"] == 8.5
    
    mock_httpx_client.get.assert_called_once_with(f"http://user_profile_service:8001/internal/profiles/{athlete_id}")

@pytest.mark.asyncio
async def test_run_personal_optimum_solver_profile_not_found(mock_httpx_client: AsyncMock):
    athlete_id = 1
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "User profile not found"
    mock_response.json.return_value = {"detail": "User profile not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_httpx_client.get.return_value = mock_response

    response = client.post(f"/internal/optimize/{athlete_id}", json={})
    assert response.status_code == 500 # Internal service error propagated as 500
    assert "Failed to fetch user profile" in response.json()["detail"]

@pytest.mark.asyncio
async def test_run_personal_optimum_solver_network_error(mock_httpx_client: AsyncMock):
    athlete_id = 1
    mock_httpx_client.get.side_effect = httpx.RequestError("Connection refused", request=httpx.Request("GET", "http://test"))

    response = client.post(f"/internal/optimize/{athlete_id}", json={})
    assert response.status_code == 503
    assert "Could not connect to user profile service" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_sensitivity_analysis_results_success(mock_httpx_client: AsyncMock):
    athlete_id = 1
    
    # Mock user profile service response
    mock_httpx_client.get.return_value = AsyncMock(
        status_code=200,
        json=lambda: {"id": 1, "user_id": athlete_id, "first_name": "John", "height_cm": 180.0, "weight_kg": 75.0},
        raise_for_status=lambda: None
    )

    response = client.get(f"/internal/optimize/{athlete_id}/sensitivity")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["athlete_id"] == athlete_id
    assert "results" in data
    assert len(data["results"]) == 4
    assert data["results"][0]["parameter_name"] == "knee_flexion_at_takeoff"
    
    mock_httpx_client.get.assert_called_once_with(f"http://user_profile_service:8001/internal/profiles/{athlete_id}")

@pytest.mark.asyncio
async def test_get_sensitivity_analysis_results_profile_not_found(mock_httpx_client: AsyncMock):
    athlete_id = 1
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "User profile not found"
    mock_response.json.return_value = {"detail": "User profile not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_httpx_client.get.return_value = mock_response

    response = client.get(f"/internal/optimize/{athlete_id}/sensitivity")
    assert response.status_code == 500
    assert "Failed to fetch user profile" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_sensitivity_analysis_results_network_error(mock_httpx_client: AsyncMock):
    athlete_id = 1
    mock_httpx_client.get.side_effect = httpx.RequestError("Connection refused", request=httpx.Request("GET", "http://test"))

    response = client.get(f"/internal/optimize/{athlete_id}/sensitivity")
    assert response.status_code == 503
    assert "Could not connect to user profile service" in response.json()["detail"]
