import pytest
from httpx import AsyncClient, Response
from unittest.mock import AsyncMock, patch
from api_gateway.main import app
from api_gateway.schemas import UserResponse, SessionSummaryReportResponse, VisualsDataResponse
import httpx

# Assuming test_client fixture is defined in a conftest.py or similar for the API Gateway tests
# For this submission, we'll define it here for completeness.
@pytest.fixture(name="test_client")
async def test_client_fixture():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

# New fixture for feedback service client
@pytest.fixture
def mock_feedback_reporting_service_client():
    with patch("api_gateway.dependencies.get_feedback_reporting_service_client", autospec=True) as mock_dep:
        mock_client = AsyncMock(spec=AsyncClient)
        mock_dep.return_value.__aenter__.return_value = mock_client
        yield mock_client

# Mock current user for authenticated endpoints
@pytest.fixture
def mock_current_user():
    with patch("api_gateway.auth.get_current_user", autospec=True) as mock_get_current_user:
        mock_user = UserResponse(id=1, email="test@example.com", is_active=True)
        mock_get_current_user.return_value = mock_user
        yield mock_get_current_user

# Test get_session_feedback_summary
async def test_get_session_feedback_summary_success(test_client, mock_feedback_reporting_service_client, mock_current_user):
    session_id = 1
    user_id = mock_current_user.return_value.id
    mock_feedback_reporting_service_client.get.return_value = Response(status_code=200, json={
        "session_id": session_id,
        "session_date": "2023-01-15T10:00:00",
        "session_type": "TRAINING",
        "total_attempts": 3,
        "successful_attempts": 2,
        "average_bar_height_cm": 177.5,
        "overall_feedback": "Great session! (Simulated feedback based on basic metrics.)",
        "key_takeaways": ["Takeaway 1", "Note: This feedback is simulated and requires advanced biomechanical analysis for production-ready insights."],
        "actionable_advice": ["Advice 1", "Note: This advice is simulated and requires advanced biomechanical analysis for production-ready insights."]
    })

    response = await test_client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 200
    assert response.json()["session_id"] == session_id
    assert "Great session!" in response.json()["overall_feedback"]
    mock_feedback_reporting_service_client.get.assert_called_once_with(f"/reports/session/{session_id}?user_id={user_id}")

async def test_get_session_feedback_summary_not_found(test_client, mock_feedback_reporting_service_client, mock_current_user):
    session_id = 999
    mock_feedback_reporting_service_client.get.return_value = Response(status_code=404, text="Not found")

    response = await test_client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 404
    assert "Session feedback not found or not authorized" in response.json()["detail"]

async def test_get_session_feedback_summary_unauthenticated(test_client):
    # Do not use mock_current_user fixture to simulate unauthenticated
    response = await test_client.get("/feedback/session/1")
    assert response.status_code == 401

async def test_get_session_feedback_summary_service_unavailable(test_client, mock_feedback_reporting_service_client, mock_current_user):
    session_id = 1
    mock_feedback_reporting_service_client.get.side_effect = httpx.RequestError("Network error", request=httpx.Request("GET", "http://test"))

    response = await test_client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 503
    assert "Could not connect to Feedback & Reporting service" in response.json()["detail"]

# Test get_attempt_feedback_visuals
async def test_get_attempt_feedback_visuals_success(test_client, mock_feedback_reporting_service_client, mock_current_user):
    attempt_id = 101
    user_id = mock_current_user.return_value.id
    mock_feedback_reporting_service_client.get.return_value = Response(status_code=200, json={
        "attempt_id": attempt_id,
        "bvh_s3_key": "bvh/101.bvh",
        "comparison_data": {"optimal_model_comparison": {"description": "Simulated optimal model comparison data."}, "note": "This comparison data is simulated"},
        "overlay_data": {"joint_angles_time_series": {"description": "Simulated joint angle data."}, "note": "This overlay data is simulated"}
    })

    response = await test_client.get(f"/feedback/attempt/{attempt_id}/visuals")
    assert response.status_code == 200
    assert response.json()["attempt_id"] == attempt_id
    assert response.json()["bvh_s3_key"] == "bvh/101.bvh"
    mock_feedback_reporting_service_client.get.assert_called_once_with(f"/reports/attempt/{attempt_id}/visuals?user_id={user_id}")

async def test_get_attempt_feedback_visuals_not_found(test_client, mock_feedback_reporting_service_client, mock_current_user):
    attempt_id = 999
    mock_feedback_reporting_service_client.get.return_value = Response(status_code=404, text="Not found")

    response = await test_client.get(f"/feedback/attempt/{attempt_id}/visuals")
    assert response.status_code == 404
    assert "Attempt visuals data not found or not authorized" in response.json()["detail"]

async def test_get_attempt_feedback_visuals_unauthenticated(test_client):
    # Do not use mock_current_user fixture to simulate unauthenticated
    response = await test_client.get("/feedback/attempt/1")
    assert response.status_code == 401

async def test_get_attempt_feedback_visuals_service_unavailable(test_client, mock_feedback_reporting_service_client, mock_current_user):
    attempt_id = 1
    mock_feedback_reporting_service_client.get.side_effect = httpx.RequestError("Network error", request=httpx.Request("GET", "http://test"))

    response = await test_client.get(f"/feedback/attempt/{attempt_id}/visuals")
    assert response.status_code == 503
    assert "Could not connect to Feedback & Reporting service" in response.json()["detail"]
