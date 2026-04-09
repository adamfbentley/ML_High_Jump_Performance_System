import pytest
from httpx import AsyncClient, Response
from unittest.mock import AsyncMock, patch
from feedback_reporting_service.main import app
from feedback_reporting_service.schemas import HealthCheckResponse, SessionSummaryReportResponse, ActionableFeedbackResponse, VisualsDataResponse, SessionTypeEnum, AttemptOutcomeEnum
from datetime import datetime
import httpx

@pytest.fixture(name="test_client")
async def test_client_fixture():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_session_bvh_client():
    with patch("feedback_reporting_service.dependencies.get_session_bvh_data_service_client", autospec=True) as mock_dep:
        mock_client = AsyncMock(spec=AsyncClient)
        mock_dep.return_value.__aenter__.return_value = mock_client
        yield mock_client

# Test Health Check
async def test_health_check(test_client):
    response = await test_client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "feedback_reporting_service"}

# Test get_session_summary_report
async def test_get_session_summary_report_success(test_client, mock_session_bvh_client):
    session_id = 1
    user_id = 123
    mock_session_bvh_client.get.side_effect = [
        Response(status_code=200, json={
            "id": session_id,
            "user_id": user_id,
            "raw_video_s3_key": "raw/video.mp4",
            "session_date": "2023-01-15T10:00:00",
            "session_type": "TRAINING",
            "notes": "Morning session",
            "created_at": "2023-01-15T10:00:00"
        }),
        Response(status_code=200, json=[
            {
                "id": 101, "session_id": session_id, "attempt_number": 1,
                "bar_height_cm": 180.0, "outcome": "SUCCESS", "bvh_file_s3_key": "bvh/101.bvh",
                "created_at": "2023-01-15T10:05:00"
            },
            {
                "id": 102, "session_id": session_id, "attempt_number": 2,
                "bar_height_cm": 185.0, "outcome": "FAIL", "bvh_file_s3_key": "bvh/102.bvh",
                "created_at": "2023-01-15T10:10:00"
            },
            {
                "id": 103, "session_id": session_id, "attempt_number": 3,
                "bar_height_cm": 175.0, "outcome": "SUCCESS", "bvh_file_s3_key": "bvh/103.bvh",
                "created_at": "2023-01-15T10:15:00"
            }
        ])
    ]

    response = await test_client.get(f"/internal/reports/session/{session_id}", params={"user_id": user_id})
    assert response.status_code == 200
    report = SessionSummaryReportResponse(**response.json())
    assert report.session_id == session_id
    assert report.total_attempts == 3
    assert report.successful_attempts == 2
    assert report.average_bar_height_cm == pytest.approx(177.5) # (180 + 175) / 2
    assert "Excellent performance" in report.overall_feedback # Based on 2/3 success rate (66%)
    assert len(report.key_takeaways) > 0
    assert len(report.actionable_advice) > 0
    assert "Simulated feedback" in report.overall_feedback
    assert "Note: This feedback is simulated" in report.key_takeaways[-1]


async def test_get_session_summary_report_no_attempts(test_client, mock_session_bvh_client):
    session_id = 2
    user_id = 123
    mock_session_bvh_client.get.side_effect = [
        Response(status_code=200, json={
            "id": session_id,
            "user_id": user_id,
            "raw_video_s3_key": None,
            "session_date": "2023-01-16T11:00:00",
            "session_type": "OTHER",
            "notes": "Warm-up session",
            "created_at": "2023-01-16T11:00:00"
        }),
        Response(status_code=200, json=[]) # No attempts
    ]

    response = await test_client.get(f"/internal/reports/session/{session_id}", params={"user_id": user_id})
    assert response.status_code == 200
    report = SessionSummaryReportResponse(**response.json())
    assert report.session_id == session_id
    assert report.total_attempts == 0
    assert report.successful_attempts == 0
    assert report.average_bar_height_cm is None
    assert "A challenging session" in report.overall_feedback # Based on 0% success rate
    assert len(report.key_takeaways) > 0
    assert len(report.actionable_advice) > 0

async def test_get_session_summary_report_session_not_found(test_client, mock_session_bvh_client):
    session_id = 999
    user_id = 123
    mock_session_bvh_client.get.return_value = Response(status_code=404, text="Session not found")

    response = await test_client.get(f"/internal/reports/session/{session_id}", params={"user_id": user_id})
    assert response.status_code == 404
    assert "Session not found or not owned by user" in response.json()["detail"]

async def test_get_session_summary_report_service_unavailable(test_client, mock_session_bvh_client):
    session_id = 1
    user_id = 123
    mock_session_bvh_client.get.side_effect = httpx.RequestError("Network error", request=httpx.Request("GET", "http://test"))

    response = await test_client.get(f"/internal/reports/session/{session_id}", params={"user_id": user_id})
    assert response.status_code == 503
    assert "Could not connect to Session & BVH Data service" in response.json()["detail"]

# Test get_attempt_actionable_feedback
@pytest.mark.parametrize("outcome, expected_message_part", [
    ("SUCCESS", "Excellent execution!"),
    ("FAIL", "The attempt resulted in a fail."),
    ("KNOCK", "The bar was knocked."),
    ("UNKNOWN", "Analysis pending or outcome unknown."),
])
async def test_get_attempt_actionable_feedback_success(test_client, mock_session_bvh_client, outcome, expected_message_part):
    attempt_id = 101
    user_id = 123
    mock_session_bvh_client.get.return_value = Response(status_code=200, json={
        "id": attempt_id, "session_id": 1, "attempt_number": 1,
        "bar_height_cm": 180.0, "outcome": outcome, "bvh_file_s3_key": "bvh/101.bvh",
        "created_at": "2023-01-15T10:05:00"
    })

    response = await test_client.get(f"/internal/reports/attempt/{attempt_id}/feedback", params={"user_id": user_id})
    assert response.status_code == 200
    feedback = ActionableFeedbackResponse(**response.json())
    assert feedback.attempt_id == attempt_id
    assert expected_message_part in feedback.feedback_message
    assert "Simulated feedback" in feedback.feedback_message
    assert "Note: This feedback is simulated" in feedback.recommendations[-1]
    assert "Note: This advice is simulated" in feedback.areas_for_improvement[-1]

async def test_get_attempt_actionable_feedback_not_found(test_client, mock_session_bvh_client):
    attempt_id = 999
    user_id = 123
    mock_session_bvh_client.get.return_value = Response(status_code=404, text="Attempt not found")

    response = await test_client.get(f"/internal/reports/attempt/{attempt_id}/feedback", params={"user_id": user_id})
    assert response.status_code == 404
    assert "Attempt not found or not owned by user's session" in response.json()["detail"]

# Test get_attempt_visuals_data
async def test_get_attempt_visuals_data_success(test_client, mock_session_bvh_client):
    attempt_id = 101
    user_id = 123
    bvh_key = "bvh/101.bvh"
    mock_session_bvh_client.get.return_value = Response(status_code=200, json={
        "id": attempt_id, "session_id": 1, "attempt_number": 1,
        "bar_height_cm": 180.0, "outcome": "SUCCESS", "bvh_file_s3_key": bvh_key,
        "created_at": "2023-01-15T10:05:00"
    })

    response = await test_client.get(f"/internal/reports/attempt/{attempt_id}/visuals", params={"user_id": user_id})
    assert response.status_code == 200
    visuals = VisualsDataResponse(**response.json())
    assert visuals.attempt_id == attempt_id
    assert visuals.bvh_s3_key == bvh_key
    assert "optimal_model_comparison" in visuals.comparison_data
    assert "joint_angles_time_series" in visuals.overlay_data
    assert "This comparison data is simulated" in visuals.comparison_data["note"]
    assert "This overlay data is simulated" in visuals.overlay_data["note"]

async def test_get_attempt_visuals_data_not_found(test_client, mock_session_bvh_client):
    attempt_id = 999
    user_id = 123
    mock_session_bvh_client.get.return_value = Response(status_code=404, text="Attempt not found")

    response = await test_client.get(f"/internal/reports/attempt/{attempt_id}/visuals", params={"user_id": user_id})
    assert response.status_code == 404
    assert "Attempt not found or not owned by user's session" in response.json()["detail"]
