import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone, timedelta
import httpx

from api_gateway.main import app
from api_gateway import schemas
from api_gateway.dependencies import get_session_bvh_data_service_client, get_feedback_reporting_service_client
from api_gateway import auth

# Create a TestClient for the FastAPI app
client = TestClient(app)

# Mock the current user for authenticated endpoints
@pytest.fixture
def mock_current_user():
    user = schemas.UserResponse(id=1, email="test@example.com", is_active=True)
    with patch("api_gateway.auth.get_current_user", return_value=user):
        yield user

# Mock the Session & BVH Data Service client
@pytest.fixture
def mock_session_bvh_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch("api_gateway.dependencies.get_session_bvh_data_service_client", return_value=mock_client):
        yield mock_client

# Mock the Video Ingestion Service client
@pytest.fixture
def mock_video_ingestion_service_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch('api_gateway.dependencies.get_video_ingestion_service_client') as mock_dependency:
        mock_dependency.return_value.__aenter__.return_value = mock_client
        yield mock_client

# Mock the Feedback & Reporting Service client
@pytest.fixture
def mock_feedback_reporting_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch("api_gateway.dependencies.get_feedback_reporting_service_client", return_value=mock_client):
        yield mock_client

# --- Schema Tests ---

def test_session_type_enum():
    assert schemas.SessionTypeEnum.TRAINING == "TRAINING"
    assert schemas.SessionTypeEnum.COMPETITION == "COMPETITION"
    assert schemas.SessionTypeEnum.OTHER == "OTHER"
    assert list(schemas.SessionTypeEnum) == ["TRAINING", "COMPETITION", "OTHER"]

def test_attempt_outcome_enum():
    assert schemas.AttemptOutcomeEnum.SUCCESS == "SUCCESS"
    assert schemas.AttemptOutcomeEnum.FAIL == "FAIL"
    assert schemas.AttemptOutcomeEnum.KNOCK == "KNOCK"
    assert schemas.AttemptOutcomeEnum.DID_NOT_ATTEMPT == "DID_NOT_ATTEMPT"
    assert schemas.AttemptOutcomeEnum.UNKNOWN == "UNKNOWN"
    assert len(list(schemas.AttemptOutcomeEnum)) == 5

def test_session_with_attempts_response_schema():
    now = datetime.now(timezone.utc)
    session_data = {
        "id": 1,
        "user_id": 10,
        "raw_video_s3_key": "s3://my-bucket/video-abc.mp4",
        "session_date": now.isoformat(),
        "session_type": "TRAINING",
        "notes": "First training session",
        "created_at": now.isoformat(),
        "updated_at": None
    }
    attempt_data_1 = {
        "id": 101,
        "session_id": 1,
        "attempt_number": 1,
        "start_time_ms": 1000,
        "end_time_ms": 5000,
        "bar_height_cm": 150.0,
        "outcome": "SUCCESS",
        "bvh_file_s3_key": "s3://my-bucket/bvh-101.bvh",
        "created_at": now.isoformat(),
        "updated_at": None
    }
    attempt_data_2 = {
        "id": 102,
        "session_id": 1,
        "attempt_number": 2,
        "start_time_ms": 6000,
        "end_time_ms": 10000,
        "bar_height_cm": 155.0,
        "outcome": "FAIL",
        "bvh_file_s3_key": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }

    session_with_attempts_payload = {
        **session_data,
        "attempts": [attempt_data_1, attempt_data_2]
    }

    session_response = schemas.SessionWithAttemptsResponse(**session_with_attempts_payload)

    assert session_response.id == 1
    assert session_response.user_id == 10
    assert session_response.raw_video_s3_key == "s3://my-bucket/video-abc.mp4"
    assert session_response.session_date.isoformat() == now.isoformat()
    assert session_response.session_type == schemas.SessionTypeEnum.TRAINING
    assert session_response.notes == "First training session"
    assert session_response.created_at.isoformat() == now.isoformat()
    assert session_response.updated_at is None

    assert len(session_response.attempts) == 2
    assert session_response.attempts[0].id == 101
    assert session_response.attempts[0].session_id == 1
    assert session_response.attempts[0].attempt_number == 1
    assert session_response.attempts[0].start_time_ms == 1000
    assert session_response.attempts[0].end_time_ms == 5000
    assert session_response.attempts[0].bar_height_cm == 150.0
    assert session_response.attempts[0].outcome == schemas.AttemptOutcomeEnum.SUCCESS
    assert session_response.attempts[0].bvh_file_s3_key == "s3://my-bucket/bvh-101.bvh"
    assert session_response.attempts[0].created_at.isoformat() == now.isoformat()
    assert session_response.attempts[0].updated_at is None

    assert session_response.attempts[1].id == 102
    assert session_response.attempts[1].session_id == 1
    assert session_response.attempts[1].attempt_number == 2
    assert session_response.attempts[1].start_time_ms == 6000
    assert session_response.attempts[1].end_time_ms == 10000
    assert session_response.attempts[1].bar_height_cm == 155.0
    assert session_response.attempts[1].outcome == schemas.AttemptOutcomeEnum.FAIL
    assert session_response.attempts[1].bvh_file_s3_key is None
    assert session_response.attempts[1].created_at.isoformat() == now.isoformat()
    assert session_response.attempts[1].updated_at.isoformat() == now.isoformat()

def test_session_with_attempts_response_schema_no_attempts():
    now = datetime.now(timezone.utc)
    session_data = {
        "id": 2,
        "user_id": 11,
        "raw_video_s3_key": "s3://my-bucket/video-def.mp4",
        "session_date": now.isoformat(),
        "session_type": "COMPETITION",
        "notes": "Competition session with no attempts yet",
        "created_at": now.isoformat(),
        "updated_at": None
    }

    session_with_attempts_payload = {
        **session_data,
        "attempts": []
    }

    session_response = schemas.SessionWithAttemptsResponse(**session_with_attempts_payload)

    assert session_response.id == 2
    assert session_response.session_type == schemas.SessionTypeEnum.COMPETITION
    assert len(session_response.attempts) == 0

# --- Endpoint Tests for api_gateway/main.py ---

# Test cases for POST /sessions (STORY-201 related - creating a session)
@pytest.mark.asyncio
async def test_create_session_endpoint_success_with_video(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_create_payload = {
        "raw_video_s3_key": "new_video_id",
        "session_type": schemas.SessionTypeEnum.TRAINING.value,
        "notes": "Test session"
    }
    mock_session_bvh_client.post.return_value = AsyncMock(
        status_code=201,
        json=lambda: {"id": 1, "user_id": user_id, **session_create_payload, "session_date": "2023-01-01T00:00:00Z", "created_at": "2023-01-01T00:00:00Z"}
    )
    mock_session_bvh_client.post.return_value.raise_for_status = AsyncMock()

    response = client.post("/sessions", json=session_create_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["raw_video_s3_key"] == session_create_payload["raw_video_s3_key"]
    assert data["user_id"] == user_id

    mock_session_bvh_client.post.assert_called_once_with(
        "/sessions",
        json={"user_id": user_id, **session_create_payload}
    )

@pytest.mark.asyncio
async def test_create_session_endpoint_success_live_session_no_video(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_create_payload = {
        "session_type": schemas.SessionTypeEnum.TRAINING.value,
        "notes": "Live session in progress"
    }
    mock_session_bvh_client.post.return_value = AsyncMock(
        status_code=201,
        json=lambda: {"id": 2, "user_id": user_id, "raw_video_s3_key": None, **session_create_payload, "session_date": "2023-01-01T00:00:00Z", "created_at": "2023-01-01T00:00:00Z"}
    )
    mock_session_bvh_client.post.return_value.raise_for_status = AsyncMock()

    response = client.post("/sessions", json=session_create_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["raw_video_s3_key"] is None
    assert data["user_id"] == user_id

    mock_session_bvh_client.post.assert_called_once_with(
        "/sessions",
        json={"user_id": user_id, **session_create_payload}
    )

@pytest.mark.asyncio
async def test_create_session_endpoint_duplicate_video_id(mock_current_user, mock_session_bvh_client):
    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.text = '{"detail": "Session with this raw video S3 key already exists for this user."}'
    mock_response.json.return_value = {"detail": "Session with this raw video S3 key already exists for this user.", "status_code": 400}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("POST", "http://test"), response=mock_response
    )
    mock_session_bvh_client.post.return_value = mock_response

    session_create_payload = {
        "raw_video_s3_key": "existing_video_id",
        "session_type": schemas.SessionTypeEnum.TRAINING.value
    }
    response = client.post("/sessions", json=session_create_payload)
    assert response.status_code == 400
    assert "Session with this raw video S3 key already exists for this user." in response.json()["detail"]

# Test cases for GET /sessions (STORY-105)
@pytest.mark.asyncio
async def test_get_all_sessions_success(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_sessions_data = [
        {
            "id": 101,
            "user_id": user_id,
            "raw_video_s3_key": "s3://bucket/video1.mp4",
            "session_date": now_str,
            "session_type": "TRAINING",
            "notes": "Morning session",
            "created_at": now_str,
            "updated_at": None
        },
        {
            "id": 102,
            "user_id": user_id,
            "raw_video_s3_key": None,
            "session_date": now_str,
            "session_type": "COMPETITION",
            "notes": "Afternoon competition (live)",
            "created_at": now_str,
            "updated_at": None
        }
    ]
    mock_session_bvh_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_sessions_data, raise_for_status=lambda: None
    )

    response = client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == 101
    assert data[1]["session_type"] == "COMPETITION"
    assert data[1]["raw_video_s3_key"] is None
    mock_session_bvh_client.get.assert_called_once_with(f"/users/{user_id}/sessions")

@pytest.mark.asyncio
async def test_get_all_sessions_no_sessions(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    mock_session_bvh_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: [], raise_for_status=lambda: None
    )

    response = client.get("/sessions")
    assert response.status_code == 200
    assert response.json() == []
    mock_session_bvh_client.get.assert_called_once_with(f"/users/{user_id}/sessions")

@pytest.mark.asyncio
async def test_get_all_sessions_service_error(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.return_value = {"detail": "Internal Server Error"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_session_bvh_client.get.return_value = mock_response

    response = client.get("/sessions")
    assert response.status_code == 500
    assert "Session & BVH Data service error" in response.json()["detail"]
    mock_session_bvh_client.get.assert_called_once_with(f"/users/{user_id}/sessions")

@pytest.mark.asyncio
async def test_get_all_sessions_connection_error(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    mock_session_bvh_client.get.side_effect = httpx.RequestError("Connection refused", request=httpx.Request("GET", "http://test"))

    response = client.get("/sessions")
    assert response.status_code == 503
    assert "Could not connect to Session & BVH Data service" in response.json()["detail"]
    mock_session_bvh_client.get.assert_called_once_with(f"/users/{user_id}/sessions")

# Test cases for GET /sessions/{session_id} (STORY-106)
@pytest.mark.asyncio
async def test_get_session_details_success(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 101
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_session_data = {
        "id": session_id,
        "user_id": user_id,
        "raw_video_s3_key": "s3://bucket/video1.mp4",
        "session_date": now_str,
        "session_type": "TRAINING",
        "notes": "Morning session",
        "created_at": now_str,
        "updated_at": None
    }
    mock_attempts_data = [
        {
            "id": 1,
            "session_id": session_id,
            "attempt_number": 1,
            "start_time_ms": 1000,
            "end_time_ms": 5000,
            "bar_height_cm": 150.0,
            "outcome": "SUCCESS",
            "bvh_file_s3_key": "s3://bucket/bvh1.bvh",
            "created_at": now_str,
            "updated_at": None
        },
        {
            "id": 2,
            "session_id": session_id,
            "attempt_number": 2,
            "start_time_ms": 6000,
            "end_time_ms": 10000,
            "bar_height_cm": 155.0,
            "outcome": "FAIL",
            "bvh_file_s3_key": None,
            "created_at": now_str,
            "updated_at": None
        }
    ]

    mock_session_bvh_client.get.side_effect = [
        AsyncMock(status_code=200, json=lambda: mock_session_data, raise_for_status=lambda: None),
        AsyncMock(status_code=200, json=lambda: mock_attempts_data, raise_for_status=lambda: None)
    ]

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["user_id"] == user_id
    assert len(data["attempts"]) == 2
    assert data["attempts"][0]["id"] == 1
    assert data["attempts"][1]["outcome"] == "FAIL"

    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}?user_id={user_id}")
    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}/attempts?user_id={user_id}")
    assert mock_session_bvh_client.get.call_count == 2

@pytest.mark.asyncio
async def test_get_session_details_session_not_found(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 999

    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "Session not found"
    mock_response.json.return_value = {"detail": "Session not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_session_bvh_client.get.return_value = mock_response

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 404
    assert "Session not found or not authorized" in response.json()["detail"]
    mock_session_bvh_client.get.assert_called_once_with(f"/sessions/{session_id}?user_id={user_id}")

@pytest.mark.asyncio
async def test_get_session_details_attempts_empty(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 101
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_session_data = {
        "id": session_id,
        "user_id": user_id,
        "raw_video_s3_key": "s3://bucket/video1.mp4",
        "session_date": now_str,
        "session_type": "TRAINING",
        "notes": "Morning session",
        "created_at": now_str,
        "updated_at": None
    }

    mock_session_bvh_client.get.side_effect = [
        AsyncMock(status_code=200, json=lambda: mock_session_data, raise_for_status=lambda: None),
        AsyncMock(status_code=200, json=lambda: [], raise_for_status=lambda: None) # No attempts
    ]

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["user_id"] == user_id
    assert len(data["attempts"]) == 0

    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}?user_id={user_id}")
    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}/attempts?user_id={user_id}")
    assert mock_session_bvh_client.get.call_count == 2

@pytest.mark.asyncio
async def test_get_session_details_service_error_session_fetch(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 101

    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.return_value = {"detail": "Internal Server Error"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_session_bvh_client.get.return_value = mock_response # First call fails

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 500
    assert "Session & BVH Data service error" in response.json()["detail"]
    mock_session_bvh_client.get.assert_called_once_with(f"/sessions/{session_id}?user_id={user_id}")

@pytest.mark.asyncio
async def test_get_session_details_service_error_attempts_fetch(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 101
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_session_data = {
        "id": session_id,
        "user_id": user_id,
        "raw_video_s3_key": "s3://bucket/video1.mp4",
        "session_date": now_str,
        "session_type": "TRAINING",
        "notes": "Morning session",
        "created_at": now_str,
        "updated_at": None
    }

    mock_response_session = AsyncMock(status_code=200, json=lambda: mock_session_data, raise_for_status=lambda: None)
    mock_response_attempts = AsyncMock()
    mock_response_attempts.status_code = 500
    mock_response_attempts.text = "Internal Server Error"
    mock_response_attempts.json.return_value = {"detail": "Internal Server Error"}
    mock_response_attempts.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=httpx.Request("GET", "http://test"), response=mock_response_attempts
    )

    mock_session_bvh_client.get.side_effect = [
        mock_response_session,
        mock_response_attempts # Second call fails
    ]

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 500
    assert "Session & BVH Data service error" in response.json()["detail"]
    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}?user_id={user_id}")
    mock_session_bvh_client.get.assert_any_call(f"/sessions/{session_id}/attempts?user_id={user_id}")
    assert mock_session_bvh_client.get.call_count == 2

@pytest.mark.asyncio
async def test_get_session_details_connection_error(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 101
    mock_session_bvh_client.get.side_effect = httpx.RequestError("Connection refused", request=httpx.Request("GET", "http://test"))

    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 503
    assert "Could not connect to Session & BVH Data service" in response.json()["detail"]
    mock_session_bvh_client.get.assert_called_once_with(f"/sessions/{session_id}?user_id={user_id}")

# Test cases for POST /sessions/{session_id}/attempts (STORY-202 related - creating an attempt)
@pytest.mark.asyncio
async def test_create_attempt_for_session_endpoint_success(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 1
    attempt_payload = {"attempt_number": 1, "bar_height_cm": 170.0, "outcome": "SUCCESS"}
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'

    mock_session_bvh_client.post.return_value = AsyncMock(
        status_code=201,
        json=lambda: {"id": 101, "session_id": session_id, **attempt_payload, "created_at": now_str},
        raise_for_status=lambda: None
    )

    response = client.post(f"/sessions/{session_id}/attempts", json=attempt_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == session_id
    assert data["attempt_number"] == 1
    assert data["bar_height_cm"] == 170.0

    mock_session_bvh_client.post.assert_called_once_with(
        f"/sessions/{session_id}/attempts?user_id={user_id}",
        json=attempt_payload
    )

@pytest.mark.asyncio
async def test_create_attempt_for_session_endpoint_session_not_found(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 999
    attempt_payload = {"attempt_number": 1, "bar_height_cm": 170.0, "outcome": "SUCCESS"}

    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = '{"detail": "Session not found or not owned by user"}'
    mock_response.json.return_value = {"detail": "Session not found or not owned by user"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("POST", "http://test"), response=mock_response
    )
    mock_session_bvh_client.post.return_value = mock_response

    response = client.post(f"/sessions/{session_id}/attempts", json=attempt_payload)
    assert response.status_code == 404
    assert "Session not found or not authorized to add attempts" in response.json()["detail"]

@pytest.mark.asyncio
async def test_create_attempt_for_session_endpoint_duplicate_attempt(mock_current_user, mock_session_bvh_client):
    user_id = mock_current_user.id
    session_id = 1
    attempt_payload = {"attempt_number": 1, "bar_height_cm": 170.0, "outcome": "SUCCESS"}

    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.text = '{"detail": "Attempt number already exists for this session."}'
    mock_response.json.return_value = {"detail": "Attempt number already exists for this session."}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("POST", "http://test"), response=mock_response
    )
    mock_session_bvh_client.post.return_value = mock_response

    response = client.post(f"/sessions/{session_id}/attempts", json=attempt_payload)
    assert response.status_code == 400
    assert "Attempt number already exists for this session." in response.json()["detail"]

# NEW TESTS FOR FEEDBACK & REPORTING ENDPOINTS

@pytest.mark.asyncio
async def test_get_feedback_session_summary_success(mock_current_user, mock_feedback_reporting_client):
    user_id = mock_current_user.id
    session_id = 1
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_summary_data = {
        "session_id": session_id,
        "session_date": now_str,
        "session_type": "TRAINING",
        "total_attempts": 3,
        "successful_attempts": 2,
        "average_bar_height_cm": 153.3,
        "key_insights": ["Total attempts: 3, Success rate: 66.7%", "Stable with good progress, focus on consistency."],
        "recommendations": ["Review fundamental techniques and consistency."]
    }
    mock_feedback_reporting_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_summary_data, raise_for_status=lambda: None
    )

    response = client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["total_attempts"] == 3
    mock_feedback_reporting_client.get.assert_called_once_with(f"/reports/session/{session_id}?user_id={user_id}")

@pytest.mark.asyncio
async def test_get_feedback_session_summary_not_found(mock_current_user, mock_feedback_reporting_client):
    user_id = mock_current_user.id
    session_id = 999
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = '{"detail": "Session not found"}'
    mock_response.json.return_value = {"detail": "Session not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "http://test"), response=mock_response
    )
    mock_feedback_reporting_client.get.return_value = mock_response

    response = client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 404
    assert "Session feedback not found or not authorized" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_feedback_attempt_visuals_success(mock_current_user, mock_feedback_reporting_client):
    user_id = mock_current_user.id
    attempt_id = 101
    session_id = 1 # Assuming session_id is 1 for this attempt
    mock_visuals_data = {
        "attempt_id": attempt_id,
        "bvh_s3_key": "s3://bvh/file.bvh",
        "video_s3_key": "s3://raw/video.mp4",
        "overlay_image_urls": [f"https://example.com/visuals/session/{session_id}/attempt/{attempt_id}/pose_overlay.png"],
        "comparison_data_urls": [f"https://example.com/visuals/session/{session_id}/attempt/{attempt_id}/comparison_graph.json"]
    }
    mock_feedback_reporting_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_visuals_data, raise_for_status=lambda: None
    )

    response = client.get(f"/feedback/attempt/{attempt_id}/visuals")
    assert response.status_code == 200
    data = response.json()
    assert data["attempt_id"] == attempt_id
    assert data["bvh_s3_key"] == "s3://bvh/file.bvh"
    assert f"https://example.com/visuals/session/{session_id}/attempt/{attempt_id}/pose_overlay.png" in data["overlay_image_urls"]
    mock_feedback_reporting_client.get.assert_called_once_with(f"/reports/attempt/{attempt_id}/visuals?user_id={user_id}")

@pytest.mark.asyncio
async def test_get_progress_dashboard_success(mock_current_user, mock_feedback_reporting_client):
    user_id = mock_current_user.id
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_dashboard_data = {
        "athlete_id": user_id,
        "total_sessions": 5,
        "total_attempts": 15,
        "personal_best_height_cm": 200.0,
        "progress_chart_data": {"labels": ["2023-01-01", "2023-01-15"], "datasets": [{"label": "Max Height (cm)", "data": [180.0, 200.0]}]},
        "recent_sessions_summary": [
            {"session_id": 1, "session_date": now_str, "session_type": "TRAINING", "total_attempts": 3, "successful_attempts": 2, "key_insights": [], "recommendations": []}
        ]
    }
    mock_feedback_reporting_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_dashboard_data, raise_for_status=lambda: None
    )

    response = client.get("/progress/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["athlete_id"] == user_id
    assert data["total_sessions"] == 5
    assert data["personal_best_height_cm"] == 200.0
    assert data["progress_chart_data"]["labels"] == ["2023-01-01", "2023-01-15"]
    mock_feedback_reporting_client.get.assert_called_once_with(f"/reports/athlete/{user_id}/progress?user_id={user_id}")

@pytest.mark.asyncio
async def test_get_coach_athlete_reports_success(mock_current_user, mock_feedback_reporting_client):
    coach_id = mock_current_user.id # This is 1 from mock_current_user fixture
    athlete_id = 10 # This is the athlete_id that the mock authorization allows
    now_str = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'
    mock_report_data = {
        "coach_id": coach_id,
        "athlete_id": athlete_id,
        "report_date": now_str,
        "athlete_summary": "Summary for athlete 10",
        "performance_trends": {"recent_performance": "Excellent consistency"},
        "custom_notes": "Great work!",
        "recommended_drills_for_athlete": ["Advanced drills"]
    }
    mock_feedback_reporting_client.get.return_value = AsyncMock(
        status_code=200, json=lambda: mock_report_data, raise_for_status=lambda: None
    )

    response = client.get(f"/coach/athletes/{athlete_id}/reports")
    assert response.status_code == 200
    data = response.json()
    assert data["coach_id"] == coach_id
    assert data["athlete_id"] == athlete_id
    assert "Summary for athlete 10" in data["athlete_summary"]
    mock_feedback_reporting_client.get.assert_called_once_with(f"/reports/coach/{coach_id}/athletes/{athlete_id}?user_id={coach_id}")

@pytest.mark.asyncio
async def test_get_coach_athlete_reports_unauthorized(mock_current_user, mock_feedback_reporting_client):
    # Simulate current_user is not the authorized coach (id=1) for athlete_id=10
    mock_current_user.id = 2 # Unauthorized user ID
    athlete_id = 10

    # The API Gateway's new authorization check should block this request.
    response = client.get(f"/coach/athletes/{athlete_id}/reports")
    assert response.status_code == 403
    assert "Not authorized to view this athlete's report (placeholder relationship check)" in response.json()["detail"]
    # The internal service should not be called in this case
    mock_feedback_reporting_client.get.assert_not_called()

@pytest.mark.asyncio
async def test_get_coach_athlete_reports_unauthorized_athlete_id(mock_current_user, mock_feedback_reporting_client):
    # Simulate current_user is the authorized coach (id=1) but requests an unauthorized athlete_id (not 10)
    mock_current_user.id = 1 # Authorized coach ID
    athlete_id = 11 # Unauthorized athlete ID for this coach

    # The API Gateway's new authorization check should block this request.
    response = client.get(f"/coach/athletes/{athlete_id}/reports")
    assert response.status_code == 403
    assert "Not authorized to view this athlete's report (placeholder relationship check)" in response.json()["detail"]
    # The internal service should not be called in this case
    mock_feedback_reporting_client.get.assert_not_called()
