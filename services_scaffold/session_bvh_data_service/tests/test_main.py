from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from session_bvh_data_service import crud, schemas, models
import pytest

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "session_bvh_data_service"}

def test_create_session_endpoint(client: TestClient, test_db_session: Session):
    # Test successful creation
    response = client.post("/internal/sessions", json={"user_id": 1, "raw_video_s3_key": "test_video_1.mp4"})
    assert response.status_code == 201
    data = response.json()
    assert data["user_id"] == 1
    assert data["raw_video_s3_key"] == "test_video_1.mp4"

    # Test duplicate raw_video_s3_key for the same user
    response = client.post("/internal/sessions", json={"user_id": 1, "raw_video_s3_key": "test_video_1.mp4"})
    assert response.status_code == 400
    assert "Session with this raw video S3 key already exists for this user." in response.json()["detail"]

    # Test different user can use same raw_video_s3_key (if not unique globally)
    response = client.post("/internal/sessions", json={"user_id": 2, "raw_video_s3_key": "test_video_1.mp4"})
    assert response.status_code == 201

    # Test session with no raw_video_s3_key (e.g., live session)
    response = client.post("/internal/sessions", json={"user_id": 1, "session_type": "TRAINING"})
    assert response.status_code == 201
    assert response.json()["raw_video_s3_key"] is None

def test_read_session_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="read_video.mp4")
    created_session = crud.create_session(test_db_session, session_create)

    response = client.get(f"/internal/sessions/{created_session.id}?user_id=1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created_session.id
    assert data["user_id"] == 1

    # Test session not found
    response = client.get("/internal/sessions/999?user_id=1")
    assert response.status_code == 404

    # Test unauthorized access
    response = client.get(f"/internal/sessions/{created_session.id}?user_id=2")
    assert response.status_code == 404 # Should be 404 or 403, current crud returns None for unauthorized

def test_read_user_sessions_endpoint(client: TestClient, test_db_session: Session):
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=1, raw_video_s3_key="user1_video1.mp4"))
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=1, raw_video_s3_key="user1_video2.mp4"))
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=2, raw_video_s3_key="user2_video1.mp4"))

    response = client.get("/internal/users/1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(s["user_id"] == 1 for s in data)

def test_read_session_by_attempt_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video_for_attempt.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    response = client.get(f"/internal/sessions/by-attempt/{created_attempt.id}?user_id=1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session.id
    assert data["user_id"] == 1

    # Test attempt not found
    response = client.get("/internal/sessions/by-attempt/999?user_id=1")
    assert response.status_code == 404

    # Test unauthorized access
    response = client.get(f"/internal/sessions/by-attempt/{created_attempt.id}?user_id=2")
    assert response.status_code == 404

def test_create_attempt_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="attempt_video.mp4")
    session = crud.create_session(test_db_session, session_create)

    # Test successful creation
    response = client.post("/internal/attempts", json={"session_id": session.id, "attempt_number": 1, "bar_height_cm": 200.0})
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == session.id
    assert data["attempt_number"] == 1

    # Test duplicate attempt number for same session
    response = client.post("/internal/attempts", json={"session_id": session.id, "attempt_number": 1, "bar_height_cm": 205.0})
    assert response.status_code == 400
    assert "Attempt number already exists for this session." in response.json()["detail"]

    # Test session not found
    response = client.post("/internal/attempts", json={"session_id": 999, "attempt_number": 1})
    assert response.status_code == 404

def test_create_attempt_for_session_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="live_session_video.mp4")
    session = crud.create_session(test_db_session, session_create)

    # Test successful creation
    response = client.post(f"/internal/sessions/{session.id}/attempts?user_id=1", json={"attempt_number": 1, "bar_height_cm": 210.0, "start_time_ms": 1000, "end_time_ms": 5000})
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == session.id
    assert data["attempt_number"] == 1
    assert data["start_time_ms"] == 1000
    assert data["end_time_ms"] == 5000

    # Test duplicate attempt number
    response = client.post(f"/internal/sessions/{session.id}/attempts?user_id=1", json={"attempt_number": 1, "bar_height_cm": 215.0})
    assert response.status_code == 400
    assert "Attempt number already exists for this session." in response.json()["detail"]

    # Test session not found/owned
    response = client.post(f"/internal/sessions/999/attempts?user_id=1", json={"attempt_number": 1})
    assert response.status_code == 404

def test_read_attempt_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="read_attempt_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    response = client.get(f"/internal/attempts/{created_attempt.id}?user_id=1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created_attempt.id
    assert data["session_id"] == session.id

    # Test attempt not found
    response = client.get("/internal/attempts/999?user_id=1")
    assert response.status_code == 404

    # Test unauthorized access
    response = client.get(f"/internal/attempts/{created_attempt.id}?user_id=2")
    assert response.status_code == 404

def test_read_session_attempts_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="read_attempts_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    crud.create_attempt(test_db_session, schemas.AttemptCreate(session_id=session.id, attempt_number=1))
    crud.create_attempt(test_db_session, schemas.AttemptCreate(session_id=session.id, attempt_number=2))

    response = client.get(f"/internal/sessions/{session.id}/attempts?user_id=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(a["session_id"] == session.id for a in data)

    # Test unauthorized access
    response = client.get(f"/internal/sessions/{session.id}/attempts?user_id=2")
    assert response.status_code == 200 # Should return empty list, not 404/403
    assert len(response.json()) == 0

def test_update_attempt_metadata_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="update_meta_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1, bar_height_cm=200.0)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    update_payload = {"bar_height_cm": 205.0, "outcome": "SUCCESS"}
    response = client.put(f"/internal/attempts/{created_attempt.id}/metadata?user_id=1", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["bar_height_cm"] == 205.0
    assert data["outcome"] == "SUCCESS"

    # Test unauthorized update
    response = client.put(f"/internal/attempts/{created_attempt.id}/metadata?user_id=2", json=update_payload)
    assert response.status_code == 404

def test_update_attempt_segmentation_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="update_seg_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    update_payload = {"start_time_ms": 100, "end_time_ms": 500}
    response = client.put(f"/internal/attempts/{created_attempt.id}/segmentation?user_id=1", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["start_time_ms"] == 100
    assert data["end_time_ms"] == 500

    # Test unauthorized update
    response = client.put(f"/internal/attempts/{created_attempt.id}/segmentation?user_id=2", json=update_payload)
    assert response.status_code == 404

def test_update_attempt_bvh_key_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="update_bvh_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    bvh_key = "s3://bucket/path/to/bvh.bvh"
    response = client.post(f"/internal/attempts/{created_attempt.id}/bvh?user_id=1", json={"bvh_file_s3_key": bvh_key})
    assert response.status_code == 200
    data = response.json()
    assert data["bvh_file_s3_key"] == bvh_key

    # Test unauthorized update
    response = client.post(f"/internal/attempts/{created_attempt.id}/bvh?user_id=2", json={"bvh_file_s3_key": "bad_key.bvh"})
    assert response.status_code == 404

def test_update_attempt_biomechanical_parameters_key_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="update_params_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    params_key = "s3://bucket/path/to/params.json"
    response = client.post(f"/internal/attempts/{created_attempt.id}/parameters?user_id=1", json={"biomechanical_parameters_s3_key": params_key})
    assert response.status_code == 200
    data = response.json()
    assert data["biomechanical_parameters_s3_key"] == params_key

    # Test unauthorized update
    response = client.post(f"/internal/attempts/{created_attempt.id}/parameters?user_id=2", json={"biomechanical_parameters_s3_key": "bad_params.json"})
    assert response.status_code == 404

def test_get_attempt_biomechanical_parameters_key_endpoint(client: TestClient, test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="get_params_video.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    # Test not found if key not set
    response = client.get(f"/internal/attempts/{created_attempt.id}/parameters?user_id=1")
    assert response.status_code == 404
    assert "Biomechanical parameters not yet available" in response.json()["detail"]

    # Set the key
    params_key = "s3://bucket/path/to/retrieved_params.json"
    crud.update_attempt_biomechanical_parameters_key(test_db_session, created_attempt.id, 1, params_key)

    # Mock S3 download for the actual endpoint call
    with patch('session_bvh_data_service.s3_utils.download_file_from_s3', new=AsyncMock(return_value=json.dumps({"test_param": 123}).encode('utf-8'))) as mock_download:
        # Test successful retrieval
        response = client.get(f"/internal/attempts/{created_attempt.id}/parameters?user_id=1")
        assert response.status_code == 200
        data = response.json()
        assert data["s3_key"] == params_key
        assert data["parameters"] == {"test_param": 123}
        mock_download.assert_called_once_with("path/to/retrieved_params.json")

    # Test unauthorized access
    response = client.get(f"/internal/attempts/{created_attempt.id}/parameters?user_id=2")
    assert response.status_code == 404
