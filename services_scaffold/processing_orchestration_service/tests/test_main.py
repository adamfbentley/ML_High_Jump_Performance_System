from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import pytest
from processing_orchestration_service.schemas import ProcessVideoRequest, SegmentAttemptRequest, ProcessExistingSessionVideoRequest
from session_bvh_data_service.models import SessionTypeEnum

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "processing_orchestration_service"}

def test_trigger_video_processing(client: TestClient, mock_process_video_pipeline_task: MagicMock):
    video_id = "test_video_abc"
    user_id = 123
    s3_key = "uploads/user123/test_video_abc.mp4"
    session_type = SessionTypeEnum.COMPETITION
    resolution = "1920x1080"
    frame_rate = 60.0
    duration_ms = 15000

    request_payload = {
        "user_id": user_id,
        "raw_video_s3_key": s3_key,
        "session_type": session_type.value,
        "resolution": resolution,
        "frame_rate": frame_rate,
        "duration_ms": duration_ms
    }

    response = client.post(f"/internal/process-video/{video_id}", json=request_payload)
    assert response.status_code == 202
    assert response.json()["status"] == "processing_initiated"
    assert response.json()["video_id"] == video_id

    mock_process_video_pipeline_task.delay.assert_called_once_with(
        video_id=video_id,
        user_id=user_id,
        raw_video_s3_key=s3_key,
        session_type=session_type.value,
        resolution=resolution,
        frame_rate=frame_rate,
        duration_ms=duration_ms
    )

    # Test with default session_type and optional fields as None
    request_payload_default = {
        "user_id": user_id,
        "raw_video_s3_key": s3_key,
    }
    response_default = client.post(f"/internal/process-video/{video_id}_default", json=request_payload_default)
    assert response_default.status_code == 202
    mock_process_video_pipeline_task.delay.assert_called_with(
        video_id=f"{video_id}_default",
        user_id=user_id,
        raw_video_s3_key=s3_key,
        session_type=SessionTypeEnum.TRAINING.value,
        resolution=None,
        frame_rate=None,
        duration_ms=None
    )

def test_trigger_processing_for_live_session_endpoint(client: TestClient, mock_trigger_processing_for_live_session_task: MagicMock):
    session_id = 1
    user_id = 123
    raw_video_s3_key = "live_uploads/user123/session1.mp4"
    resolution = "1280x720"
    frame_rate = 30.0
    duration_ms = 10000

    request_payload = {
        "user_id": user_id,
        "raw_video_s3_key": raw_video_s3_key,
        "resolution": resolution,
        "frame_rate": frame_rate,
        "duration_ms": duration_ms
    }

    response = client.post(f"/internal/process-existing-session-video/{session_id}", json=request_payload)
    assert response.status_code == 202
    assert response.json()["status"] == "processing_initiated"
    assert response.json()["video_id"] == str(session_id)

    mock_trigger_processing_for_live_session_task.delay.assert_called_once_with(
        session_id=session_id,
        user_id=user_id,
        raw_video_s3_key=raw_video_s3_key,
        resolution=resolution,
        frame_rate=frame_rate,
        duration_ms=duration_ms
    )

def test_trigger_attempt_segmentation(client: TestClient, mock_segment_attempt_task: MagicMock):
    attempt_id = 456
    user_id = 123
    start_time = 1000
    end_time = 2500

    request_payload = {
        "user_id": user_id,
        "start_time_ms": start_time,
        "end_time_ms": end_time
    }

    response = client.post(f"/internal/segment-attempt/{attempt_id}", json=request_payload)
    assert response.status_code == 202
    assert "segmentation update acknowledged" in response.json()["message"]

    mock_segment_attempt_task.delay.assert_called_once_with(
        attempt_id=attempt_id,
        user_id=user_id,
        start_time_ms=start_time,
        end_time_ms=end_time
    )
