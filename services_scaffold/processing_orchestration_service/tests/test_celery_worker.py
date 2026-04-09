import pytest
from unittest.mock import patch, AsyncMock
from processing_orchestration_service.celery_worker import (
    trigger_video_processing_task,
    trigger_pose_estimation_task,
    trigger_pinn_gnn_inference_task,
    trigger_anomaly_detection_task
)
from processing_orchestration_service import schemas
from session_bvh_data_service.models import AttemptOutcomeEnum # Import for enum access
import httpx
import json

# Mock Celery's delay method for chaining tasks
@pytest.fixture(autouse=True)
def mock_celery_delay():
    with patch('processing_orchestration_service.celery_worker.trigger_pose_estimation_task.delay') as mock_pose_delay,
         patch('processing_orchestration_service.celery_worker.trigger_pinn_gnn_inference_task.delay') as mock_pinn_delay,
         patch('processing_orchestration_service.celery_worker.trigger_anomaly_detection_task.delay') as mock_anomaly_delay:
        yield mock_pose_delay, mock_pinn_delay, mock_anomaly_delay

# Mock httpx.AsyncClient for internal service calls
@pytest.fixture(autouse=True)
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        yield mock_client

@pytest.mark.asyncio
async def test_trigger_video_processing_task_success(mock_celery_delay, mock_httpx_client):
    mock_pose_delay, _, _ = mock_celery_delay

    # Mock Session & BVH Data Service call to create attempt
    # Simulate multiple attempts being created
    mock_httpx_client.post.side_effect = [
        AsyncMock(return_value=httpx.Response(201, request=httpx.Request("POST", "http://test"), json={
            "id": 1, "session_id": 101, "attempt_number": 1, "bar_height_cm": 200.0, "outcome": "SUCCESS", "start_time_ms": 1000, "end_time_ms": 5000
        })),
        AsyncMock(return_value=httpx.Response(201, request=httpx.Request("POST", "http://test"), json={
            "id": 2, "session_id": 101, "attempt_number": 2, "bar_height_cm": 210.0, "outcome": "FAIL", "start_time_ms": 7000, "end_time_ms": 11000
        }))
    ]
    mock_httpx_client.post.return_value.raise_for_status.return_value = None

    request_data = {
        "video_id": 100,
        "user_id": 1,
        "session_id": 101,
        "raw_video_s3_key": "s3://bucket/raw_video.mp4"
    }
    result = trigger_video_processing_task(request_data)

    assert result["status"] == "SUCCESS"
    assert "Video processing initiated" in result["message"]
    assert "Created 2 attempts" in result["message"]

    # Verify two calls to create attempt endpoint
    assert mock_httpx_client.post.call_count == 2
    
    # Verify two calls to trigger_pose_estimation_task.delay
    assert mock_pose_delay.call_count == 2

    # Check first call to create attempt
    first_post_call_args, first_post_call_kwargs = mock_httpx_client.post.call_args_list[0]
    assert first_post_call_args[0] == 'http://session_bvh_data_service:8003/internal/sessions/101/attempts?user_id=1'
    assert 'attempt_number' in first_post_call_kwargs['json']
    assert 'bar_height_cm' in first_post_call_kwargs['json']
    assert 'outcome' in first_post_call_kwargs['json']
    assert 'start_time_ms' in first_post_call_kwargs['json']
    assert 'end_time_ms' in first_post_call_kwargs['json']

    # Check first call to pose estimation task
    first_pose_delay_call_args, _ = mock_pose_delay.call_args_list[0]
    assert first_pose_delay_call_args[0]['attempt_id'] == 1
    assert first_pose_delay_call_args[0]['user_id'] == 1
    assert first_pose_delay_call_args[0]['video_s3_key'] == "s3://bucket/raw_video.mp4"
    assert first_pose_delay_call_args[0]['start_time_ms'] == 1000 # Now dynamic
    assert first_pose_delay_call_args[0]['end_time_ms'] == 5000   # Now dynamic

    # Check second call to pose estimation task
    second_pose_delay_call_args, _ = mock_pose_delay.call_args_list[1]
    assert second_pose_delay_call_args[0]['attempt_id'] == 2
    assert second_pose_delay_call_args[0]['user_id'] == 1
    assert second_pose_delay_call_args[0]['video_s3_key'] == "s3://bucket/raw_video.mp4"
    assert second_pose_delay_call_args[0]['start_time_ms'] == 7000 # Now dynamic
    assert second_pose_delay_call_args[0]['end_time_ms'] == 11000  # Now dynamic

@pytest.mark.asyncio
async def test_trigger_video_processing_task_no_attempts_simulated(mock_celery_delay, mock_httpx_client):
    mock_pose_delay, _, _ = mock_celery_delay

    # Simulate a very short video that yields no attempts
    with patch('processing_orchestration_service.celery_worker.random.randint', side_effect=[1, 1, 100, 100, 100, 100]): # Forces short durations
        request_data = {
            "video_id": 100,
            "user_id": 1,
            "session_id": 101,
            "raw_video_s3_key": "s3://bucket/raw_video.mp4"
        }
        result = trigger_video_processing_task(request_data)

        assert result["status"] == "SKIPPED"
        assert "No attempts detected/simulated" in result["message"]
        mock_httpx_client.post.assert_not_called()
        mock_pose_delay.assert_not_called()

@pytest.mark.asyncio
async def test_trigger_pose_estimation_task_success(mock_celery_delay, mock_httpx_client):
    _, mock_pinn_delay, _ = mock_celery_delay

    # Mock Pose Estimation Service call
    mock_httpx_client.post.return_value.json.return_value = {"status": "COMPLETED", "bvh_file_s3_key": "s3://bvh/file.bvh"}
    mock_httpx_client.post.return_value.raise_for_status.return_value = None

    request_data = schemas.TriggerPoseEstimationRequest(
        attempt_id=1,
        user_id=1,
        video_s3_key="s3://bucket/raw_video.mp4",
        start_time_ms=0,
        end_time_ms=10000
    ).dict()
    result = trigger_pose_estimation_task(request_data)

    assert result["status"] == "SUCCESS"
    assert "Pose estimation completed" in result["message"]

    mock_httpx_client.post.assert_called_once_with(
        'http://pose_estimation_service:8006/internal/pose-estimate',
        json=request_data
    )
    mock_pinn_delay.assert_called_once()
    called_args, _ = mock_pinn_delay.call_args
    assert called_args[0]['attempt_id'] == 1
    assert called_args[0]['user_id'] == 1

@pytest.mark.asyncio
async def test_trigger_pinn_gnn_inference_task_success(mock_celery_delay, mock_httpx_client):
    _, _, mock_anomaly_delay = mock_celery_delay

    # Mock PINN & GNN Inference Service call
    mock_httpx_client.post.return_value.json.return_value = {"status": "COMPLETED", "biomechanical_parameters_s3_key": "s3://params/file.json"}
    mock_httpx_client.post.return_value.raise_for_status.return_value = None

    request_data = schemas.TriggerBiomechanicsAnalysisRequest(
        attempt_id=1,
        user_id=1
    ).dict()
    result = trigger_pinn_gnn_inference_task(request_data)

    assert result["status"] == "SUCCESS"
    assert "Biomechanics analysis completed" in result["message"]

    mock_httpx_client.post.assert_called_once_with(
        'http://pinn_gnn_inference_service:8008/internal/analyze/biomechanics',
        json=request_data
    )
    mock_anomaly_delay.assert_called_once()
    called_args, _ = mock_anomaly_delay.call_args
    assert called_args[0]['attempt_id'] == 1
    assert called_args[0]['user_id'] == 1

@pytest.mark.asyncio
async def test_trigger_anomaly_detection_task_success(mock_celery_delay, mock_httpx_client):
    _, _, _ = mock_celery_delay # No further tasks to delay

    # Mock Anomaly Detection Service call
    mock_httpx_client.post.return_value.json.return_value = {"status": "COMPLETED", "anomalies_detected": True}
    mock_httpx_client.post.return_value.raise_for_status.return_value = None

    request_data = schemas.TriggerAnomalyDetectionRequest(
        attempt_id=1,
        user_id=1
    ).dict()
    result = trigger_anomaly_detection_task(request_data)

    assert result["status"] == "SUCCESS"
    assert "Anomaly detection completed" in result["message"]

    mock_httpx_client.post.assert_called_once_with(
        'http://anomaly_detection_service:8009/internal/detect/anomalies',
        json=request_data
    )

@pytest.mark.asyncio
async def test_trigger_video_processing_task_http_error(mock_celery_delay, mock_httpx_client):
    mock_httpx_client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("POST", "http://test"), response=httpx.Response(400, request=httpx.Request("POST", "http://test"), content=b'{"detail":"Error"}')
    )

    request_data = {"video_id": 100, "user_id": 1, "session_id": 101, "raw_video_s3_key": "s3://bucket/raw_video.mp4"}
    with pytest.raises(httpx.HTTPStatusError):
        trigger_video_processing_task(request_data)

@pytest.mark.asyncio
async def test_trigger_pose_estimation_task_http_error(mock_celery_delay, mock_httpx_client):
    mock_httpx_client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=httpx.Request("POST", "http://test"), response=httpx.Response(500, request=httpx.Request("POST", "http://test"), content=b'{"detail":"Error"}')
    )

    request_data = schemas.TriggerPoseEstimationRequest(
        attempt_id=1,
        user_id=1,
        video_s3_key="s3://bucket/raw_video.mp4",
        start_time_ms=0,
        end_time_ms=10000
    ).dict()
    with pytest.raises(httpx.HTTPStatusError):
        trigger_pose_estimation_task(request_data)

@pytest.mark.asyncio
async def test_trigger_pinn_gnn_inference_task_http_error(mock_celery_delay, mock_httpx_client):
    mock_httpx_client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=httpx.Request("POST", "http://test"), response=httpx.Response(500, request=httpx.Request("POST", "http://test"), content=b'{"detail":"Error"}')
    )

    request_data = schemas.TriggerBiomechanicsAnalysisRequest(
        attempt_id=1,
        user_id=1
    ).dict()
    with pytest.raises(httpx.HTTPStatusError):
        trigger_pinn_gnn_inference_task(request_data)

@pytest.mark.asyncio
async def test_trigger_anomaly_detection_task_http_error(mock_celery_delay, mock_httpx_client):
    mock_httpx_client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=httpx.Request("POST", "http://test"), response=httpx.Response(500, request=httpx.Request("POST", "http://test"), content=b'{"detail":"Error"}')
    )

    request_data = schemas.TriggerAnomalyDetectionRequest(
        attempt_id=1,
        user_id=1
    ).dict()
    with pytest.raises(httpx.HTTPStatusError):
        trigger_anomaly_detection_task(request_data)
