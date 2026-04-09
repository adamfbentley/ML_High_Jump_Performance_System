import pytest
from celery import Celery
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

# Mock Celery app for testing
@pytest.fixture(scope='session')
def celery_app():
    app = Celery('test_celery_worker', broker='memory://', backend='memory://')
    app.conf.update(TASK_ALWAYS_EAGER=True, TASK_EAGER_PROPAGATES=True)
    return app

@pytest.fixture(autouse=True)
def mock_celery_imports(celery_app):
    # Patch the Celery app in the worker module to use our test app
    with patch('processing_orchestration_service.celery_worker.celery_app', new=celery_app):
        yield

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_instance = AsyncMock(spec=httpx.AsyncClient)
        mock_client_class.return_value.__aenter__.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_pose_estimation_service_response():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"bvh_s3_key": "mock/bvh/key.bvh"}
    mock_response.raise_for_status = MagicMock()
    return mock_response

@pytest.fixture
def mock_pinn_gnn_inference_service_response():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 201
    mock_response.json.return_value = {"attempt_id": 1, "parameters_data": {"dummy": "data"}}
    mock_response.raise_for_status = MagicMock()
    return mock_response

@pytest.fixture
def mock_anomaly_detection_service_response():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 201
    mock_response.json.return_value = {"attempt_id": 1, "report_data": {"anomalies_detected": True}}
    mock_response.raise_for_status = MagicMock()
    return mock_response
