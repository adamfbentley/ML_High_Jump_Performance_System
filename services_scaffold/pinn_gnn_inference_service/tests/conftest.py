import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from pinn_gnn_inference_service.main import app, get_session_bvh_data_service_client
from pinn_gnn_inference_service import s3_utils
import httpx

@pytest.fixture(name="mock_session_bvh_client")
def mock_session_bvh_client_fixture():
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        app.dependency_overrides[get_session_bvh_data_service_client] = lambda: mock_client
        yield mock_client
    app.dependency_overrides = {}

@pytest.fixture(name="mock_s3_utils")
def mock_s3_utils_fixture():
    with patch('pinn_gnn_inference_service.s3_utils.upload_file_to_s3', new_callable=AsyncMock) as mock_upload,
         patch('pinn_gnn_inference_service.s3_utils.download_file_from_s3', new_callable=AsyncMock) as mock_download:
        yield mock_upload, mock_download

@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client
