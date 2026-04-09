import pytest
from fastapi.testclient import TestClient
from api_gateway.main import app
from unittest.mock import AsyncMock, patch
from api_gateway import schemas

@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_current_user():
    user = schemas.UserResponse(id=1, email="test@example.com", is_active=True)
    with patch("api_gateway.auth.get_current_user", return_value=user):
        yield user

@pytest.fixture
def mock_session_bvh_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch("api_gateway.dependencies.get_session_bvh_data_service_client", return_value=mock_client):
        yield mock_client

@pytest.fixture
def mock_video_ingestion_service_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch('api_gateway.dependencies.get_video_ingestion_service_client') as mock_dependency:
        mock_dependency.return_value.__aenter__.return_value = mock_client
        yield mock_client

# Mock the Optimization Engine Service client
@pytest.fixture
def mock_optimization_engine_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch("api_gateway.dependencies.get_optimization_engine_service_client", return_value=mock_client):
        yield mock_client
