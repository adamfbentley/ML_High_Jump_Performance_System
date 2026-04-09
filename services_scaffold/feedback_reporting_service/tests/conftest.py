import pytest
from fastapi.testclient import TestClient
from feedback_reporting_service.main import app
from unittest.mock import AsyncMock, patch
import httpx

@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_session_bvh_data_service_client():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    with patch("feedback_reporting_service.main.get_session_bvh_data_service_client", return_value=mock_client):
        yield mock_client
