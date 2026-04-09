import pytest
from fastapi.testclient import TestClient
from optimization_engine_service.main import app
from unittest.mock import AsyncMock, patch
import httpx

@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_async_client:
        mock_instance = AsyncMock(spec=httpx.AsyncClient)
        mock_async_client.return_value.__aenter__.return_value = mock_instance
        yield mock_instance
