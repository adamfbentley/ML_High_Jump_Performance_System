from fastapi.testclient import TestClient
from api_gateway.main import app
from unittest.mock import patch, MagicMock
from api_gateway import auth
import httpx

client = TestClient(app)

# Mock the authentication dependency
def override_get_current_user():
    return auth.schemas.UserResponse(id=1, email="test@example.com", is_active=True)

app.dependency_overrides[auth.get_current_user] = override_get_current_user

@patch('httpx.AsyncClient.post')
async def test_trigger_personal_model_training_endpoint(mock_post):
    athlete_id = 1
    request_payload = {"retrain_epochs": 50, "learning_rate": 0.0005}
    mock_response_data = {
        "status": "training_initiated",
        "message": f"Personal model training initiated for athlete {athlete_id}. Task ID: mock_task_id_123",
        "athlete_id": athlete_id,
        "model_version": None,
        "model_artifact_s3_path": None
    }
    mock_post.return_value = httpx.Response(202, json=mock_response_data)

    response = client.post(f"/models/personal/train/{athlete_id}", json=request_payload)

    assert response.status_code == 202
    assert response.json() == mock_response_data
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"/train/personal-model/{athlete_id}"
    assert mock_post.call_args[1]['json'] == request_payload

@patch('httpx.AsyncClient.get')
async def test_get_personal_optimum_technique(mock_get):
    athlete_id = 1
    mock_response_data = {
        "athlete_id": athlete_id,
        "optimal_parameters": {"takeoff_angle_deg": 20.0, "approach_speed_mps": 9.5},
        "target_height_cm": 200.0,
        "message": f"Optimal technique parameters generated for athlete {athlete_id}."
    }
    mock_get.return_value = httpx.Response(200, json=mock_response_data)

    response = client.get("/models/personal/optimum")

    assert response.status_code == 200
    assert response.json() == mock_response_data
    mock_get.assert_called_once_with(f"/optimize/{athlete_id}")

@patch('httpx.AsyncClient.get')
async def test_get_personal_optimum_sensitivity(mock_get):
    athlete_id = 1
    mock_response_data = {
        "athlete_id": athlete_id,
        "sensitivity_results": {"takeoff_angle_impact": "High"},
        "message": f"Sensitivity analysis results generated for athlete {athlete_id}."
    }
    mock_get.return_value = httpx.Response(200, json=mock_response_data)

    response = client.get("/models/personal/optimum/sensitivity")

    assert response.status_code == 200
    assert response.json() == mock_response_data
    mock_get.assert_called_once_with(f"/optimize/{athlete_id}/sensitivity")

def test_trigger_personal_model_training_unauthorized():
    # User ID in token (1) does not match athlete_id (2)
    response = client.post("/models/personal/train/2", json={})
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to train this personal model."
