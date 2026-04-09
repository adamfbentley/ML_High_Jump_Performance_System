from fastapi.testclient import TestClient
from ai_model_training_service.main import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai_model_training_service"}

@patch('ai_model_training_service.celery_worker.train_personal_model_task.delay')
def test_trigger_personal_model_training(mock_delay):
    athlete_id = 1
    request_payload = {"retrain_epochs": 50, "learning_rate": 0.0005}
    mock_delay.return_value.id = "mock_task_id_123"

    response = client.post(f"/internal/train/personal-model/{athlete_id}", json=request_payload)

    assert response.status_code == 202
    response_json = response.json()
    assert response_json["status"] == "training_initiated"
    assert f"Personal model training initiated for athlete {athlete_id}. Task ID: mock_task_id_123" in response_json["message"]
    assert response_json["athlete_id"] == athlete_id
    assert response_json["model_version"] is None
    assert response_json["model_artifact_s3_path"] is None
    mock_delay.assert_called_once_with(athlete_id=athlete_id, retrain_epochs=50, learning_rate=0.0005)

@patch('ai_model_training_service.celery_worker.train_personal_model_task.delay')
def test_trigger_personal_model_training_default_params(mock_delay):
    athlete_id = 2
    mock_delay.return_value.id = "mock_task_id_456"

    response = client.post(f"/internal/train/personal-model/{athlete_id}")

    assert response.status_code == 202
    response_json = response.json()
    assert response_json["status"] == "training_initiated"
    assert response_json["athlete_id"] == athlete_id
    mock_delay.assert_called_once_with(athlete_id=athlete_id, retrain_epochs=100, learning_rate=0.001)
