from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from user_profile_service import crud, schemas, models
import pytest
from datetime import datetime, timezone

def test_health_check(client: TestClient):
    response = client.get("/internal/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "user_profile_service"}

def test_create_user_endpoint(client: TestClient, test_db_session: Session):
    # Test successful creation
    response = client.post("/internal/users", json={"email": "newuser@example.com", "password": "securepassword"})
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data
    assert data["is_active"] is True

    # Test duplicate email
    response = client.post("/internal/users", json={"email": "newuser@example.com", "password": "anotherpassword"})
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]

def test_read_user_endpoint(client: TestClient, test_db_session: Session):
    user_create = schemas.UserCreate(email="readuser@example.com", password="password123")
    crud.create_user(test_db_session, user_create)

    response = client.get(f"/internal/users/{created_user.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created_user.id
    assert data["email"] == created_user.email

    # Test user not found
    response = client.get("/internal/users/999")
    assert response.status_code == 404

def test_verify_user_credentials_endpoint(client: TestClient, test_db_session: Session):
    user_create = schemas.UserCreate(email="verify@example.com", password="verifypass")
    crud.create_user(test_db_session, user_create)

    # Test successful verification
    response = client.post("/internal/auth/verify", json={"email": "verify@example.com", "password": "verifypass"})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "verify@example.com"
    assert "id" in data

    # Test incorrect password
    response = client.post("/internal/auth/verify", json={"email": "verify@example.com", "password": "wrongpass"})
    assert response.status_code == 401

    # Test non-existent email
    response = client.post("/internal/auth/verify", json={"email": "nonexistent@example.com", "password": "anypass"})
    assert response.status_code == 401

def test_read_user_profile_endpoint(client: TestClient, test_db_session: Session):
    user_create = schemas.UserCreate(email="profile_read@example.com", password="password123")
    created_user = crud.create_user(test_db_session, user_create)

    response = client.get(f"/internal/profiles/{created_user.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == created_user.id
    assert data["first_name"] is None # Initially empty

    # Test profile not found (should not happen if user creation creates profile)
    response = client.get("/internal/profiles/999")
    assert response.status_code == 404

def test_update_user_profile_endpoint(client: TestClient, test_db_session: Session):
    user_create = schemas.UserCreate(email="profile_update@example.com", password="password123")
    created_user = crud.create_user(test_db_session, user_create)

    update_payload = {
        "first_name": "Jane",
        "last_name": "Doe",
        "height_cm": 165.0,
        "gender": "FEMALE",
        "injury_status": "Healthy",
        "injury_date": None,
        "recovery_date": None
    }
    response = client.put(f"/internal/profiles/{created_user.id}", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["first_name"] == "Jane"
    assert data["height_cm"] == 165.0
    assert data["gender"] == "FEMALE"
    assert data["injury_status"] == "Healthy"

    # Test partial update
    partial_update_payload = {"weight_kg": 60.0, "injury_status": "Minor ankle sprain", "injury_date": datetime.now(timezone.utc).isoformat()}
    response = client.put(f"/internal/profiles/{created_user.id}", json=partial_update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["first_name"] == "Jane" # Should remain
    assert data["weight_kg"] == 60.0
    assert data["injury_status"] == "Minor ankle sprain"
    assert data["injury_date"] is not None

    # Test profile not found
    response = client.put("/internal/profiles/999", json=update_payload)
    assert response.status_code == 404
