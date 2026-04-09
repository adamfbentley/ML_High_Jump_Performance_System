import pytest
from sqlalchemy.orm import Session
from user_profile_service import crud, schemas, models
from datetime import datetime, timezone

def test_create_user(test_db_session: Session):
    user_create = schemas.UserCreate(email="test@example.com", password="password123")
    user = crud.create_user(test_db_session, user_create)
    assert user.id is not None
    assert user.email == "test@example.com"
    assert hasattr(user, "hashed_password")
    assert user.is_active is True
    assert user.profile is not None
    assert user.profile.user_id == user.id

def test_get_user_by_email(test_db_session: Session):
    user_create = schemas.UserCreate(email="get@example.com", password="password123")
    crud.create_user(test_db_session, user_create)
    user = crud.get_user_by_email(test_db_session, "get@example.com")
    assert user is not None
    assert user.email == "get@example.com"

def test_get_user_by_id(test_db_session: Session):
    user_create = schemas.UserCreate(email="getid@example.com", password="password123")
    created_user = crud.create_user(test_db_session, user_create)
    user = crud.get_user_by_id(test_db_session, created_user.id)
    assert user is not None
    assert user.id == created_user.id

def test_verify_password():
    hashed_password = crud.get_password_hash("testpassword")
    assert crud.verify_password("testpassword", hashed_password)
    assert not crud.verify_password("wrongpassword", hashed_password)

def test_get_user_profile(test_db_session: Session):
    user_create = schemas.UserCreate(email="profile@example.com", password="password123")
    created_user = crud.create_user(test_db_session, user_create)
    profile = crud.get_user_profile(test_db_session, created_user.id)
    assert profile is not None
    assert profile.user_id == created_user.id

def test_update_user_profile(test_db_session: Session):
    user_create = schemas.UserCreate(email="update@example.com", password="password123")
    created_user = crud.create_user(test_db_session, user_create)

    update_data = schemas.UserProfileUpdate(
        first_name="John",
        last_name="Doe",
        height_cm=180.5,
        gender=models.GenderEnum.MALE,
        injury_status="Recovering from knee injury",
        injury_date=datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    updated_profile = crud.update_user_profile(test_db_session, created_user.id, update_data)

    assert updated_profile is not None
    assert updated_profile.first_name == "John"
    assert updated_profile.last_name == "Doe"
    assert updated_profile.height_cm == 180.5
    assert updated_profile.gender == models.GenderEnum.MALE
    assert updated_profile.injury_status == "Recovering from knee injury"
    assert updated_profile.injury_date == datetime(2023, 1, 1, tzinfo=timezone.utc)

    # Test partial update
    partial_update_data = schemas.UserProfileUpdate(
        weight_kg=75.0,
        recovery_date=datetime(2023, 6, 1, tzinfo=timezone.utc)
    )
    updated_profile_partial = crud.update_user_profile(test_db_session, created_user.id, partial_update_data)

    assert updated_profile_partial is not None
    assert updated_profile_partial.first_name == "John" # Should remain unchanged
    assert updated_profile_partial.weight_kg == 75.0
    assert updated_profile_partial.recovery_date == datetime(2023, 6, 1, tzinfo=timezone.utc)

    # Test update for non-existent user
    non_existent_update = crud.update_user_profile(test_db_session, 999, update_data)
    assert non_existent_update is None
