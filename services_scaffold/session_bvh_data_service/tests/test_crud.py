import pytest
from sqlalchemy.orm import Session
from session_bvh_data_service import crud, schemas, models
from datetime import datetime, timezone

def test_create_session(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video1.mp4", session_type=models.SessionTypeEnum.TRAINING)
    session = crud.create_session(test_db_session, session_create)
    assert session.id is not None
    assert session.user_id == 1
    assert session.raw_video_s3_key == "video1.mp4"

def test_get_session_by_id(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video2.mp4")
    created_session = crud.create_session(test_db_session, session_create)
    session = crud.get_session_by_id(test_db_session, created_session.id, 1)
    assert session is not None
    assert session.id == created_session.id
    assert crud.get_session_by_id(test_db_session, created_session.id, 2) is None # Unauthorized user

def test_get_sessions_by_user_id(test_db_session: Session):
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=1, raw_video_s3_key="video3.mp4"))
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=1, raw_video_s3_key="video4.mp4"))
    crud.create_session(test_db_session, schemas.SessionCreate(user_id=2, raw_video_s3_key="video5.mp4"))
    sessions = crud.get_sessions_by_user_id(test_db_session, 1)
    assert len(sessions) == 2
    assert all(s.user_id == 1 for s in sessions)

def test_create_attempt(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video6.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1, bar_height_cm=200.0)
    attempt = crud.create_attempt(test_db_session, attempt_create)
    assert attempt.id is not None
    assert attempt.session_id == session.id
    assert attempt.attempt_number == 1

def test_create_attempt_for_session(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video_live.mp4")
    session = crud.create_session(test_db_session, session_create)

    attempt_data = schemas.AttemptCreateLive(attempt_number=1, bar_height_cm=210.0, outcome=models.AttemptOutcomeEnum.SUCCESS, start_time_ms=1000, end_time_ms=5000)
    attempt = crud.create_attempt_for_session(test_db_session, session.id, 1, attempt_data)
    assert attempt is not None
    assert attempt.session_id == session.id
    assert attempt.attempt_number == 1
    assert attempt.bar_height_cm == 210.0
    assert attempt.start_time_ms == 1000
    assert attempt.end_time_ms == 5000

    # Test duplicate attempt number
    duplicate_attempt = crud.create_attempt_for_session(test_db_session, session.id, 1, attempt_data)
    assert duplicate_attempt is None

    # Test session not found/owned
    non_existent_attempt = crud.create_attempt_for_session(test_db_session, 999, 1, attempt_data)
    assert non_existent_attempt is None

def test_get_attempt_by_id(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video7.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)
    attempt = crud.get_attempt_by_id(test_db_session, created_attempt.id, 1)
    assert attempt is not None
    assert attempt.id == created_attempt.id
    assert crud.get_attempt_by_id(test_db_session, created_attempt.id, 2) is None # Unauthorized user

def test_get_attempts_by_session_id(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video8.mp4")
    session = crud.create_session(test_db_session, session_create)
    crud.create_attempt(test_db_session, schemas.AttemptCreate(session_id=session.id, attempt_number=1))
    crud.create_attempt(test_db_session, schemas.AttemptCreate(session_id=session.id, attempt_number=2))
    attempts = crud.get_attempts_by_session_id(test_db_session, session.id, 1)
    assert len(attempts) == 2
    assert all(a.session_id == session.id for a in attempts)
    assert crud.get_attempts_by_session_id(test_db_session, session.id, 2) == [] # Unauthorized user

def test_update_attempt_metadata(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video9.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1, bar_height_cm=200.0, outcome=models.AttemptOutcomeEnum.UNKNOWN)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    update_data = schemas.AttemptUpdateMetadata(bar_height_cm=205.0, outcome=models.AttemptOutcomeEnum.SUCCESS)
    updated_attempt = crud.update_attempt_metadata(test_db_session, created_attempt.id, 1, update_data)
    assert updated_attempt is not None
    assert updated_attempt.bar_height_cm == 205.0
    assert updated_attempt.outcome == models.AttemptOutcomeEnum.SUCCESS

    # Test unauthorized update
    unauthorized_update = crud.update_attempt_metadata(test_db_session, created_attempt.id, 2, update_data)
    assert unauthorized_update is None

def test_update_attempt_segmentation(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video10.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    update_data = schemas.AttemptUpdateSegmentation(start_time_ms=1000, end_time_ms=5000)
    updated_attempt = crud.update_attempt_segmentation(test_db_session, created_attempt.id, 1, update_data)
    assert updated_attempt is not None
    assert updated_attempt.start_time_ms == 1000
    assert updated_attempt.end_time_ms == 5000

    # Test unauthorized update
    unauthorized_update = crud.update_attempt_segmentation(test_db_session, created_attempt.id, 2, update_data)
    assert unauthorized_update is None

def test_update_attempt_bvh_key(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video11.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    bvh_key = "path/to/bvh/file.bvh"
    updated_attempt = crud.update_attempt_bvh_key(test_db_session, created_attempt.id, 1, bvh_key)
    assert updated_attempt is not None
    assert updated_attempt.bvh_file_s3_key == bvh_key

    # Test unauthorized update
    unauthorized_update = crud.update_attempt_bvh_key(test_db_session, created_attempt.id, 2, "another/bvh.bvh")
    assert unauthorized_update is None

def test_update_attempt_biomechanical_parameters_key(test_db_session: Session):
    session_create = schemas.SessionCreate(user_id=1, raw_video_s3_key="video12.mp4")
    session = crud.create_session(test_db_session, session_create)
    attempt_create = schemas.AttemptCreate(session_id=session.id, attempt_number=1)
    created_attempt = crud.create_attempt(test_db_session, attempt_create)

    params_key = "path/to/params/file.json"
    updated_attempt = crud.update_attempt_biomechanical_parameters_key(test_db_session, created_attempt.id, 1, params_key)
    assert updated_attempt is not None
    assert updated_attempt.biomechanical_parameters_s3_key == params_key

    # Test unauthorized update
    unauthorized_update = crud.update_attempt_biomechanical_parameters_key(test_db_session, created_attempt.id, 2, "another/params.json")
    assert unauthorized_update is None
