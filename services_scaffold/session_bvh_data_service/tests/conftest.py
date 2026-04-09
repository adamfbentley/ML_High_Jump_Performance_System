import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from session_bvh_data_service.database import Base, get_db
from session_bvh_data_service.main import app
from session_bvh_data_service import models, crud, schemas
from datetime import datetime

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(name="db_session")
def db_session_fixture():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="client")
def client_fixture(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def test_user_id():
    return 1

@pytest.fixture
def test_session(db_session, test_user_id):
    session_create = schemas.SessionCreate(
        user_id=test_user_id,
        raw_video_s3_key="test/video/key.mp4",
        session_date=datetime.now(),
        session_type=models.SessionTypeEnum.TRAINING,
        notes="Test session notes"
    )
    return crud.create_session(db_session, session_create)

@pytest.fixture
def test_attempt(db_session, test_session, test_user_id):
    attempt_create = schemas.AttemptCreate(
        session_id=test_session.id,
        attempt_number=1,
        bar_height_cm=200.0,
        outcome=models.AttemptOutcomeEnum.SUCCESS,
        bvh_file_s3_key="test/bvh/key.bvh"
    )
    return crud.create_attempt(db_session, attempt_create)

@pytest.fixture
def test_biomechanical_analysis(db_session, test_attempt, test_user_id):
    analysis_create = schemas.BiomechanicalAnalysisCreate(
        attempt_id=test_attempt.id,
        user_id=test_user_id,
        parameters_data={
            "joint_angles": [{"time": 0.1, "hip": 90}, {"time": 0.2, "hip": 85}],
            "velocities": [{"time": 0.1, "hip_vel": 10}, {"time": 0.2, "hip_vel": 12}]
        }
    )
    return crud.create_biomechanical_analysis(db_session, analysis_create)

@pytest.fixture
def test_anomaly_report(db_session, test_attempt, test_user_id):
    report_create = schemas.AnomalyReportCreate(
        attempt_id=test_attempt.id,
        user_id=test_user_id,
        report_data={
            "anomalies_detected": [
                {"joint": "knee", "deviation": "high", "score": 0.9},
                {"joint": "ankle", "deviation": "low", "score": 0.7}
            ],
            "overall_deviation_score": 0.85,
            "fatigue_pattern_detected": False
        }
    )
    return crud.create_anomaly_report(db_session, report_create)
