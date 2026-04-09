from sqlalchemy.orm import Session
from sqlalchemy import and_
from session_bvh_data_service import models, schemas
from typing import List, Optional, Dict, Any

def create_session(db: Session, session_create: schemas.SessionCreate) -> models.Session:
    db_session = models.Session(**session_create.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_session_by_id(db: Session, session_id: int, user_id: int) -> Optional[models.Session]:
    return db.query(models.Session).filter(and_(models.Session.id == session_id, models.Session.user_id == user_id)).first()

def get_sessions_by_user_id(db: Session, user_id: int) -> List[models.Session]:
    return db.query(models.Session).filter(models.Session.user_id == user_id).all()

def get_session_by_attempt_id(db: Session, attempt_id: int, user_id: int) -> Optional[models.Session]:
    return db.query(models.Session).join(models.Attempt).filter(
        and_(models.Attempt.id == attempt_id, models.Session.user_id == user_id)
    ).first()

def create_attempt(db: Session, attempt_create: schemas.AttemptCreate) -> models.Attempt:
    db_attempt = models.Attempt(**attempt_create.dict())
    db.add(db_attempt)
    db.commit()
    db.refresh(db_attempt)
    return db_attempt

def create_attempt_for_session(db: Session, session_id: int, user_id: int, attempt_create_data: schemas.AttemptCreateLive) -> Optional[models.Attempt]:
    # First, verify the session exists and belongs to the user
    db_session = get_session_by_id(db, session_id, user_id)
    if not db_session:
        return None # Session not found or not owned by user

    # Ensure attempt_number is unique within the session
    existing_attempt = db.query(models.Attempt).filter(
        models.Attempt.session_id == session_id,
        models.Attempt.attempt_number == attempt_create_data.attempt_number
    ).first()
    if existing_attempt:
        return None # Attempt number already exists

    # Create the attempt
    db_attempt = models.Attempt(
        session_id=session_id,
        attempt_number=attempt_create_data.attempt_number,
        bar_height_cm=attempt_create_data.bar_height_cm,
        outcome=attempt_create_data.outcome,
        start_time_ms=attempt_create_data.start_time_ms,
        end_time_ms=attempt_create_data.end_time_ms
    )
    db.add(db_attempt)
    db.commit()
    db.refresh(db_attempt)
    return db_attempt

def get_attempt_by_id(db: Session, attempt_id: int, user_id: int) -> Optional[models.Attempt]:
    # Join with Session to check user_id ownership
    return db.query(models.Attempt).join(models.Session).filter(
        and_(models.Attempt.id == attempt_id, models.Session.user_id == user_id)
    ).first()

def get_attempts_by_session_id(db: Session, session_id: int, user_id: int) -> List[models.Attempt]:
    # Join with Session to check user_id ownership
    return db.query(models.Attempt).join(models.Session).filter(
        and_(models.Attempt.session_id == session_id, models.Session.user_id == user_id)
    ).all()

def update_attempt_metadata(db: Session, attempt_id: int, user_id: int, metadata_update: schemas.AttemptUpdateMetadata) -> Optional[models.Attempt]:
    db_attempt = get_attempt_by_id(db, attempt_id, user_id) # Ensures ownership
    if db_attempt:
        update_data = metadata_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_attempt, key, value)
        db.add(db_attempt)
        db.commit()
        db.refresh(db_attempt)
    return db_attempt

def update_attempt_segmentation(db: Session, attempt_id: int, user_id: int, segmentation_update: schemas.AttemptUpdateSegmentation) -> Optional[models.Attempt]:
    db_attempt = get_attempt_by_id(db, attempt_id, user_id) # Ensures ownership
    if db_attempt:
        update_data = segmentation_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_attempt, key, value)
        db.add(db_attempt)
        db.commit()
        db.refresh(db_attempt)
    return db_attempt

def update_attempt_bvh_key(db: Session, attempt_id: int, user_id: int, bvh_file_s3_key: str) -> Optional[models.Attempt]:
    db_attempt = get_attempt_by_id(db, attempt_id, user_id) # Ensures ownership
    if db_attempt:
        db_attempt.bvh_file_s3_key = bvh_file_s3_key
        db.add(db_attempt)
        db.commit()
        db.refresh(db_attempt)
    return db_attempt

def update_attempt_biomechanical_parameters_key(db: Session, attempt_id: int, user_id: int, biomechanical_parameters_s3_key: str) -> Optional[models.Attempt]:
    db_attempt = get_attempt_by_id(db, attempt_id, user_id) # Ensures ownership
    if db_attempt:
        db_attempt.biomechanical_parameters_s3_key = biomechanical_parameters_s3_key
        db.add(db_attempt)
        db.commit()
        db.refresh(db_attempt)
    return db_attempt
