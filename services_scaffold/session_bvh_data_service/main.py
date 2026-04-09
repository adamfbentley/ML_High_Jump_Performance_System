from fastapi import FastAPI, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from session_bvh_data_service import models, schemas, crud
from session_bvh_data_service.database import engine, get_db

app = FastAPI(
    title="Session & BVH Data Service",
    description="Manages session metadata, individual jump attempt details, and stores references to BVH files.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.on_event("startup")
def on_startup():
    # CQ-008: Removed informational comment about database schema migrations.
    pass

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/sessions", response_model=schemas.SessionResponseWithRawVideoS3Key, status_code=status.HTTP_201_CREATED)
def create_session_endpoint(session: schemas.SessionCreate, db: Session = Depends(get_db)):
    # If raw_video_s3_key is provided, check for uniqueness for that user.
    # If raw_video_s3_key is None (e.g., for live sessions), allow multiple sessions with None.
    if session.raw_video_s3_key:
        existing_session = db.query(models.Session).filter(
            models.Session.raw_video_s3_key == session.raw_video_s3_key,
            models.Session.user_id == session.user_id
        ).first()
        if existing_session:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session with this raw video S3 key already exists for this user.")
    return crud.create_session(db=db, session_create=session)

@app.get("/internal/sessions/{session_id}", response_model=schemas.SessionResponseWithRawVideoS3Key)
def read_session_endpoint(session_id: int, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_session = crud.get_session_by_id(db, session_id=session_id, user_id=user_id)
    if db_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not owned by user")
    return db_session

@app.get("/internal/users/{user_id}/sessions", response_model=List[schemas.SessionResponseWithRawVideoS3Key])
def read_user_sessions_endpoint(user_id: int, db: Session = Depends(get_db)):
    sessions = crud.get_sessions_by_user_id(db, user_id=user_id)
    return sessions

@app.get("/internal/sessions/by-attempt/{attempt_id}", response_model=schemas.SessionResponseWithRawVideoS3Key)
def read_session_by_attempt_endpoint(attempt_id: int, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_session = crud.get_session_by_attempt_id(db, attempt_id=attempt_id, user_id=user_id)
    if db_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found for this attempt or not owned by user")
    return db_session

@app.post("/internal/attempts", response_model=schemas.AttemptResponse, status_code=status.HTTP_201_CREATED)
def create_attempt_endpoint(attempt: schemas.AttemptCreate, db: Session = Depends(get_db)):
    # Ensure the session exists and belongs to the user (implicitly checked by API Gateway)
    # For internal call, we trust the session_id is valid and authorized.
    db_session = db.query(models.Session).filter(models.Session.id == attempt.session_id).first()
    if not db_session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Ensure attempt_number is unique within the session
    existing_attempt = db.query(models.Attempt).filter(
        models.Attempt.session_id == attempt.session_id,
        models.Attempt.attempt_number == attempt.attempt_number
    ).first()
    if existing_attempt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attempt number already exists for this session.")

    return crud.create_attempt(db=db, attempt_create=attempt)

@app.post("/internal/sessions/{session_id}/attempts", response_model=schemas.AttemptResponse, status_code=status.HTTP_201_CREATED)
def create_attempt_for_session_endpoint(
    session_id: int,
    attempt_create_data: schemas.AttemptCreateLive,
    user_id: int = Query(..., description="User ID for authorization"),
    db: Session = Depends(get_db)
):
    """
    Creates a new jump attempt for a specific session, ensuring user ownership.
    """
    db_attempt = crud.create_attempt_for_session(db, session_id, user_id, attempt_create_data)
    if db_attempt is None:
        # The crud function returns None if session not found/owned or attempt_number duplicate
        # We need to distinguish these cases for proper HTTP status codes
        db_session = crud.get_session_by_id(db, session_id, user_id)
        if not db_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not owned by user")
        else:
            # If session exists and is owned, then it must be a duplicate attempt number
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attempt number already exists for this session.")
    return db_attempt

@app.get("/internal/attempts/{attempt_id}", response_model=schemas.AttemptResponse)
def read_attempt_endpoint(attempt_id: int, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_attempt = crud.get_attempt_by_id(db, attempt_id=attempt_id, user_id=user_id)
    if db_attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not owned by user's session")
    return db_attempt

@app.get("/internal/sessions/{session_id}/attempts", response_model=List[schemas.AttemptResponse])
def read_session_attempts_endpoint(session_id: int, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_attempts = crud.get_attempts_by_session_id(db, session_id=session_id, user_id=user_id)
    if not db_attempts and not crud.get_session_by_id(db, session_id, user_id): # Check if session exists and is owned, even if no attempts
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not owned by user")
    return db_attempts

@app.put("/internal/attempts/{attempt_id}/metadata", response_model=schemas.AttemptResponse)
def update_attempt_metadata_endpoint(attempt_id: int, metadata_update: schemas.AttemptUpdateMetadata, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_attempt = crud.update_attempt_metadata(db, attempt_id=attempt_id, user_id=user_id, metadata_update=metadata_update)
    if db_attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not owned by user's session")
    return db_attempt

@app.put("/internal/attempts/{attempt_id}/segmentation", response_model=schemas.AttemptResponse)
def update_attempt_segmentation_endpoint(attempt_id: int, segmentation_update: schemas.AttemptUpdateSegmentation, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_attempt = crud.update_attempt_segmentation(db, attempt_id=attempt_id, user_id=user_id, segmentation_update=segmentation_update)
    if db_attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not owned by user's session")
    return db_attempt

@app.post("/internal/attempts/{attempt_id}/bvh", response_model=schemas.AttemptResponse)
def store_attempt_bvh_endpoint(attempt_id: int, bvh_update: schemas.AttemptUpdateBvhKey, user_id: int = Query(..., description="User ID for authorization"), db: Session = Depends(get_db)):
    db_attempt = crud.update_attempt_bvh_key(db, attempt_id=attempt_id, user_id=user_id, bvh_file_s3_key=bvh_update.bvh_file_s3_key)
    if db_attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not owned by user's session")
    return db_attempt
