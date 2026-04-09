from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from user_profile_service import models, schemas, crud
from user_profile_service.database import engine, get_db
from typing import List

app = FastAPI(
    title="User & Profile Service",
    description="Manages user authentication, authorization, and athlete profiles.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.on_event("startup")
def on_startup():
    # As per QA feedback CQ-002, removed automatic schema creation on startup.
    # Database schema migrations should be handled by dedicated tools like Alembic.
    pass

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/users", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def create_user_endpoint(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.get("/internal/users/{user_id}", response_model=schemas.UserResponse)
def read_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return db_user

@app.post("/internal/auth/verify", response_model=schemas.UserAuthResponseInternal)
def verify_user_credentials(user_login: schemas.UserLoginInternal, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user_login.email)
    if not db_user or not crud.verify_password(user_login.password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    return db_user

@app.get("/internal/profiles/{user_id}", response_model=schemas.UserProfileResponse)
def read_user_profile_endpoint(user_id: int, db: Session = Depends(get_db)):
    db_profile = crud.get_user_profile(db, user_id=user_id)
    if db_profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")
    return db_profile

@app.put("/internal/profiles/{user_id}", response_model=schemas.UserProfileResponse)
def update_user_profile_endpoint(user_id: int, profile_update: schemas.UserProfileUpdate, db: Session = Depends(get_db)):
    db_profile = crud.update_user_profile(db, user_id=user_id, profile_update=profile_update)
    if db_profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_NOT_FOUND, detail="User profile not found")
    return db_profile
