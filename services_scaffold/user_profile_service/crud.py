from sqlalchemy.orm import Session
from user_profile_service import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    # Create an empty profile for the new user
    db_profile = models.UserProfile(user_id=db_user.id)
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_user

def get_user_profile(db: Session, user_id: int):
    return db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()

def update_user_profile(db: Session, user_id: int, profile_update: schemas.UserProfileUpdate):
    db_profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    if db_profile:
        update_data = profile_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_profile, key, value)
        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)
    return db_profile
