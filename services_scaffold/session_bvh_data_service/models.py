from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from session_bvh_data_service.database import Base
import enum

class SessionTypeEnum(str, enum.Enum):
    TRAINING = "TRAINING"
    COMPETITION = "COMPETITION"
    OTHER = "OTHER"

class AttemptOutcomeEnum(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    KNOCK = "KNOCK"
    DID_NOT_ATTEMPT = "DID_NOT_ATTEMPT"
    UNKNOWN = "UNKNOWN"

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True) # Foreign key to User Service's User.id
    raw_video_s3_key = Column(String, index=True, nullable=True) # Changed to nullable=True, removed unique=True
    session_date = Column(DateTime(timezone=True), server_default=func.now())
    session_type = Column(Enum(SessionTypeEnum), default=SessionTypeEnum.TRAINING)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    attempts = relationship("Attempt", back_populates="session", cascade="all, delete-orphan")

class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    start_time_ms = Column(Integer, nullable=True) # Start time of the jump segment in milliseconds
    end_time_ms = Column(Integer, nullable=True)   # End time of the jump segment in milliseconds
    bar_height_cm = Column(Float, nullable=True)
    outcome = Column(Enum(AttemptOutcomeEnum), default=AttemptOutcomeEnum.UNKNOWN)
    bvh_file_s3_key = Column(String, nullable=True) # Reference to processed BVH file in S3
    biomechanical_parameters_s3_key = Column(String, nullable=True) # NEW: Reference to biomechanical parameters (e.g., JSON) in S3
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    session = relationship("Session", back_populates="attempts")
