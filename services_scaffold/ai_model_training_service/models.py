from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ai_model_training_service.database import Base

class PopulationModelVersion(Base):
    __tablename__ = "population_model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_number = Column(String, unique=True, nullable=False)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    dataset_size = Column(Integer, nullable=False, default=0)
    metrics = Column(JSONB, nullable=True) # e.g., {"loss": 0.01, "accuracy": 0.99}
    s3_path = Column(String, nullable=True) # S3 path to the trained model artifact
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    personal_models = relationship("PersonalModelVersion", back_populates="population_model")

class PersonalModelVersion(Base):
    __tablename__ = "personal_model_versions"

    id = Column(Integer, primary_key=True, index=True)
    athlete_id = Column(Integer, nullable=False, index=True) # Foreign key to User Service's User.id
    population_model_version_id = Column(Integer, ForeignKey("population_model_versions.id"), nullable=False)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    metrics = Column(JSONB, nullable=True) # e.g., {"loss": 0.005, "fine_tune_epochs": 10}
    s3_path = Column(String, nullable=True) # S3 path to the fine-tuned model artifact (LoRA weights)
    optimal_technique_parameters = Column(JSONB, nullable=True) # e.g., {"takeoff_angle": 45.0, "run_up_speed": 8.5}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    population_model = relationship("PopulationModelVersion", back_populates="personal_models")

class PopulationCohort(Base):
    __tablename__ = "population_cohorts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    criteria = Column(JSONB, nullable=True) # e.g., {"gender": "MALE", "age_range": [18, 25], "height_range": [170, 180]}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
