from sqlalchemy.orm import Session
from sqlalchemy import desc
from ai_model_training_service import models, schemas
from typing import List, Optional, Dict, Any

def create_population_model_version(db: Session, model_create: schemas.PopulationModelCreate) -> models.PopulationModelVersion:
    db_model = models.PopulationModelVersion(**model_create.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_latest_population_model_version(db: Session) -> Optional[models.PopulationModelVersion]:
    return db.query(models.PopulationModelVersion).order_by(desc(models.PopulationModelVersion.trained_at)).first()

def create_personal_model_version(db: Session, model_create: schemas.PersonalModelCreate) -> models.PersonalModelVersion:
    db_model = models.PersonalModelVersion(**model_create.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_latest_personal_model_version(db: Session, athlete_id: int) -> Optional[models.PersonalModelVersion]:
    return db.query(models.PersonalModelVersion).filter(models.PersonalModelVersion.athlete_id == athlete_id).order_by(desc(models.PersonalModelVersion.trained_at)).first()

def create_population_cohort(db: Session, cohort_create: schemas.PopulationCohortCreate) -> models.PopulationCohort:
    db_cohort = models.PopulationCohort(**cohort_create.dict())
    db.add(db_cohort)
    db.commit()
    db.refresh(db_cohort)
    return db_cohort

def get_population_cohort_by_name(db: Session, name: str) -> Optional[models.PopulationCohort]:
    return db.query(models.PopulationCohort).filter(models.PopulationCohort.name == name).first()

def get_all_population_cohorts(db: Session) -> List[models.PopulationCohort]:
    return db.query(models.PopulationCohort).all()

def update_population_model_metrics(db: Session, model_id: int, metrics: Dict[str, Any], s3_path: str, dataset_size: int) -> Optional[models.PopulationModelVersion]:
    db_model = db.query(models.PopulationModelVersion).filter(models.PopulationModelVersion.id == model_id).first()
    if db_model:
        db_model.metrics = metrics
        db_model.s3_path = s3_path
        db_model.dataset_size = dataset_size
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
    return db_model

def update_personal_model_results(db: Session, model_id: int, metrics: Dict[str, Any], s3_path: str, optimal_params: Dict[str, Any]) -> Optional[models.PersonalModelVersion]:
    db_model = db.query(models.PersonalModelVersion).filter(models.PersonalModelVersion.id == model_id).first()
    if db_model:
        db_model.metrics = metrics
        db_model.s3_path = s3_path
        db_model.optimal_technique_parameters = optimal_params
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
    return db_model
