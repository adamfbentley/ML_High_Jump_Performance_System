# Microservices Deployment Scaffolding

This directory contains the **microservices architecture scaffolding** — 12+ FastAPI services designed for eventual production deployment. These are separated from the core research code because:

1. **All AI/ML logic is simulated** (mock data, `time.sleep()` placeholders)
2. **Over-engineered for research phase** (13 services for a 3-person team)
3. **Infrastructure not yet created** (no Docker Compose, no Dockerfiles, no Alembic migrations)

## When to use this code

After the research produces working models (pose estimation, PINNs, GNN, optimization), these services provide the API contracts and pipeline orchestration needed to deploy the system as a scalable web application.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| api_gateway | 8000 | JWT auth, request routing |
| user_profile_service | 8001 | User accounts, anthropometrics |
| video_ingestion_service | 8002 | S3 presigned uploads, triggers pipeline |
| processing_orchestration_service | 8003 | Celery pipeline chain |
| session_bvh_data_service | 8004 | Session/attempt/BVH storage |
| pose_estimation_service | 8005 | Pose estimation endpoint (simulated) |
| feedback_reporting_service | 8006 | Reports, dashboards, coaching cues |
| ai_model_training_service | 8007 | Model training endpoint (simulated) |
| pinn_gnn_inference_service | 8008 | PINN/GNN inference (simulated) |
| anomaly_detection_service | 8009 | Anomaly detection (simulated) |
| optimization_engine_service | 8010 | Technique optimization (heuristic) |
| population_model_service | 8011 | Cohort matching (heuristic) |

## Known issues

- `ai_model_training_service/config.py` missing `DATABASE_URL`
- `ai_model_training_service/requirements.txt` missing `sqlalchemy`, `psycopg2-binary`
- Port defaults inconsistent across service configs
- `https://` used for internal service URLs in some configs (should be `http://`)
- Missing `docker-compose.yml` and `Dockerfile`s

## To remove this scaffolding

If you don't need the deployment scaffolding:

```bash
rm -rf services_scaffold/
```

The research code in `src/` is fully independent.
