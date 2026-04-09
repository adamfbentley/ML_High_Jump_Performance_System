import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

PROCESSING_ORCHESTRATION_SERVICE_URL = os.getenv("PROCESSING_ORCHESTRATION_SERVICE_URL", "http://processing_orchestration_service:8003/internal")
SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8004/internal") # NEW
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
    raise ValueError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME environment variables must be set for S3 operations.")
