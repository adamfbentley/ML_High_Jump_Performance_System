import os
from dotenv import load_dotenv

load_dotenv()

USER_PROFILE_SERVICE_URL = os.getenv("USER_PROFILE_SERVICE_URL", "https://user_profile_service:8001/internal")
SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "https://session_bvh_data_service:8004/internal")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# S3 details for model artifact storage
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Optional: Raise error if critical URLs are not set, similar to other services
if not USER_PROFILE_SERVICE_URL or not SESSION_BVH_DATA_SERVICE_URL:
    raise ValueError("USER_PROFILE_SERVICE_URL and SESSION_BVH_DATA_SERVICE_URL must be set.")
