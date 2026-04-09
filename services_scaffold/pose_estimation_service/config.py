import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

USER_PROFILE_SERVICE_URL = os.getenv("USER_PROFILE_SERVICE_URL", "http://user_profile_service:8001/internal")
SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8004/internal")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
    raise ValueError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME environment variables must be set for S3 operations.")
