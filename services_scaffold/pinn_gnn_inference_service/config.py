import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "highjump-biomechanical-data")

SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8003/internal")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    # This check might be too strict for local development without AWS credentials
    # Consider mocking S3 in tests or providing dummy credentials for local dev
    # raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables must be set.")
    pass # Allow running without AWS credentials for now, will fail on S3 operations
