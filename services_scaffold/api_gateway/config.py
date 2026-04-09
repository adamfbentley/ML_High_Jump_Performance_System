import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set. This is critical for JWT security.")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
USER_PROFILE_SERVICE_URL = os.getenv("USER_PROFILE_SERVICE_URL", "http://user_profile_service:8001/internal")
VIDEO_INGESTION_SERVICE_URL = os.getenv("VIDEO_INGESTION_SERVICE_URL", "http://video_ingestion_service:8002/internal")
SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8004/internal")
FEEDBACK_REPORTING_SERVICE_URL = os.getenv("FEEDBACK_REPORTING_SERVICE_URL", "http://feedback_reporting_service:8006/internal")
