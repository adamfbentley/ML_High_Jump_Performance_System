import os
from dotenv import load_dotenv

load_dotenv()

SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8004/internal")
