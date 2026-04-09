import os
from dotenv import load_dotenv

load_dotenv()

USER_PROFILE_SERVICE_URL = os.getenv("USER_PROFILE_SERVICE_URL", "http://user_profile_service:8001/internal")
