import httpx
from fastapi import Depends, HTTPException, status
from api_gateway.config import USER_PROFILE_SERVICE_URL, VIDEO_INGESTION_SERVICE_URL, SESSION_BVH_DATA_SERVICE_URL, FEEDBACK_REPORTING_SERVICE_URL

async def get_user_profile_service_client():
    async with httpx.AsyncClient(base_url=USER_PROFILE_SERVICE_URL) as client:
        yield client

async def get_video_ingestion_service_client():
    async with httpx.AsyncClient(base_url=VIDEO_INGESTION_SERVICE_URL) as client:
        yield client

async def get_session_bvh_data_service_client():
    async with httpx.AsyncClient(base_url=SESSION_BVH_DATA_SERVICE_URL) as client:
        yield client

# NEW DEPENDENCY FOR FEEDBACK & REPORTING SERVICE
async def get_feedback_reporting_service_client():
    async with httpx.AsyncClient(base_url=FEEDBACK_REPORTING_SERVICE_URL) as client:
        yield client
