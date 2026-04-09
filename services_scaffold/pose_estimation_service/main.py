from fastapi import FastAPI, Depends, HTTPException, status
import httpx
import logging
import os
import tempfile
from pose_estimation_service import schemas, s3_utils
from pose_estimation_service.config import USER_PROFILE_SERVICE_URL, SESSION_BVH_DATA_SERVICE_URL, S3_BUCKET_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pose Estimation Service",
    description="Performs pose estimation and BVH generation on video segments.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.post("/internal/pose-estimate", response_model=schemas.PoseEstimationResponse, status_code=status.HTTP_202_ACCEPTED)
async def pose_estimate_endpoint(request: schemas.PoseEstimationRequest):
    logger.info(f"Received pose estimation request for attempt_id: {request.attempt_id}, user_id: {request.user_id}")

    async with httpx.AsyncClient() as client:
        # 1. Fetch user profile for anthropometric data (BE-02)
        try:
            profile_response = await client.get(f"{USER_PROFILE_SERVICE_URL}/profiles/{request.user_id}")
            profile_response.raise_for_status()
            user_profile = schemas.UserProfileAnthropometrics(**profile_response.json())
            logger.info(f"Fetched user profile for user {request.user_id}: {user_profile.dict()}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch user profile for user {request.user_id}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Network error fetching user profile for user {request.user_id}: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user profile service: {e}")

        # Create temporary files for video and BVH
        with tempfile.TemporaryDirectory() as temp_dir:
            video_local_path = os.path.join(temp_dir, f"video_{request.attempt_id}.mp4")
            bvh_local_path = os.path.join(temp_dir, f"bvh_{request.attempt_id}.bvh")

            # 2. Download video segment from S3 (DS-01)
            try:
                # In a real scenario, we would download the full video and then extract the segment
                # For this sprint, we assume the video_s3_key points to the full video.
                s3_utils.download_file_from_s3(request.video_s3_key, video_local_path)
                logger.info(f"Downloaded video from S3: {request.video_s3_key}")
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to download video from S3: {e}")

            # 3. Perform Pose Estimation and BVH Generation (Placeholder)
            logger.info(f"Simulating pose estimation and BVH generation for video segment from {request.start_time_ms}ms to {request.end_time_ms}ms using profile: {user_profile.dict()}")
            # In a real implementation, this would involve:
            # - Loading a pose estimation model (e.g., MediaPipe BlazePose, OpenPose)
            # - Processing the video segment (e.g., using OpenCV) to extract 2D keypoints
            # - Performing 3D pose reconstruction
            # - Retargeting to a personalized skeleton rig using anthropometric data
            # - Generating BVH data
            # For now, we'll just create a dummy BVH file.
            dummy_bvh_content = f"HIERARCHY\nROOT Hips\n{{\n  OFFSET 0.00 0.00 0.00\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n  JOINT LeftHip\n  {{\n    OFFSET 10.00 0.00 0.00\n    CHANNELS 3 Zrotation Xrotation Yrotation\n    End Site\n    {{\n      OFFSET 0.00 -10.00 0.00\n    }}\n  }}\n}}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
            with open(bvh_local_path, "w") as f:
                f.write(dummy_bvh_content)
            logger.info(f"Simulated BVH file created at {bvh_local_path}")

            # 4. Upload generated BVH file to S3 (DS-01)
            bvh_s3_key = f"{request.user_id}/bvh/{request.attempt_id}.bvh"
            try:
                s3_utils.upload_file_to_s3(bvh_local_path, bvh_s3_key, content_type='application/bvh')
                logger.info(f"Uploaded BVH to S3: {bvh_s3_key}")
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload BVH to S3: {e}")

            # 5. Update Attempt record with BVH S3 key in Session & BVH Data Service (BE-04)
            try:
                update_bvh_payload = {"bvh_file_s3_key": bvh_s3_key}
                update_response = await client.post(
                    f"{SESSION_BVH_DATA_SERVICE_URL}/attempts/{request.attempt_id}/bvh?user_id={request.user_id}",
                    json=update_bvh_payload
                )
                update_response.raise_for_status()
                logger.info(f"Updated attempt {request.attempt_id} with BVH S3 key: {bvh_s3_key}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to update attempt {request.attempt_id} with BVH S3 key: {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update attempt BVH key: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Network error updating attempt {request.attempt_id} with BVH S3 key: {e}")
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session BVH data service: {e}")

        return schemas.PoseEstimationResponse(
            status="success",
            message=f"Pose estimation and BVH generation complete for attempt {request.attempt_id}.",
            attempt_id=request.attempt_id,
            bvh_s3_key=bvh_s3_key
        )
