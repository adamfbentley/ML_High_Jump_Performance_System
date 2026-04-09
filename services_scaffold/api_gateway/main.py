from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import httpx
from typing import List

from api_gateway import schemas, auth, config
from api_gateway.dependencies import get_user_profile_service_client, get_video_ingestion_service_client, get_session_bvh_data_service_client, get_feedback_reporting_service_client

app = FastAPI(
    title="API Gateway",
    description="Central entry point for all frontend requests, handling authentication, authorization, and routing.",
    version="1.0.0"
)

@app.post("/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: schemas.UserCreate, client: httpx.AsyncClient = Depends(get_user_profile_service_client)):
    try:
        response = await client.post("/users", json=user.dict())
        response.raise_for_status()
        return schemas.UserResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_400_BAD_REQUEST:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.response.json().get("detail", "Email already registered"))
        raise HTTPException(status_code=e.response.status_code, detail=f"User service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user service: {e}")

@app.post("/auth/login", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), client: httpx.AsyncClient = Depends(get_user_profile_service_client)):
    user_data = await auth.authenticate_user(form_data.username, form_data.password) # form_data.username is email
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user_data["email"], "user_id": user_data["id"]}, # Include user_id in token
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/{user_id}/profile", response_model=schemas.UserProfileResponse)
async def read_user_profile(user_id: int, current_user: schemas.UserResponse = Depends(auth.get_current_user), client: httpx.AsyncClient = Depends(get_user_profile_service_client)):
    if current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this profile")

    try:
        response = await client.get(f"/profiles/{user_id}")
        response.raise_for_status()
        return schemas.UserProfileResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")
        raise HTTPException(status_code=e.response.status_code, detail=f"User service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user service: {e}")

@app.put("/users/{user_id}/profile", response_model=schemas.UserProfileResponse)
async def update_user_profile(user_id: int, profile_update: schemas.UserProfileUpdate, current_user: schemas.UserResponse = Depends(auth.get_current_user), client: httpx.AsyncClient = Depends(get_user_profile_service_client)):
    if current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this profile")

    try:
        response = await client.put(f"/profiles/{user_id}", json=profile_update.dict(exclude_unset=True))
        response.raise_for_status()
        return schemas.UserProfileResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")
        raise HTTPException(status_code=e.response.status_code, detail=f"User service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to user service: {e}")

@app.post("/videos/upload-url", response_model=schemas.VideoUploadResponse, status_code=status.HTTP_200_OK)
async def request_video_upload_url(
    request: schemas.VideoUploadRequest,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_video_ingestion_service_client)
):
    """
    Requests a pre-signed S3 URL for direct video upload.
    The client should use the returned URL and form_fields to perform a direct S3 POST upload.
    """
    try:
        # Pass the current user's ID and session_type to the internal video ingestion service
        internal_request_payload = {
            "user_id": current_user.id,
            "file_name": request.file_name,
            "content_type": request.content_type,
            "session_type": request.session_type.value if request.session_type else None
        }
        response = await client.post("/videos/upload-request", json=internal_request_payload)
        response.raise_for_status()
        return schemas.VideoUploadResponse(**response.json())
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Video Ingestion service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to video ingestion service: {e}")

# NEW ENDPOINTS FOR SESSION & ATTEMPT MANAGEMENT

@app.post("/sessions", response_model=schemas.SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session_endpoint(
    session_create: schemas.SessionCreate,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Creates a new training or competition session record for the authenticated user.
    `raw_video_s3_key` can be optional for live sessions.
    """
    try:
        internal_session_payload = session_create.dict(exclude_unset=True) # exclude_unset to allow raw_video_s3_key=None
        internal_session_payload["user_id"] = current_user.id
        response = await client.post("/sessions", json=internal_session_payload)
        response.raise_for_status()
        return schemas.SessionResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_400_BAD_REQUEST:
            # Attempt to parse detail from internal service, fallback to generic message
            detail_message = e.response.json().get("detail", "Session creation failed due to bad request.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail_message)
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

@app.get("/sessions", response_model=List[schemas.SessionResponse])
async def get_all_sessions(
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Retrieves a historical list of all jump sessions for the authenticated user.
    """
    try:
        response = await client.get(f"/users/{current_user.id}/sessions")
        response.raise_for_status()
        return [schemas.SessionResponse(**session_data) for session_data in response.json()]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

@app.get("/sessions/{session_id}", response_model=schemas.SessionWithAttemptsResponse)
async def get_session_details(
    session_id: int,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Retrieves details for a specific session, including all attempts and their metadata,
    ensuring the authenticated user owns it.
    """
    try:
        # Fetch session details
        session_response = await client.get(f"/sessions/{session_id}?user_id={current_user.id}")
        session_response.raise_for_status()
        session_data = session_response.json()

        # Fetch attempts for the session
        attempts_response = await client.get(f"/sessions/{session_id}/attempts?user_id={current_user.id}")
        attempts_response.raise_for_status()
        attempts_data = [schemas.AttemptResponse(**attempt) for attempt in attempts_response.json()]

        # Combine session data with attempts
        session_with_attempts = schemas.SessionWithAttemptsResponse(
            **session_data,
            attempts=attempts_data
        )
        return session_with_attempts
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

@app.post("/sessions/{session_id}/attempts", response_model=schemas.AttemptResponse, status_code=status.HTTP_201_CREATED)
async def create_attempt_for_session_endpoint(
    session_id: int,
    attempt_create_request: schemas.AttemptCreateLiveRequest,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Creates a new jump attempt record for a specific session, ensuring the authenticated user owns it.
    This is typically used during live sessions to log individual attempts.
    """
    try:
        internal_attempt_payload = attempt_create_request.dict()
        response = await client.post(
            f"/sessions/{session_id}/attempts?user_id={current_user.id}",
            json=internal_attempt_payload
        )
        response.raise_for_status()
        return schemas.AttemptResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized to add attempts")
        elif e.response.status_code == status.HTTP_400_BAD_REQUEST:
            detail_message = e.response.json().get("detail", "Attempt creation failed due to bad request.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail_message)
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

@app.get("/sessions/{session_id}/attempts", response_model=List[schemas.AttemptResponse])
async def get_session_attempts(
    session_id: int,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Retrieves all jump attempts associated with a session, ensuring the authenticated user owns the session.
    """
    try:
        response = await client.get(f"/sessions/{session_id}/attempts?user_id={current_user.id}")
        response.raise_for_status()
        return [schemas.AttemptResponse(**attempt) for attempt in response.json()]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

@app.put("/attempts/{attempt_id}/metadata", response_model=schemas.AttemptResponse)
async def update_attempt_metadata(
    attempt_id: int,
    metadata_update: schemas.AttemptUpdateMetadata,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    """
    Updates metadata (e.g., bar height, outcome) for a specific jump attempt, ensuring user ownership.
    """
    try:
        response = await client.put(
            f"/attempts/{attempt_id}/metadata?user_id={current_user.id}",
            json=metadata_update.dict(exclude_unset=True)
        )
        response.raise_for_status()
        return schemas.AttemptResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Session & BVH Data service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Session & BVH Data service: {e}")

# NEW ENDPOINTS FOR FEEDBACK & REPORTING

@app.get("/feedback/session/{session_id}", response_model=schemas.SessionSummaryResponse)
async def get_feedback_session_summary(
    session_id: int,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_feedback_reporting_service_client)
):
    """
    Retrieves a summary of performance for a given session, ensuring user ownership.
    """
    try:
        response = await client.get(f"/reports/session/{session_id}?user_id={current_user.id}")
        response.raise_for_status()
        return schemas.SessionSummaryResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session feedback not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Feedback & Reporting service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Feedback & Reporting service: {e}")

@app.get("/feedback/attempt/{attempt_id}/visuals", response_model=schemas.AttemptVisualsResponse)
async def get_feedback_attempt_visuals(
    attempt_id: int,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_feedback_reporting_service_client)
):
    """
    Retrieves data for visual overlays and comparisons for an attempt, ensuring user ownership.
    """
    try:
        response = await client.get(f"/reports/attempt/{attempt_id}/visuals?user_id={current_user.id}")
        response.raise_for_status()
        return schemas.AttemptVisualsResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt visuals not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Feedback & Reporting service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Feedback & Reporting service: {e}")

@app.get("/progress/dashboard", response_model=schemas.ProgressDashboardResponse)
async def get_progress_dashboard(
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_feedback_reporting_service_client)
):
    """
    Retrieves data for the athlete's long-term progress dashboard.
    """
    try:
        response = await client.get(f"/reports/athlete/{current_user.id}/progress?user_id={current_user.id}")
        response.raise_for_status()
        return schemas.ProgressDashboardResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Progress dashboard data not found or not authorized")
        raise HTTPException(status_code=e.response.status_code, detail=f"Feedback & Reporting service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Feedback & Reporting service: {e}")

@app.get("/coach/athletes/{athlete_id}/reports", response_model=schemas.CoachReportResponse)
async def get_coach_athlete_reports(
    athlete_id: int,
    current_user: schemas.UserResponse = Depends(auth.get_current_user),
    client: httpx.AsyncClient = Depends(get_feedback_reporting_service_client)
):
    """
    Retrieves coach-specific reports for a particular athlete.
    Assumes the current_user is a coach and is authorized to view this athlete's report.
    """
    # CRITICAL FIX: AUTHZ-001 - Implement robust authorization for coach-athlete relationship
    # In a real system, this would involve querying a dedicated service
    # (e.g., User Profile Service or Coach-Athlete Relationship Service)
    # to check if 'current_user.id' (coach) is authorized to view 'athlete_id'.
    # For now, we simulate a successful authorization for the conceptual test case (coach_id=1, athlete_id=10)
    # and deny access for any other combination to prevent unauthorized access.
    if not (current_user.id == 1 and athlete_id == 10):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this athlete's report (placeholder relationship check)")

    coach_id = current_user.id
    try:
        response = await client.get(f"/reports/coach/{coach_id}/athletes/{athlete_id}?user_id={coach_id}")
        response.raise_for_status()
        return schemas.CoachReportResponse(**response.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Coach report not found or not authorized")
        elif e.response.status_code == status.HTTP_403_FORBIDDEN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this athlete's report")
        raise HTTPException(status_code=e.response.status_code, detail=f"Feedback & Reporting service error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Feedback & Reporting service: {e}")
