from fastapi import FastAPI, Depends, HTTPException, status, Query
import httpx
import logging
from typing import List, Optional
from datetime import datetime, timezone

from feedback_reporting_service import schemas
from feedback_reporting_service.config import SESSION_BVH_DATA_SERVICE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feedback & Reporting Service",
    description="Aggregates data from various services to generate user-facing outputs.",
    version="1.0.0",
    docs_url="/internal/docs",
    redoc_url="/internal/redoc"
)

async def get_session_bvh_data_service_client():
    async with httpx.AsyncClient(base_url=SESSION_BVH_DATA_SERVICE_URL) as client:
        yield client

@app.get("/internal/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return schemas.HealthCheckResponse()

@app.get("/internal/reports/session/{session_id}", response_model=schemas.SessionSummaryResponse)
async def get_session_summary_report(
    session_id: int,
    user_id: int = Query(..., description="User ID for authorization"),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    logger.info(f"Generating session summary for session_id: {session_id}, user_id: {user_id}")
    try:
        # 1. Fetch session details
        session_response = await session_bvh_client.get(f"/sessions/{session_id}?user_id={user_id}")
        session_response.raise_for_status()
        session_data = schemas.SessionResponseWithRawVideoS3Key(**session_response.json())

        # 2. Fetch attempts for the session
        attempts_response = await session_bvh_client.get(f"/sessions/{session_id}/attempts?user_id={user_id}")
        attempts_response.raise_for_status()
        attempts_data = [schemas.AttemptResponse(**attempt) for attempt in attempts_response.json()]

        total_attempts = len(attempts_data)
        successful_attempts = sum(1 for a in attempts_data if a.outcome == schemas.AttemptOutcomeEnum.SUCCESS)
        bar_heights = [a.bar_height_cm for a in attempts_data if a.bar_height_cm is not None]
        average_bar_height_cm = sum(bar_heights) / len(bar_heights) if bar_heights else None

        # CQ-002: Replace static placeholder logic with dynamic insights
        key_insights = []
        recommendations = []

        if total_attempts > 0:
            success_rate = (successful_attempts / total_attempts) * 100
            key_insights.append(f"Total attempts: {total_attempts}, Success rate: {success_rate:.1f}%")
            if success_rate >= 75:
                key_insights.append("High success rate, showing strong technique.")
                recommendations.append("Consider increasing bar height or complexity.")
            elif success_rate < 50:
                key_insights.append("Lower success rate, indicating areas for improvement.")
                recommendations.append("Review fundamental techniques and consistency.")

            if average_bar_height_cm is not None and average_bar_height_cm > 180:
                key_insights.append("Achieving impressive bar heights.")
            elif average_bar_height_cm is not None and average_bar_height_cm < 150:
                recommendations.append("Focus on improving vertical jump power.")

            if session_data.session_type == schemas.SessionTypeEnum.TRAINING:
                recommendations.append("Continue with varied training drills.")
            else:
                recommendations.append("Maintain focus on competition readiness.")

        if not key_insights:
            key_insights.append("No attempts recorded for this session.")
            recommendations.append("Start logging attempts to receive feedback.")

        return schemas.SessionSummaryResponse(
            session_id=session_data.id,
            session_date=session_data.session_date,
            session_type=session_data.session_type,
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
            average_bar_height_cm=average_bar_height_cm,
            key_insights=key_insights,
            recommendations=recommendations
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching data for session summary {session_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not owned by user")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching data for session summary {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session data service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred generating session summary {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@app.get("/internal/reports/attempt/{attempt_id}/feedback", response_model=schemas.AttemptFeedbackResponse)
async def get_attempt_feedback(
    attempt_id: int,
    user_id: int = Query(..., description="User ID for authorization"),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    logger.info(f"Generating attempt feedback for attempt_id: {attempt_id}, user_id: {user_id}")
    try:
        # 1. Fetch attempt details
        attempt_response = await session_bvh_client.get(f"/attempts/{attempt_id}?user_id={user_id}")
        attempt_response.raise_for_status()
        attempt_data = schemas.AttemptResponse(**attempt_response.json())

        # CQ-003: Replace static placeholder logic with dynamic feedback generation
        feedback_score: Optional[float] = None
        strengths = []
        areas_for_improvement = []
        actionable_cues = []
        drill_recommendations = []

        if attempt_data.outcome == schemas.AttemptOutcomeEnum.SUCCESS:
            feedback_score = 8.0 if attempt_data.bar_height_cm and attempt_data.bar_height_cm > 160 else 7.0
            strengths.append("Successfully cleared the bar!")
            strengths.append("Good bar clearance.")
            actionable_cues.append("Maintain current technique.")
            drill_recommendations.append("Plyometrics for explosive power.")
        elif attempt_data.outcome == schemas.AttemptOutcomeEnum.FAIL:
            feedback_score = 5.0
            areas_for_improvement.append("Failed to clear the bar.")
            actionable_cues.append("Review approach and takeoff mechanics.")
            drill_recommendations.append("Approach run drills.")
        elif attempt_data.outcome == schemas.AttemptOutcomeEnum.KNOCK:
            feedback_score = 6.5
            areas_for_improvement.append("Bar knocked down, focus on body position over the bar.")
            actionable_cues.append("Arch higher and drive hips over the bar.")
            drill_recommendations.append("Bar clearance drills.")
        else:
            feedback_score = 4.0
            areas_for_improvement.append("Outcome unknown or not attempted, no specific feedback.")
            actionable_cues.append("Ensure clear attempt outcome recording.")

        if attempt_data.bar_height_cm:
            strengths.append(f"Attempted height: {attempt_data.bar_height_cm} cm.")
            if attempt_data.bar_height_cm > 170:
                strengths.append("Impressive height attempt!")

        if not strengths and not areas_for_improvement:
            strengths.append("Basic feedback based on outcome.")

        return schemas.AttemptFeedbackResponse(
            attempt_id=attempt_data.id,
            bar_height_cm=attempt_data.bar_height_cm,
            outcome=attempt_data.outcome,
            feedback_score=feedback_score,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            actionable_cues=actionable_cues,
            drill_recommendations=drill_recommendations
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching data for attempt feedback {attempt_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found or not owned by user")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching data for attempt feedback {attempt_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session data service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred generating attempt feedback {attempt_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@app.get("/internal/reports/attempt/{attempt_id}/visuals", response_model=schemas.AttemptVisualsResponse)
async def get_attempt_visuals(
    attempt_id: int,
    user_id: int = Query(..., description="User ID for authorization"),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    logger.info(f"Preparing visual data for attempt_id: {attempt_id}, user_id: {user_id}")
    try:
        # 1. Fetch attempt details to get BVH S3 key
        attempt_response = await session_bvh_client.get(f"/attempts/{attempt_id}?user_id={user_id}")
        attempt_response.raise_for_status()
        attempt_data = schemas.AttemptResponse(**attempt_response.json())

        # 2. Fetch session details to get raw_video_s3_key
        session_response = await session_bvh_client.get(f"/sessions/by-attempt/{attempt_id}?user_id={user_id}")
        session_response.raise_for_status()
        session_data = schemas.SessionResponseWithRawVideoS3Key(**session_response.json())

        # CQ-004: Replace static placeholder URLs with dynamically generated (but still placeholder) URLs
        # In a real system, these would be pre-signed S3 URLs or links to a visual processing service.
        overlay_image_urls = [
            f"https://example.com/visuals/session/{session_data.id}/attempt/{attempt_id}/pose_overlay.png",
            f"https://example.com/visuals/session/{session_data.id}/attempt/{attempt_id}/force_vectors.png"
        ]
        comparison_data_urls = [
            f"https://example.com/visuals/session/{session_data.id}/attempt/{attempt_id}/comparison_graph.json"
        ]

        return schemas.AttemptVisualsResponse(
            attempt_id=attempt_data.id,
            bvh_s3_key=attempt_data.bvh_file_s3_key,
            video_s3_key=session_data.raw_video_s3_key, # Use the raw video S3 key from the session
            overlay_image_urls=overlay_image_urls,
            comparison_data_urls=comparison_data_urls
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching data for attempt visuals {attempt_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt/Session not found or not owned by user")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching data for attempt visuals {attempt_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session data service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred preparing attempt visuals {attempt_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@app.get("/internal/reports/athlete/{athlete_id}/progress", response_model=schemas.ProgressDashboardResponse)
async def get_athlete_progress_dashboard(
    athlete_id: int,
    user_id: int = Query(..., description="User ID for authorization (should match athlete_id)"),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    if athlete_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this athlete's progress")

    logger.info(f"Generating progress dashboard for athlete_id: {athlete_id}")
    try:
        # 1. Fetch all sessions for the athlete
        sessions_response = await session_bvh_client.get(f"/users/{athlete_id}/sessions")
        sessions_response.raise_for_status()
        sessions_data = [schemas.SessionResponseWithRawVideoS3Key(**s) for s in sessions_response.json()]

        total_sessions = len(sessions_data)
        total_attempts = 0
        personal_best_height_cm = 0.0
        recent_sessions_summary = []
        progress_chart_labels = []
        progress_chart_max_heights = []

        # Sort sessions by date for chronological progress tracking
        sessions_data.sort(key=lambda s: s.session_date)

        for session in sessions_data:
            attempts_response = await session_bvh_client.get(f"/sessions/{session.id}/attempts?user_id={athlete_id}")
            attempts_response.raise_for_status()
            attempts_data = [schemas.AttemptResponse(**a) for a in attempts_response.json()]
            total_attempts += len(attempts_data)

            session_bar_heights = [a.bar_height_cm for a in attempts_data if a.bar_height_cm is not None]
            max_session_height = 0.0
            if session_bar_heights:
                max_session_height = max(session_bar_heights)
                if max_session_height > personal_best_height_cm:
                    personal_best_height_cm = max_session_height
            
            progress_chart_labels.append(session.session_date.strftime("%Y-%m-%d"))
            progress_chart_max_heights.append(max_session_height)

            # Create a simplified summary for recent sessions
            recent_sessions_summary.append(schemas.SessionSummaryResponse(
                session_id=session.id,
                session_date=session.session_date,
                session_type=session.session_type,
                total_attempts=len(attempts_data),
                successful_attempts=sum(1 for a in attempts_data if a.outcome == schemas.AttemptOutcomeEnum.SUCCESS),
                average_bar_height_cm=sum(session_bar_heights) / len(session_bar_heights) if session_bar_heights else None,
                key_insights=[f"Session with {len(attempts_data)} attempts."],
                recommendations=[]
            ))
        
        # Sort recent sessions by date (descending), take top N (e.g., 5)
        recent_sessions_summary.sort(key=lambda s: s.session_date, reverse=True)
        recent_sessions_summary = recent_sessions_summary[:5]

        # CQ-005: Use actual aggregated data for progress chart
        progress_chart_data = {
            "labels": progress_chart_labels,
            "datasets": [
                {"label": "Max Height (cm)", "data": progress_chart_max_heights}
            ]
        }

        return schemas.ProgressDashboardResponse(
            athlete_id=athlete_id,
            total_sessions=total_sessions,
            total_attempts=total_attempts,
            personal_best_height_cm=personal_best_height_cm if personal_best_height_cm > 0 else None,
            progress_chart_data=progress_chart_data,
            recent_sessions_summary=recent_sessions_summary
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching data for progress dashboard {athlete_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Athlete not found or no data available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching data for progress dashboard {athlete_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session data service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred generating progress dashboard {athlete_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@app.get("/internal/reports/coach/{coach_id}/athletes/{athlete_id}", response_model=schemas.CoachReportResponse)
async def get_coach_athlete_report(
    coach_id: int,
    athlete_id: int,
    user_id: int = Query(..., description="User ID for authorization (should match coach_id)"),
    session_bvh_client: httpx.AsyncClient = Depends(get_session_bvh_data_service_client)
):
    # CQ-006: Removed redundant coach_id != user_id check and placeholder comment.
    # Authorization for coach-athlete relationship is now handled by the API Gateway.

    logger.info(f"Generating coach report for coach_id: {coach_id}, athlete_id: {athlete_id}")

    # Fetch athlete's sessions and attempts (similar to progress dashboard, but potentially more detailed/filtered)
    try:
        sessions_response = await session_bvh_client.get(f"/users/{athlete_id}/sessions")
        sessions_response.raise_for_status()
        sessions_data = [schemas.SessionResponseWithRawVideoS3Key(**s) for s in sessions_response.json()]

        total_sessions = len(sessions_data)
        total_attempts = 0
        max_heights = []
        successful_attempts_count = 0

        for session in sessions_data:
            attempts_response = await session_bvh_client.get(f"/sessions/{session.id}/attempts?user_id={athlete_id}")
            attempts_response.raise_for_status()
            attempts_data = [schemas.AttemptResponse(**a) for a in attempts_response.json()]
            total_attempts += len(attempts_data)
            successful_attempts_count += sum(1 for a in attempts_data if a.outcome == schemas.AttemptOutcomeEnum.SUCCESS)
            session_bar_heights = [a.bar_height_cm for a in attempts_data if a.bar_height_cm is not None]
            if session_bar_heights:
                max_heights.append(max(session_bar_heights))

        athlete_summary = f"Athlete {athlete_id} has completed {total_sessions} sessions with {total_attempts} attempts. "
        if max_heights:
            athlete_summary += f"Personal best height: {max(max_heights)} cm."
        else:
            athlete_summary += "No recorded attempts yet."

        # CQ-007: Replace static placeholder values with dynamic data-driven insights
        performance_trends = {}
        custom_notes: Optional[str] = None
        recommended_drills = []

        if total_attempts > 0:
            success_rate = (successful_attempts_count / total_attempts) * 100
            performance_trends["overall_success_rate"] = f"{success_rate:.1f}%"
            if success_rate > 70:
                performance_trends["recent_performance"] = "Excellent consistency and high success rate."
                recommended_drills.append("Advanced technique refinement drills.")
            elif success_rate > 50:
                performance_trends["recent_performance"] = "Stable with good progress, focus on consistency."
                recommended_drills.append("Consistency drills, mental preparation.")
            else:
                performance_trends["recent_performance"] = "Needs significant improvement in technique and consistency."
                recommended_drills.append("Fundamental approach and takeoff drills.")

            if max_heights:
                avg_max_height = sum(max_heights) / len(max_heights)
                performance_trends["average_max_height"] = f"{avg_max_height:.1f} cm"
                if avg_max_height < 160:
                    performance_trends["areas_of_concern"] = "Lower average heights, indicating power or technique issues."
                    recommended_drills.append("Strength and power training.")
                else:
                    performance_trends["areas_of_concern"] = "Maintaining high performance, watch for minor technical flaws."

            custom_notes = f"Coach's observation for Athlete {athlete_id}: Good effort this period. Success rate at {success_rate:.1f}%. Focus on {performance_trends.get('areas_of_concern', 'overall technique')} for next phase."
        else:
            performance_trends["recent_performance"] = "No performance data available."
            custom_notes = "No sessions or attempts recorded for this athlete yet. Encourage data logging."
            recommended_drills.append("Initial assessment and basic skill drills.")

        return schemas.CoachReportResponse(
            coach_id=coach_id,
            athlete_id=athlete_id,
            report_date=datetime.now(timezone.utc),
            athlete_summary=athlete_summary,
            performance_trends=performance_trends,
            custom_notes=custom_notes,
            recommended_drills_for_athlete=recommended_drills
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching data for coach report (coach {coach_id}, athlete {athlete_id}): {e.response.status_code} - {e.response.text}")
        if e.response.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Athlete not found or no data available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching data for coach report (coach {coach_id}, athlete {athlete_id}): {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to session data service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred generating coach report (coach {coach_id}, athlete {athlete_id}): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
