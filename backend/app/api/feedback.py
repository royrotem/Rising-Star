"""
Anomaly Feedback API

Endpoints for submitting and retrieving user feedback on detected anomalies.
This is an additive feature module - removing this router does not affect core functionality.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.feedback_store import feedback_store

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ---------- Request / Response Models ----------

class FeedbackCreate(BaseModel):
    anomaly_id: str
    anomaly_title: str
    anomaly_type: str
    severity: str
    feedback_type: str  # 'relevant' | 'false_positive' | 'already_known'
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    system_id: str
    anomaly_id: str
    anomaly_title: str
    anomaly_type: str
    severity: str
    feedback_type: str
    comment: Optional[str]
    created_at: str


# ---------- Endpoints ----------

@router.post("/systems/{system_id}", response_model=FeedbackResponse)
async def submit_feedback(system_id: str, body: FeedbackCreate):
    """Submit feedback for an anomaly detected in a system."""
    if body.feedback_type not in ("relevant", "false_positive", "already_known"):
        raise HTTPException(
            status_code=400,
            detail="feedback_type must be one of: relevant, false_positive, already_known",
        )

    entry = feedback_store.add_feedback(
        system_id=system_id,
        anomaly_id=body.anomaly_id,
        anomaly_title=body.anomaly_title,
        anomaly_type=body.anomaly_type,
        severity=body.severity,
        feedback_type=body.feedback_type,
        comment=body.comment,
    )
    return entry


@router.get("/systems/{system_id}")
async def get_feedback(
    system_id: str,
    anomaly_id: Optional[str] = None,
    feedback_type: Optional[str] = None,
):
    """Get all feedback entries for a system, optionally filtered."""
    return feedback_store.get_feedback(
        system_id=system_id,
        anomaly_id=anomaly_id,
        feedback_type=feedback_type,
    )


@router.get("/systems/{system_id}/summary")
async def get_feedback_summary(system_id: str):
    """Get aggregated feedback statistics and confidence metrics for a system."""
    return feedback_store.get_feedback_summary(system_id)


@router.get("/systems/{system_id}/adaptive-thresholds")
async def get_adaptive_thresholds(system_id: str):
    """Get threshold adjustment suggestions based on accumulated feedback."""
    return feedback_store.get_adaptive_thresholds(system_id)


@router.delete("/systems/{system_id}/{feedback_id}")
async def delete_feedback(system_id: str, feedback_id: str):
    """Delete a specific feedback entry."""
    deleted = feedback_store.delete_feedback(system_id, feedback_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Feedback entry not found")
    return {"status": "deleted", "id": feedback_id}
