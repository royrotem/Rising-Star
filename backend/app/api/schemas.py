"""
API Request/Response Schemas

Pydantic models used across API endpoints for request validation
and response serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ─── System Schemas ──────────────────────────────────────────────────

class SystemCreate(BaseModel):
    name: str
    system_type: str
    serial_number: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    analysis_id: Optional[str] = None


class SystemResponse(BaseModel):
    id: str
    name: str
    system_type: str
    status: str
    health_score: float
    created_at: str
    anomaly_count: Optional[int] = None
    last_analysis_at: Optional[str] = None


# ─── Field Confirmation Schemas ──────────────────────────────────────

class FieldConfirmation(BaseModel):
    field_name: str
    confirmed_type: Optional[str] = None
    confirmed_unit: Optional[str] = None
    confirmed_meaning: Optional[str] = None
    is_correct: bool


# ─── Analysis Schemas ────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    include_anomaly_detection: bool = True
    include_root_cause: bool = True
    include_blind_spots: bool = True
    time_range_hours: int = 24


# ─── Conversation Schemas ────────────────────────────────────────────

class ConversationQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}


# ─── Schedule Schemas ───────────────────────────────────────────────

class ScheduleRequest(BaseModel):
    enabled: bool = True
    interval: str = Field(default="24h", description="One of: 1h, 6h, 12h, 24h, 7d")


class ScheduleResponse(BaseModel):
    system_id: str
    enabled: bool
    interval: str = "24h"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    last_error: Optional[str] = None
    run_count: int = 0
