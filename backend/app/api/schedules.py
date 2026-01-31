"""
Schedule API Endpoints

CRUD for per-system watchdog (auto-analysis) schedules.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..services.data_store import data_store
from ..services.scheduler import scheduler, VALID_INTERVALS
from .schemas import ScheduleRequest, ScheduleResponse

router = APIRouter(prefix="/schedules", tags=["Schedules"])


@router.get("/", response_model=List[ScheduleResponse])
async def list_schedules():
    """List all configured schedules."""
    return scheduler.list_schedules()


@router.get("/{system_id}", response_model=ScheduleResponse)
async def get_schedule(system_id: str):
    """Get schedule configuration for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    sched = scheduler.get_schedule(system_id)
    if not sched:
        # Return a default disabled schedule
        return ScheduleResponse(
            system_id=system_id,
            enabled=False,
            interval="24h",
            run_count=0,
        )
    return sched


@router.put("/{system_id}", response_model=ScheduleResponse)
async def set_schedule(system_id: str, request: ScheduleRequest):
    """Create or update a schedule for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    try:
        sched = scheduler.set_schedule(
            system_id=system_id,
            enabled=request.enabled,
            interval=request.interval,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return sched


@router.delete("/{system_id}")
async def delete_schedule(system_id: str):
    """Delete a system's schedule."""
    deleted = scheduler.delete_schedule(system_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No schedule found for this system")
    return {"status": "deleted", "system_id": system_id}
