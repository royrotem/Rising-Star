"""
AI Chat API

Full conversational interface with persistent history and Claude-powered
responses grounded in system telemetry data.
This is an additive feature module — removing this router does not affect
existing query or analysis endpoints.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.data_store import data_store
from ..services.chat_service import chat_service, conversation_store
from ..utils import load_saved_analysis

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Request / Response Models ────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    ai_powered: bool
    data: Optional[dict] = None


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/systems/{system_id}", response_model=ChatResponse)
async def send_message(system_id: str, body: ChatMessage):
    """Send a message to the AI chat and get a response."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id, limit=5000)
    schema = system.get("discovered_schema", [])
    analysis = load_saved_analysis(system_id)

    result = await chat_service.chat(
        system_id=system_id,
        user_message=body.message,
        system=system,
        records=records,
        schema=schema,
        analysis=analysis,
    )

    # Get the last assistant message from the store
    history = conversation_store.get_messages(system_id, limit=1)
    last_msg = history[-1] if history else {}

    return {
        "id": last_msg.get("id", "unknown"),
        "role": "assistant",
        "content": result["content"],
        "timestamp": last_msg.get("timestamp", ""),
        "ai_powered": result.get("ai_powered", False),
        "data": result.get("data"),
    }


@router.get("/systems/{system_id}/history")
async def get_history(system_id: str, limit: int = 50):
    """Get conversation history for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    messages = conversation_store.get_messages(system_id, limit=limit)
    return {"system_id": system_id, "messages": messages}


@router.delete("/systems/{system_id}/history")
async def clear_history(system_id: str):
    """Clear conversation history for a system."""
    conversation_store.clear(system_id)
    return {"status": "cleared", "system_id": system_id}
