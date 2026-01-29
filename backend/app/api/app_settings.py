"""
Application Settings API

Manages runtime configuration such as API keys and feature toggles.
Settings are persisted to disk so they survive restarts.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path
import threading

router = APIRouter(prefix="/settings", tags=["Settings"])


# ─── Persistent settings store ───────────────────────────────────────

class _SettingsStore:
    """Simple file-based settings persistence."""

    def __init__(self):
        data_dir = os.environ.get("DATA_DIR", "/app/data")
        if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self._path = Path(data_dir) / "app_settings.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(json.dumps(self._cache, indent=2))

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._cache.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = value
            self._save()

    def update(self, values: Dict[str, Any]):
        with self._lock:
            self._cache.update(values)
            self._save()


settings_store = _SettingsStore()


# ─── Pydantic models ─────────────────────────────────────────────────

class AISettings(BaseModel):
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key for Claude LLM agents")
    enable_ai_agents: bool = Field(True, description="Enable AI multi-agent analysis")
    enable_web_grounding: bool = Field(True, description="Allow agents to search the web for engineering context")
    extended_thinking_budget: int = Field(10000, ge=1000, le=50000, description="Token budget for extended thinking agent")


class SettingsResponse(BaseModel):
    ai: AISettings
    status: str = "ok"


class SettingsUpdate(BaseModel):
    ai: Optional[AISettings] = None


# ─── Endpoints ────────────────────────────────────────────────────────

@router.get("/", response_model=SettingsResponse)
async def get_settings():
    """Get all application settings. API keys are masked in the response."""
    raw = settings_store.get_all()
    ai_raw = raw.get("ai", {})

    # Mask API key in response
    api_key = ai_raw.get("anthropic_api_key", "") or ""
    masked_key = ""
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***configured***"

    return SettingsResponse(
        ai=AISettings(
            anthropic_api_key=masked_key,
            enable_ai_agents=ai_raw.get("enable_ai_agents", True),
            enable_web_grounding=ai_raw.get("enable_web_grounding", True),
            extended_thinking_budget=ai_raw.get("extended_thinking_budget", 10000),
        )
    )


@router.put("/", response_model=SettingsResponse)
async def update_settings(update: SettingsUpdate):
    """Update application settings."""
    if update.ai is not None:
        ai_data = update.ai.model_dump(exclude_none=False)

        # If the masked key is sent back, don't overwrite the real one
        existing_key = settings_store.get("ai", {}).get("anthropic_api_key", "")
        if ai_data.get("anthropic_api_key") and "..." in (ai_data["anthropic_api_key"] or ""):
            ai_data["anthropic_api_key"] = existing_key
        if ai_data.get("anthropic_api_key") == "***configured***":
            ai_data["anthropic_api_key"] = existing_key

        settings_store.set("ai", ai_data)

    # Return updated settings (masked)
    return await get_settings()


@router.get("/ai/status")
async def get_ai_status():
    """Check if AI is properly configured and ready."""
    ai_settings = settings_store.get("ai", {})
    api_key = ai_settings.get("anthropic_api_key", "")
    enabled = ai_settings.get("enable_ai_agents", True)

    has_key = bool(api_key and len(api_key) > 10)

    return {
        "configured": has_key,
        "enabled": enabled,
        "ready": has_key and enabled,
        "web_grounding_enabled": ai_settings.get("enable_web_grounding", True),
        "message": (
            "AI agents are ready" if has_key and enabled
            else "AI agents disabled" if not enabled
            else "Set your Anthropic API key in Settings to enable AI agents"
        ),
    }


def get_anthropic_api_key() -> str:
    """Helper: get the current Anthropic API key from settings (or env fallback)."""
    ai = settings_store.get("ai", {})
    key = ai.get("anthropic_api_key", "")
    if key:
        return key
    # Fallback to environment variable
    return os.environ.get("ANTHROPIC_API_KEY", "")


def get_ai_settings() -> dict:
    """Helper: get all AI settings."""
    return settings_store.get("ai", {
        "enable_ai_agents": True,
        "enable_web_grounding": True,
        "extended_thinking_budget": 10000,
    })
