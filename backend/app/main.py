"""
UAIE - Universal Autonomous Insight Engine

Main FastAPI application entry point.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn

from .core.config import settings
from .api.systems import router as systems_router
from .api.app_settings import router as settings_router
from .api.feedback import router as feedback_router
from .api.streaming import router as streaming_router
from .api.chat import router as chat_router
from .api.reports import router as reports_router
from .api.baselines import router as baselines_router
from .api.schedules import router as schedules_router
from .agents.orchestrator import orchestrator
from .services.scheduler import scheduler

# Path to the built frontend (populated in production Docker image)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    # orchestrator.start() would be called here in production
    await scheduler.start()
    yield
    # Shutdown
    print("👋 Shutting down...")
    await scheduler.stop()
    await orchestrator.stop()


app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Universal Autonomous Insight Engine (UAIE)
    
    The "Tesla Standard" – Democratized as a Service
    
    ### Core Capabilities:
    - **Zero-Knowledge Ingestion**: Upload raw data, we learn the structure
    - **Physics-Aware Anomaly Detection**: Beyond simple thresholds
    - **Root Cause Analysis**: Natural language explanations
    - **AI Ready Lifecycle**: Design the next generation
    
    ### Key Features:
    - Human-in-the-Loop calibration for trust
    - 80/20 Impact Radar for prioritization
    - Conversational interface for engineers
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(systems_router, prefix=settings.API_PREFIX)
app.include_router(settings_router, prefix=settings.API_PREFIX)
app.include_router(feedback_router, prefix=settings.API_PREFIX)
app.include_router(streaming_router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(reports_router, prefix=settings.API_PREFIX)
app.include_router(baselines_router, prefix=settings.API_PREFIX)
app.include_router(schedules_router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "agents": orchestrator.get_agent_status(),
    }


@app.get(f"{settings.API_PREFIX}/agents/status")
async def get_agents_status():
    """Get status of all AI agents."""
    return orchestrator.get_agent_status()


# ── Serve built frontend (production) ──────────────────────────────────────
# Mount static assets (JS, CSS, images) if the build directory exists.
# The SPA catch-all below returns index.html for any non-API route so that
# React Router can handle client-side routing.

if STATIC_DIR.is_dir():
    # Serve /assets/* directly (Vite puts hashed bundles here)
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_spa_root():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{path:path}")
    async def serve_spa(request: Request, path: str):
        """Serve static file if it exists, otherwise return index.html for SPA routing."""
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    async def root():
        """Root endpoint with API information (dev mode — no frontend build)."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": "The Tesla Standard – Democratized as a Service",
            "docs": "/docs",
            "health": "/health",
        }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
