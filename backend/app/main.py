"""
UAIE - Universal Autonomous Insight Engine

Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://frontend:80",
    ],
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "The Tesla Standard – Democratized as a Service",
        "docs": "/docs",
        "health": "/health",
    }


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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
