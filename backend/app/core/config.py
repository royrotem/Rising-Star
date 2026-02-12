from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings."""

    # Application
    APP_NAME: str = "UAIE - Universal Autonomous Insight Engine"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:5173,http://frontend:80"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/uaie"
    DATABASE_POOL_SIZE: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Ingestion - File Size Limits
    MAX_FILE_SIZE_MB: int = 5000  # 5GB for single files
    MAX_ARCHIVE_SIZE_MB: int = 5000  # 5GB for compressed archives
    MAX_EXTRACTED_SIZE_GB: int = 50  # 50GB total extracted from archives
    MAX_FILES_PER_ARCHIVE: int = 10000  # Zip bomb protection
    SUPPORTED_FORMATS: list = ["csv", "json", "parquet", "can", "bin", "xlsx", "xml", "yaml"]

    # Ingestion - Chunking for Large Files
    CHUNK_SIZE_RECORDS: int = 100_000  # Records per chunk
    CHUNK_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB per chunk
    STREAM_BUFFER_SIZE: int = 8 * 1024 * 1024  # 8MB streaming buffer

    # Ingestion - Storage Thresholds (use PostgreSQL above these)
    USE_DB_THRESHOLD_RECORDS: int = 50_000  # Use PostgreSQL above this
    USE_DB_THRESHOLD_MB: int = 100  # Use PostgreSQL above this size

    # Data Directory
    DATA_DIR: str = "/app/data"

    # Anomaly Detection
    ANOMALY_THRESHOLD: float = 0.95
    DETECTION_WINDOW_HOURS: int = 24

    # ML Models
    MODELS_DIR: Optional[str] = None  # Defaults to backend/models/ if unset

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
