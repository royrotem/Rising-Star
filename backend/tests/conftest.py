"""
Shared pytest fixtures for UAIE test suite.
"""

import pytest
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.data_store import DataStore


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def data_store_instance(temp_data_dir: Path) -> DataStore:
    """Provide a DataStore instance with temporary directory."""
    return DataStore(data_dir=str(temp_data_dir))


@pytest.fixture
def sample_system_data() -> Dict[str, Any]:
    """Provide sample system data for testing."""
    system_id = str(uuid.uuid4())
    return {
        "id": system_id,
        "name": "Test System",
        "system_type": "vehicle",
        "serial_number": "TEST-001",
        "model": "TestModel-X",
        "metadata": {
            "manufacturer": "Test Manufacturer",
            "year": 2024
        },
        "status": "active",
        "health_score": 95.0,
        "discovered_schema": [],
        "confirmed_fields": {},
        "is_demo": False
    }


@pytest.fixture
def sample_ingested_records() -> list:
    """Provide sample ingested data records."""
    return [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "temperature": 25.5,
            "voltage": 12.6,
            "current": 1.5
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "temperature": 26.0,
            "voltage": 12.5,
            "current": 1.6
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "temperature": 26.5,
            "voltage": 12.4,
            "current": 1.7
        }
    ]


@pytest.fixture
def sample_discovered_schema() -> Dict[str, Any]:
    """Provide sample discovered schema."""
    return {
        "fields": [
            {
                "name": "timestamp",
                "inferred_type": "timestamp",
                "physical_unit": None,
                "confidence": 0.95
            },
            {
                "name": "temperature",
                "inferred_type": "numeric",
                "physical_unit": "celsius",
                "confidence": 0.90
            },
            {
                "name": "voltage",
                "inferred_type": "numeric",
                "physical_unit": "volts",
                "confidence": 0.92
            },
            {
                "name": "current",
                "inferred_type": "numeric",
                "physical_unit": "amperes",
                "confidence": 0.91
            }
        ],
        "relationships": []
    }


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
