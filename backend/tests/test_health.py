"""
Tests for health check endpoint.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(test_client: AsyncClient):
    """Test that /health endpoint returns 200 status code."""
    response = await test_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_has_status(test_client: AsyncClient):
    """Test that /health endpoint returns status field."""
    response = await test_client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint_has_version(test_client: AsyncClient):
    """Test that /health endpoint returns version field."""
    response = await test_client.get("/health")
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


@pytest.mark.asyncio
async def test_health_endpoint_has_agents(test_client: AsyncClient):
    """Test that /health endpoint returns agents status."""
    response = await test_client.get("/health")
    data = response.json()
    assert "agents" in data
