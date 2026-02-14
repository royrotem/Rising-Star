"""
Tests for Systems API endpoints.
"""

import pytest
import uuid
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_system(test_client: AsyncClient):
    """Test POST /api/v1/systems creates a new system."""
    system_data = {
        "name": "API Test System",
        "system_type": "vehicle",
        "serial_number": "API-TEST-001",
        "model": "TestModel-API",
        "metadata": {
            "manufacturer": "API Test Manufacturer",
            "year": 2024
        }
    }

    response = await test_client.post("/api/v1/systems", json=system_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == system_data["name"]
    assert data["system_type"] == system_data["system_type"]
    assert data["serial_number"] == system_data["serial_number"]
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_systems(test_client: AsyncClient):
    """Test GET /api/v1/systems lists all systems."""
    # Create a test system first
    system_data = {
        "name": "List Test System",
        "system_type": "robot",
        "serial_number": "LIST-TEST-001",
        "model": "ListTestModel"
    }
    await test_client.post("/api/v1/systems", json=system_data)

    # List systems
    response = await test_client.get("/api/v1/systems")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify our system is in the list
    system_names = [s["name"] for s in data]
    assert "List Test System" in system_names


@pytest.mark.asyncio
async def test_get_system_by_id(test_client: AsyncClient):
    """Test GET /api/v1/systems/{id} retrieves a specific system."""
    # Create a system first
    system_data = {
        "name": "Get Test System",
        "system_type": "vehicle",
        "serial_number": "GET-TEST-001",
        "model": "GetTestModel"
    }
    create_response = await test_client.post("/api/v1/systems", json=system_data)
    created_system = create_response.json()
    system_id = created_system["id"]

    # Get the system by ID
    response = await test_client.get(f"/api/v1/systems/{system_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == system_id
    assert data["name"] == system_data["name"]
    assert data["system_type"] == system_data["system_type"]


@pytest.mark.asyncio
async def test_get_system_nonexistent(test_client: AsyncClient):
    """Test GET /api/v1/systems/{id} with non-existent ID returns 404."""
    nonexistent_id = str(uuid.uuid4())
    response = await test_client.get(f"/api/v1/systems/{nonexistent_id}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_system(test_client: AsyncClient):
    """Test DELETE /api/v1/systems/{id} deletes a system."""
    # Create a system first
    system_data = {
        "name": "Delete Test System",
        "system_type": "vehicle",
        "serial_number": "DEL-TEST-001",
        "model": "DeleteTestModel"
    }
    create_response = await test_client.post("/api/v1/systems", json=system_data)
    created_system = create_response.json()
    system_id = created_system["id"]

    # Delete the system
    response = await test_client.delete(f"/api/v1/systems/{system_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "System deleted successfully"

    # Verify it's gone
    get_response = await test_client.get(f"/api/v1/systems/{system_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_system_nonexistent(test_client: AsyncClient):
    """Test DELETE /api/v1/systems/{id} with non-existent ID returns 404."""
    nonexistent_id = str(uuid.uuid4())
    response = await test_client.delete(f"/api/v1/systems/{nonexistent_id}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_system_validation(test_client: AsyncClient):
    """Test POST /api/v1/systems with invalid data returns error."""
    # Missing required fields
    invalid_data = {
        "name": "Invalid System"
        # Missing system_type, serial_number, model
    }

    response = await test_client.post("/api/v1/systems", json=invalid_data)

    # Should return validation error (422 Unprocessable Entity)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_systems_empty(test_client: AsyncClient):
    """Test GET /api/v1/systems returns empty list when no systems exist."""
    # This test assumes a fresh database or clean state
    # In practice, demo systems might exist
    response = await test_client.get("/api/v1/systems")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # We can't assert empty because demo systems might exist
    # Just verify it's a valid list


@pytest.mark.asyncio
async def test_system_roundtrip(test_client: AsyncClient):
    """Test creating, retrieving, and deleting a system (full roundtrip)."""
    # Create
    system_data = {
        "name": "Roundtrip Test System",
        "system_type": "robot",
        "serial_number": "RT-TEST-001",
        "model": "RoundtripModel",
        "metadata": {"test": "roundtrip"}
    }
    create_response = await test_client.post("/api/v1/systems", json=system_data)
    assert create_response.status_code == 201
    created = create_response.json()
    system_id = created["id"]

    # Read
    get_response = await test_client.get(f"/api/v1/systems/{system_id}")
    assert get_response.status_code == 200
    retrieved = get_response.json()
    assert retrieved["id"] == system_id
    assert retrieved["name"] == system_data["name"]

    # List (verify it's there)
    list_response = await test_client.get("/api/v1/systems")
    assert list_response.status_code == 200
    systems = list_response.json()
    assert any(s["id"] == system_id for s in systems)

    # Delete
    delete_response = await test_client.delete(f"/api/v1/systems/{system_id}")
    assert delete_response.status_code == 200

    # Verify deletion
    final_get = await test_client.get(f"/api/v1/systems/{system_id}")
    assert final_get.status_code == 404
