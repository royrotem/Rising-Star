"""
Tests for DataStore class.
"""

import pytest
import uuid
from typing import Dict, Any

from app.services.data_store import DataStore


def test_create_system(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test creating a new system."""
    system = data_store_instance.create_system(sample_system_data)

    assert system is not None
    assert system["id"] == sample_system_data["id"]
    assert system["name"] == sample_system_data["name"]
    assert system["system_type"] == sample_system_data["system_type"]
    assert "created_at" in system
    assert "updated_at" in system


def test_get_system(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test retrieving a system by ID."""
    # Create system first
    created = data_store_instance.create_system(sample_system_data)

    # Retrieve it
    retrieved = data_store_instance.get_system(created["id"])

    assert retrieved is not None
    assert retrieved["id"] == created["id"]
    assert retrieved["name"] == created["name"]


def test_get_system_nonexistent(data_store_instance: DataStore):
    """Test retrieving a non-existent system returns None."""
    result = data_store_instance.get_system("nonexistent-id")
    assert result is None


def test_list_systems(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test listing all systems."""
    # Create a few systems
    system1 = sample_system_data.copy()
    system1["id"] = str(uuid.uuid4())
    data_store_instance.create_system(system1)

    system2 = sample_system_data.copy()
    system2["id"] = str(uuid.uuid4())
    system2["name"] = "Test System 2"
    data_store_instance.create_system(system2)

    # List all systems
    systems = data_store_instance.list_systems()

    assert len(systems) >= 2
    assert any(s["id"] == system1["id"] for s in systems)
    assert any(s["id"] == system2["id"] for s in systems)


def test_list_systems_exclude_demo(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test listing systems excluding demo systems."""
    # Create regular system
    regular_system = sample_system_data.copy()
    regular_system["id"] = str(uuid.uuid4())
    regular_system["is_demo"] = False
    data_store_instance.create_system(regular_system)

    # Create demo system
    demo_system = sample_system_data.copy()
    demo_system["id"] = str(uuid.uuid4())
    demo_system["is_demo"] = True
    data_store_instance.create_system(demo_system)

    # List without demo
    systems = data_store_instance.list_systems(include_demo=False)

    assert any(s["id"] == regular_system["id"] for s in systems)
    assert not any(s["id"] == demo_system["id"] for s in systems)


def test_update_system(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test updating a system."""
    # Create system
    system = data_store_instance.create_system(sample_system_data)
    original_name = system["name"]

    # Update it
    updates = {"name": "Updated System Name", "health_score": 85.0}
    updated = data_store_instance.update_system(system["id"], updates)

    assert updated is not None
    assert updated["name"] == "Updated System Name"
    assert updated["name"] != original_name
    assert updated["health_score"] == 85.0
    assert "updated_at" in updated


def test_update_system_nonexistent(data_store_instance: DataStore):
    """Test updating a non-existent system returns None."""
    result = data_store_instance.update_system("nonexistent-id", {"name": "New Name"})
    assert result is None


def test_delete_system(data_store_instance: DataStore, sample_system_data: Dict[str, Any]):
    """Test deleting a system."""
    # Create system
    system = data_store_instance.create_system(sample_system_data)
    system_id = system["id"]

    # Verify it exists
    assert data_store_instance.get_system(system_id) is not None

    # Delete it
    result = data_store_instance.delete_system(system_id)
    assert result is True

    # Verify it's gone
    assert data_store_instance.get_system(system_id) is None


def test_delete_system_nonexistent(data_store_instance: DataStore):
    """Test deleting a non-existent system returns False."""
    result = data_store_instance.delete_system("nonexistent-id")
    assert result is False


def test_store_ingested_data(
    data_store_instance: DataStore,
    sample_system_data: Dict[str, Any],
    sample_ingested_records: list,
    sample_discovered_schema: Dict[str, Any]
):
    """Test storing ingested data."""
    # Create system first
    system = data_store_instance.create_system(sample_system_data)

    # Store ingested data
    source_id = str(uuid.uuid4())
    metadata = data_store_instance.store_ingested_data(
        system_id=system["id"],
        source_id=source_id,
        source_name="test_data.csv",
        records=sample_ingested_records,
        discovered_schema=sample_discovered_schema
    )

    assert metadata is not None
    assert metadata["source_id"] == source_id
    assert metadata["source_name"] == "test_data.csv"
    assert metadata["record_count"] == len(sample_ingested_records)
    assert "ingested_at" in metadata


def test_get_ingested_records(
    data_store_instance: DataStore,
    sample_system_data: Dict[str, Any],
    sample_ingested_records: list,
    sample_discovered_schema: Dict[str, Any]
):
    """Test retrieving ingested records."""
    # Create system and store data
    system = data_store_instance.create_system(sample_system_data)
    source_id = str(uuid.uuid4())

    data_store_instance.store_ingested_data(
        system_id=system["id"],
        source_id=source_id,
        source_name="test_data.csv",
        records=sample_ingested_records,
        discovered_schema=sample_discovered_schema
    )

    # Retrieve records
    records = data_store_instance.get_ingested_records(
        system_id=system["id"],
        source_id=source_id
    )

    assert len(records) == len(sample_ingested_records)
    assert records[0]["timestamp"] == sample_ingested_records[0]["timestamp"]
    assert records[0]["temperature"] == sample_ingested_records[0]["temperature"]


def test_get_ingested_records_with_pagination(
    data_store_instance: DataStore,
    sample_system_data: Dict[str, Any],
    sample_ingested_records: list,
    sample_discovered_schema: Dict[str, Any]
):
    """Test retrieving ingested records with pagination."""
    # Create system and store data
    system = data_store_instance.create_system(sample_system_data)
    source_id = str(uuid.uuid4())

    data_store_instance.store_ingested_data(
        system_id=system["id"],
        source_id=source_id,
        source_name="test_data.csv",
        records=sample_ingested_records,
        discovered_schema=sample_discovered_schema
    )

    # Get with limit
    records = data_store_instance.get_ingested_records(
        system_id=system["id"],
        source_id=source_id,
        limit=2
    )
    assert len(records) == 2

    # Get with offset
    records = data_store_instance.get_ingested_records(
        system_id=system["id"],
        source_id=source_id,
        limit=2,
        offset=1
    )
    assert len(records) == 2
    assert records[0]["timestamp"] == sample_ingested_records[1]["timestamp"]


def test_store_temp_analysis(data_store_instance: DataStore):
    """Test storing temporary analysis data."""
    analysis_id = str(uuid.uuid4())
    records = [{"field1": "value1", "field2": "value2"}]
    file_summaries = [{"filename": "test.csv", "row_count": 1}]
    discovered_fields = [{"name": "field1", "type": "string"}]
    file_records_map = {"test.csv": records}

    data_store_instance.store_temp_analysis(
        analysis_id=analysis_id,
        records=records,
        file_summaries=file_summaries,
        discovered_fields=discovered_fields,
        file_records_map=file_records_map
    )

    # Should not raise an error
    assert True


def test_get_temp_analysis(data_store_instance: DataStore):
    """Test retrieving temporary analysis data."""
    analysis_id = str(uuid.uuid4())
    records = [{"field1": "value1", "field2": "value2"}]
    file_summaries = [{"filename": "test.csv", "row_count": 1}]
    discovered_fields = [{"name": "field1", "type": "string"}]
    file_records_map = {"test.csv": records}

    # Store analysis
    data_store_instance.store_temp_analysis(
        analysis_id=analysis_id,
        records=records,
        file_summaries=file_summaries,
        discovered_fields=discovered_fields,
        file_records_map=file_records_map
    )

    # Retrieve it
    analysis = data_store_instance.get_temp_analysis(analysis_id)

    assert analysis is not None
    assert analysis["analysis_id"] == analysis_id
    assert len(analysis["records"]) == 1
    assert len(analysis["file_summaries"]) == 1
    assert "created_at" in analysis


def test_get_temp_analysis_nonexistent(data_store_instance: DataStore):
    """Test retrieving non-existent temp analysis returns None."""
    result = data_store_instance.get_temp_analysis("nonexistent-id")
    assert result is None


def test_delete_temp_analysis(data_store_instance: DataStore):
    """Test deleting temporary analysis data."""
    analysis_id = str(uuid.uuid4())
    records = [{"field1": "value1"}]
    file_summaries = [{"filename": "test.csv"}]
    discovered_fields = [{"name": "field1"}]
    file_records_map = {"test.csv": records}

    # Store analysis
    data_store_instance.store_temp_analysis(
        analysis_id=analysis_id,
        records=records,
        file_summaries=file_summaries,
        discovered_fields=discovered_fields,
        file_records_map=file_records_map
    )

    # Verify it exists
    assert data_store_instance.get_temp_analysis(analysis_id) is not None

    # Delete it
    result = data_store_instance.delete_temp_analysis(analysis_id)
    assert result is True

    # Verify it's gone
    assert data_store_instance.get_temp_analysis(analysis_id) is None


def test_delete_temp_analysis_nonexistent(data_store_instance: DataStore):
    """Test deleting non-existent temp analysis returns False."""
    result = data_store_instance.delete_temp_analysis("nonexistent-id")
    assert result is False
