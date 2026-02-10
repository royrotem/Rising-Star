"""
Data Storage Service

Simple file-based storage for ingested data and system configurations.
In production, this would use PostgreSQL/TimescaleDB.
"""

import json
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DataStore:
    """
    File-based data store for persisting system data and ingested records.
    Thread-safe implementation for concurrent access.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use /app/data in Docker, ./data locally
            data_dir = os.environ.get("DATA_DIR", "/app/data")
            if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.systems_dir = self.data_dir / "systems"
        self.systems_dir.mkdir(exist_ok=True)

        self.ingested_dir = self.data_dir / "ingested"
        self.ingested_dir.mkdir(exist_ok=True)

        self.schemas_dir = self.data_dir / "schemas"
        self.schemas_dir.mkdir(exist_ok=True)

        self.temp_dir = self.data_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        # In-memory cache
        self._systems_cache: Dict[str, Dict] = {}
        self._temp_analysis_cache: Dict[str, Dict] = {}  # Cache for temp analysis data
        self._load_systems()

    def _load_systems(self):
        """Load all systems from disk into memory cache."""
        with self._lock:
            for system_file in self.systems_dir.glob("*.json"):
                try:
                    with open(system_file) as f:
                        system = json.load(f)
                        self._systems_cache[system["id"]] = system
                except Exception as e:
                    print(f"Error loading system {system_file}: {e}")

    # ============ System Operations ============

    def create_system(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new system record."""
        with self._lock:
            system_id = system_data["id"]
            system_data["created_at"] = system_data.get("created_at", datetime.utcnow().isoformat())
            system_data["updated_at"] = datetime.utcnow().isoformat()

            # Save to file
            system_file = self.systems_dir / f"{system_id}.json"
            with open(system_file, "w") as f:
                json.dump(system_data, f, indent=2, default=str)

            # Update cache
            self._systems_cache[system_id] = system_data

            # Create system data directory
            system_data_dir = self.ingested_dir / system_id
            system_data_dir.mkdir(exist_ok=True)

            return system_data

    def get_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get a system by ID."""
        with self._lock:
            return self._systems_cache.get(system_id)

    def list_systems(self, include_demo: bool = True) -> List[Dict[str, Any]]:
        """List all systems."""
        with self._lock:
            systems = list(self._systems_cache.values())
            if not include_demo:
                systems = [s for s in systems if not s.get("is_demo", False)]
            return systems

    def update_system(self, system_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a system."""
        with self._lock:
            if system_id not in self._systems_cache:
                return None

            system = self._systems_cache[system_id]
            system.update(updates)
            system["updated_at"] = datetime.utcnow().isoformat()

            # Save to file
            system_file = self.systems_dir / f"{system_id}.json"
            with open(system_file, "w") as f:
                json.dump(system, f, indent=2, default=str)

            return system

    def delete_system(self, system_id: str) -> bool:
        """Delete a system and all its data."""
        with self._lock:
            if system_id not in self._systems_cache:
                return False

            # Remove from cache
            del self._systems_cache[system_id]

            # Remove files
            system_file = self.systems_dir / f"{system_id}.json"
            if system_file.exists():
                system_file.unlink()

            # Remove ingested data
            system_data_dir = self.ingested_dir / system_id
            if system_data_dir.exists():
                shutil.rmtree(system_data_dir)

            return True

    # ============ Ingested Data Operations ============

    def store_ingested_data(
        self,
        system_id: str,
        source_id: str,
        source_name: str,
        records: List[Dict],
        discovered_schema: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Store ingested data for a system."""
        with self._lock:
            system_data_dir = self.ingested_dir / system_id
            system_data_dir.mkdir(exist_ok=True)

            # Store records
            records_file = system_data_dir / f"{source_id}_records.json"
            with open(records_file, "w") as f:
                json.dump(records, f, default=str)

            # Store schema
            schema_file = self.schemas_dir / f"{system_id}_{source_id}_schema.json"
            with open(schema_file, "w") as f:
                json.dump(discovered_schema, f, indent=2, default=str)

            # Store metadata
            source_metadata = {
                "source_id": source_id,
                "source_name": source_name,
                "system_id": system_id,
                "record_count": len(records),
                "ingested_at": datetime.utcnow().isoformat(),
                "schema": discovered_schema,
                **(metadata or {})
            }

            metadata_file = system_data_dir / f"{source_id}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(source_metadata, f, indent=2, default=str)

            # Update system with data source info
            if system_id in self._systems_cache:
                system = self._systems_cache[system_id]
                if "data_sources" not in system:
                    system["data_sources"] = []
                system["data_sources"].append({
                    "source_id": source_id,
                    "source_name": source_name,
                    "record_count": len(records),
                    "ingested_at": source_metadata["ingested_at"]
                })
                self.update_system(system_id, system)

            return source_metadata

    def get_ingested_records(
        self,
        system_id: str,
        source_id: str = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict]:
        """Get ingested records for a system."""
        with self._lock:
            system_data_dir = self.ingested_dir / system_id
            if not system_data_dir.exists():
                return []

            all_records = []

            if source_id:
                # Get specific source
                records_file = system_data_dir / f"{source_id}_records.json"
                if records_file.exists():
                    with open(records_file) as f:
                        all_records = json.load(f)
            else:
                # Get all sources
                for records_file in system_data_dir.glob("*_records.json"):
                    with open(records_file) as f:
                        all_records.extend(json.load(f))

            return all_records[offset:offset + limit]

    def get_data_sources(self, system_id: str) -> List[Dict]:
        """Get all data sources for a system."""
        with self._lock:
            system_data_dir = self.ingested_dir / system_id
            if not system_data_dir.exists():
                return []

            sources = []
            for metadata_file in system_data_dir.glob("*_metadata.json"):
                with open(metadata_file) as f:
                    sources.append(json.load(f))

            return sources

    def get_schema(self, system_id: str, source_id: str = None) -> Optional[Dict]:
        """Get discovered schema for a system/source."""
        with self._lock:
            if source_id:
                schema_file = self.schemas_dir / f"{system_id}_{source_id}_schema.json"
                if schema_file.exists():
                    with open(schema_file) as f:
                        return json.load(f)
            else:
                # Return combined schema from all sources
                schemas = {}
                for schema_file in self.schemas_dir.glob(f"{system_id}_*_schema.json"):
                    with open(schema_file) as f:
                        schema = json.load(f)
                        schemas.update(schema)
                return schemas if schemas else None

            return None

    # ============ Statistics Operations ============

    def get_system_statistics(self, system_id: str) -> Dict[str, Any]:
        """Get statistics for a system's data."""
        sources = self.get_data_sources(system_id)

        # Get actual total record count from source metadata (not limited)
        total_records = sum(s.get("record_count", 0) for s in sources)

        # Get records for field statistics (limited sample is fine for stats)
        records = self.get_ingested_records(system_id, limit=10000)

        if not records:
            return {
                "total_records": total_records,
                "total_sources": len(sources),
                "field_count": 0,
                "fields": []
            }

        df = pd.DataFrame(records)

        field_stats = []
        for col in df.columns:
            stat = {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique())
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                stat.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                })

            field_stats.append(stat)

        return {
            "total_records": total_records,  # Use actual count from metadata, not limited records
            "total_sources": len(sources),
            "field_count": len(df.columns),
            "fields": field_stats
        }

    # ============ Temporary Analysis Storage ============

    def store_temp_analysis(
        self,
        analysis_id: str,
        records: List[Dict],
        file_summaries: List[Dict],
        discovered_fields: List[Dict],
        file_records_map: Dict[str, List[Dict]],
    ) -> None:
        """Store temporary analysis data before system creation."""
        with self._lock:
            analysis_data = {
                "analysis_id": analysis_id,
                "records": records,
                "file_summaries": file_summaries,
                "discovered_fields": discovered_fields,
                "file_records_map": file_records_map,
                "created_at": datetime.utcnow().isoformat(),
            }

            # Store in memory cache
            self._temp_analysis_cache[analysis_id] = analysis_data

            # Also persist to disk
            analysis_file = self.temp_dir / f"{analysis_id}.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis_data, f, default=str)

    def get_temp_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get temporary analysis data."""
        with self._lock:
            # Check memory cache first
            if analysis_id in self._temp_analysis_cache:
                return self._temp_analysis_cache[analysis_id]

            # Try loading from disk
            analysis_file = self.temp_dir / f"{analysis_id}.json"
            if analysis_file.exists():
                with open(analysis_file) as f:
                    data = json.load(f)
                    self._temp_analysis_cache[analysis_id] = data
                    return data

            return None

    def move_temp_to_system(self, analysis_id: str, system_id: str) -> bool:
        """Move temporary analysis data to a system."""
        with self._lock:
            analysis_data = self.get_temp_analysis(analysis_id)
            if not analysis_data:
                return False

            # Store each file's records as a separate source
            file_records_map = analysis_data.get("file_records_map", {})
            file_summaries = analysis_data.get("file_summaries", [])

            for summary in file_summaries:
                filename = summary.get("filename", "unknown")
                records = file_records_map.get(filename, [])

                if records:
                    source_id = str(uuid.uuid4())

                    self.store_ingested_data(
                        system_id=system_id,
                        source_id=source_id,
                        source_name=filename,
                        records=records,
                        discovered_schema={
                            "fields": [f for f in analysis_data.get("discovered_fields", [])
                                       if f.get("source_file") == filename],
                            "relationships": summary.get("relationships", []),
                        },
                        metadata={"filename": filename}
                    )

            # Update system with schema
            self.update_system(system_id, {
                "discovered_schema": analysis_data.get("discovered_fields", []),
                "status": "data_ingested"
            })

            # Clean up temp data
            self.delete_temp_analysis(analysis_id)

            return True

    def delete_temp_analysis(self, analysis_id: str) -> bool:
        """Delete temporary analysis data."""
        with self._lock:
            # Remove from cache
            if analysis_id in self._temp_analysis_cache:
                del self._temp_analysis_cache[analysis_id]

            # Remove from disk
            analysis_file = self.temp_dir / f"{analysis_id}.json"
            if analysis_file.exists():
                analysis_file.unlink()
                return True

            return False


# Global data store instance
data_store = DataStore()
