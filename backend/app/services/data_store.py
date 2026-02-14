"""
Data Storage Service

Hybrid storage for ingested data and system configurations.
- Small datasets: File-based JSON storage (fast, simple)
- Large datasets: PostgreSQL with chunked JSONB storage (scalable to 50GB+)

The storage backend is chosen automatically based on thresholds.
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Callable

import pandas as pd

from ..core.config import settings

# Try to import asyncpg for PostgreSQL support
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None


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

        # In-memory cache — atomic write prevents corruption on crash
        self._systems_cache: Dict[str, Dict] = {}
        self._temp_analysis_cache: Dict[str, Dict] = {}  # Cache for temp analysis data
        self._load_systems()

    @staticmethod
    def _atomic_write(file_path: Path, data: Any, indent: int = None) -> None:
        """Write JSON data atomically: write to temp file, then rename.

        This prevents data corruption if the process crashes mid-write.
        """
        dir_path = file_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=indent, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(file_path))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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

            # Save to file (atomic write)
            system_file = self.systems_dir / f"{system_id}.json"
            self._atomic_write(system_file, system_data, indent=2)

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

            # Save to file (atomic write)
            system_file = self.systems_dir / f"{system_id}.json"
            self._atomic_write(system_file, system, indent=2)

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

            # Store records (atomic write)
            records_file = system_data_dir / f"{source_id}_records.json"
            self._atomic_write(records_file, records)

            # Store schema (atomic write)
            schema_file = self.schemas_dir / f"{system_id}_{source_id}_schema.json"
            self._atomic_write(schema_file, discovered_schema, indent=2)

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

            # Atomic write for metadata
            metadata_file = system_data_dir / f"{source_id}_metadata.json"
            self._atomic_write(metadata_file, source_metadata, indent=2)

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

            # Also persist to disk (atomic write)
            analysis_file = self.temp_dir / f"{analysis_id}.json"
            self._atomic_write(analysis_file, analysis_data)

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


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Data Store — Automatic File/PostgreSQL Selection
# ═══════════════════════════════════════════════════════════════════════════


class HybridDataStore:
    """
    Hybrid data store that automatically selects storage backend.

    - Small data (< 50K records or < 100MB): File-based JSON storage
    - Large data: PostgreSQL with chunked JSONB storage

    This enables handling datasets up to 50GB while maintaining fast
    performance for smaller datasets.
    """

    # Storage thresholds (from config)
    RECORD_THRESHOLD = getattr(settings, 'USE_DB_THRESHOLD_RECORDS', 50_000)
    SIZE_THRESHOLD_MB = getattr(settings, 'USE_DB_THRESHOLD_MB', 100)
    CHUNK_SIZE = getattr(settings, 'CHUNK_SIZE_RECORDS', 100_000)

    def __init__(
        self,
        file_store: DataStore = None,
        db_url: str = None,
    ):
        """
        Initialize hybrid data store.

        Args:
            file_store: Existing DataStore instance for file-based storage
            db_url: PostgreSQL connection URL (from DATABASE_URL env var)
        """
        self.file_store = file_store or DataStore()
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self._db_pool: Optional[asyncpg.Pool] = None
        self._lock = threading.RLock()

        # Track which sources use which backend
        self._backend_map: Dict[str, str] = {}  # source_id -> "file" | "postgres"

    async def initialize(self) -> None:
        """Initialize database connection pool if available."""
        if HAS_ASYNCPG and self.db_url:
            try:
                self._db_pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=2,
                    max_size=10,
                )
                print("HybridDataStore: PostgreSQL connection pool initialized")
            except Exception as e:
                print(f"HybridDataStore: Failed to connect to PostgreSQL: {e}")
                print("HybridDataStore: Falling back to file-based storage only")
                self._db_pool = None
        else:
            if not HAS_ASYNCPG:
                print("HybridDataStore: asyncpg not installed, using file storage only")
            if not self.db_url:
                print("HybridDataStore: DATABASE_URL not set, using file storage only")

    async def close(self) -> None:
        """Close database connections."""
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None

    def _should_use_db(
        self,
        record_count: int = 0,
        file_size_mb: float = 0,
    ) -> bool:
        """Determine if PostgreSQL should be used based on thresholds."""
        if not self._db_pool:
            return False

        return (
            record_count > self.RECORD_THRESHOLD or
            file_size_mb > self.SIZE_THRESHOLD_MB
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Chunked Storage for Large Datasets
    # ═══════════════════════════════════════════════════════════════════════

    async def store_chunk(
        self,
        system_id: str,
        source_id: str,
        chunk_index: int,
        records: List[Dict],
        field_stats: Dict[str, Any] = None,
    ) -> None:
        """
        Store a chunk of records to PostgreSQL.

        Args:
            system_id: System identifier
            source_id: Data source identifier
            chunk_index: Index of this chunk (0-based)
            records: List of record dictionaries
            field_stats: Optional field statistics for this chunk
        """
        if not self._db_pool:
            raise RuntimeError("PostgreSQL not available for chunked storage")

        async with self._db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO time_series_chunks
                    (system_id, source_id, chunk_index, data, record_count, field_stats)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb)
                ON CONFLICT (system_id, source_id, chunk_index)
                DO UPDATE SET
                    data = EXCLUDED.data,
                    record_count = EXCLUDED.record_count,
                    field_stats = EXCLUDED.field_stats
            """,
                system_id,
                source_id,
                chunk_index,
                json.dumps(records, default=str),
                len(records),
                json.dumps(field_stats, default=str) if field_stats else None,
            )

        with self._lock:
            self._backend_map[source_id] = "postgres"

    async def store_large_data_streaming(
        self,
        system_id: str,
        source_id: str,
        source_name: str,
        chunk_iterator: AsyncIterator[pd.DataFrame],
        total_records: int = None,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> Dict[str, Any]:
        """
        Store large dataset by streaming chunks to PostgreSQL.

        Args:
            system_id: System identifier
            source_id: Data source identifier
            source_name: Human-readable source name
            chunk_iterator: Async iterator yielding DataFrame chunks
            total_records: Optional total record count for progress
            progress_callback: Optional callback(processed, total, status)

        Returns:
            Metadata about the stored data
        """
        if not self._db_pool:
            raise RuntimeError("PostgreSQL not available for large data storage")

        chunk_index = 0
        total_stored = 0
        start_time = datetime.utcnow()

        # Create ingestion job
        job_id = str(uuid.uuid4())
        async with self._db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ingestion_jobs
                    (id, system_id, source_id, source_name, status, started_at)
                VALUES ($1, $2, $3, $4, 'in_progress', NOW())
            """, job_id, system_id, source_id, source_name)

        try:
            async for chunk_df in chunk_iterator:
                records = chunk_df.to_dict(orient='records')

                await self.store_chunk(
                    system_id=system_id,
                    source_id=source_id,
                    chunk_index=chunk_index,
                    records=records,
                )

                total_stored += len(records)
                chunk_index += 1

                if progress_callback:
                    progress_callback(
                        total_stored,
                        total_records or 0,
                        f"Stored chunk {chunk_index}"
                    )

            # Update job as completed
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = 'completed',
                        completed_at = NOW(),
                        total_records = $2,
                        chunk_count = $3
                    WHERE id = $1
                """, job_id, total_stored, chunk_index)

                # Store source metadata
                await conn.execute("""
                    INSERT INTO data_source_metadata
                        (system_id, source_id, source_name, total_records,
                         chunk_count, storage_type, ingested_at)
                    VALUES ($1, $2, $3, $4, $5, 'chunked', NOW())
                    ON CONFLICT (system_id, source_id)
                    DO UPDATE SET
                        total_records = EXCLUDED.total_records,
                        chunk_count = EXCLUDED.chunk_count,
                        ingested_at = NOW()
                """, system_id, source_id, source_name, total_stored, chunk_index)

        except Exception as e:
            # Update job as failed
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = 'failed', error_message = $2, completed_at = NOW()
                    WHERE id = $1
                """, job_id, str(e))
            raise

        return {
            "source_id": source_id,
            "source_name": source_name,
            "system_id": system_id,
            "record_count": total_stored,
            "chunk_count": chunk_index,
            "storage_type": "chunked",
            "ingested_at": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
        }

    async def get_records_streaming(
        self,
        system_id: str,
        source_id: str,
        chunk_size: int = None,
    ) -> AsyncIterator[List[Dict]]:
        """
        Stream records from PostgreSQL in chunks.

        Args:
            system_id: System identifier
            source_id: Data source identifier
            chunk_size: Optional chunk size (uses stored chunk size if not specified)

        Yields:
            Lists of record dictionaries
        """
        if not self._db_pool:
            # Fall back to file-based
            records = self.file_store.get_ingested_records(system_id, source_id)
            yield records
            return

        async with self._db_pool.acquire() as conn:
            # Get total chunks
            row = await conn.fetchrow("""
                SELECT MAX(chunk_index) + 1 as chunk_count
                FROM time_series_chunks
                WHERE system_id = $1 AND source_id = $2
            """, system_id, source_id)

            if not row or row['chunk_count'] is None:
                return

            chunk_count = row['chunk_count']

            for chunk_idx in range(chunk_count):
                row = await conn.fetchrow("""
                    SELECT data FROM time_series_chunks
                    WHERE system_id = $1 AND source_id = $2 AND chunk_index = $3
                """, system_id, source_id, chunk_idx)

                if row:
                    yield json.loads(row['data'])

    async def get_total_record_count(
        self,
        system_id: str,
        source_id: str = None,
    ) -> int:
        """Get total record count for a system/source."""
        if not self._db_pool:
            # Fall back to file store
            sources = self.file_store.get_data_sources(system_id)
            if source_id:
                for s in sources:
                    if s.get("source_id") == source_id:
                        return s.get("record_count", 0)
                return 0
            return sum(s.get("record_count", 0) for s in sources)

        async with self._db_pool.acquire() as conn:
            if source_id:
                row = await conn.fetchrow("""
                    SELECT total_records FROM data_source_metadata
                    WHERE system_id = $1 AND source_id = $2
                """, system_id, source_id)
                return row['total_records'] if row else 0
            else:
                row = await conn.fetchrow("""
                    SELECT SUM(total_records) as total FROM data_source_metadata
                    WHERE system_id = $1
                """, system_id)
                return row['total'] if row and row['total'] else 0

    # ═══════════════════════════════════════════════════════════════════════
    # Unified Interface (auto-select backend)
    # ═══════════════════════════════════════════════════════════════════════

    async def store_ingested_data(
        self,
        system_id: str,
        source_id: str,
        source_name: str,
        records: List[Dict],
        discovered_schema: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        file_size_mb: float = 0,
    ) -> Dict[str, Any]:
        """
        Store ingested data, automatically selecting backend.

        For small datasets: Uses file-based storage
        For large datasets: Uses PostgreSQL chunked storage
        """
        record_count = len(records)

        if self._should_use_db(record_count, file_size_mb):
            # Use PostgreSQL with chunking
            async def chunk_generator():
                for i in range(0, len(records), self.CHUNK_SIZE):
                    chunk = records[i:i + self.CHUNK_SIZE]
                    yield pd.DataFrame(chunk)

            return await self.store_large_data_streaming(
                system_id=system_id,
                source_id=source_id,
                source_name=source_name,
                chunk_iterator=chunk_generator(),
                total_records=record_count,
            )
        else:
            # Use file-based storage
            with self._lock:
                self._backend_map[source_id] = "file"

            return self.file_store.store_ingested_data(
                system_id=system_id,
                source_id=source_id,
                source_name=source_name,
                records=records,
                discovered_schema=discovered_schema,
                metadata=metadata,
            )

    async def get_ingested_records(
        self,
        system_id: str,
        source_id: str = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Get ingested records, using appropriate backend.

        For chunked data, this loads all required chunks.
        For large datasets, consider using get_records_streaming instead.
        """
        backend = self._backend_map.get(source_id, "file")

        if backend == "postgres" and self._db_pool:
            # Collect records from chunks
            all_records = []
            async for chunk in self.get_records_streaming(system_id, source_id):
                all_records.extend(chunk)
                if len(all_records) >= offset + limit:
                    break
            return all_records[offset:offset + limit]
        else:
            return self.file_store.get_ingested_records(
                system_id, source_id, limit, offset
            )

    def get_storage_info(self, source_id: str) -> Dict[str, Any]:
        """Get storage information for a source."""
        backend = self._backend_map.get(source_id, "unknown")
        return {
            "source_id": source_id,
            "backend": backend,
            "db_available": self._db_pool is not None,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Passthrough methods to file_store
    # ═══════════════════════════════════════════════════════════════════════

    def create_system(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.file_store.create_system(system_data)

    def get_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        return self.file_store.get_system(system_id)

    def list_systems(self, include_demo: bool = True) -> List[Dict[str, Any]]:
        return self.file_store.list_systems(include_demo)

    def update_system(self, system_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.file_store.update_system(system_id, updates)

    def delete_system(self, system_id: str) -> bool:
        return self.file_store.delete_system(system_id)

    def get_data_sources(self, system_id: str) -> List[Dict]:
        return self.file_store.get_data_sources(system_id)

    def get_schema(self, system_id: str, source_id: str = None) -> Optional[Dict]:
        return self.file_store.get_schema(system_id, source_id)

    def store_temp_analysis(self, *args, **kwargs):
        return self.file_store.store_temp_analysis(*args, **kwargs)

    def get_temp_analysis(self, analysis_id: str):
        return self.file_store.get_temp_analysis(analysis_id)

    def move_temp_to_system(self, analysis_id: str, system_id: str) -> bool:
        return self.file_store.move_temp_to_system(analysis_id, system_id)


# Global data store instances
data_store = DataStore()
hybrid_store = HybridDataStore(file_store=data_store)
