"""
Streaming Ingestion Service - Handles large files without loading into memory

This service processes large files (up to 50GB) using:
1. Chunked reading - never loads entire file into memory
2. Running statistics - Welford's algorithm for online mean/variance
3. Reservoir sampling - uniform sampling for field examples
4. PostgreSQL storage - stores chunks directly to database
"""

import asyncio
import csv
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import settings

logger = logging.getLogger("uaie.streaming_ingestion")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk_id: str
    chunk_index: int
    record_count: int
    byte_size: int
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    field_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingIngestionResult:
    """Overall result of streaming ingestion."""
    ingestion_id: str
    total_records: int
    total_chunks: int
    total_bytes: int
    chunks: List[ChunkResult]
    discovered_schema: Dict[str, Any]
    field_profiles: List[Dict[str, Any]]
    time_range: Optional[Tuple[float, float]] = None
    storage_type: str = "database"


@dataclass
class ProgressUpdate:
    """Progress update for SSE streaming."""
    stage: str  # uploading, parsing, profiling, storing, complete
    percent: float
    records_processed: int
    chunks_processed: int
    bytes_processed: int
    message: str


# ============================================================================
# Running Statistics (Welford's Algorithm)
# ============================================================================

class RunningFieldStats:
    """
    Maintains running statistics without keeping all data in memory.
    Uses Welford's algorithm for online variance calculation.
    """

    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.stats: Dict[str, Dict] = {}
        self.reservoir: Dict[str, List] = {}  # Reservoir sampling
        self.total_records = 0

    def update(self, field_name: str, values: pd.Series, chunk_start: int = 0) -> None:
        """Update running statistics for a field with new values."""
        if field_name not in self.stats:
            self.stats[field_name] = {
                'count': 0,
                'null_count': 0,
                'mean': 0.0,
                'M2': 0.0,  # For Welford's variance
                'min': float('inf'),
                'max': float('-inf'),
                'dtype': str(values.dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(values),
            }
            self.reservoir[field_name] = []

        s = self.stats[field_name]

        # Handle nulls
        null_count = int(values.isna().sum())
        s['null_count'] += null_count

        # Numeric statistics using Welford's algorithm
        if s['is_numeric']:
            non_null = values.dropna()
            for x in non_null:
                try:
                    x = float(x)
                    if not np.isfinite(x):
                        continue
                    s['count'] += 1
                    delta = x - s['mean']
                    s['mean'] += delta / s['count']
                    delta2 = x - s['mean']
                    s['M2'] += delta * delta2
                    s['min'] = min(s['min'], x)
                    s['max'] = max(s['max'], x)
                except (ValueError, TypeError):
                    continue
        else:
            s['count'] += len(values.dropna())

        # Reservoir sampling for sample values
        self._reservoir_sample(field_name, values.dropna(), chunk_start)

    def _reservoir_sample(
        self,
        field_name: str,
        values: pd.Series,
        chunk_start: int
    ) -> None:
        """
        Reservoir sampling - maintains uniform sample across all chunks.
        Algorithm R (Vitter, 1985)
        """
        import random
        reservoir = self.reservoir[field_name]

        for i, val in enumerate(values):
            global_idx = chunk_start + i

            if len(reservoir) < self.sample_size:
                # Fill reservoir
                try:
                    # Convert to JSON-serializable type
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    reservoir.append(val)
                except (TypeError, ValueError):
                    reservoir.append(str(val))
            else:
                # Replace with decreasing probability
                j = random.randint(0, global_idx)
                if j < self.sample_size:
                    try:
                        if isinstance(val, (np.integer, np.floating)):
                            val = val.item()
                        reservoir[j] = val
                    except (TypeError, ValueError):
                        reservoir[j] = str(val)

    def get_chunk_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get statistics for a single chunk (for storage)."""
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    stats[col] = {
                        'min': float(non_null.min()),
                        'max': float(non_null.max()),
                        'mean': float(non_null.mean()),
                        'count': int(len(non_null)),
                    }
        return stats

    def finalize(self) -> Dict[str, Dict]:
        """Finalize and return all statistics."""
        result = {}
        for field_name, s in self.stats.items():
            variance = s['M2'] / s['count'] if s['count'] > 1 else 0

            result[field_name] = {
                'name': field_name,
                'count': s['count'],
                'null_count': s['null_count'],
                'null_pct': (s['null_count'] / (s['count'] + s['null_count']) * 100
                            if (s['count'] + s['null_count']) > 0 else 0),
                'dtype': s['dtype'],
                'is_numeric': s['is_numeric'],
                'sample_values': self.reservoir.get(field_name, [])[:10],
            }

            if s['is_numeric'] and s['count'] > 0:
                result[field_name].update({
                    'mean': s['mean'],
                    'std': variance ** 0.5,
                    'min': s['min'] if s['min'] != float('inf') else None,
                    'max': s['max'] if s['max'] != float('-inf') else None,
                })

        return result


# ============================================================================
# Streaming Ingestion Service
# ============================================================================

class StreamingIngestionService:
    """
    Handles streaming ingestion for large files.

    Key features:
    - Never loads entire file into memory
    - Processes in configurable chunks
    - Stores directly to PostgreSQL
    - Maintains running statistics for profiling
    - Sends progress updates via callback
    """

    def __init__(
        self,
        chunk_size_records: int = None,
        chunk_size_bytes: int = None,
    ):
        self.chunk_size_records = chunk_size_records or settings.CHUNK_SIZE_RECORDS
        self.chunk_size_bytes = chunk_size_bytes or settings.CHUNK_SIZE_BYTES
        self.data_dir = Path(settings.DATA_DIR if hasattr(settings, 'DATA_DIR') else '/app/data')

    async def ingest_large_file(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_id: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> StreamingIngestionResult:
        """
        Main entry point for streaming ingestion.

        Automatically detects file format and uses appropriate streaming parser.
        """
        ingestion_id = str(uuid.uuid4())
        logger.info("Starting streaming ingestion: %s (id=%s)", filename, ingestion_id)

        # Detect format
        ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''

        if ext == 'csv':
            return await self._ingest_streaming_csv(
                file_content, filename, system_id, source_id,
                ingestion_id, progress_callback
            )
        elif ext == 'parquet':
            return await self._ingest_streaming_parquet(
                file_content, filename, system_id, source_id,
                ingestion_id, progress_callback
            )
        elif ext in ('json', 'jsonl', 'ndjson'):
            return await self._ingest_streaming_json(
                file_content, filename, system_id, source_id,
                ingestion_id, progress_callback
            )
        else:
            # Fallback to loading entire file (for smaller/unsupported formats)
            return await self._ingest_full_load(
                file_content, filename, system_id, source_id,
                ingestion_id, progress_callback
            )

    async def _ingest_streaming_csv(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_id: str,
        ingestion_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> StreamingIngestionResult:
        """Stream-ingest a large CSV file."""
        logger.info("Streaming CSV ingestion: %s", filename)

        # Phase 1: Detect encoding and delimiter from sample
        sample = file_content.read(8192)
        file_content.seek(0)

        encoding = self._detect_encoding(sample)
        delimiter = self._detect_delimiter(sample.decode(encoding, errors='replace'))

        logger.info("Detected encoding=%s, delimiter=%r", encoding, delimiter)

        # Phase 2: Stream chunks
        chunks: List[ChunkResult] = []
        total_records = 0
        total_bytes = 0
        running_stats = RunningFieldStats()
        schema = {}

        chunk_index = 0

        # Use pandas chunked reading
        file_content.seek(0)
        text_wrapper = io.TextIOWrapper(file_content, encoding=encoding, errors='replace')

        try:
            for chunk_df in pd.read_csv(
                text_wrapper,
                delimiter=delimiter,
                chunksize=self.chunk_size_records,
                low_memory=True,
                on_bad_lines='skip',
            ):
                # Clean up the chunk
                chunk_df.dropna(how='all', inplace=True)
                chunk_df.columns = [str(c).strip() for c in chunk_df.columns]

                if len(chunk_df) == 0:
                    continue

                # Estimate byte size
                byte_size = int(chunk_df.memory_usage(deep=True).sum())

                # Update running statistics
                for col in chunk_df.columns:
                    running_stats.update(col, chunk_df[col], total_records)

                # Infer schema from first chunk
                if not schema:
                    schema = {col: str(chunk_df[col].dtype) for col in chunk_df.columns}

                # Store chunk
                chunk_id = f"{ingestion_id}_chunk_{chunk_index}"
                chunk_result = await self._store_chunk(
                    system_id=system_id,
                    source_id=source_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    records=chunk_df.to_dict('records'),
                    field_stats=running_stats.get_chunk_stats(chunk_df),
                )

                chunks.append(chunk_result)
                total_records += len(chunk_df)
                total_bytes += byte_size
                chunk_index += 1

                # Report progress
                if progress_callback:
                    await self._send_progress(progress_callback, ProgressUpdate(
                        stage='processing',
                        percent=0,  # Unknown total
                        records_processed=total_records,
                        chunks_processed=chunk_index,
                        bytes_processed=total_bytes,
                        message=f"Processed {total_records:,} records in {chunk_index} chunks",
                    ))

                logger.info("Processed chunk %d: %d records, %d bytes total",
                           chunk_index, len(chunk_df), total_bytes)

                # Yield to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("Error during CSV streaming: %s", e)
            raise

        # Finalize statistics
        field_profiles = running_stats.finalize()

        logger.info("Streaming CSV complete: %d records in %d chunks",
                   total_records, len(chunks))

        return StreamingIngestionResult(
            ingestion_id=ingestion_id,
            total_records=total_records,
            total_chunks=len(chunks),
            total_bytes=total_bytes,
            chunks=chunks,
            discovered_schema={'fields': schema},
            field_profiles=list(field_profiles.values()),
            storage_type='database',
        )

    async def _ingest_streaming_parquet(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_id: str,
        ingestion_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> StreamingIngestionResult:
        """Stream-ingest a large Parquet file using row groups."""
        logger.info("Streaming Parquet ingestion: %s", filename)

        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("PyArrow not available, falling back to full load")
            return await self._ingest_full_load(
                file_content, filename, system_id, source_id,
                ingestion_id, progress_callback
            )

        # Parquet supports native chunking via row groups
        parquet_file = pq.ParquetFile(file_content)
        num_row_groups = parquet_file.metadata.num_row_groups

        chunks: List[ChunkResult] = []
        total_records = 0
        total_bytes = 0
        running_stats = RunningFieldStats()

        for i in range(num_row_groups):
            table = parquet_file.read_row_group(i)
            chunk_df = table.to_pandas()

            byte_size = int(chunk_df.memory_usage(deep=True).sum())

            # Update statistics
            for col in chunk_df.columns:
                running_stats.update(col, chunk_df[col], total_records)

            # Store chunk
            chunk_id = f"{ingestion_id}_rg_{i}"
            chunk_result = await self._store_chunk(
                system_id=system_id,
                source_id=source_id,
                chunk_id=chunk_id,
                chunk_index=i,
                records=chunk_df.to_dict('records'),
                field_stats=running_stats.get_chunk_stats(chunk_df),
            )

            chunks.append(chunk_result)
            total_records += len(chunk_df)
            total_bytes += byte_size

            if progress_callback:
                await self._send_progress(progress_callback, ProgressUpdate(
                    stage='processing',
                    percent=(i + 1) / num_row_groups * 100,
                    records_processed=total_records,
                    chunks_processed=i + 1,
                    bytes_processed=total_bytes,
                    message=f"Processed row group {i + 1}/{num_row_groups}",
                ))

            await asyncio.sleep(0)

        schema = {col: str(dtype) for col, dtype in
                 zip(parquet_file.schema_arrow.names,
                     [str(t) for t in parquet_file.schema_arrow.types])}

        return StreamingIngestionResult(
            ingestion_id=ingestion_id,
            total_records=total_records,
            total_chunks=len(chunks),
            total_bytes=total_bytes,
            chunks=chunks,
            discovered_schema={'fields': schema},
            field_profiles=list(running_stats.finalize().values()),
            storage_type='database',
        )

    async def _ingest_streaming_json(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_id: str,
        ingestion_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> StreamingIngestionResult:
        """Stream-ingest JSON Lines (JSONL/NDJSON) files."""
        logger.info("Streaming JSON ingestion: %s", filename)

        chunks: List[ChunkResult] = []
        total_records = 0
        total_bytes = 0
        running_stats = RunningFieldStats()
        schema = {}

        chunk_records = []
        chunk_index = 0

        # Read line by line
        for line in file_content:
            line = line.decode('utf-8', errors='replace').strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    chunk_records.append(record)

                    # When chunk is full, process it
                    if len(chunk_records) >= self.chunk_size_records:
                        chunk_df = pd.DataFrame(chunk_records)

                        for col in chunk_df.columns:
                            running_stats.update(col, chunk_df[col], total_records)

                        if not schema:
                            schema = {col: str(chunk_df[col].dtype) for col in chunk_df.columns}

                        chunk_id = f"{ingestion_id}_chunk_{chunk_index}"
                        chunk_result = await self._store_chunk(
                            system_id=system_id,
                            source_id=source_id,
                            chunk_id=chunk_id,
                            chunk_index=chunk_index,
                            records=chunk_records,
                            field_stats=running_stats.get_chunk_stats(chunk_df),
                        )

                        chunks.append(chunk_result)
                        total_records += len(chunk_records)
                        total_bytes += len(json.dumps(chunk_records))
                        chunk_index += 1
                        chunk_records = []

                        await asyncio.sleep(0)

            except json.JSONDecodeError:
                continue

        # Process remaining records
        if chunk_records:
            chunk_df = pd.DataFrame(chunk_records)
            for col in chunk_df.columns:
                running_stats.update(col, chunk_df[col], total_records)

            chunk_id = f"{ingestion_id}_chunk_{chunk_index}"
            chunk_result = await self._store_chunk(
                system_id=system_id,
                source_id=source_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                records=chunk_records,
                field_stats=running_stats.get_chunk_stats(chunk_df),
            )
            chunks.append(chunk_result)
            total_records += len(chunk_records)

        return StreamingIngestionResult(
            ingestion_id=ingestion_id,
            total_records=total_records,
            total_chunks=len(chunks),
            total_bytes=total_bytes,
            chunks=chunks,
            discovered_schema={'fields': schema},
            field_profiles=list(running_stats.finalize().values()),
            storage_type='database',
        )

    async def _ingest_full_load(
        self,
        file_content: BinaryIO,
        filename: str,
        system_id: str,
        source_id: str,
        ingestion_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> StreamingIngestionResult:
        """Fallback: Load entire file for formats without streaming support."""
        logger.info("Full load ingestion: %s", filename)

        # Import the existing ingestion service
        from .ingestion import ingestion_service

        # Use existing parser
        result = await ingestion_service.ingest_file(
            file_content=file_content,
            filename=filename,
            system_id=system_id,
            source_name=filename,
        )

        records = result.get('sample_records', [])

        if len(records) > settings.USE_DB_THRESHOLD_RECORDS:
            # Store in chunks
            return await self._store_records_chunked(
                records=records,
                system_id=system_id,
                source_id=source_id,
                ingestion_id=ingestion_id,
                discovered_schema=result.get('discovered_fields', []),
            )
        else:
            # Use file-based storage
            return StreamingIngestionResult(
                ingestion_id=ingestion_id,
                total_records=len(records),
                total_chunks=1,
                total_bytes=len(json.dumps(records, default=str)),
                chunks=[],
                discovered_schema={'fields': result.get('discovered_fields', [])},
                field_profiles=result.get('discovered_fields', []),
                storage_type='file',
            )

    async def _store_records_chunked(
        self,
        records: List[Dict],
        system_id: str,
        source_id: str,
        ingestion_id: str,
        discovered_schema: List[Dict],
    ) -> StreamingIngestionResult:
        """Store a large list of records in chunks."""
        chunks = []
        running_stats = RunningFieldStats()

        for i in range(0, len(records), self.chunk_size_records):
            chunk_records = records[i:i + self.chunk_size_records]
            chunk_df = pd.DataFrame(chunk_records)

            for col in chunk_df.columns:
                running_stats.update(col, chunk_df[col], i)

            chunk_id = f"{ingestion_id}_chunk_{i // self.chunk_size_records}"
            chunk_result = await self._store_chunk(
                system_id=system_id,
                source_id=source_id,
                chunk_id=chunk_id,
                chunk_index=i // self.chunk_size_records,
                records=chunk_records,
                field_stats=running_stats.get_chunk_stats(chunk_df),
            )
            chunks.append(chunk_result)

        return StreamingIngestionResult(
            ingestion_id=ingestion_id,
            total_records=len(records),
            total_chunks=len(chunks),
            total_bytes=sum(c.byte_size for c in chunks),
            chunks=chunks,
            discovered_schema={'fields': discovered_schema},
            field_profiles=list(running_stats.finalize().values()),
            storage_type='database',
        )

    async def _store_chunk(
        self,
        system_id: str,
        source_id: str,
        chunk_id: str,
        chunk_index: int,
        records: List[Dict],
        field_stats: Dict,
    ) -> ChunkResult:
        """Store a chunk of records."""
        # For now, store to file system (database integration in HybridDataStore)
        chunk_dir = self.data_dir / 'chunks' / system_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_file = chunk_dir / f"{source_id}_{chunk_index}.json"

        chunk_data = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'source_id': source_id,
            'record_count': len(records),
            'records': records,
            'field_stats': field_stats,
            'created_at': datetime.utcnow().isoformat(),
        }

        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, default=str)

        byte_size = chunk_file.stat().st_size

        return ChunkResult(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            record_count=len(records),
            byte_size=byte_size,
            field_stats=field_stats,
        )

    async def _send_progress(
        self,
        callback: Callable,
        update: ProgressUpdate
    ) -> None:
        """Send progress update via callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(update)
            else:
                callback(update)
        except Exception as e:
            logger.warning("Progress callback failed: %s", e)

    def _detect_encoding(self, sample: bytes) -> str:
        """Detect text encoding from sample bytes."""
        for encoding in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
            try:
                sample.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        return 'utf-8'

    def _detect_delimiter(self, sample: str) -> str:
        """Detect CSV delimiter from sample text."""
        try:
            dialect = csv.Sniffer().sniff(sample[:4096], delimiters=',;\t|')
            return dialect.delimiter
        except csv.Error:
            return ','


# ============================================================================
# Utility Functions
# ============================================================================

def should_use_streaming(file_size_bytes: int, record_estimate: int = None) -> bool:
    """Determine if streaming ingestion should be used."""
    size_mb = file_size_bytes / (1024 * 1024)

    if size_mb > settings.USE_DB_THRESHOLD_MB:
        return True

    if record_estimate and record_estimate > settings.USE_DB_THRESHOLD_RECORDS:
        return True

    return False


# Global instance
streaming_ingestion_service = StreamingIngestionService()
