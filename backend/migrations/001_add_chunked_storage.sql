-- Migration: Add chunked storage tables for large datasets
-- Version: 001
-- Description: Creates tables for storing large datasets in chunks with PostgreSQL

-- ============================================================================
-- Table: time_series_chunks
-- Stores data in chunks for memory-efficient processing of large files
-- ============================================================================

CREATE TABLE IF NOT EXISTS time_series_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(255) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,

    -- Data stored as JSONB array of records
    data JSONB NOT NULL,
    record_count INTEGER NOT NULL,

    -- Time range for this chunk (if applicable)
    min_timestamp DOUBLE PRECISION,
    max_timestamp DOUBLE PRECISION,

    -- Per-chunk statistics for efficient querying
    field_stats JSONB,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure unique chunks per source
    CONSTRAINT unique_chunk UNIQUE (system_id, source_id, chunk_index)
);

-- Indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_system_id
    ON time_series_chunks(system_id);

CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON time_series_chunks(system_id, source_id);

CREATE INDEX IF NOT EXISTS idx_chunks_time_range
    ON time_series_chunks(min_timestamp, max_timestamp);

CREATE INDEX IF NOT EXISTS idx_chunks_created
    ON time_series_chunks(created_at);

-- ============================================================================
-- Table: ingestion_jobs
-- Tracks progress of large file ingestion operations
-- ============================================================================

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(255) NOT NULL,
    source_name VARCHAR(500) NOT NULL,

    -- Job status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    -- pending, uploading, processing, profiling, complete, error

    -- Size tracking
    total_bytes BIGINT,
    processed_bytes BIGINT DEFAULT 0,
    total_records BIGINT,
    processed_records BIGINT DEFAULT 0,

    -- Chunk tracking
    chunks_total INTEGER,
    chunks_processed INTEGER DEFAULT 0,

    -- Error handling
    error_message TEXT,
    error_details JSONB,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Progress percentage (computed)
    progress_percent DOUBLE PRECISION GENERATED ALWAYS AS (
        CASE
            WHEN total_bytes > 0 THEN (processed_bytes::DOUBLE PRECISION / total_bytes) * 100
            WHEN total_records > 0 THEN (processed_records::DOUBLE PRECISION / total_records) * 100
            ELSE 0
        END
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_jobs_system
    ON ingestion_jobs(system_id);

CREATE INDEX IF NOT EXISTS idx_jobs_status
    ON ingestion_jobs(status);

CREATE INDEX IF NOT EXISTS idx_jobs_created
    ON ingestion_jobs(created_at);

-- ============================================================================
-- Table: data_source_metadata
-- Extended metadata for data sources (especially large ones)
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_source_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(255) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    source_name VARCHAR(500) NOT NULL,

    -- Storage info
    storage_type VARCHAR(50) NOT NULL DEFAULT 'file',
    -- 'file' for JSON files, 'database' for chunked PostgreSQL storage

    -- Size info
    total_records BIGINT,
    total_bytes BIGINT,
    chunk_count INTEGER,

    -- Schema info (discovered fields)
    discovered_schema JSONB,
    field_statistics JSONB,

    -- Complex type detections
    complex_types JSONB,

    -- Timestamps
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ,

    CONSTRAINT unique_source UNIQUE (system_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_metadata_system
    ON data_source_metadata(system_id);

-- ============================================================================
-- Helper function: Get records from chunks with pagination
-- ============================================================================

CREATE OR REPLACE FUNCTION get_chunked_records(
    p_system_id VARCHAR(255),
    p_source_id VARCHAR(255) DEFAULT NULL,
    p_limit INTEGER DEFAULT 1000,
    p_offset INTEGER DEFAULT 0
)
RETURNS JSONB AS $$
DECLARE
    result JSONB := '[]'::JSONB;
    chunk_record RECORD;
    records_needed INTEGER := p_limit;
    records_skipped INTEGER := 0;
BEGIN
    FOR chunk_record IN
        SELECT data, record_count
        FROM time_series_chunks
        WHERE system_id = p_system_id
          AND (p_source_id IS NULL OR source_id = p_source_id)
        ORDER BY chunk_index
    LOOP
        -- Skip records for offset
        IF records_skipped < p_offset THEN
            IF records_skipped + chunk_record.record_count <= p_offset THEN
                records_skipped := records_skipped + chunk_record.record_count;
                CONTINUE;
            ELSE
                -- Partial skip within this chunk
                DECLARE
                    skip_in_chunk INTEGER := p_offset - records_skipped;
                    take_from_chunk INTEGER := LEAST(records_needed, chunk_record.record_count - skip_in_chunk);
                BEGIN
                    result := result || (
                        SELECT jsonb_agg(elem)
                        FROM (
                            SELECT elem
                            FROM jsonb_array_elements(chunk_record.data) WITH ORDINALITY AS t(elem, idx)
                            WHERE idx > skip_in_chunk
                            LIMIT take_from_chunk
                        ) sub
                    );
                    records_needed := records_needed - take_from_chunk;
                    records_skipped := p_offset;
                END;
            END IF;
        ELSE
            -- Take records from this chunk
            DECLARE
                take_from_chunk INTEGER := LEAST(records_needed, chunk_record.record_count);
            BEGIN
                result := result || (
                    SELECT jsonb_agg(elem)
                    FROM (
                        SELECT elem
                        FROM jsonb_array_elements(chunk_record.data) WITH ORDINALITY AS t(elem, idx)
                        LIMIT take_from_chunk
                    ) sub
                );
                records_needed := records_needed - take_from_chunk;
            END;
        END IF;

        EXIT WHEN records_needed <= 0;
    END LOOP;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Helper function: Get total record count for a system/source
-- ============================================================================

CREATE OR REPLACE FUNCTION get_total_record_count(
    p_system_id VARCHAR(255),
    p_source_id VARCHAR(255) DEFAULT NULL
)
RETURNS BIGINT AS $$
BEGIN
    RETURN (
        SELECT COALESCE(SUM(record_count), 0)
        FROM time_series_chunks
        WHERE system_id = p_system_id
          AND (p_source_id IS NULL OR source_id = p_source_id)
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Cleanup: Remove old chunks when system is deleted
-- ============================================================================

CREATE OR REPLACE FUNCTION cleanup_system_chunks()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM time_series_chunks WHERE system_id = OLD.system_id;
    DELETE FROM ingestion_jobs WHERE system_id = OLD.system_id;
    DELETE FROM data_source_metadata WHERE system_id = OLD.system_id;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Note: Trigger should be created on the systems table if it exists
-- CREATE TRIGGER cleanup_chunks_on_delete
--     AFTER DELETE ON systems
--     FOR EACH ROW
--     EXECUTE FUNCTION cleanup_system_chunks();
