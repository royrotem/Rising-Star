from sqlalchemy import Column, String, Text, ForeignKey, Float, BigInteger, JSON, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import enum
from .base import Base


class DataSourceType(str, enum.Enum):
    """Types of data sources supported."""
    CAN_BUS = "can_bus"
    JSON_LOG = "json_log"
    CSV = "csv"
    PARQUET = "parquet"
    BINARY = "binary"
    API_STREAM = "api_stream"
    MQTT = "mqtt"


class IngestionStatus(str, enum.Enum):
    """Status of data ingestion."""
    PENDING = "pending"
    PROCESSING = "processing"
    MAPPING = "mapping"  # AI is mapping the schema
    AWAITING_VERIFICATION = "awaiting_verification"
    COMPLETED = "completed"
    FAILED = "failed"


class DataSource(Base):
    """Represents a data source for a system."""

    __tablename__ = "data_sources"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)
    name = Column(String(255), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    status = Column(Enum(IngestionStatus), default=IngestionStatus.PENDING)

    # Schema information (discovered or provided)
    discovered_schema = Column(JSONB)  # AI-discovered schema
    verified_schema = Column(JSONB)  # Human-verified schema

    # Statistics
    total_records = Column(BigInteger, default=0)
    first_timestamp = Column(Float)  # Unix timestamp
    last_timestamp = Column(Float)
    file_path = Column(String(500))

    # Processing metadata
    processing_errors = Column(JSON, default=[])
    ingestion_metadata = Column(JSON, default={})

    # Relationships
    system = relationship("System", back_populates="data_sources")
    time_series = relationship("TimeSeries", back_populates="data_source", cascade="all, delete-orphan")


class TimeSeries(Base):
    """Time series data storage for system fields."""

    __tablename__ = "time_series"

    data_source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)
    field_id = Column(UUID(as_uuid=True), ForeignKey("system_fields.id"), nullable=False)

    # Time range for this chunk
    start_time = Column(Float, nullable=False)  # Unix timestamp
    end_time = Column(Float, nullable=False)

    # Aggregated data stored as arrays (for efficient querying)
    timestamps = Column(JSONB)  # Array of timestamps
    values = Column(JSONB)  # Array of values
    sample_count = Column(BigInteger, default=0)

    # Statistics for this chunk
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)
    std_deviation = Column(Float)

    # Relationships
    data_source = relationship("DataSource", back_populates="time_series")

    __table_args__ = (
        Index("idx_timeseries_field_time", "field_id", "start_time", "end_time"),
    )


class DataPoint(Base):
    """Individual data point for real-time streaming data."""

    __tablename__ = "data_points"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)
    field_id = Column(UUID(as_uuid=True), ForeignKey("system_fields.id"), nullable=False)
    timestamp = Column(Float, nullable=False, index=True)
    value = Column(Float, nullable=False)
    metadata = Column(JSONB)

    __table_args__ = (
        Index("idx_datapoint_field_time", "field_id", "timestamp"),
        Index("idx_datapoint_system_time", "system_id", "timestamp"),
    )
