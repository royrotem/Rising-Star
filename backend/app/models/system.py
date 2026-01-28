from sqlalchemy import Column, String, Text, ForeignKey, Boolean, Float, Integer, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
import enum
from .base import Base


class SystemStatus(str, enum.Enum):
    """System operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ANOMALY_DETECTED = "anomaly_detected"


class System(Base):
    """Represents a monitored hardware system (vehicle, robot, device, etc.)."""

    __tablename__ = "systems"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)

    # Identification
    name = Column(String(255), nullable=False)
    serial_number = Column(String(255), unique=True, index=True)
    system_type = Column(String(100))  # "vehicle", "robot", "medical_device", etc.
    model = Column(String(255))
    firmware_version = Column(String(100))

    # Status
    status = Column(Enum(SystemStatus), default=SystemStatus.ACTIVE)
    health_score = Column(Float, default=100.0)  # 0-100 overall health

    # Discovered schema (learned by AI agents)
    discovered_schema = Column(JSONB, default={})  # AI-discovered data structure
    confirmed_fields = Column(JSONB, default={})  # Human-confirmed field mappings
    field_relationships = Column(JSONB, default={})  # Discovered correlations

    # Metadata
    metadata = Column(JSONB, default={})
    tags = Column(ARRAY(String))

    # Relationships
    organization = relationship("Organization", back_populates="systems")
    data_sources = relationship("DataSource", back_populates="system", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="system", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="system", cascade="all, delete-orphan")


class DataSource(Base):
    """A data source connected to a system."""

    __tablename__ = "data_sources"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Source identification
    name = Column(String(255), nullable=False)
    source_type = Column(String(100))  # "can_bus", "csv_upload", "api_stream", "mqtt"
    connection_config = Column(JSONB, default={})

    # Schema discovery status
    discovery_status = Column(String(50), default="pending")  # "pending", "discovering", "discovered", "confirmed"
    discovered_fields = Column(JSONB, default={})
    sample_data = Column(JSONB)  # Sample records for reference

    # Statistics
    total_records = Column(Integer, default=0)
    last_data_received = Column(String(50))
    ingestion_rate = Column(Float)  # Records per second

    is_active = Column(Boolean, default=True)

    # Relationships
    system = relationship("System", back_populates="data_sources")
    time_series = relationship("TimeSeriesData", back_populates="data_source", cascade="all, delete-orphan")


class TimeSeriesData(Base):
    """Time series data storage (reference to actual data in TimescaleDB/InfluxDB)."""

    __tablename__ = "time_series_metadata"

    data_source_id = Column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)

    # Time range
    start_time = Column(String(50), nullable=False)
    end_time = Column(String(50), nullable=False)

    # Data location
    storage_location = Column(String(500))  # Path or reference to actual data
    compression = Column(String(50))
    record_count = Column(Integer)
    size_bytes = Column(Integer)

    # Statistics
    field_stats = Column(JSONB)  # Min, max, mean, std for each field

    # Relationships
    data_source = relationship("DataSource", back_populates="time_series")
