from sqlalchemy import Column, String, Text, ForeignKey, Float, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import enum
from .base import Base


class InsightType(str, enum.Enum):
    """Types of insights generated."""
    ANOMALY = "anomaly"
    TREND = "trend"
    RECOMMENDATION = "recommendation"
    BLIND_SPOT = "blind_spot"
    OPTIMIZATION = "optimization"


class InsightPriority(str, enum.Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Insight(Base):
    """AI-generated insight about a system."""

    __tablename__ = "insights"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Classification
    insight_type = Column(Enum(InsightType), nullable=False)
    priority = Column(Enum(InsightPriority), default=InsightPriority.MEDIUM)

    # Content
    title = Column(String(500), nullable=False)
    description = Column(Text)
    natural_language_summary = Column(Text)

    # Impact
    impact_score = Column(Float)
    affected_components = Column(ARRAY(String))

    # Recommendations
    recommendations = Column(JSONB)
    
    # Status
    is_acknowledged = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)

    # Relationships
    system = relationship("System", back_populates="insights")


class DataGap(Base):
    """Identified gap in data coverage (blind spot)."""

    __tablename__ = "data_gaps"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Description
    title = Column(String(500), nullable=False)
    description = Column(Text)

    # What's missing
    missing_data_type = Column(String(100))
    recommended_sensor = Column(JSONB)

    # Impact
    diagnostic_coverage_improvement = Column(Float)
    related_anomalies = Column(ARRAY(UUID(as_uuid=True)))

    # Resolution
    is_addressed = Column(Boolean, default=False)
