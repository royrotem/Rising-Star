from sqlalchemy import Column, String, Text, ForeignKey, Float, Integer, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import enum
from .base import Base


class AnomalySeverity(str, enum.Enum):
    """Anomaly severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyStatus(str, enum.Enum):
    """Anomaly resolution status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AnomalyType(str, enum.Enum):
    """Types of detected anomalies."""
    BEHAVIORAL_DEVIATION = "behavioral_deviation"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_CHANGE = "pattern_change"
    CORRELATION_BREAK = "correlation_break"
    MARGIN_WARNING = "margin_warning"
    PREDICTIVE_FAILURE = "predictive_failure"


class Anomaly(Base):
    """Detected anomaly in system behavior."""

    __tablename__ = "anomalies"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Classification
    anomaly_type = Column(Enum(AnomalyType), nullable=False)
    severity = Column(Enum(AnomalySeverity), default=AnomalySeverity.MEDIUM)
    status = Column(Enum(AnomalyStatus), default=AnomalyStatus.OPEN)

    # Description
    title = Column(String(500), nullable=False)
    description = Column(Text)
    natural_language_explanation = Column(Text)  # AI-generated plain language explanation

    # Technical details
    affected_fields = Column(ARRAY(String))
    expected_values = Column(JSONB)
    actual_values = Column(JSONB)
    deviation_percentage = Column(Float)

    # Time context
    detected_at = Column(String(50))
    occurrence_start = Column(String(50))
    occurrence_end = Column(String(50))

    # Engineering context
    engineering_margin = Column(Float)  # How close to breaking point (0-100)
    margin_trend = Column(String(20))  # "decreasing", "stable", "increasing"

    # Root cause analysis
    root_cause_analysis = Column(JSONB)
    correlated_events = Column(JSONB)  # Related events (firmware updates, etc.)
    contributing_factors = Column(JSONB)

    # Recommendations
    recommendations = Column(JSONB)  # AI-generated recommendations
    similar_past_incidents = Column(ARRAY(UUID(as_uuid=True)))

    # Impact assessment
    impact_score = Column(Float)  # 80/20 impact score
    affected_fleet_percentage = Column(Float)
    estimated_downtime_hours = Column(Float)

    # Relationships
    from sqlalchemy.orm import relationship
    system = relationship("System", back_populates="anomalies")


class EngineeringMargin(Base):
    """Tracks engineering margins over time for a system."""

    __tablename__ = "engineering_margins"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Component identification
    component_name = Column(String(255), nullable=False)
    parameter_name = Column(String(255), nullable=False)

    # Margin data
    design_limit = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    margin_percentage = Column(Float, nullable=False)  # Distance from limit

    # Trend analysis
    trend_direction = Column(String(20))  # "improving", "stable", "degrading"
    trend_rate = Column(Float)  # Rate of change per day
    projected_breach_date = Column(String(50))  # When margin will hit zero

    # Safety classification
    safety_critical = Column(String(10), default="false")
    optimization_potential = Column(Float)  # Hidden buffer that can be utilized

    # Context
    environmental_factors = Column(JSONB)  # Temperature, humidity, etc.
    operating_conditions = Column(JSONB)
