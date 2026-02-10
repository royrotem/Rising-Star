from sqlalchemy import Column, String, Text, ForeignKey, Float, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import enum
from .base import Base


class CorrelationType(str, enum.Enum):
    """Types of correlations detected."""
    DIRECT = "direct"
    INVERSE = "inverse"
    LAGGED = "lagged"
    CONDITIONAL = "conditional"
    CAUSAL = "causal"


class RootCauseConfidence(str, enum.Enum):
    """Confidence level for root cause analysis."""
    CONFIRMED = "confirmed"
    HIGHLY_LIKELY = "highly_likely"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    SPECULATIVE = "speculative"


class RootCause(Base):
    """Root cause analysis for an anomaly."""

    __tablename__ = "root_causes"

    anomaly_id = Column(UUID(as_uuid=True), ForeignKey("anomalies.id"), nullable=False)

    # Analysis results
    title = Column(String(255), nullable=False)
    explanation = Column(Text, nullable=False)  # Natural language explanation
    technical_details = Column(Text)

    confidence = Column(Enum(RootCauseConfidence), default=RootCauseConfidence.POSSIBLE)
    confidence_score = Column(Float)  # 0-1

    # Causal chain
    causal_chain = Column(JSONB)  # Ordered list of events leading to anomaly
    contributing_factors = Column(JSONB)  # Additional factors

    # Evidence
    evidence = Column(JSONB)  # Supporting data points
    related_events = Column(JSONB)  # Related system events (firmware updates, etc.)

    # Recommendations
    recommended_actions = Column(JSONB)  # List of recommended fixes
    prevention_measures = Column(JSONB)

    # Verification
    is_verified = Column(Boolean, default=False)
    verified_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    verification_notes = Column(Text)

    # Relationships
    anomaly = relationship("Anomaly", back_populates="root_causes")


class Correlation(Base):
    """Discovered correlation between fields/events."""

    __tablename__ = "correlations"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)
    source_field_id = Column(UUID(as_uuid=True), ForeignKey("system_fields.id"), nullable=False)
    target_field_id = Column(UUID(as_uuid=True), ForeignKey("system_fields.id"), nullable=False)

    # Correlation details
    correlation_type = Column(Enum(CorrelationType), nullable=False)
    strength = Column(Float, nullable=False)  # -1 to 1
    p_value = Column(Float)  # Statistical significance

    # For lagged correlations
    lag_seconds = Column(Float)  # Time lag between source and target

    # Conditional correlations
    conditions = Column(JSONB)  # Under what conditions this correlation holds

    # Description
    description = Column(Text)  # AI-generated explanation
    is_significant = Column(Boolean, default=True)

    # Learning
    sample_count = Column(Float, default=0)
    last_updated = Column(Float)


class EngineeringMargin(Base):
    """Engineering margin analysis for system fields."""

    __tablename__ = "engineering_margins"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)
    field_id = Column(UUID(as_uuid=True), ForeignKey("system_fields.id"), nullable=False)

    # Limits
    absolute_min = Column(Float)  # Physical/safety minimum
    absolute_max = Column(Float)  # Physical/safety maximum
    operational_min = Column(Float)  # Normal operational minimum
    operational_max = Column(Float)  # Normal operational maximum

    # Current margins
    current_value = Column(Float)
    margin_to_min = Column(Float)  # Distance to minimum (percentage)
    margin_to_max = Column(Float)  # Distance to maximum (percentage)
    margin_percentage = Column(Float)  # Overall margin health (0-100)

    # Trends
    trend_direction = Column(String(20))  # "increasing", "decreasing", "stable"
    trend_rate = Column(Float)  # Rate of change per hour
    estimated_breach_time = Column(Float)  # Unix timestamp when limit might be reached

    # Analysis
    risk_level = Column(String(20))  # "low", "medium", "high", "critical"
    optimization_potential = Column(Float)  # How much margin can be safely reduced
    recommendations = Column(JSONB)

    # Historical
    historical_min = Column(Float)
    historical_max = Column(Float)
    breach_count = Column(Float, default=0)


class BlindSpot(Base):
    """Identified data gaps and blind spots in monitoring."""

    __tablename__ = "blind_spots"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)

    # Gap identification
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    gap_type = Column(String(100))  # "missing_sensor", "insufficient_resolution", "coverage_gap"

    # Impact
    affected_diagnoses = Column(JSONB)  # What can't be diagnosed due to this gap
    severity = Column(String(20))  # "low", "medium", "high", "critical"

    # Recommendation
    recommended_sensor = Column(String(255))
    recommended_spec = Column(JSONB)  # Specifications for the recommended sensor
    estimated_value = Column(Text)  # Business value of filling this gap

    # Status
    is_addressed = Column(Boolean, default=False)
    addressed_in_version = Column(String(50))  # e.g., "Model Z", "v2.0"
