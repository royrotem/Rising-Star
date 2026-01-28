from sqlalchemy import Column, String, Text, ForeignKey, Float, Integer, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import enum
from .base import Base


class InsightType(str, enum.Enum):
    """Types of generated insights."""
    BLIND_SPOT = "blind_spot"
    OPTIMIZATION = "optimization"
    DESIGN_RECOMMENDATION = "design_recommendation"
    PATTERN_DISCOVERY = "pattern_discovery"
    CROSS_DOMAIN_CORRELATION = "cross_domain_correlation"
    NEXT_GEN_SPEC = "next_gen_spec"


class InsightPriority(str, enum.Enum):
    """Priority levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Insight(Base):
    """AI-generated insight about system behavior or design."""

    __tablename__ = "insights"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"))
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))

    # Classification
    insight_type = Column(Enum(InsightType), nullable=False)
    priority = Column(Enum(InsightPriority), default=InsightPriority.MEDIUM)

    # Content
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False)
    detailed_analysis = Column(Text)
    natural_language_explanation = Column(Text)

    # 80/20 Impact
    impact_score = Column(Float)  # How much this affects overall fleet
    affected_systems_count = Column(Integer)
    potential_savings = Column(Float)  # Estimated cost/time savings

    # Supporting data
    evidence = Column(JSONB)  # Data points supporting this insight
    visualizations = Column(JSONB)  # Chart configurations
    related_fields = Column(ARRAY(String))

    # Recommendations
    recommendations = Column(JSONB)
    action_items = Column(JSONB)

    # For blind spot detection
    missing_data_description = Column(Text)
    recommended_sensors = Column(JSONB)

    # For next-gen specs
    next_gen_requirements = Column(JSONB)
    sensor_specifications = Column(JSONB)
    data_architecture_changes = Column(JSONB)

    # Status
    status = Column(String(50), default="new")  # "new", "reviewed", "actioned", "dismissed"
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    actioned_at = Column(String(50))

    # Relationships
    system = relationship("System", back_populates="insights")


class DataGap(Base):
    """Identified gaps in data collection (blind spots)."""

    __tablename__ = "data_gaps"

    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"), nullable=False)
    insight_id = Column(UUID(as_uuid=True), ForeignKey("insights.id"))

    # Gap description
    title = Column(String(500), nullable=False)
    description = Column(Text)

    # What's missing
    missing_measurement = Column(String(255))
    affected_diagnoses = Column(ARRAY(String))  # What can't be diagnosed
    recurring_issues = Column(JSONB)  # Issues that can't be explained

    # Recommended solution
    recommended_sensor = Column(JSONB)
    estimated_cost = Column(Float)
    implementation_complexity = Column(String(20))  # "low", "medium", "high"

    # Impact
    diagnostic_coverage_improvement = Column(Float)  # % improvement if fixed
    priority_score = Column(Float)
