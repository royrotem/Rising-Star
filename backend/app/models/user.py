from sqlalchemy import Column, String, Text, ForeignKey, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
import enum
from .base import Base


class UserRole(str, enum.Enum):
    """User roles in the system."""
    ADMIN = "admin"
    ENGINEER = "engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Organization(Base):
    """Organization/company using the platform."""

    __tablename__ = "organizations"

    name = Column(String(255), nullable=False)
    description = Column(Text)
    industry = Column(String(100))  # "aerospace", "automotive", "robotics", "medtech"

    # Settings
    settings = Column(JSON, default={})
    features_enabled = Column(ARRAY(String))  # Enabled platform features

    # Subscription
    subscription_tier = Column(String(50), default="free")  # "free", "pro", "enterprise"
    is_active = Column(Boolean, default=True)

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    systems = relationship("System", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    """User account."""

    __tablename__ = "users"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)

    # Profile
    full_name = Column(String(255))
    role = Column(Enum(UserRole), default=UserRole.VIEWER)

    # Preferences
    preferences = Column(JSON, default={})
    notification_settings = Column(JSON, default={})

    # Relationships
    organization = relationship("Organization", back_populates="users")


class ConversationHistory(Base):
    """Stores conversation history with the AI assistant."""

    __tablename__ = "conversation_history"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    system_id = Column(UUID(as_uuid=True), ForeignKey("systems.id"))

    # Conversation
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    context = Column(JSONB)  # Additional context (referenced data, etc.)

    # For assistant responses
    query_type = Column(String(100))  # Type of query answered
    data_retrieved = Column(JSONB)  # Summary of data used in response
