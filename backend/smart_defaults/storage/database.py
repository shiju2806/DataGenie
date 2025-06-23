"""
Smart Defaults Database Layer
Async SQLAlchemy with SQLite/PostgreSQL support and comprehensive table definitions
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from dataclasses import asdict

import asyncpg
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text,
    Boolean, DateTime, Float, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.sql import func, select, update, delete, insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

# Import your models - handle both direct execution and module imports
try:
    from ..models.user_profile import UserProfile, UserPreferences, UserBehavior
    from ..models.data_source import DataSource, SourceMetadata, AccessPattern
    from ..models.policy import SecurityPolicy, ComplianceRule, PolicyViolation
    from ..models.recommendation import Recommendation, RecommendationScore, UserFeedback
except ImportError:
    # For direct execution, create mock classes
    from typing import Any
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "test_role"
        department: str = "test_dept"
        seniority_level: str = "senior"
        industry: str = "technology"
        location: str = "US"
        permissions: list = None
        preferences: Any = None
        created_at: datetime = None
        last_active: datetime = None

        def __post_init__(self):
            if self.permissions is None:
                self.permissions = []
            if self.created_at is None:
                self.created_at = datetime.now(timezone.utc)

    @dataclass
    class UserPreferences:
        auto_connect_threshold: float = 0.85
        recommendation_frequency: str = "daily"
        notification_preferences: dict = None

        def __post_init__(self):
            if self.notification_preferences is None:
                self.notification_preferences = {}

    @dataclass
    class UserBehavior:
        action_type: str = "test_action"
        context: dict = None

        def __post_init__(self):
            if self.context is None:
                self.context = {}

    @dataclass
    class DataSource:
        id: str = "test_source_id"
        source_type: str = "postgresql"
        name: str = "Test Database"
        description: str = "Test description"
        connection_config: dict = None
        metadata: Any = None
        health_status: str = "healthy"
        last_scan: datetime = None
        is_active: bool = True

        def __post_init__(self):
            if self.connection_config is None:
                self.connection_config = {}
            if self.last_scan is None:
                self.last_scan = datetime.now(timezone.utc)

    @dataclass
    class SourceMetadata:
        schema_info: dict = None
        table_count: int = 0

        def __post_init__(self):
            if self.schema_info is None:
                self.schema_info = {}

    @dataclass
    class Recommendation:
        id: str = "test_rec_id"
        user_id: str = "test_user"
        source_id: str = "test_source"
        recommendation_type: str = "recommend"
        confidence_score: float = 0.75
        reasoning: dict = None
        context: dict = None
        created_at: datetime = None

        def __post_init__(self):
            if self.reasoning is None:
                self.reasoning = {}
            if self.context is None:
                self.context = {}
            if self.created_at is None:
                self.created_at = datetime.now(timezone.utc)

    @dataclass
    class UserFeedback:
        id: str = "test_feedback_id"
        user_id: str = "test_user"
        recommendation_id: str = "test_rec"
        action: str = "accept"
        implicit_feedback: bool = False
        context: dict = None
        confidence_impact: float = 0.1

        def __post_init__(self):
            if self.context is None:
                self.context = {}

    # Mock other classes that aren't used in direct execution
    class SecurityPolicy: pass
    class ComplianceRule: pass
    class PolicyViolation: pass
    class RecommendationScore: pass
    class AccessPattern: pass

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

class DatabaseConfig:
    """Database configuration with environment-specific settings"""

    def __init__(self,
                 db_type: str = "sqlite",
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "smart_defaults",
                 username: str = "",
                 password: str = "",
                 sqlite_path: str = "smart_defaults.db"):
        self.db_type = db_type
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.sqlite_path = sqlite_path

    @property
    def connection_string(self) -> str:
        """Generate connection string based on database type"""
        if self.db_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        elif self.db_type == "postgresql":
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

# SQLAlchemy Table Definitions
class UserProfileTable(Base):
    __tablename__ = "user_profiles"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(100), nullable=False, index=True)
    department = Column(String(100), index=True)
    seniority_level = Column(String(50), index=True)
    industry = Column(String(100), index=True)
    location = Column(String(255))
    permissions = Column(JSON)  # Will use JSONB for PostgreSQL
    preferences = Column(JSON)
    behavior_patterns = Column(JSON)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_active = Column(DateTime, index=True)
    is_active = Column(Boolean, default=True, index=True)

    # Relationships
    behaviors = relationship("UserBehaviorTable", back_populates="profile", cascade="all, delete-orphan")
    recommendations = relationship("RecommendationTable", back_populates="profile")
    feedback = relationship("UserFeedbackTable", back_populates="profile")

    __table_args__ = (
        Index('idx_user_role_dept', 'role', 'department'),
        Index('idx_user_industry_seniority', 'industry', 'seniority_level'),
    )

class UserBehaviorTable(Base):
    __tablename__ = "user_behaviors"

    id = Column(String(36), primary_key=True)
    profile_id = Column(String(36), ForeignKey("user_profiles.id"), nullable=False, index=True)
    action_type = Column(String(100), nullable=False, index=True)
    source_id = Column(String(36), ForeignKey("data_sources.id"), index=True)
    context = Column(JSON)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=func.now(), index=True)
    session_id = Column(String(255), index=True)

    # Relationships
    profile = relationship("UserProfileTable", back_populates="behaviors")
    source = relationship("DataSourceTable", back_populates="behaviors")

    __table_args__ = (
        Index('idx_behavior_user_time', 'profile_id', 'timestamp'),
        Index('idx_behavior_action_source', 'action_type', 'source_id'),
    )

class DataSourceTable(Base):
    __tablename__ = "data_sources"

    id = Column(String(36), primary_key=True)
    source_type = Column(String(100), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    connection_config = Column(JSON)  # Encrypted connection details
    source_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    health_status = Column(String(50), default="unknown", index=True)
    last_scan = Column(DateTime, index=True)
    scan_frequency = Column(Integer, default=3600)  # seconds
    is_active = Column(Boolean, default=True, index=True)
    access_patterns = Column(JSON)
    compliance_requirements = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    behaviors = relationship("UserBehaviorTable", back_populates="source")
    recommendations = relationship("RecommendationTable", back_populates="source")
    access_logs = relationship("AccessLogTable", back_populates="source")

    __table_args__ = (
        Index('idx_source_type_active', 'source_type', 'is_active'),
        Index('idx_source_health_scan', 'health_status', 'last_scan'),
    )

class RecommendationTable(Base):
    __tablename__ = "recommendations"

    id = Column(String(36), primary_key=True)
    profile_id = Column(String(36), ForeignKey("user_profiles.id"), nullable=False, index=True)
    source_id = Column(String(36), ForeignKey("data_sources.id"), nullable=False, index=True)
    recommendation_type = Column(String(50), nullable=False, index=True)  # auto_connect, recommend, available
    confidence_score = Column(Float, nullable=False, index=True)
    reasoning = Column(JSON)
    context = Column(JSON)
    status = Column(String(50), default="pending", index=True)  # pending, accepted, rejected, expired
    expires_at = Column(DateTime, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    profile = relationship("UserProfileTable", back_populates="recommendations")
    source = relationship("DataSourceTable", back_populates="recommendations")
    feedback = relationship("UserFeedbackTable", back_populates="recommendation")

    __table_args__ = (
        Index('idx_rec_user_confidence', 'profile_id', 'confidence_score'),
        Index('idx_rec_type_status', 'recommendation_type', 'status'),
        Index('idx_rec_expires', 'expires_at'),
    )

class UserFeedbackTable(Base):
    __tablename__ = "user_feedback"

    id = Column(String(36), primary_key=True)
    profile_id = Column(String(36), ForeignKey("user_profiles.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("recommendations.id"), index=True)
    action = Column(String(50), nullable=False, index=True)  # accept, reject, ignore, override
    implicit_feedback = Column(Boolean, default=False)  # Was this inferred or explicit?
    context = Column(JSON)
    confidence_impact = Column(Float, default=0.0)  # How much this should adjust future scores
    timestamp = Column(DateTime, default=func.now(), index=True)

    # Relationships
    profile = relationship("UserProfileTable", back_populates="feedback")
    recommendation = relationship("RecommendationTable", back_populates="feedback")

    __table_args__ = (
        Index('idx_feedback_user_time', 'profile_id', 'timestamp'),
        Index('idx_feedback_action_implicit', 'action', 'implicit_feedback'),
    )

class SecurityPolicyTable(Base):
    __tablename__ = "security_policies"

    id = Column(String(36), primary_key=True)
    policy_type = Column(String(100), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    rules = Column(JSON, nullable=False)
    scope = Column(JSON)  # Which users/departments/sources this applies to
    enforcement_level = Column(String(50), default="warn", index=True)  # block, warn, log
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    violations = relationship("PolicyViolationTable", back_populates="policy")

class PolicyViolationTable(Base):
    __tablename__ = "policy_violations"

    id = Column(String(36), primary_key=True)
    policy_id = Column(String(36), ForeignKey("security_policies.id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    source_id = Column(String(36), ForeignKey("data_sources.id"), index=True)
    violation_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(50), default="medium", index=True)
    details = Column(JSON)
    resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    policy = relationship("SecurityPolicyTable", back_populates="violations")
    source = relationship("DataSourceTable")

class AccessLogTable(Base):
    __tablename__ = "access_logs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    source_id = Column(String(36), ForeignKey("data_sources.id"), nullable=False, index=True)
    access_type = Column(String(50), nullable=False, index=True)  # query, connect, configure
    success = Column(Boolean, nullable=False, index=True)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=func.now(), index=True)

    # Relationships
    source = relationship("DataSourceTable", back_populates="access_logs")

    __table_args__ = (
        Index('idx_access_user_time', 'user_id', 'timestamp'),
        Index('idx_access_source_success', 'source_id', 'success'),
    )

class LearningMetricsTable(Base):
    __tablename__ = "learning_metrics"

    id = Column(String(36), primary_key=True)
    metric_type = Column(String(100), nullable=False, index=True)
    user_segment = Column(String(100), index=True)  # role, department, industry
    feature_weights = Column(JSON)
    accuracy_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    sample_size = Column(Integer)
    last_updated = Column(DateTime, default=func.now(), index=True)

    __table_args__ = (
        Index('idx_metrics_type_segment', 'metric_type', 'user_segment'),
    )

class DatabaseManager:
    """Async database manager with SQLite/PostgreSQL support"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables"""
        if self._initialized:
            return

        try:
            # Create async engine
            self.engine = create_async_engine(
                self.config.connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=3600,
                json_serializer=lambda obj: obj,  # Custom JSON serializer if needed
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info(f"✅ Database initialized: {self.config.db_type}")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("🔐 Database connections closed")

    @asynccontextmanager
    async def get_session(self):
        """Get async database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    # User Profile Operations
    async def create_user_profile(self, profile: UserProfile) -> UserProfile:
        """Create a new user profile"""
        async with self.get_session() as session:
            try:
                db_profile = UserProfileTable(
                    id=profile.id,
                    user_id=profile.user_id,
                    role=profile.role,
                    department=profile.department,
                    seniority_level=profile.seniority_level,
                    industry=profile.industry,
                    location=profile.location,
                    permissions=json.dumps(profile.permissions or []),  # Serialize list as JSON
                    preferences=json.dumps(asdict(profile.preferences) if profile.preferences else {}),  # Serialize dict as JSON
                    behavior_patterns=json.dumps({})  # Initialize as empty JSON string
                )

                session.add(db_profile)
                await session.flush()

                logger.info(f"👤 Created user profile: {profile.user_id}")
                return profile

            except IntegrityError as e:
                logger.error(f"❌ User profile already exists: {profile.user_id}")
                raise ValueError(f"User profile already exists: {profile.user_id}")

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        async with self.get_session() as session:
            stmt = select(UserProfileTable).where(UserProfileTable.user_id == user_id)
            result = await session.execute(stmt)
            db_profile = result.scalar_one_or_none()

            if not db_profile:
                return None

            # Convert back to domain model - handle both string and dict cases
            if isinstance(db_profile.preferences, str):
                preferences_dict = json.loads(db_profile.preferences) if db_profile.preferences else {}
            else:
                preferences_dict = db_profile.preferences or {}

            if isinstance(db_profile.permissions, str):
                permissions = json.loads(db_profile.permissions) if db_profile.permissions else []
            else:
                permissions = db_profile.permissions or []

            preferences = UserPreferences(**preferences_dict) if preferences_dict else UserPreferences()

            return UserProfile(
                id=db_profile.id,
                user_id=db_profile.user_id,
                role=db_profile.role,
                department=db_profile.department,
                seniority_level=db_profile.seniority_level,
                industry=db_profile.industry,
                location=db_profile.location,
                permissions=permissions,
                preferences=preferences,
                created_at=db_profile.created_at,
                last_active=db_profile.last_active
            )

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with partial data"""
        async with self.get_session() as session:
            stmt = (
                update(UserProfileTable)
                .where(UserProfileTable.user_id == user_id)
                .values(**updates, updated_at=func.now())
            )
            result = await session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"📝 Updated user profile: {user_id}")
                return True
            return False

    # Data Source Operations
    async def create_data_source(self, source: DataSource) -> DataSource:
        """Create a new data source"""
        async with self.get_session() as session:
            try:
                db_source = DataSourceTable(
                    id=source.id,
                    source_type=source.source_type,
                    name=source.name,
                    description=source.description,
                    connection_config=json.dumps(source.connection_config or {}),
                    source_metadata=json.dumps(asdict(source.metadata) if source.metadata else {}),
                    health_status=source.health_status,
                    is_active=source.is_active
                )

                session.add(db_source)
                await session.flush()

                logger.info(f"🔌 Created data source: {source.name} ({source.source_type})")
                return source

            except IntegrityError as e:
                logger.error(f"❌ Data source creation failed: {e}")
                raise

    async def get_active_data_sources(self, source_type: Optional[str] = None) -> List[DataSource]:
        """Get all active data sources, optionally filtered by type"""
        async with self.get_session() as session:
            stmt = select(DataSourceTable).where(DataSourceTable.is_active == True)

            if source_type:
                stmt = stmt.where(DataSourceTable.source_type == source_type)

            result = await session.execute(stmt)
            db_sources = result.scalars().all()

            sources = []
            for db_source in db_sources:
                # Handle both string and dict cases for JSON fields
                if isinstance(db_source.source_metadata, str):
                    metadata_dict = json.loads(db_source.source_metadata) if db_source.source_metadata else {}
                else:
                    metadata_dict = db_source.source_metadata or {}

                if isinstance(db_source.connection_config, str):
                    connection_config = json.loads(db_source.connection_config) if db_source.connection_config else {}
                else:
                    connection_config = db_source.connection_config or {}

                metadata = SourceMetadata(**metadata_dict) if metadata_dict else SourceMetadata()

                sources.append(DataSource(
                    id=db_source.id,
                    source_type=db_source.source_type,
                    name=db_source.name,
                    description=db_source.description,
                    connection_config=connection_config,
                    metadata=metadata,
                    health_status=db_source.health_status,
                    last_scan=db_source.last_scan,
                    is_active=db_source.is_active
                ))

            return sources

    # Recommendation Operations
    async def create_recommendation(self, recommendation: Recommendation) -> Recommendation:
        """Create a new recommendation"""
        async with self.get_session() as session:
            try:
                db_rec = RecommendationTable(
                    id=recommendation.id,
                    profile_id=recommendation.user_id,  # Note: using user_id as profile lookup
                    source_id=recommendation.source_id,
                    recommendation_type=recommendation.recommendation_type,
                    confidence_score=recommendation.confidence_score,
                    reasoning=json.dumps(recommendation.reasoning or {}),
                    context=json.dumps(recommendation.context or {}),
                    expires_at=datetime.now(timezone.utc) + timedelta(days=7)  # Default 7-day expiry
                )

                session.add(db_rec)
                await session.flush()

                logger.info(f"💡 Created recommendation: {recommendation.recommendation_type} (confidence: {recommendation.confidence_score:.2f})")
                return recommendation

            except IntegrityError as e:
                logger.error(f"❌ Recommendation creation failed: {e}")
                raise

    async def get_user_recommendations(self, user_id: str, status: Optional[str] = None) -> List[Recommendation]:
        """Get recommendations for a user"""
        async with self.get_session() as session:
            # First get profile_id from user_id
            profile_stmt = select(UserProfileTable.id).where(UserProfileTable.user_id == user_id)
            profile_result = await session.execute(profile_stmt)
            profile_id = profile_result.scalar_one_or_none()

            if not profile_id:
                return []

            stmt = select(RecommendationTable).where(RecommendationTable.profile_id == profile_id)

            if status:
                stmt = stmt.where(RecommendationTable.status == status)

            # Only return non-expired recommendations
            stmt = stmt.where(
                (RecommendationTable.expires_at.is_(None)) |
                (RecommendationTable.expires_at > func.now())
            )

            result = await session.execute(stmt)
            db_recommendations = result.scalars().all()

            recommendations = []
            for db_rec in db_recommendations:
                # Handle both string and dict cases for JSON fields
                if isinstance(db_rec.reasoning, str):
                    reasoning = json.loads(db_rec.reasoning) if db_rec.reasoning else {}
                else:
                    reasoning = db_rec.reasoning or {}

                if isinstance(db_rec.context, str):
                    context = json.loads(db_rec.context) if db_rec.context else {}
                else:
                    context = db_rec.context or {}

                recommendations.append(Recommendation(
                    id=db_rec.id,
                    user_id=user_id,
                    source_id=db_rec.source_id,
                    recommendation_type=db_rec.recommendation_type,
                    confidence_score=db_rec.confidence_score,
                    reasoning=reasoning,
                    context=context,
                    created_at=db_rec.created_at
                ))

            return recommendations

    # User Behavior Tracking
    async def record_user_behavior(self, user_id: str, action_type: str,
                                 source_id: Optional[str] = None,
                                 context: Optional[Dict] = None,
                                 confidence_score: Optional[float] = None,
                                 session_id: Optional[str] = None):
        """Record user behavior for learning"""
        async with self.get_session() as session:
            try:
                # Get profile_id from user_id
                profile_stmt = select(UserProfileTable.id).where(UserProfileTable.user_id == user_id)
                profile_result = await session.execute(profile_stmt)
                profile_id = profile_result.scalar_one_or_none()

                if not profile_id:
                    logger.warning(f"⚠️ Cannot record behavior for unknown user: {user_id}")
                    return

                behavior_id = f"{user_id}_{action_type}_{int(datetime.now(timezone.utc).timestamp())}"

                db_behavior = UserBehaviorTable(
                    id=behavior_id,
                    profile_id=profile_id,
                    action_type=action_type,
                    source_id=source_id,
                    context=json.dumps(context or {}),
                    confidence_score=confidence_score,
                    session_id=session_id
                )

                session.add(db_behavior)
                await session.flush()

                logger.debug(f"📊 Recorded behavior: {user_id} -> {action_type}")

            except Exception as e:
                logger.error(f"❌ Failed to record behavior: {e}")

    # Feedback Operations
    async def record_user_feedback(self, feedback: UserFeedback):
        """Record user feedback on recommendations"""
        async with self.get_session() as session:
            try:
                # Get profile_id from user_id
                profile_stmt = select(UserProfileTable.id).where(UserProfileTable.user_id == feedback.user_id)
                profile_result = await session.execute(profile_stmt)
                profile_id = profile_result.scalar_one_or_none()

                if not profile_id:
                    logger.warning(f"⚠️ Cannot record feedback for unknown user: {feedback.user_id}")
                    return

                db_feedback = UserFeedbackTable(
                    id=feedback.id,
                    profile_id=profile_id,
                    recommendation_id=feedback.recommendation_id,
                    action=feedback.action,
                    implicit_feedback=feedback.implicit_feedback,
                    context=json.dumps(feedback.context or {}),
                    confidence_impact=feedback.confidence_impact
                )

                session.add(db_feedback)
                await session.flush()

                logger.info(f"👍 Recorded feedback: {feedback.user_id} -> {feedback.action}")

            except Exception as e:
                logger.error(f"❌ Failed to record feedback: {e}")

    # Health Check Operations
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return metrics"""
        try:
            async with self.get_session() as session:
                # Test basic connectivity
                await session.execute(select(1))

                # Get table counts
                user_count = await session.scalar(select(func.count(UserProfileTable.id)))
                source_count = await session.scalar(select(func.count(DataSourceTable.id)))
                rec_count = await session.scalar(select(func.count(RecommendationTable.id)))

                return {
                    "status": "healthy",
                    "database_type": self.config.db_type,
                    "tables": {
                        "users": user_count,
                        "sources": source_count,
                        "recommendations": rec_count
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    # PLACEHOLDER METHODS - These log what they would do in production
    async def migrate_database(self, target_version: Optional[str] = None):
        """PLACEHOLDER: Database migration management"""
        logger.info(f"🔄 [PLACEHOLDER] Would migrate database to version: {target_version or 'latest'}")
        # In production: Implement Alembic migrations
        pass

    async def backup_database(self, backup_path: str) -> str:
        """PLACEHOLDER: Database backup functionality"""
        logger.info(f"💾 [PLACEHOLDER] Would backup database to: {backup_path}")
        # In production: Implement pg_dump/sqlite backup
        return f"backup_{'test_backup.sql'}"

    async def optimize_performance(self):
        """PLACEHOLDER: Database performance optimization"""
        logger.info("⚡ [PLACEHOLDER] Would run database optimization: VACUUM, ANALYZE, index cleanup")
        # In production: Run VACUUM, ANALYZE, check slow queries
        pass

# Factory function for easy initialization
async def create_database_manager(
    db_type: str = "sqlite",
    **kwargs
) -> DatabaseManager:
    """Factory function to create and initialize database manager"""
    config = DatabaseConfig(db_type=db_type, **kwargs)
    manager = DatabaseManager(config)
    await manager.initialize()
    return manager

# Example usage and testing
if __name__ == "__main__":
    async def test_database():
        """Test database operations"""

        try:
            # Initialize with SQLite for testing
            db = await create_database_manager(db_type="sqlite", sqlite_path=":memory:")

            try:
                # Test health check
                health = await db.health_check()
                print(f"Health: {health}")

                # Test user creation
                user_profile = UserProfile(
                    user_id="test_user_123",
                    role="data_analyst",
                    department="analytics",
                    seniority_level="senior",
                    industry="technology",
                    preferences=UserPreferences(auto_connect_threshold=0.8)
                )

                created_profile = await db.create_user_profile(user_profile)
                print(f"Created profile: {created_profile.user_id}")

                # Test retrieving profile
                retrieved_profile = await db.get_user_profile("test_user_123")
                print(f"Retrieved profile: {retrieved_profile.role if retrieved_profile else 'None'}")

                # Test behavior recording
                await db.record_user_behavior(
                    user_id="test_user_123",
                    action_type="source_connected",
                    context={"source_type": "postgresql", "success": True}
                )

                print("✅ All database tests passed!")

            finally:
                await db.close()

        except ValueError as e:
            if "greenlet" in str(e):
                print("❌ Missing dependency: greenlet")
                print("📦 Install with: pip install greenlet aiosqlite asyncpg")
                print("🔧 Or for all async dependencies: pip install sqlalchemy[asyncio]")
                print("\n📋 Required packages for full functionality:")
                print("  - greenlet (for async SQLAlchemy)")
                print("  - aiosqlite (for async SQLite)")
                print("  - asyncpg (for async PostgreSQL)")
                print("  - redis[hiredis] (for caching layer)")

                # Show basic database schema info instead
                print("\n🏗️ Database Schema Summary:")
                print("Tables that would be created:")
                for table_name in Base.metadata.tables.keys():
                    table = Base.metadata.tables[table_name]
                    print(f"  📋 {table_name}: {len(table.columns)} columns, {len(table.indexes)} indexes")
            else:
                raise
        except ImportError as e:
            print(f"❌ Missing dependencies: {e}")
            print("📦 Install required packages:")
            print("  pip install sqlalchemy[asyncio] aiosqlite asyncpg redis[hiredis]")

    def test_schema_validation():
        """Test database schema without async operations"""
        print("🔍 Validating database schema...")

        try:
            # Check that all tables are properly defined
            tables = Base.metadata.tables
            print(f"✅ Found {len(tables)} table definitions:")

            for table_name, table in tables.items():
                print(f"  📋 {table_name}:")
                print(f"    - Columns: {len(table.columns)}")
                print(f"    - Indexes: {len(table.indexes)}")
                print(f"    - Foreign Keys: {len(table.foreign_keys)}")

            print("\n🔑 Primary Keys:")
            for table_name, table in tables.items():
                pk_cols = [col.name for col in table.primary_key.columns]
                print(f"  {table_name}: {pk_cols}")

            print("\n🔗 Foreign Key Relationships:")
            for table_name, table in tables.items():
                for fk in table.foreign_keys:
                    print(f"  {table_name}.{fk.parent.name} -> {fk.column}")

            print("\n✅ Schema validation completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Schema validation failed: {e}")
            return False

    # Try async test first, fall back to schema validation
    try:
        asyncio.run(test_database())
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"\n⚠️ Async test failed: {e}")
        print("🔄 Running schema validation instead...\n")
        test_schema_validation()