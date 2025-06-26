# core/models.py - Core Pydantic Models for DataGenie Multi-Source Analytics
"""
Production-ready Pydantic models for type safety and validation
Comprehensive models for multi-source data operations
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid
from decimal import Decimal


# ===============================
# Base Models
# ===============================

class TimestampedModel(BaseModel):
    """Base model with timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def mark_updated(self):
        self.updated_at = datetime.utcnow()


class IdentifiedModel(TimestampedModel):
    """Base model with ID and timestamps"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ===============================
# Enums
# ===============================

class DataSourceType(str, Enum):
    DATABASE = "database"
    FILE = "file"
    API = "api"
    STREAM = "stream"
    CACHE = "cache"


class DataFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"


class DataSensitivityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AccessPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class CachePolicy(str, Enum):
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    PRECOMPUTE = "precompute"
    SMART_REFRESH = "smart_refresh"


class ConflictResolutionStrategy(str, Enum):
    AUTHORITY_BASED = "authority_based"
    RECENCY_BASED = "recency_based"
    CONSENSUS_BASED = "consensus_based"
    BUSINESS_RULE_BASED = "business_rule_based"
    USER_CHOICE = "user_choice"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class JoinStrategy(str, Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


# ===============================
# User and Security Models
# ===============================

class User(IdentifiedModel):
    """User model with comprehensive profile"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: str = Field(default="analyst")
    is_active: bool = Field(default=True)

    # Expertise and preferences
    expertise_level: Literal["novice", "intermediate", "expert"] = "intermediate"
    preferred_domains: List[str] = Field(default_factory=list)

    # Security
    hashed_password: str
    last_login: Optional[datetime] = None
    failed_login_attempts: int = Field(default=0)
    is_locked: bool = Field(default=False)

    # Permissions
    permissions: List[AccessPermission] = Field(default_factory=list)
    data_access_level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL


class UserSession(IdentifiedModel):
    """User session tracking"""
    user_id: str
    session_token: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    expires_at: datetime
    is_active: bool = Field(default=True)


# ===============================
# Data Source Models
# ===============================

class DataSourceConnection(BaseModel):
    """Data source connection configuration"""
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None  # Will be encrypted in storage
    database: Optional[str] = None
    connection_string: Optional[str] = None
    ssl_enabled: bool = Field(default=True)
    timeout: int = Field(default=30)

    # API-specific settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    # File-specific settings
    file_path: Optional[str] = None
    encoding: str = Field(default="utf-8")


class DataSource(IdentifiedModel):
    """Comprehensive data source model"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    source_type: DataSourceType
    data_format: DataFormat

    # Connection details
    connection: DataSourceConnection

    # Data characteristics
    estimated_rows: Optional[int] = None
    estimated_size_mb: Optional[float] = None
    schema_fields: List[str] = Field(default_factory=list)

    # Metadata
    owner: str
    tags: List[str] = Field(default_factory=list)

    # Access control
    sensitivity_level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL
    allowed_users: List[str] = Field(default_factory=list)
    allowed_roles: List[str] = Field(default_factory=list)

    # Operational
    is_active: bool = Field(default=True)
    last_accessed: Optional[datetime] = None
    health_status: Literal["healthy", "warning", "error", "unknown"] = "unknown"

    # Caching
    cache_policy: CachePolicy = CachePolicy.SMART_REFRESH
    cache_ttl_seconds: Optional[int] = None

    # Data freshness
    update_frequency: Optional[str] = None  # "daily", "hourly", "real-time"
    last_updated: Optional[datetime] = None


class DataSourceRelationship(IdentifiedModel):
    """Relationships between data sources"""
    source_id: str
    target_id: str
    relationship_type: Literal["parent", "child", "sibling", "derived", "backup"]
    join_fields: Dict[str, str] = Field(default_factory=dict)  # source_field -> target_field
    strength: float = Field(ge=0.0, le=1.0, default=0.5)  # Relationship strength
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)  # Confidence in relationship


# ===============================
# Query and Analysis Models
# ===============================

class QueryContext(BaseModel):
    """Context information for a query"""
    user_id: str
    session_id: Optional[str] = None
    query_text: str
    intended_domain: Optional[str] = None
    urgency_level: Literal["low", "medium", "high", "critical"] = "medium"
    expected_result_size: Optional[Literal["small", "medium", "large"]] = None

    # Query characteristics
    requires_real_time: bool = Field(default=False)
    max_staleness_seconds: Optional[int] = None
    confidence_threshold: float = Field(default=0.7)


class DataField(BaseModel):
    """Individual data field information"""
    name: str
    data_type: str  # "string", "integer", "float", "datetime", "boolean"
    is_nullable: bool = Field(default=True)
    is_unique: bool = Field(default=False)
    is_primary_key: bool = Field(default=False)

    # Semantic information
    semantic_type: Optional[str] = None  # "person_name", "email", "phone", "address", etc.
    business_meaning: Optional[str] = None
    sensitivity_level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL

    # Statistics
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    sample_values: List[Any] = Field(default_factory=list)


class DataSchema(BaseModel):
    """Schema information for a dataset"""
    source_id: str
    table_name: Optional[str] = None
    fields: List[DataField]
    row_count: Optional[int] = None

    # Schema metadata
    version: str = Field(default="1.0")
    last_analyzed: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# ===============================
# Cache Models
# ===============================

class CacheKey(BaseModel):
    """Cache key structure"""
    query_hash: str
    source_ids: List[str]
    parameters_hash: str
    user_context_hash: str

    def generate_key(self) -> str:
        """Generate unique cache key string"""
        sources_str = "_".join(sorted(self.source_ids))
        return f"{self.query_hash}_{sources_str}_{self.parameters_hash}_{self.user_context_hash}"


class CacheEntry(IdentifiedModel):
    """Cache entry with metadata"""
    key: str
    data: Any  # Serialized data
    size_bytes: int

    # Cache metadata
    cache_policy: CachePolicy
    ttl_seconds: int
    expires_at: datetime

    # Access tracking
    hit_count: int = Field(default=0)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)

    # Quality metrics
    generation_time_ms: float
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Source tracking
    source_ids: List[str]
    source_versions: Dict[str, str] = Field(default_factory=dict)


class CacheStats(BaseModel):
    """Cache performance statistics"""
    total_entries: int
    total_size_mb: float
    hit_rate: float = Field(ge=0.0, le=1.0)

    # Performance metrics
    avg_hit_time_ms: float
    avg_miss_time_ms: float
    eviction_count: int

    # Time-based stats
    period_start: datetime
    period_end: datetime


# ===============================
# Conflict Resolution Models
# ===============================

class ConflictingValue(BaseModel):
    """A single conflicting value from a source"""
    source_id: str
    value: Any
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataConflict(IdentifiedModel):
    """Data conflict between multiple sources"""
    field_name: str
    record_identifier: str  # Primary key or unique identifier
    conflicting_values: List[ConflictingValue]

    # Conflict metadata
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    conflict_type: Literal["value", "format", "semantic", "temporal"] = "value"
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Resolution
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_value: Optional[Any] = None
    resolution_confidence: Optional[float] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None  # user_id or "system"

    # Learning
    user_feedback: Optional[Literal["correct", "incorrect", "partial"]] = None
    feedback_timestamp: Optional[datetime] = None


class ResolutionRule(IdentifiedModel):
    """Business rule for conflict resolution"""
    name: str
    description: str
    field_pattern: str  # Regex or field name pattern
    source_priority: List[str] = Field(default_factory=list)  # Ordered list of source IDs

    # Rule conditions
    conditions: Dict[str, Any] = Field(default_factory=dict)
    strategy: ConflictResolutionStrategy

    # Rule metadata
    priority: int = Field(default=100)  # Lower number = higher priority
    is_active: bool = Field(default=True)
    created_by: str

    # Performance tracking
    times_applied: int = Field(default=0)
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)


class SourceTrustScore(IdentifiedModel):
    """Trust score for a data source"""
    source_id: str
    overall_score: float = Field(ge=0.0, le=1.0, default=0.8)

    # Domain-specific scores
    domain_scores: Dict[str, float] = Field(default_factory=dict)

    # Score history
    score_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Factors
    accuracy_score: float = Field(ge=0.0, le=1.0, default=0.8)
    freshness_score: float = Field(ge=0.0, le=1.0, default=0.8)
    completeness_score: float = Field(ge=0.0, le=1.0, default=0.8)
    consistency_score: float = Field(ge=0.0, le=1.0, default=0.8)

    # Tracking
    evaluations_count: int = Field(default=0)
    last_evaluated: datetime = Field(default_factory=datetime.utcnow)


# ===============================
# Audit and Governance Models
# ===============================

class AuditAction(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY_EXECUTE = "query_execute"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"
    CONFLICT_RESOLVE = "conflict_resolve"
    CACHE_INVALIDATE = "cache_invalidate"


class AuditLog(IdentifiedModel):
    """Comprehensive audit logging"""
    user_id: str
    session_id: Optional[str] = None
    action: AuditAction

    # Event details
    resource_type: str  # "data_source", "query", "cache", etc.
    resource_id: Optional[str] = None
    description: str

    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Results
    success: bool = Field(default=True)
    error_message: Optional[str] = None

    # Data governance
    data_accessed: List[str] = Field(default_factory=list)  # List of source IDs
    sensitivity_levels: List[DataSensitivityLevel] = Field(default_factory=list)

    # Performance
    duration_ms: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataAccessPermission(IdentifiedModel):
    """Granular data access permissions"""
    user_id: str
    source_id: str
    field_name: Optional[str] = None  # None means all fields

    # Permission details
    permission_type: AccessPermission
    granted_by: str
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Conditions
    conditions: Dict[str, Any] = Field(default_factory=dict)  # Time-based, IP-based, etc.
    max_records: Optional[int] = None

    # Tracking
    is_active: bool = Field(default=True)
    last_used: Optional[datetime] = None
    usage_count: int = Field(default=0)


# ===============================
# Multi-Source Analysis Models
# ===============================

class JoinConfiguration(BaseModel):
    """Configuration for joining datasets"""
    left_source_id: str
    right_source_id: str
    join_strategy: JoinStrategy

    # Join keys
    left_keys: List[str]
    right_keys: List[str]

    # Join options
    fuzzy_threshold: Optional[float] = None  # For fuzzy joins
    case_sensitive: bool = Field(default=True)
    null_handling: Literal["drop", "keep", "fill"] = "keep"

    # Performance hints
    estimated_result_size: Optional[int] = None
    optimization_hints: List[str] = Field(default_factory=list)


class DataFusionResult(IdentifiedModel):
    """Result of multi-source data fusion"""
    query_context: QueryContext
    source_ids: List[str]
    join_configurations: List[JoinConfiguration]

    # Results
    result_row_count: int
    result_column_count: int

    # Quality metrics
    join_success_rate: float = Field(ge=0.0, le=1.0)
    data_completeness: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Conflicts encountered
    conflicts_detected: int = Field(default=0)
    conflicts_resolved: int = Field(default=0)

    # Performance
    processing_time_ms: float
    cache_hit_rate: float = Field(ge=0.0, le=1.0, default=0.0)

    # Metadata
    data_lineage: List[str] = Field(default_factory=list)
    transformations_applied: List[str] = Field(default_factory=list)


# ===============================
# API Response Models
# ===============================

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    processing_time_ms: Optional[float] = None

    # Pagination (when applicable)
    total_count: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None


class ValidationError(BaseModel):
    """Detailed validation error"""
    field: str
    message: str
    invalid_value: Any
    validation_rule: str


# ===============================
# Configuration Models
# ===============================

class SystemHealth(BaseModel):
    """System health status"""
    overall_status: Literal["healthy", "degraded", "unhealthy"]

    # Component health
    database_healthy: bool
    cache_healthy: bool
    external_apis_healthy: bool

    # Performance metrics
    avg_response_time_ms: float
    error_rate: float = Field(ge=0.0, le=1.0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)

    # Resource usage
    cpu_usage_percent: float = Field(ge=0.0, le=100.0)
    memory_usage_percent: float = Field(ge=0.0, le=100.0)
    disk_usage_percent: float = Field(ge=0.0, le=100.0)

    # Timestamp
    checked_at: datetime = Field(default_factory=datetime.utcnow)


# ===============================
# Validators and Custom Types
# ===============================

class DecimalField(BaseModel):
    """Custom decimal field for financial data"""
    value: Decimal
    currency: str = Field(default="USD")
    precision: int = Field(default=2)

    @field_validator('value')
    @classmethod
    def validate_decimal(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


# Model registry for dynamic model resolution
MODEL_REGISTRY = {
    'User': User,
    'DataSource': DataSource,
    'DataConflict': DataConflict,
    'AuditLog': AuditLog,
    'CacheEntry': CacheEntry,
    'DataFusionResult': DataFusionResult,
    # Add more models as needed
}


def get_model_by_name(model_name: str) -> Optional[BaseModel]:
    """Get model class by name"""
    return MODEL_REGISTRY.get(model_name)