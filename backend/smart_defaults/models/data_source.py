# smart_defaults/models/data_source.py
"""
Comprehensive Data Source Management for Smart Defaults

This module handles all data source-related models including:
- Data source definitions and metadata
- Connection configurations and credentials
- Schema discovery and data profiling
- Performance and quality metrics
- Access control and security classifications
- Smart data analysis with automatic column understanding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import re


class DataSourceType(Enum):
    """Types of data sources supported"""
    DATABASE = "database"
    API = "api"
    FILE_SYSTEM = "file_system"
    CLOUD_STORAGE = "cloud_storage"
    REAL_TIME_STREAM = "real_time_stream"
    WEB_SERVICE = "web_service"
    MESSAGE_QUEUE = "message_queue"
    DATA_WAREHOUSE = "data_warehouse"
    SPREADSHEET = "spreadsheet"
    DOCUMENT_STORE = "document_store"


class ConnectionStatus(Enum):
    """Connection status states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"
    PENDING_APPROVAL = "pending_approval"


class DataClassification(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class DataFreshness(Enum):
    """Data freshness levels"""
    REAL_TIME = "real_time"  # < 1 minute
    NEAR_REAL_TIME = "near_real_time"  # 1-15 minutes
    RECENT = "recent"  # 15 minutes - 1 hour
    HOURLY = "hourly"  # 1-24 hours
    DAILY = "daily"  # 1-7 days
    WEEKLY = "weekly"  # 7-30 days
    MONTHLY = "monthly"  # 30+ days
    HISTORICAL = "historical"  # Archive data


@dataclass
class ConnectionConfiguration:
    """Configuration for connecting to a data source"""
    # Basic connection info
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    # Authentication
    username: Optional[str] = None
    password_encrypted: Optional[str] = None  # Encrypted password
    auth_method: str = "basic"  # basic, oauth, api_key, certificate
    api_key_encrypted: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
    certificate_path: Optional[str] = None

    # Connection pooling and performance
    max_connections: int = 10
    connection_timeout: int = 30  # seconds
    query_timeout: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds

    # SSL/TLS settings
    use_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_verify: bool = True

    # Additional parameters
    connection_params: Dict[str, Any] = field(default_factory=dict)
    driver_settings: Dict[str, Any] = field(default_factory=dict)

    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration without sensitive information"""
        safe_config = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'schema': self.schema,
            'auth_method': self.auth_method,
            'max_connections': self.max_connections,
            'connection_timeout': self.connection_timeout,
            'use_ssl': self.use_ssl
        }
        return {k: v for k, v in safe_config.items() if v is not None}


@dataclass
class SchemaInfo:
    """Database/source schema information"""
    # Basic schema structure
    tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    views: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    indexes: Dict[str, List[str]] = field(default_factory=dict)

    # Column information
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # table.column -> info
    data_types: Dict[str, str] = field(default_factory=dict)
    nullable_columns: Set[str] = field(default_factory=set)
    primary_keys: Dict[str, List[str]] = field(default_factory=dict)
    foreign_keys: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    # Business context
    business_names: Dict[str, str] = field(default_factory=dict)  # technical_name -> business_name
    descriptions: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, List[str]] = field(default_factory=dict)

    # Data profiling
    row_counts: Dict[str, int] = field(default_factory=dict)
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_quality_scores: Dict[str, float] = field(default_factory=dict)

    # Discovery metadata
    last_discovered: Optional[datetime] = None
    discovery_method: str = "automatic"
    discovery_confidence: float = 0.0

    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a table"""
        if table_name not in self.tables:
            return None

        table_info = self.tables[table_name].copy()
        table_info.update({
            'columns': [col for col in self.columns.keys() if col.startswith(f"{table_name}.")],
            'row_count': self.row_counts.get(table_name, 0),
            'primary_key': self.primary_keys.get(table_name, []),
            'foreign_keys': self.foreign_keys.get(table_name, []),
            'data_quality_score': self.data_quality_scores.get(table_name, 0.0)
        })
        return table_info

    def search_columns(self, search_term: str) -> List[str]:
        """Search for columns matching a term"""
        search_term = search_term.lower()
        matching_columns = []

        for column in self.columns.keys():
            # Check technical name
            if search_term in column.lower():
                matching_columns.append(column)
            # Check business name
            elif column in self.business_names and search_term in self.business_names[column].lower():
                matching_columns.append(column)
            # Check description
            elif column in self.descriptions and search_term in self.descriptions[column].lower():
                matching_columns.append(column)

        return matching_columns

    def get_business_context(self, technical_name: str) -> Dict[str, Any]:
        """Get business context for a technical element"""
        return {
            'technical_name': technical_name,
            'business_name': self.business_names.get(technical_name, technical_name),
            'description': self.descriptions.get(technical_name, ''),
            'tags': self.tags.get(technical_name, [])
        }


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    # Completeness metrics
    completeness_score: float = 0.0
    missing_value_percentage: float = 0.0
    null_value_percentage: float = 0.0

    # Accuracy metrics
    accuracy_score: float = 0.0
    data_type_violations: int = 0
    format_violations: int = 0
    range_violations: int = 0

    # Consistency metrics
    consistency_score: float = 0.0
    duplicate_records: int = 0
    inconsistent_formats: int = 0
    referential_integrity_violations: int = 0

    # Timeliness metrics
    timeliness_score: float = 0.0
    last_update: Optional[datetime] = None
    update_frequency: Optional[str] = None
    staleness_hours: float = 0.0

    # Validity metrics
    validity_score: float = 0.0
    constraint_violations: int = 0
    business_rule_violations: int = 0

    # Overall metrics
    overall_quality_score: float = 0.0
    quality_trend: str = "stable"  # improving, stable, declining

    # Assessment metadata
    assessed_at: datetime = field(default_factory=datetime.now)
    assessment_method: str = "automatic"
    sample_size: int = 0
    confidence_level: float = 0.0

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score from component scores"""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score,
            self.timeliness_score,
            self.validity_score
        ]

        # Filter out zero scores and calculate average
        valid_scores = [s for s in scores if s > 0]
        if not valid_scores:
            return 0.0

        self.overall_quality_score = sum(valid_scores) / len(valid_scores)
        return self.overall_quality_score

    def get_quality_issues(self) -> List[Dict[str, Any]]:
        """Get list of quality issues requiring attention"""
        issues = []

        if self.completeness_score < 0.8:
            issues.append({
                'type': 'completeness',
                'severity': 'high' if self.completeness_score < 0.6 else 'medium',
                'description': f'High missing data rate: {self.missing_value_percentage:.1f}%'
            })

        if self.accuracy_score < 0.8:
            issues.append({
                'type': 'accuracy',
                'severity': 'high' if self.accuracy_score < 0.6 else 'medium',
                'description': f'Data accuracy concerns: {self.data_type_violations} violations'
            })

        if self.staleness_hours > 24:
            issues.append({
                'type': 'timeliness',
                'severity': 'medium' if self.staleness_hours < 72 else 'high',
                'description': f'Data is stale: {self.staleness_hours:.1f} hours old'
            })

        if self.duplicate_records > 0:
            issues.append({
                'type': 'consistency',
                'severity': 'low',
                'description': f'Found {self.duplicate_records} duplicate records'
            })

        return issues


@dataclass
class PerformanceMetrics:
    """Performance metrics for data source operations"""
    # Response time metrics
    avg_response_time: float = 0.0  # milliseconds
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0

    # Throughput metrics
    queries_per_second: float = 0.0
    data_transfer_rate: float = 0.0  # MB/s
    concurrent_connections: int = 0

    # Reliability metrics
    uptime_percentage: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0

    # Capacity metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_utilization: float = 0.0

    # Cost metrics
    query_cost: Optional[float] = None  # cost per query
    data_transfer_cost: Optional[float] = None
    storage_cost: Optional[float] = None

    # Trend data
    performance_trend: str = "stable"  # improving, stable, declining
    baseline_comparison: float = 0.0  # % change from baseline

    # Measurement metadata
    measured_at: datetime = field(default_factory=datetime.now)
    measurement_period: timedelta = field(default_factory=lambda: timedelta(hours=1))
    sample_count: int = 0

    def get_performance_grade(self) -> str:
        """Get overall performance grade"""
        if self.avg_response_time < 100 and self.error_rate < 0.01 and self.uptime_percentage > 99.9:
            return "A"
        elif self.avg_response_time < 500 and self.error_rate < 0.05 and self.uptime_percentage > 99.0:
            return "B"
        elif self.avg_response_time < 1000 and self.error_rate < 0.1 and self.uptime_percentage > 95.0:
            return "C"
        elif self.avg_response_time < 2000 and self.error_rate < 0.2 and self.uptime_percentage > 90.0:
            return "D"
        else:
            return "F"

    def is_performance_acceptable(self) -> bool:
        """Check if performance meets acceptable thresholds"""
        return (
                self.avg_response_time < 1000 and
                self.error_rate < 0.1 and
                self.uptime_percentage > 95.0
        )


@dataclass
class AccessControlInfo:
    """Access control and security information for data source"""
    # Security classification
    data_classification: DataClassification = DataClassification.INTERNAL
    sensitivity_level: str = "medium"  # low, medium, high, critical

    # Access control
    required_permissions: Set[str] = field(default_factory=set)
    restricted_roles: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)
    approval_required: bool = False
    approval_roles: Set[str] = field(default_factory=set)

    # Compliance requirements
    regulatory_requirements: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    audit_requirements: List[str] = field(default_factory=list)

    # Data masking and privacy
    contains_pii: bool = False
    contains_phi: bool = False
    contains_financial_data: bool = False
    masking_required: bool = False
    anonymization_available: bool = False

    # Geographic restrictions
    geographic_restrictions: List[str] = field(default_factory=list)
    data_residency_requirements: List[str] = field(default_factory=list)

    # Temporal restrictions
    access_hours: Optional[Dict[str, Any]] = None
    embargo_periods: List[Dict[str, Any]] = field(default_factory=list)
    retention_period: Optional[timedelta] = None

    def can_user_access(self, user_role: str, user_permissions: Set[str]) -> bool:
        """Check if user can access this data source"""
        # Check role restrictions
        if self.restricted_roles and user_role in self.restricted_roles:
            return False

        if self.allowed_roles and user_role not in self.allowed_roles:
            return False

        # Check required permissions
        if self.required_permissions and not self.required_permissions.issubset(user_permissions):
            return False

        return True

    def get_access_requirements(self) -> Dict[str, Any]:
        """Get comprehensive access requirements"""
        return {
            'classification': self.data_classification.value,
            'sensitivity': self.sensitivity_level,
            'approval_required': self.approval_required,
            'required_permissions': list(self.required_permissions),
            'compliance_frameworks': self.compliance_frameworks,
            'contains_sensitive_data': self.contains_pii or self.contains_phi or self.contains_financial_data,
            'geographic_restrictions': self.geographic_restrictions
        }


@dataclass
class BusinessContext:
    """Business context and metadata for data source"""
    # Business identification
    business_name: str
    business_description: str
    business_owner: Optional[str] = None
    technical_owner: Optional[str] = None

    # Business classification
    business_domain: str = "general"  # finance, sales, operations, hr, etc.
    business_criticality: str = "medium"  # low, medium, high, critical
    business_value: str = "medium"  # low, medium, high, strategic

    # Usage context
    primary_use_cases: List[str] = field(default_factory=list)
    common_analyses: List[str] = field(default_factory=list)
    typical_users: List[str] = field(default_factory=list)
    business_processes: List[str] = field(default_factory=list)

    # KPIs and metrics
    key_metrics: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    data_lineage: List[str] = field(default_factory=list)

    # Operational context
    update_schedule: Optional[str] = None
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    downstream_systems: List[str] = field(default_factory=list)

    # Documentation
    documentation_links: List[str] = field(default_factory=list)
    training_materials: List[str] = field(default_factory=list)
    contact_information: Dict[str, str] = field(default_factory=dict)

    def get_business_summary(self) -> Dict[str, Any]:
        """Get business summary for the data source"""
        return {
            'name': self.business_name,
            'description': self.business_description,
            'domain': self.business_domain,
            'criticality': self.business_criticality,
            'value': self.business_value,
            'primary_uses': self.primary_use_cases[:3],  # Top 3 use cases
            'key_metrics': self.key_metrics[:5],  # Top 5 metrics
            'owners': {
                'business': self.business_owner,
                'technical': self.technical_owner
            }
        }


@dataclass
class DataSource:
    """Comprehensive data source definition"""
    # Basic identification
    source_id: str
    name: str
    display_name: str
    description: str
    source_type: DataSourceType

    # Connection and configuration
    connection_config: ConnectionConfiguration
    connection_status: ConnectionStatus = ConnectionStatus.INACTIVE

    # Schema and structure
    schema_info: Optional[SchemaInfo] = None
    data_freshness: DataFreshness = DataFreshness.DAILY
    estimated_size: Optional[str] = None  # "small", "medium", "large", "very_large"
    record_count: Optional[int] = None

    # Business context
    business_context: BusinessContext = field(default_factory=lambda: BusinessContext("", ""))

    # Access control and security
    access_control: AccessControlInfo = field(default_factory=AccessControlInfo)

    # Quality and performance
    data_quality: Optional[DataQualityMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None

    # Operational information
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    last_schema_discovery: Optional[datetime] = None

    # Usage and popularity
    access_count: int = 0
    unique_users: Set[str] = field(default_factory=set)
    popularity_score: float = 0.0
    user_ratings: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration and metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    version: str = "1.0"
    is_active: bool = True

    def __post_init__(self):
        """Initialize computed fields after object creation"""
        self.source_hash = self._calculate_source_hash()
        if not self.business_context.business_name:
            self.business_context.business_name = self.display_name
        if not self.business_context.business_description:
            self.business_context.business_description = self.description

    def _calculate_source_hash(self) -> str:
        """Calculate hash for change detection"""
        source_data = {
            'source_id': self.source_id,
            'name': self.name,
            'source_type': self.source_type.value,
            'connection_host': self.connection_config.host,
            'updated_at': self.updated_at.isoformat()
        }
        return hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest()

    def update_access_info(self, user_id: str):
        """Update access tracking information"""
        self.access_count += 1
        self.unique_users.add(user_id)
        self.last_accessed = datetime.now()
        self.updated_at = datetime.now()

        # Update popularity score (simple algorithm)
        days_since_creation = (datetime.now() - self.created_at).days or 1
        self.popularity_score = len(self.unique_users) / days_since_creation

    def get_recommendation_score(self, user_profile: Any, query_context: Dict[str, Any]) -> float:
        """Calculate recommendation score for this source given user and context"""
        score = 0.0

        # Base score from business value and criticality
        criticality_score = {
            'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0
        }.get(self.business_context.business_criticality, 0.5)

        value_score = {
            'low': 0.2, 'medium': 0.5, 'high': 0.8, 'strategic': 1.0
        }.get(self.business_context.business_value, 0.5)

        score += (criticality_score + value_score) * 0.3

        # Popularity and usage score
        if self.popularity_score > 0:
            score += min(self.popularity_score / 10.0, 0.2)  # Cap at 0.2

        # Data quality score
        if self.data_quality:
            score += self.data_quality.overall_quality_score * 0.2

        # Performance score
        if self.performance_metrics and self.performance_metrics.is_performance_acceptable():
            score += 0.1

        # Freshness score based on query needs
        freshness_score = self._calculate_freshness_score(query_context.get('time_sensitivity', 'medium'))
        score += freshness_score * 0.2

        return min(score, 1.0)

    def _calculate_freshness_score(self, time_sensitivity: str) -> float:
        """Calculate freshness score based on data freshness and requirements"""
        freshness_values = {
            DataFreshness.REAL_TIME: 1.0,
            DataFreshness.NEAR_REAL_TIME: 0.9,
            DataFreshness.RECENT: 0.8,
            DataFreshness.HOURLY: 0.6,
            DataFreshness.DAILY: 0.4,
            DataFreshness.WEEKLY: 0.2,
            DataFreshness.MONTHLY: 0.1,
            DataFreshness.HISTORICAL: 0.05
        }

        sensitivity_multipliers = {
            'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0
        }

        base_score = freshness_values.get(self.data_freshness, 0.5)
        multiplier = sensitivity_multipliers.get(time_sensitivity, 1.0)

        return min(base_score * multiplier, 1.0)

    def is_accessible_to_user(self, user_profile: Any) -> tuple[bool, str]:
        """Check if user can access this data source"""
        # Check basic access control
        user_role = getattr(user_profile, 'primary_role', 'user')
        user_permissions = getattr(user_profile, 'permissions', set())

        if hasattr(user_permissions, 'allowed_data_types'):
            user_perms_set = user_permissions.allowed_data_types
        else:
            user_perms_set = set()

        can_access = self.access_control.can_user_access(user_role, user_perms_set)

        if not can_access:
            return False, "Insufficient permissions"

        # Check if approval is required
        if self.access_control.approval_required:
            return False, "Approval required"

        # Check connection status
        if self.connection_status != ConnectionStatus.ACTIVE:
            return False, f"Source unavailable: {self.connection_status.value}"

        return True, "Access granted"

    def get_connection_summary(self) -> Dict[str, Any]:
        """Get safe connection summary without sensitive information"""
        return {
            'source_id': self.source_id,
            'name': self.display_name,
            'type': self.source_type.value,
            'status': self.connection_status.value,
            'data_freshness': self.data_freshness.value,
            'business_domain': self.business_context.business_domain,
            'classification': self.access_control.data_classification.value,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'popularity_score': self.popularity_score,
            'quality_score': self.data_quality.overall_quality_score if self.data_quality else None,
            'performance_grade': self.performance_metrics.get_performance_grade() if self.performance_metrics else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert data source to dictionary for storage/serialization"""
        return {
            'source_id': self.source_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'source_type': self.source_type.value,
            'connection_config': self.connection_config.get_safe_config(),
            'connection_status': self.connection_status.value,
            'data_freshness': self.data_freshness.value,
            'business_context': self.business_context.get_business_summary(),
            'access_control': self.access_control.get_access_requirements(),
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
                'access_count': self.access_count,
                'unique_users_count': len(self.unique_users),
                'popularity_score': self.popularity_score,
                'tags': self.tags,
                'categories': self.categories,
                'version': self.version,
                'is_active': self.is_active
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Create DataSource from dictionary"""
        # Parse connection config
        connection_config = ConnectionConfiguration()
        config_data = data.get('connection_config', {})
        for key, value in config_data.items():
            if hasattr(connection_config, key):
                setattr(connection_config, key, value)

        # Parse business context
        business_data = data.get('business_context', {})
        business_context = BusinessContext(
            business_name=business_data.get('name', ''),
            business_description=business_data.get('description', ''),
            business_domain=business_data.get('domain', 'general'),
            business_criticality=business_data.get('criticality', 'medium'),
            business_value=business_data.get('value', 'medium')
        )

        # Parse access control
        access_data = data.get('access_control', {})
        access_control = AccessControlInfo(
            data_classification=DataClassification(access_data.get('classification', 'internal')),
            sensitivity_level=access_data.get('sensitivity', 'medium'),
            approval_required=access_data.get('approval_required', False)
        )

        # Create data source
        metadata = data.get('metadata', {})
        source = cls(
            source_id=data['source_id'],
            name=data['name'],
            display_name=data['display_name'],
            description=data['description'],
            source_type=DataSourceType(data['source_type']),
            connection_config=connection_config,
            connection_status=ConnectionStatus(data.get('connection_status', 'inactive')),
            data_freshness=DataFreshness(data.get('data_freshness', 'daily')),
            business_context=business_context,
            access_control=access_control,
            created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get('updated_at', datetime.now().isoformat())),
            access_count=metadata.get('access_count', 0),
            popularity_score=metadata.get('popularity_score', 0.0),
            tags=metadata.get('tags', []),
            categories=metadata.get('categories', []),
            version=metadata.get('version', '1.0'),
            is_active=metadata.get('is_active', True)
        )

        if metadata.get('last_accessed'):
            source.last_accessed = datetime.fromisoformat(metadata['last_accessed'])

        return source


class DataSourceManager:
    """Manager class for data source operations"""

    def __init__(self, storage_backend=None):
        self.storage_backend = storage_backend
        self.sources_cache: Dict[str, DataSource] = {}
        self.schema_cache: Dict[str, SchemaInfo] = {}

    async def register_data_source(self, source: DataSource) -> bool:
        """Register a new data source"""
        try:
            # Validate source configuration
            if not self._validate_source_config(source):
                return False

            # Test connection
            connection_test = await self._test_connection(source)
            if not connection_test:
                source.connection_status = ConnectionStatus.ERROR
            else:
                source.connection_status = ConnectionStatus.ACTIVE

            # Discover schema if possible
            if connection_test:
                schema_info = await self._discover_schema(source)
                if schema_info:
                    source.schema_info = schema_info
                    source.last_schema_discovery = datetime.now()

            # Cache and save
            self.sources_cache[source.source_id] = source

            # PLACEHOLDER: Save to database
            # await self.storage_backend.save_data_source(source.to_dict())

            return True

        except Exception as e:
            # PLACEHOLDER: Proper error handling
            print(f"Error registering data source: {e}")
            return False

    async def get_data_source(self, source_id: str) -> Optional[DataSource]:
        """Get data source by ID"""
        # Check cache first
        if source_id in self.sources_cache:
            return self.sources_cache[source_id]

        # PLACEHOLDER: Load from database
        # source_data = await self.storage_backend.get_data_source(source_id)
        # if source_data:
        #     source = DataSource.from_dict(source_data)
        #     self.sources_cache[source_id] = source
        #     return source

        return None

    async def get_accessible_sources(self, user_profile: Any) -> List[DataSource]:
        """Get all data sources accessible to a user"""
        accessible_sources = []

        # PLACEHOLDER: In production, query database with user permissions
        for source in self.sources_cache.values():
            can_access, _ = source.is_accessible_to_user(user_profile)
            if can_access:
                accessible_sources.append(source)

        return accessible_sources

    async def search_sources(self, query: str, user_profile: Any) -> List[DataSource]:
        """Search for data sources matching query"""
        query = query.lower()
        matching_sources = []

        accessible_sources = await self.get_accessible_sources(user_profile)

        for source in accessible_sources:
            # Search in name, description, business context
            if (query in source.name.lower() or
                    query in source.description.lower() or
                    query in source.business_context.business_name.lower() or
                    any(query in tag.lower() for tag in source.tags)):
                matching_sources.append(source)

            # Search in schema if available
            elif source.schema_info:
                matching_columns = source.schema_info.search_columns(query)
                if matching_columns:
                    matching_sources.append(source)

        # Sort by relevance and popularity
        matching_sources.sort(key=lambda s: s.popularity_score, reverse=True)

        return matching_sources

    def _validate_source_config(self, source: DataSource) -> bool:
        """Validate data source configuration"""
        # Basic validation
        if not source.source_id or not source.name:
            return False

        # Connection config validation
        if source.source_type == DataSourceType.DATABASE:
            if not source.connection_config.host or not source.connection_config.database:
                return False

        return True

    async def _test_connection(self, source: DataSource) -> bool:
        """Test connection to data source"""
        # PLACEHOLDER: Implement actual connection testing
        # This would depend on the source type and connection method
        return True  # Simplified for development

    async def _discover_schema(self, source: DataSource) -> Optional[SchemaInfo]:
        """Discover schema information for data source"""
        # PLACEHOLDER: Implement schema discovery based on source type
        # This would connect to the source and introspect its structure
        return None  # Simplified for development


# ====================================================================
# SMART DATA ANALYZER - NEW ADDITION
# ====================================================================

class SmartDataAnalyzer:
    """Enhanced analyzer that uses Smart Defaults for intelligent column discovery"""

    def __init__(self, data_source_manager: Optional[DataSourceManager] = None, smart_defaults_engine=None):
        self.data_source_manager = data_source_manager
        self.smart_defaults_engine = smart_defaults_engine

    def analyze_dataset_with_smart_discovery(self, df: pd.DataFrame, query: str, query_interpretation: dict) -> dict:
        """Analyze dataset using smart schema discovery and business context"""

        # Step 1: Automatic Schema Discovery
        schema_info = self._discover_schema_automatically(df)

        # Step 2: Query Intent Analysis
        intent_analysis = self._analyze_query_intent(query, query_interpretation, schema_info)

        # Step 3: Smart Column Mapping
        relevant_columns = self._find_relevant_columns(query, schema_info, intent_analysis)

        # Step 4: Intelligent Analysis
        analysis_result = self._perform_intelligent_analysis(df, query, relevant_columns, intent_analysis)

        return analysis_result

    def _discover_schema_automatically(self, df: pd.DataFrame) -> dict:
        """Automatically discover schema and business context for columns"""

        schema_info = {
            'columns': {},
            'business_context': {},
            'data_types': {},
            'semantic_types': {},
            'relationships': {}
        }

        for column in df.columns:
            col_info = self._analyze_column(df[column], column)
            schema_info['columns'][column] = col_info
            schema_info['business_context'][column] = col_info['business_context']
            schema_info['semantic_types'][column] = col_info['semantic_type']
            schema_info['data_types'][column] = col_info['data_type']

        return schema_info

    def _analyze_column(self, series: pd.Series, column_name: str) -> dict:
        """Analyze individual column to determine its business meaning"""

        col_name_lower = column_name.lower()

        # Basic data type
        if pd.api.types.is_numeric_dtype(series):
            data_type = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            data_type = 'datetime'
        else:
            data_type = 'categorical'

        # Semantic type detection
        semantic_type = self._detect_semantic_type(series, column_name)

        # Business context detection
        business_context = self._detect_business_context(series, column_name)

        # Statistical summary
        stats = self._get_column_stats(series)

        return {
            'data_type': data_type,
            'semantic_type': semantic_type,
            'business_context': business_context,
            'stats': stats,
            'sample_values': series.dropna().head(5).tolist() if len(series.dropna()) > 0 else []
        }

    def _detect_semantic_type(self, series: pd.Series, column_name: str) -> str:
        """Detect semantic meaning of column (what it represents)"""

        col_name_lower = column_name.lower()

        # Financial/Sales patterns
        if any(keyword in col_name_lower for keyword in
               ['sales', 'revenue', 'amount', 'price', 'cost', 'value', 'total']):
            if pd.api.types.is_numeric_dtype(series):
                return 'financial_amount'

        # Temporal patterns
        if any(keyword in col_name_lower for keyword in ['date', 'time', 'year', 'month', 'day', 'created', 'updated']):
            return 'temporal'

        # Geographic patterns
        if any(keyword in col_name_lower for keyword in ['region', 'country', 'state', 'city', 'location', 'address']):
            return 'geographic'

        # Identifier patterns
        if any(keyword in col_name_lower for keyword in ['id', 'key', 'code', 'number']) and not any(
                keyword in col_name_lower for keyword in ['phone', 'mobile']):
            return 'identifier'

        # Category patterns
        if any(keyword in col_name_lower for keyword in ['type', 'category', 'class', 'group', 'segment']):
            return 'category'

        # Quantity patterns
        if any(keyword in col_name_lower for keyword in
               ['count', 'quantity', 'qty', 'num', 'total']) and pd.api.types.is_numeric_dtype(series):
            return 'quantity'

        # Performance/Metric patterns
        if any(keyword in col_name_lower for keyword in ['score', 'rating', 'performance', 'metric', 'kpi']):
            return 'metric'

        return 'general'

    def _detect_business_context(self, series: pd.Series, column_name: str) -> dict:
        """Detect business domain and context"""

        col_name_lower = column_name.lower()

        context = {
            'domain': 'general',
            'business_name': column_name.replace('_', ' ').title(),
            'description': '',
            'importance': 'medium'
        }

        # Sales domain
        if any(keyword in col_name_lower for keyword in ['sales', 'revenue', 'deals', 'orders', 'customers']):
            context['domain'] = 'sales'
            context['importance'] = 'high'

        # Finance domain
        elif any(keyword in col_name_lower for keyword in ['cost', 'expense', 'profit', 'budget', 'price']):
            context['domain'] = 'finance'
            context['importance'] = 'high'

        # Operations domain
        elif any(keyword in col_name_lower for keyword in ['inventory', 'stock', 'production', 'supply']):
            context['domain'] = 'operations'

        # Marketing domain
        elif any(keyword in col_name_lower for keyword in ['campaign', 'lead', 'conversion', 'engagement']):
            context['domain'] = 'marketing'

        # HR domain
        elif any(keyword in col_name_lower for keyword in ['employee', 'staff', 'salary', 'department']):
            context['domain'] = 'hr'

        # Generate description based on semantic understanding
        context['description'] = self._generate_column_description(column_name, context['domain'], series)

        return context

    def _generate_column_description(self, column_name: str, domain: str, series: pd.Series) -> str:
        """Generate human-readable description for column"""

        col_name_clean = column_name.replace('_', ' ').title()

        if pd.api.types.is_numeric_dtype(series):
            if domain == 'sales':
                return f"{col_name_clean} - Sales performance metric"
            elif domain == 'finance':
                return f"{col_name_clean} - Financial value or amount"
            else:
                return f"{col_name_clean} - Numeric measurement or quantity"
        else:
            if domain == 'sales':
                return f"{col_name_clean} - Sales-related category or identifier"
            else:
                return f"{col_name_clean} - Categorical classification or label"

    def _get_column_stats(self, series: pd.Series) -> dict:
        """Get comprehensive statistics for column"""

        stats = {
            'count': len(series),
            'missing': series.isnull().sum(),
            'missing_pct': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique()
        }

        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': float(series.mean()) if not series.empty else 0,
                'median': float(series.median()) if not series.empty else 0,
                'std': float(series.std()) if not series.empty else 0,
                'min': float(series.min()) if not series.empty else 0,
                'max': float(series.max()) if not series.empty else 0,
                'q25': float(series.quantile(0.25)) if not series.empty else 0,
                'q75': float(series.quantile(0.75)) if not series.empty else 0
            })
        else:
            # For categorical data
            top_values = series.value_counts().head(5).to_dict()
            stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}

        return stats

    def _analyze_query_intent(self, query: str, query_interpretation: dict, schema_info: dict) -> dict:
        """Enhanced query intent analysis using schema context"""

        query_lower = query.lower()

        intent_analysis = {
            'primary_intent': query_interpretation.get('intent', 'explore'),
            'query_type': 'exploratory',
            'target_metric': None,
            'target_dimensions': [],
            'temporal_aspect': None,
            'aggregation_needed': False,
            'comparison_needed': False
        }

        # Detect specific question types
        if any(word in query_lower for word in ['highest', 'maximum', 'max', 'largest', 'biggest']):
            intent_analysis['query_type'] = 'find_maximum'
            intent_analysis['aggregation_needed'] = True

        elif any(word in query_lower for word in ['lowest', 'minimum', 'min', 'smallest']):
            intent_analysis['query_type'] = 'find_minimum'
            intent_analysis['aggregation_needed'] = True

        elif any(word in query_lower for word in ['average', 'mean', 'typical']):
            intent_analysis['query_type'] = 'find_average'
            intent_analysis['aggregation_needed'] = True

        elif any(word in query_lower for word in ['total', 'sum', 'overall']):
            intent_analysis['query_type'] = 'find_total'
            intent_analysis['aggregation_needed'] = True

        elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs', 'between']):
            intent_analysis['query_type'] = 'comparison'
            intent_analysis['comparison_needed'] = True

        elif any(word in query_lower for word in ['trend', 'over time', 'change', 'growth']):
            intent_analysis['query_type'] = 'trend_analysis'
            intent_analysis['temporal_aspect'] = 'trend'

        # Find target metric from schema
        target_metric = self._find_target_metric(query, schema_info)
        if target_metric:
            intent_analysis['target_metric'] = target_metric

        # Find relevant dimensions
        target_dimensions = self._find_target_dimensions(query, schema_info)
        intent_analysis['target_dimensions'] = target_dimensions

        # Detect temporal filters
        temporal_aspect = self._detect_temporal_filters(query)
        if temporal_aspect:
            intent_analysis['temporal_aspect'] = temporal_aspect

        return intent_analysis

    def _find_target_metric(self, query: str, schema_info: dict) -> Optional[str]:
        """Find the main metric/measure the user is asking about"""

        query_lower = query.lower()

        # Look for financial metrics first
        for column, context in schema_info['business_context'].items():
            col_lower = column.lower()

            # Direct mentions
            if any(word in query_lower for word in [col_lower, context['business_name'].lower()]):
                if schema_info['semantic_types'][column] == 'financial_amount':
                    return column

        # Look for semantic matches
        for column, semantic_type in schema_info['semantic_types'].items():
            if semantic_type == 'financial_amount':
                col_words = column.lower().split('_')
                if any(word in query_lower for word in col_words):
                    return column

        # Look for quantity metrics
        for column, semantic_type in schema_info['semantic_types'].items():
            if semantic_type == 'quantity':
                col_words = column.lower().split('_')
                if any(word in query_lower for word in col_words):
                    return column

        return None

    def _find_target_dimensions(self, query: str, schema_info: dict) -> List[str]:
        """Find dimensional columns that might be relevant for grouping"""

        query_lower = query.lower()
        dimensions = []

        # Look for explicitly mentioned dimensions
        for column, context in schema_info['business_context'].items():
            col_lower = column.lower()

            if any(word in query_lower for word in [col_lower, context['business_name'].lower()]):
                if schema_info['semantic_types'][column] in ['category', 'geographic', 'temporal']:
                    dimensions.append(column)

        # Look for temporal dimensions if time-based query
        if any(word in query_lower for word in ['year', 'month', 'quarter', 'time', 'date']):
            for column, semantic_type in schema_info['semantic_types'].items():
                if semantic_type == 'temporal':
                    dimensions.append(column)

        return dimensions

    def _detect_temporal_filters(self, query: str) -> Optional[dict]:
        """Detect temporal filters in the query"""

        query_lower = query.lower()

        # Year detection
        year_match = re.search(r'\b(19|20)\d{2}\b', query_lower)
        if year_match:
            return {
                'type': 'year',
                'value': int(year_match.group()),
                'filter_expression': f"year == {year_match.group()}"
            }

        # Month detection
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        for i, month in enumerate(months, 1):
            if month in query_lower:
                return {
                    'type': 'month',
                    'value': i,
                    'name': month.title(),
                    'filter_expression': f"month == {i}"
                }

        # Quarter detection
        if 'q1' in query_lower or 'quarter 1' in query_lower:
            return {'type': 'quarter', 'value': 1, 'filter_expression': "quarter == 1"}
        elif 'q2' in query_lower or 'quarter 2' in query_lower:
            return {'type': 'quarter', 'value': 2, 'filter_expression': "quarter == 2"}
        elif 'q3' in query_lower or 'quarter 3' in query_lower:
            return {'type': 'quarter', 'value': 3, 'filter_expression': "quarter == 3"}
        elif 'q4' in query_lower or 'quarter 4' in query_lower:
            return {'type': 'quarter', 'value': 4, 'filter_expression': "quarter == 4"}

        return None

    def _find_relevant_columns(self, query: str, schema_info: dict, intent_analysis: dict) -> dict:
        """Find all columns relevant to the query"""

        relevant_columns = {
            'primary_metric': intent_analysis.get('target_metric'),
            'dimensions': intent_analysis.get('target_dimensions', []),
            'temporal_columns': [],
            'additional_context': []
        }

        # Find temporal columns
        for column, semantic_type in schema_info['semantic_types'].items():
            if semantic_type == 'temporal':
                relevant_columns['temporal_columns'].append(column)

        # Find additional context columns based on domain
        primary_domain = None
        if relevant_columns['primary_metric']:
            primary_domain = schema_info['business_context'][relevant_columns['primary_metric']]['domain']

        if primary_domain:
            for column, context in schema_info['business_context'].items():
                if (context['domain'] == primary_domain and
                        column not in [relevant_columns['primary_metric']] + relevant_columns['dimensions']):
                    relevant_columns['additional_context'].append(column)

        return relevant_columns

    def _perform_intelligent_analysis(self, df: pd.DataFrame, query: str, relevant_columns: dict,
                                      intent_analysis: dict) -> dict:
        """Perform intelligent analysis based on discovered schema and intent"""

        analysis_result = {
            'analysis_type': intent_analysis['query_type'],
            'summary': '',
            'data': [],
            'insights': [],
            'metadata': {
                'schema_discovery_used': True,
                'relevant_columns': relevant_columns,
                'intent_analysis': intent_analysis
            }
        }

        primary_metric = relevant_columns['primary_metric']

        if not primary_metric:
            # Fallback to exploratory analysis
            return self._exploratory_analysis(df, query)

        # Execute specific analysis based on query type
        if intent_analysis['query_type'] == 'find_maximum':
            result = self._find_maximum_analysis(df, primary_metric, relevant_columns, intent_analysis)
        elif intent_analysis['query_type'] == 'find_minimum':
            result = self._find_minimum_analysis(df, primary_metric, relevant_columns, intent_analysis)
        elif intent_analysis['query_type'] == 'find_average':
            result = self._find_average_analysis(df, primary_metric, relevant_columns, intent_analysis)
        elif intent_analysis['query_type'] == 'find_total':
            result = self._find_total_analysis(df, primary_metric, relevant_columns, intent_analysis)
        elif intent_analysis['query_type'] == 'comparison':
            result = self._comparison_analysis(df, primary_metric, relevant_columns, intent_analysis)
        elif intent_analysis['query_type'] == 'trend_analysis':
            result = self._trend_analysis(df, primary_metric, relevant_columns, intent_analysis)
        else:
            result = self._exploratory_analysis(df, query)

        analysis_result.update(result)
        return analysis_result

    def _find_maximum_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                               intent_analysis: dict) -> dict:
        """Find maximum value analysis"""

        # Apply temporal filters if specified
        filtered_df = self._apply_temporal_filters(df, intent_analysis.get('temporal_aspect'),
                                                   relevant_columns['temporal_columns'])

        if filtered_df.empty:
            return {
                'summary': f"No data found for the specified time period",
                'data': [],
                'insights': ["No records match the specified criteria"]
            }

        # Find maximum value
        max_value = filtered_df[metric_column].max()
        max_row = filtered_df[filtered_df[metric_column] == max_value].iloc[0]

        # Build summary
        metric_name = metric_column.replace('_', ' ').title()
        filter_desc = self._get_filter_description(intent_analysis.get('temporal_aspect'))

        summary = f"The highest {metric_name.lower()}{filter_desc} was {self._format_value(max_value)}"

        # Build insights
        insights = [
            f"💰 Maximum {metric_name}: {self._format_value(max_value)}",
            f"📊 Found in {len(filtered_df)} total records{filter_desc}"
        ]

        # Add context from other columns
        for col in relevant_columns['additional_context'][:3]:  # Limit to 3 context columns
            if col in max_row.index and not pd.isna(max_row[col]):
                col_name = col.replace('_', ' ').title()
                insights.append(f"📍 {col_name}: {max_row[col]}")

        # Prepare data for visualization
        data = [
            {
                'type': 'maximum_result',
                'metric': metric_column,
                'value': float(max_value),
                'context': max_row.to_dict(),
                'total_records': len(filtered_df)
            }
        ]

        # Add aggregated data for charts if dimensions available
        if relevant_columns['dimensions']:
            agg_data = self._get_aggregated_data(filtered_df, metric_column, relevant_columns['dimensions'][:2])
            data.extend(agg_data)

        return {
            'summary': summary,
            'data': data,
            'insights': insights
        }

    def _apply_temporal_filters(self, df: pd.DataFrame, temporal_aspect: dict,
                                temporal_columns: List[str]) -> pd.DataFrame:
        """Apply temporal filters to dataframe"""

        if not temporal_aspect or not temporal_columns:
            return df

        filtered_df = df.copy()

        for col in temporal_columns:
            if temporal_aspect['type'] == 'year':
                if 'year' in col.lower():
                    filtered_df = filtered_df[filtered_df[col] == temporal_aspect['value']]
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    filtered_df = filtered_df[filtered_df[col].dt.year == temporal_aspect['value']]

        return filtered_df

    def _get_filter_description(self, temporal_aspect: dict) -> str:
        """Get human-readable filter description"""

        if not temporal_aspect:
            return ""

        if temporal_aspect['type'] == 'year':
            return f" in {temporal_aspect['value']}"
        elif temporal_aspect['type'] == 'month':
            return f" in {temporal_aspect['name']}"
        elif temporal_aspect['type'] == 'quarter':
            return f" in Q{temporal_aspect['value']}"

        return ""

    def _format_value(self, value) -> str:
        """Format value for display"""

        if isinstance(value, (int, float)):
            if value >= 1000000:
                return f"${value / 1000000:.1f}M"
            elif value >= 1000:
                return f"${value / 1000:.1f}K"
            else:
                return f"${value:,.2f}"

        return str(value)

    def _get_aggregated_data(self, df: pd.DataFrame, metric_column: str, dimensions: List[str]) -> List[dict]:
        """Get aggregated data for visualization"""

        agg_data = []

        for dim in dimensions:
            if dim in df.columns:
                grouped = df.groupby(dim)[metric_column].agg(['sum', 'mean', 'count']).reset_index()

                for _, row in grouped.iterrows():
                    agg_data.append({
                        'type': 'aggregation',
                        'dimension': dim,
                        'dimension_value': row[dim],
                        'sum': float(row['sum']),
                        'mean': float(row['mean']),
                        'count': int(row['count'])
                    })

        return agg_data

    def _exploratory_analysis(self, df: pd.DataFrame, query: str) -> dict:
        """Fallback exploratory analysis"""

        return {
            'analysis_type': 'exploratory',
            'summary': f"Exploratory analysis of dataset for query: {query}",
            'data': df.head(100).to_dict('records'),
            'insights': [
                f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                f"Found {len(df.select_dtypes(include=[np.number]).columns)} numeric columns",
                f"Found {len(df.select_dtypes(include=['object']).columns)} categorical columns"
            ],
            'metadata': {
                'schema_discovery_used': False,
                'fallback_reason': 'Could not identify specific metric or intent'
            }
        }

    # Placeholder methods for other analysis types
    def _find_minimum_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                               intent_analysis: dict) -> dict:
        """Find minimum value analysis"""
        # Apply temporal filters if specified
        filtered_df = self._apply_temporal_filters(df, intent_analysis.get('temporal_aspect'),
                                                   relevant_columns['temporal_columns'])

        if filtered_df.empty:
            return {
                'summary': f"No data found for the specified time period",
                'data': [],
                'insights': ["No records match the specified criteria"]
            }

        # Find minimum value
        min_value = filtered_df[metric_column].min()
        min_row = filtered_df[filtered_df[metric_column] == min_value].iloc[0]

        # Build summary
        metric_name = metric_column.replace('_', ' ').title()
        filter_desc = self._get_filter_description(intent_analysis.get('temporal_aspect'))

        summary = f"The lowest {metric_name.lower()}{filter_desc} was {self._format_value(min_value)}"

        # Build insights
        insights = [
            f"💰 Minimum {metric_name}: {self._format_value(min_value)}",
            f"📊 Found in {len(filtered_df)} total records{filter_desc}"
        ]

        # Add context from other columns
        for col in relevant_columns.get('additional_context', [])[:3]:
            if col in min_row.index and not pd.isna(min_row[col]):
                col_name = col.replace('_', ' ').title()
                insights.append(f"📍 {col_name}: {min_row[col]}")

        return {
            'summary': summary,
            'data': [{'type': 'minimum_result', 'metric': metric_column, 'value': float(min_value),
                      'context': min_row.to_dict()}],
            'insights': insights
        }

    def _find_average_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                               intent_analysis: dict) -> dict:
        """Find average value analysis"""
        # Apply temporal filters if specified
        filtered_df = self._apply_temporal_filters(df, intent_analysis.get('temporal_aspect'),
                                                   relevant_columns['temporal_columns'])

        if filtered_df.empty:
            return {
                'summary': f"No data found for the specified time period",
                'data': [],
                'insights': ["No records match the specified criteria"]
            }

        # Calculate average
        avg_value = filtered_df[metric_column].mean()
        metric_name = metric_column.replace('_', ' ').title()
        filter_desc = self._get_filter_description(intent_analysis.get('temporal_aspect'))

        summary = f"The average {metric_name.lower()}{filter_desc} was {self._format_value(avg_value)}"

        insights = [
            f"📊 Average {metric_name}: {self._format_value(avg_value)}",
            f"📈 Based on {len(filtered_df)} records{filter_desc}",
            f"📉 Standard deviation: {self._format_value(filtered_df[metric_column].std())}"
        ]

        return {
            'summary': summary,
            'data': [{'type': 'average_result', 'metric': metric_column, 'value': float(avg_value)}],
            'insights': insights
        }

    def _find_total_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                             intent_analysis: dict) -> dict:
        """Find total/sum analysis"""
        # Apply temporal filters if specified
        filtered_df = self._apply_temporal_filters(df, intent_analysis.get('temporal_aspect'),
                                                   relevant_columns['temporal_columns'])

        if filtered_df.empty:
            return {
                'summary': f"No data found for the specified time period",
                'data': [],
                'insights': ["No records match the specified criteria"]
            }

        # Calculate total
        total_value = filtered_df[metric_column].sum()
        metric_name = metric_column.replace('_', ' ').title()
        filter_desc = self._get_filter_description(intent_analysis.get('temporal_aspect'))

        summary = f"The total {metric_name.lower()}{filter_desc} was {self._format_value(total_value)}"

        insights = [
            f"💰 Total {metric_name}: {self._format_value(total_value)}",
            f"📊 Sum of {len(filtered_df)} records{filter_desc}",
            f"📈 Average per record: {self._format_value(total_value / len(filtered_df))}"
        ]

        return {
            'summary': summary,
            'data': [{'type': 'total_result', 'metric': metric_column, 'value': float(total_value)}],
            'insights': insights
        }

    def _comparison_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                             intent_analysis: dict) -> dict:
        """Comparison analysis"""
        # Apply temporal filters if specified
        filtered_df = self._apply_temporal_filters(df, intent_analysis.get('temporal_aspect'),
                                                   relevant_columns['temporal_columns'])

        if filtered_df.empty:
            return self._exploratory_analysis(df, "comparison analysis - no data")

        # Use first dimension for comparison if available
        dimensions = relevant_columns.get('dimensions', [])
        if not dimensions:
            return self._exploratory_analysis(df, "comparison analysis - no dimensions")

        dimension = dimensions[0]
        grouped = filtered_df.groupby(dimension)[metric_column].agg(['sum', 'mean', 'count']).reset_index()

        metric_name = metric_column.replace('_', ' ').title()
        dim_name = dimension.replace('_', ' ').title()

        summary = f"Comparison of {metric_name.lower()} by {dim_name.lower()}"

        insights = [
            f"📊 Comparing {metric_name} across {len(grouped)} {dim_name} categories",
            f"🏆 Highest: {grouped.loc[grouped['sum'].idxmax(), dimension]} ({self._format_value(grouped['sum'].max())})",
            f"📉 Lowest: {grouped.loc[grouped['sum'].idxmin(), dimension]} ({self._format_value(grouped['sum'].min())})"
        ]

        return {
            'summary': summary,
            'data': grouped.to_dict('records'),
            'insights': insights
        }

    def _trend_analysis(self, df: pd.DataFrame, metric_column: str, relevant_columns: dict,
                        intent_analysis: dict) -> dict:
        """Trend analysis"""
        temporal_columns = relevant_columns.get('temporal_columns', [])

        if not temporal_columns:
            return self._exploratory_analysis(df, "trend analysis - no temporal columns")

        temporal_col = temporal_columns[0]

        # Group by temporal dimension
        if pd.api.types.is_datetime64_any_dtype(df[temporal_col]):
            df_sorted = df.sort_values(temporal_col)
            grouped = df_sorted.groupby(df_sorted[temporal_col].dt.date)[metric_column].sum().reset_index()
        else:
            grouped = df.groupby(temporal_col)[metric_column].sum().reset_index()

        metric_name = metric_column.replace('_', ' ').title()

        summary = f"Trend analysis of {metric_name.lower()} over time"

        insights = [
            f"📈 Trend analysis across {len(grouped)} time periods",
            f"🔝 Peak value: {self._format_value(grouped[metric_column].max())}",
            f"📊 Average: {self._format_value(grouped[metric_column].mean())}"
        ]

        return {
            'summary': summary,
            'data': grouped.to_dict('records'),
            'insights': insights
        }