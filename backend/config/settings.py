# config/settings.py - Production Configuration Management
"""
Centralized configuration management for DataGenie Multi-Source Analytics
Supports environment-based configuration with secure defaults
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from enum import Enum
import os
from pathlib import Path


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    # Core database settings
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    username: str = Field(default="datagenie", description="Database username")
    password: str = Field(default="", description="Database password")
    database: str = Field(default="datagenie_db", description="Database name")

    # Connection pool settings
    min_connections: int = Field(default=5, description="Minimum connections in pool")
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    pool_timeout: int = Field(default=30, description="Connection timeout in seconds")

    # Advanced settings
    echo_queries: bool = Field(default=False, description="Log SQL queries")
    ssl_mode: str = Field(default="prefer", description="SSL mode for connections")

    @property
    def url(self) -> str:
        """Generate database URL"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        """Generate synchronous database URL for migrations"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    model_config = {"env_prefix": "DB_"}


class RedisSettings(BaseSettings):
    """Redis configuration for caching"""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    database: int = Field(default=0, description="Redis database number")

    # Connection pool settings
    max_connections: int = Field(default=50, description="Maximum Redis connections")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")

    # Cluster settings (for production scaling)
    cluster_mode: bool = Field(default=False, description="Enable Redis cluster mode")
    cluster_nodes: List[str] = Field(default_factory=list, description="Redis cluster nodes")

    @property
    def url(self) -> str:
        """Generate Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"

    model_config = {"env_prefix": "REDIS_"}


class SecuritySettings(BaseSettings):
    """Security and authentication settings"""

    # JWT Configuration
    secret_key: str = Field(default="dev-secret-key-change-in-production-min-32-chars", description="Secret key for JWT token generation")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")

    # Password policies
    min_password_length: int = Field(default=8, description="Minimum password length")
    require_uppercase: bool = Field(default=True, description="Require uppercase letters")
    require_lowercase: bool = Field(default=True, description="Require lowercase letters")
    require_numbers: bool = Field(default=True, description="Require numbers")
    require_special_chars: bool = Field(default=True, description="Require special characters")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="API calls per minute per user")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")

    # Data governance
    default_data_retention_days: int = Field(default=365, description="Default data retention period")
    audit_log_retention_days: int = Field(default=2555, description="Audit log retention (7 years)")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    model_config = {"env_prefix": "SECURITY_"}


class CacheSettings(BaseSettings):
    """Caching configuration"""

    # Default TTL settings (in seconds)
    default_ttl: int = Field(default=3600, description="Default cache TTL (1 hour)")
    short_ttl: int = Field(default=300, description="Short cache TTL (5 minutes)")
    long_ttl: int = Field(default=86400, description="Long cache TTL (24 hours)")

    # Cache size limits
    max_memory_cache_size: int = Field(default=1000, description="Max items in memory cache")
    max_cache_size_mb: int = Field(default=512, description="Max cache size in MB")

    # Cache strategies
    enable_memory_cache: bool = Field(default=True, description="Enable in-memory caching")
    enable_redis_cache: bool = Field(default=True, description="Enable Redis caching")
    enable_disk_cache: bool = Field(default=False, description="Enable disk caching")

    # Invalidation settings
    enable_smart_invalidation: bool = Field(default=True, description="Enable smart cache invalidation")
    invalidation_batch_size: int = Field(default=100, description="Batch size for cache invalidation")

    # Precompute settings
    enable_precompute: bool = Field(default=True, description="Enable precomputation")
    precompute_threshold: float = Field(default=0.1, description="Query frequency threshold for precompute")

    model_config = {"env_prefix": "CACHE_"}


class GovernanceSettings(BaseSettings):
    """Data governance and compliance settings"""

    # Access control
    enable_rbac: bool = Field(default=True, description="Enable role-based access control")
    enable_abac: bool = Field(default=True, description="Enable attribute-based access control")
    strict_mode: bool = Field(default=True, description="Strict permission checking")

    # Data classification
    auto_classify_sensitivity: bool = Field(default=True, description="Auto-classify data sensitivity")
    default_classification: str = Field(default="internal", description="Default data classification")

    # Audit settings
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    audit_all_queries: bool = Field(default=True, description="Audit all queries")
    audit_data_access: bool = Field(default=True, description="Audit data access")
    audit_admin_actions: bool = Field(default=True, description="Audit admin actions")

    # Compliance
    gdpr_compliance: bool = Field(default=True, description="Enable GDPR compliance features")
    ccpa_compliance: bool = Field(default=True, description="Enable CCPA compliance features")
    sox_compliance: bool = Field(default=False, description="Enable SOX compliance features")

    # Data masking
    enable_auto_masking: bool = Field(default=True, description="Enable automatic data masking")
    mask_pii_data: bool = Field(default=True, description="Mask PII data")
    mask_financial_data: bool = Field(default=True, description="Mask financial data")

    model_config = {"env_prefix": "GOVERNANCE_"}


class ConflictResolutionSettings(BaseSettings):
    """Conflict resolution configuration"""

    # Resolution strategies
    default_strategy: str = Field(default="authority_based", description="Default resolution strategy")
    confidence_threshold: float = Field(default=0.7, description="Auto-resolution confidence threshold")
    user_choice_threshold: float = Field(default=0.5, description="Threshold for presenting user choice")

    # Source trust scoring
    enable_trust_scoring: bool = Field(default=True, description="Enable source trust scoring")
    initial_trust_score: float = Field(default=0.8, description="Initial trust score for new sources")
    trust_decay_factor: float = Field(default=0.95, description="Trust decay factor for conflicts")

    # Learning settings
    enable_learning: bool = Field(default=True, description="Enable learning from user choices")
    learning_rate: float = Field(default=0.1, description="Learning rate for trust score updates")
    min_samples_for_learning: int = Field(default=10, description="Minimum samples before learning")

    # Business rules
    enable_business_rules: bool = Field(default=True, description="Enable business rule evaluation")
    rules_override_authority: bool = Field(default=False, description="Allow rules to override authority")

    model_config = {"env_prefix": "CONFLICT_"}


class APISettings(BaseSettings):
    """API configuration"""

    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Request/Response settings
    max_request_size: int = Field(default=100 * 1024 * 1024, description="Max request size (100MB)")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")

    # Feature flags
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable request tracing")

    # OpenAI Integration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_timeout: int = Field(default=30, description="OpenAI request timeout")
    openai_max_retries: int = Field(default=3, description="OpenAI max retries")

    model_config = {"env_prefix": "API_"}


class LoggingSettings(BaseSettings):
    """Logging configuration"""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")

    # File logging
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_file_path: str = Field(default="logs/datagenie.log", description="Log file path")
    max_file_size_mb: int = Field(default=100, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of log file backups")

    # External logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    log_correlation_id: bool = Field(default=True, description="Include correlation IDs")

    model_config = {"env_prefix": "LOG_"}


class Settings(BaseSettings):
    """Main application settings"""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    app_name: str = Field(default="DataGenie Multi-Source Analytics", description="Application name")
    version: str = Field(default="5.1.1", description="Application version")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    governance: GovernanceSettings = Field(default_factory=GovernanceSettings)
    conflict_resolution: ConflictResolutionSettings = Field(default_factory=ConflictResolutionSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Directory paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    def model_post_init(self, __context: Any) -> None:
        """Create directories if they don't exist"""
        self.logs_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()


# Configuration factory for testing/different environments
def get_settings(env_file: Optional[str] = None) -> Settings:
    """Get settings with optional custom env file"""
    if env_file:
        return Settings(_env_file=env_file)
    return settings


# Validation function for startup
def validate_settings() -> List[str]:
    """Validate critical settings and return any errors"""
    errors = []

    # Check required environment variables
    if not settings.database.password and settings.is_production:
        errors.append("Database password is required in production")

    if settings.security.secret_key == "dev-secret-key-change-in-production-min-32-chars" and settings.is_production:
        errors.append("Security secret key must be changed in production")

    if settings.governance.enable_audit_logging and not settings.database.password and settings.is_production:
        errors.append("Audit logging requires database configuration in production")

    if settings.cache.enable_redis_cache and not settings.redis.host:
        errors.append("Redis cache enabled but no Redis host configured")

    # Production-specific validations
    if settings.is_production:
        if settings.debug:
            errors.append("Debug mode should not be enabled in production")

        if settings.security.cors_origins == ["*"]:
            errors.append("CORS origins should be restricted in production")

        if not settings.database.ssl_mode or settings.database.ssl_mode == "disable":
            errors.append("SSL should be enabled for database in production")

    return errors