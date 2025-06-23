# smart_defaults/models/user_profile.py
"""
Comprehensive User Profile Management for Smart Defaults

This module handles all user-related data models including:
- User profiles and role management
- Permission and access control
- Behavioral learning and preferences
- Department and organizational context
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path


class RoleCategory(Enum):
    """User role categories for classification"""
    EXECUTIVE = "executive"
    MANAGEMENT = "management"
    ANALYST = "analyst"
    SPECIALIST = "specialist"
    USER = "user"
    ADMIN = "admin"


class SecurityClearance(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class DataScope(Enum):
    """Data access scope levels"""
    PERSONAL = "personal"
    TEAM = "team"
    DEPARTMENTAL = "departmental"
    CROSS_DEPARTMENT = "cross_department"
    ENTERPRISE_WIDE = "enterprise_wide"


@dataclass
class UserPermissions:
    """Comprehensive user permissions model"""
    # Security clearance
    security_clearance: SecurityClearance = SecurityClearance.INTERNAL
    data_scope: DataScope = DataScope.DEPARTMENTAL

    # System-level permissions
    can_auto_connect: bool = True
    can_approve_connections: bool = False
    can_override_security: bool = False
    can_access_audit_logs: bool = False

    # Data-level permissions
    allowed_data_types: Set[str] = field(default_factory=set)
    restricted_data_types: Set[str] = field(default_factory=set)
    allowed_systems: Set[str] = field(default_factory=set)
    restricted_systems: Set[str] = field(default_factory=set)

    # Time-based restrictions
    access_hours: Optional[Dict[str, Any]] = None
    access_days: Optional[List[str]] = None
    session_timeout: Optional[int] = None  # minutes

    # Geographic restrictions
    allowed_locations: Optional[List[str]] = None
    restricted_locations: Optional[List[str]] = None

    # Special permissions
    emergency_access: bool = False
    audit_exempt: bool = False
    privacy_officer: bool = False

    def has_permission(self, permission_type: str, resource: str = None) -> bool:
        """Check if user has specific permission"""
        # PLACEHOLDER: In production, implement complex permission logic
        # considering role, security clearance, time, location, etc.
        return True  # Simplified for development

    def can_access_data_type(self, data_type: str) -> bool:
        """Check if user can access specific data type"""
        if data_type in self.restricted_data_types:
            return False
        if self.allowed_data_types and data_type not in self.allowed_data_types:
            return False
        return True

    def can_access_system(self, system_name: str) -> bool:
        """Check if user can access specific system"""
        if system_name in self.restricted_systems:
            return False
        if self.allowed_systems and system_name not in self.allowed_systems:
            return False
        return True


@dataclass
class UserBehavior:
    """User behavior tracking for learning algorithms"""
    # Connection behavior
    connection_acceptance_rate: float = 0.0
    connection_override_rate: float = 0.0
    manual_connection_rate: float = 0.0

    # Usage patterns
    daily_usage_hours: Dict[int, float] = field(default_factory=dict)  # hour -> usage_duration
    weekly_usage_pattern: Dict[str, float] = field(default_factory=dict)  # day -> usage_count
    seasonal_usage_patterns: Dict[str, float] = field(default_factory=dict)

    # Query patterns
    common_query_types: Dict[str, int] = field(default_factory=dict)
    preferred_data_sources: Dict[str, float] = field(default_factory=dict)
    analysis_complexity_preference: str = "moderate"  # simple, moderate, complex

    # Success metrics
    analysis_success_rate: float = 0.0
    query_completion_rate: float = 0.0
    data_quality_satisfaction: float = 0.0

    # Learning indicators
    learning_velocity: float = 0.5  # how quickly user adapts to new features
    help_seeking_frequency: float = 0.0
    feature_adoption_rate: float = 0.0

    # Last activity tracking
    last_login: Optional[datetime] = None
    last_analysis: Optional[datetime] = None
    last_connection_change: Optional[datetime] = None

    def update_connection_behavior(self, accepted: bool, was_override: bool = False):
        """Update connection behavior metrics"""
        # PLACEHOLDER: Implement exponential moving average for behavior tracking
        alpha = 0.1  # Learning rate
        if accepted:
            self.connection_acceptance_rate = (1 - alpha) * self.connection_acceptance_rate + alpha * 1.0
        else:
            self.connection_acceptance_rate = (1 - alpha) * self.connection_acceptance_rate + alpha * 0.0

        if was_override:
            self.connection_override_rate = (1 - alpha) * self.connection_override_rate + alpha * 1.0

    def update_usage_pattern(self, session_duration: float, current_time: datetime):
        """Update usage patterns with new session data"""
        hour = current_time.hour
        day = current_time.strftime('%A')

        # Update hourly usage
        if hour not in self.daily_usage_hours:
            self.daily_usage_hours[hour] = 0.0
        self.daily_usage_hours[hour] = (0.9 * self.daily_usage_hours[hour]) + (0.1 * session_duration)

        # Update daily usage
        if day not in self.weekly_usage_pattern:
            self.weekly_usage_pattern[day] = 0.0
        self.weekly_usage_pattern[day] += 1

    def get_preferred_analysis_time(self) -> Optional[int]:
        """Get user's preferred analysis time based on usage patterns"""
        if not self.daily_usage_hours:
            return None
        return max(self.daily_usage_hours, key=self.daily_usage_hours.get)


@dataclass
class UserPreferences:
    """User preferences for smart defaults behavior"""
    # Connection preferences
    auto_connect_preference: str = "smart_defaults"  # "always", "never", "smart_defaults", "ask_first"
    risk_tolerance: str = "medium"  # "low", "medium", "high"
    approval_threshold: float = 0.7  # confidence level requiring approval

    # Data preferences
    preferred_data_freshness: str = "recent"  # "real_time", "recent", "historical", "any"
    data_quality_priority: str = "balanced"  # "speed", "quality", "balanced"
    preferred_analysis_depth: str = "standard"  # "quick", "standard", "comprehensive"

    # Notification preferences
    notify_new_sources: bool = True
    notify_connection_changes: bool = True
    notify_security_alerts: bool = True
    notification_frequency: str = "important_only"  # "all", "important_only", "critical_only", "none"

    # UI/UX preferences
    dashboard_complexity: str = "standard"  # "simple", "standard", "advanced"
    default_chart_types: List[str] = field(default_factory=lambda: ["bar", "line", "table"])
    color_scheme: str = "professional"
    language: str = "en"
    timezone: str = "UTC"

    # Privacy preferences
    allow_usage_analytics: bool = True
    allow_behavioral_learning: bool = True
    allow_cross_user_learning: bool = False
    data_retention_preference: str = "standard"  # "minimal", "standard", "extended"

    def update_preference(self, key: str, value: Any):
        """Update a specific preference"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f"Unknown preference: {key}")


@dataclass
class OrganizationalContext:
    """User's organizational context and relationships"""
    # Basic organizational info
    organization_id: str
    department: str
    sub_department: Optional[str] = None
    team: Optional[str] = None
    location: Optional[str] = None

    # Reporting structure
    manager_id: Optional[str] = None
    direct_reports: List[str] = field(default_factory=list)
    peer_users: List[str] = field(default_factory=list)

    # Organizational roles
    cost_center: Optional[str] = None
    business_unit: Optional[str] = None
    geographic_region: Optional[str] = None

    # Compliance context
    regulatory_requirements: List[str] = field(default_factory=list)
    audit_scope: List[str] = field(default_factory=list)
    compliance_officer: Optional[str] = None

    # Project and initiative context
    active_projects: List[str] = field(default_factory=list)
    budget_responsibility: List[str] = field(default_factory=list)
    vendor_relationships: List[str] = field(default_factory=list)

    def get_organizational_hierarchy_level(self) -> int:
        """Calculate hierarchy level based on reporting structure"""
        # PLACEHOLDER: In production, calculate based on actual org chart
        if not self.manager_id:
            return 0  # Top level
        elif self.direct_reports:
            return 2  # Management level
        else:
            return 3  # Individual contributor


@dataclass
class UserProfile:
    """Comprehensive user profile for smart defaults system"""
    # Basic identification - REQUIRED FIELDS FIRST
    user_id: str
    username: str
    email: str
    full_name: str

    # Role and position information - REQUIRED FIELDS
    primary_role: str
    role_category: RoleCategory
    job_title: str

    # Organizational context - REQUIRED FIELD
    organizational_context: OrganizationalContext

    # OPTIONAL FIELDS WITH DEFAULTS COME AFTER REQUIRED FIELDS
    seniority_level: str = "mid"  # "entry", "mid", "senior", "executive"

    # Permissions and security
    permissions: UserPermissions = field(default_factory=UserPermissions)

    # Behavioral data and learning
    behavior: UserBehavior = field(default_factory=UserBehavior)

    # User preferences
    preferences: UserPreferences = field(default_factory=UserPreferences)

    # Profile metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    profile_version: str = "1.0"

    # Dynamic attributes
    current_session_id: Optional[str] = None
    active_connections: Set[str] = field(default_factory=set)
    recent_queries: List[Dict[str, Any]] = field(default_factory=list)

    # Learning and adaptation
    learning_profile: Dict[str, Any] = field(default_factory=dict)
    adaptation_score: float = 0.5
    onboarding_complete: bool = False

    def __post_init__(self):
        """Initialize computed fields after object creation"""
        self.profile_hash = self._calculate_profile_hash()

    def _calculate_profile_hash(self) -> str:
        """Calculate hash for profile change detection"""
        profile_data = {
            'user_id': self.user_id,
            'primary_role': self.primary_role,
            'department': self.organizational_context.department,
            'permissions': str(self.permissions.security_clearance.value)
        }
        return hashlib.md5(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()

    def update_profile(self, **kwargs):
        """Update profile with new information"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.updated_at = datetime.now()
        self.profile_hash = self._calculate_profile_hash()

    def add_recent_query(self, query: str, data_sources: List[str], success: bool):
        """Add a recent query to the profile"""
        query_record = {
            'query': query,
            'data_sources': data_sources,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session_id
        }

        self.recent_queries.append(query_record)

        # Keep only last 50 queries
        if len(self.recent_queries) > 50:
            self.recent_queries = self.recent_queries[-50:]

        # Update behavior metrics
        self.behavior.update_usage_pattern(1.0, datetime.now())
        if success:
            alpha = 0.1
            self.behavior.analysis_success_rate = (
                    (1 - alpha) * self.behavior.analysis_success_rate + alpha * 1.0
            )

    def get_role_template_match_score(self, role_template: Dict[str, Any]) -> float:
        """Calculate how well this user matches a role template"""
        score = 0.0

        # Direct role match
        if self.primary_role == role_template.get('name'):
            score += 1.0
        elif self.role_category.value == role_template.get('category'):
            score += 0.8

        # Department/function match
        if self.organizational_context.department.lower() in role_template.get('typical_departments', []):
            score += 0.3

        # Seniority match
        if self.seniority_level in role_template.get('seniority_levels', []):
            score += 0.2

        return min(score, 1.0)

    def get_data_access_capability(self) -> Dict[str, Any]:
        """Get comprehensive data access capabilities"""
        return {
            'security_clearance': self.permissions.security_clearance.value,
            'data_scope': self.permissions.data_scope.value,
            'allowed_data_types': list(self.permissions.allowed_data_types),
            'restricted_data_types': list(self.permissions.restricted_data_types),
            'emergency_access': self.permissions.emergency_access,
            'approval_authority': self.permissions.can_approve_connections
        }

    def is_similar_to(self, other_user: 'UserProfile') -> float:
        """Calculate similarity score with another user"""
        similarity_score = 0.0

        # Role similarity
        if self.primary_role == other_user.primary_role:
            similarity_score += 0.4
        elif self.role_category == other_user.role_category:
            similarity_score += 0.3

        # Department similarity
        if self.organizational_context.department == other_user.organizational_context.department:
            similarity_score += 0.3

        # Seniority similarity
        if self.seniority_level == other_user.seniority_level:
            similarity_score += 0.1

        # Behavioral similarity (simplified)
        behavior_similarity = 0.0
        if self.behavior.preferred_data_sources and other_user.behavior.preferred_data_sources:
            common_sources = set(self.behavior.preferred_data_sources.keys()) & set(
                other_user.behavior.preferred_data_sources.keys()
            )
            total_sources = set(self.behavior.preferred_data_sources.keys()) | set(
                other_user.behavior.preferred_data_sources.keys()
            )
            if total_sources:
                behavior_similarity = len(common_sources) / len(total_sources)

        similarity_score += 0.2 * behavior_similarity

        return min(similarity_score, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage/serialization"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'primary_role': self.primary_role,
            'role_category': self.role_category.value,
            'job_title': self.job_title,
            'seniority_level': self.seniority_level,
            'organizational_context': {
                'organization_id': self.organizational_context.organization_id,
                'department': self.organizational_context.department,
                'sub_department': self.organizational_context.sub_department,
                'team': self.organizational_context.team,
                'location': self.organizational_context.location,
                'manager_id': self.organizational_context.manager_id,
                'direct_reports': self.organizational_context.direct_reports,
                'cost_center': self.organizational_context.cost_center,
                'business_unit': self.organizational_context.business_unit
            },
            'permissions': {
                'security_clearance': self.permissions.security_clearance.value,
                'data_scope': self.permissions.data_scope.value,
                'can_auto_connect': self.permissions.can_auto_connect,
                'can_approve_connections': self.permissions.can_approve_connections
            },
            'preferences': {
                'auto_connect_preference': self.preferences.auto_connect_preference,
                'risk_tolerance': self.preferences.risk_tolerance,
                'approval_threshold': self.preferences.approval_threshold,
                'notification_frequency': self.preferences.notification_frequency
            },
            'behavior': {
                'connection_acceptance_rate': self.behavior.connection_acceptance_rate,
                'analysis_success_rate': self.behavior.analysis_success_rate,
                'preferred_data_sources': self.behavior.preferred_data_sources,
                'last_login': self.behavior.last_login.isoformat() if self.behavior.last_login else None
            },
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'profile_version': self.profile_version,
                'profile_hash': self.profile_hash,
                'onboarding_complete': self.onboarding_complete
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from dictionary"""
        # Parse organizational context
        org_context = OrganizationalContext(
            organization_id=data['organizational_context']['organization_id'],
            department=data['organizational_context']['department'],
            sub_department=data['organizational_context'].get('sub_department'),
            team=data['organizational_context'].get('team'),
            location=data['organizational_context'].get('location'),
            manager_id=data['organizational_context'].get('manager_id'),
            direct_reports=data['organizational_context'].get('direct_reports', []),
            cost_center=data['organizational_context'].get('cost_center'),
            business_unit=data['organizational_context'].get('business_unit')
        )

        # Parse permissions
        permissions = UserPermissions(
            security_clearance=SecurityClearance(data['permissions']['security_clearance']),
            data_scope=DataScope(data['permissions']['data_scope']),
            can_auto_connect=data['permissions']['can_auto_connect'],
            can_approve_connections=data['permissions']['can_approve_connections']
        )

        # Parse preferences
        preferences = UserPreferences(
            auto_connect_preference=data['preferences']['auto_connect_preference'],
            risk_tolerance=data['preferences']['risk_tolerance'],
            approval_threshold=data['preferences']['approval_threshold'],
            notification_frequency=data['preferences']['notification_frequency']
        )

        # Parse behavior
        behavior = UserBehavior(
            connection_acceptance_rate=data['behavior']['connection_acceptance_rate'],
            analysis_success_rate=data['behavior']['analysis_success_rate'],
            preferred_data_sources=data['behavior']['preferred_data_sources']
        )
        if data['behavior']['last_login']:
            behavior.last_login = datetime.fromisoformat(data['behavior']['last_login'])

        # Create profile
        profile = cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            full_name=data['full_name'],
            primary_role=data['primary_role'],
            role_category=RoleCategory(data['role_category']),
            job_title=data['job_title'],
            seniority_level=data['seniority_level'],
            organizational_context=org_context,
            permissions=permissions,
            preferences=preferences,
            behavior=behavior,
            created_at=datetime.fromisoformat(data['metadata']['created_at']),
            updated_at=datetime.fromisoformat(data['metadata']['updated_at']),
            profile_version=data['metadata']['profile_version'],
            onboarding_complete=data['metadata']['onboarding_complete']
        )

        return profile


class UserProfileManager:
    """Manager class for user profile operations"""

    def __init__(self, storage_backend=None):
        self.storage_backend = storage_backend
        self.profiles_cache: Dict[str, UserProfile] = {}
        self.similarity_cache: Dict[str, float] = {}

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        # Check cache first
        if user_id in self.profiles_cache:
            return self.profiles_cache[user_id]

        # PLACEHOLDER: In production, load from database
        # profile_data = await self.storage_backend.get_user_profile(user_id)
        # if profile_data:
        #     profile = UserProfile.from_dict(profile_data)
        #     self.profiles_cache[user_id] = profile
        #     return profile

        return None

    async def save_profile(self, profile: UserProfile) -> bool:
        """Save user profile"""
        try:
            # Update cache
            self.profiles_cache[profile.user_id] = profile

            # PLACEHOLDER: In production, save to database
            # await self.storage_backend.save_user_profile(profile.to_dict())

            return True
        except Exception as e:
            # PLACEHOLDER: Proper error handling
            print(f"Error saving profile: {e}")
            return False

    async def find_similar_users(self, user_id: str, limit: int = 10) -> List[tuple[str, float]]:
        """Find users similar to the given user"""
        target_profile = await self.get_profile(user_id)
        if not target_profile:
            return []

        # PLACEHOLDER: In production, implement efficient similarity search
        similar_users = []

        # For now, return empty list with placeholder logic
        # In production, this would:
        # 1. Query database for users in same role/department
        # 2. Calculate similarity scores
        # 3. Return top matches

        return similar_users

    async def update_user_behavior(self, user_id: str, behavior_data: Dict[str, Any]) -> bool:
        """Update user behavior data"""
        profile = await self.get_profile(user_id)
        if not profile:
            return False

        # Update behavior metrics
        for key, value in behavior_data.items():
            if hasattr(profile.behavior, key):
                setattr(profile.behavior, key, value)

        # Save updated profile
        return await self.save_profile(profile)

    def create_default_profile(self, user_id: str, username: str, email: str,
                               role: str, department: str) -> UserProfile:
        """Create a default user profile for new users"""
        org_context = OrganizationalContext(
            organization_id="default_org",
            department=department
        )

        # Determine role category
        role_category = RoleCategory.USER
        if "ceo" in role.lower() or "cfo" in role.lower() or "cto" in role.lower():
            role_category = RoleCategory.EXECUTIVE
        elif "manager" in role.lower() or "director" in role.lower():
            role_category = RoleCategory.MANAGEMENT
        elif "analyst" in role.lower():
            role_category = RoleCategory.ANALYST

        return UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=username,  # Default to username
            primary_role=role,
            role_category=role_category,
            job_title=role,
            organizational_context=org_context
        )