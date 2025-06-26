# Evaluate policies in priority order
decision_reason = "No applicable allow rules found"
conditions = []
masked_fields = []
row_limit = None
confidence = 1.0  # governance/access_control.py - Production RBAC + ABAC Implementation
"""
Comprehensive Role-Based and Attribute-Based Access Control System
Supports context-aware permissions with fine-grained control
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from functools import wraps
import logging

# Note: These imports would need to be adjusted based on your actual project structure
# from core.models import (
#     User, DataSource, AccessPermission, DataSensitivityLevel,
#     QueryContext, DataAccessPermission
# )
# from core.exceptions import (
#     AccessDeniedException, InsufficientPermissionsException,
#     ResourceNotFoundException
# )

logger = logging.getLogger(__name__)


# ===============================
# Temporary Models for Demo
# ===============================

class AccessPermission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class DataSensitivityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class User:
    id: str
    username: str
    email: str
    role: str = "analyst"
    is_active: bool = True


# ===============================
# Access Control Types
# ===============================

class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class ContextType(Enum):
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    PURPOSE_BASED = "purpose_based"
    DATA_CLASSIFICATION = "data_classification"
    AGGREGATION_LEVEL = "aggregation_level"


@dataclass
class AccessContext:
    """Context information for access decisions"""
    user_id: str
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    purpose: Optional[str] = None
    aggregation_level: Optional[str] = None
    time_of_request: Optional[datetime] = None

    def __post_init__(self):
        if self.time_of_request is None:
            self.time_of_request = datetime.now(timezone.utc)


@dataclass
class PolicyRule:
    """Individual policy rule"""
    rule_id: str
    name: str
    effect: AccessDecision  # ALLOW or DENY
    priority: int  # Lower number = higher priority

    # Conditions
    subjects: List[str] = field(default_factory=list)  # User IDs, roles, groups
    resources: List[str] = field(default_factory=list)  # Resource patterns
    actions: List[AccessPermission] = field(default_factory=list)
    contexts: Dict[ContextType, Any] = field(default_factory=dict)

    # Time constraints
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Additional metadata
    created_by: str = ""
    created_at: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class AccessEvaluation:
    """Result of access evaluation"""
    decision: AccessDecision
    reason: str
    confidence: float  # 0.0 to 1.0
    applied_rules: List[str]  # Rule IDs that were applied

    # Conditional access details
    conditions: List[str] = field(default_factory=list)  # Required conditions for CONDITIONAL
    masked_fields: List[str] = field(default_factory=list)  # Fields to mask
    row_limit: Optional[int] = None

    # Metadata
    evaluation_time_ms: float = 0.0
    cached: bool = False


# ===============================
# Access Control Engine
# ===============================

class AccessControlEngine:
    """
    Production access control engine implementing RBAC + ABAC
    """

    def __init__(self):
        self._policies: List[PolicyRule] = []
        self._role_permissions: Dict[str, Set[AccessPermission]] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._evaluation_cache: Dict[str, AccessEvaluation] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default security policies"""

        # Admin allow all policy
        admin_allow = PolicyRule(
            rule_id="admin_allow_all",
            name="Admin Allow All",
            effect=AccessDecision.ALLOW,
            priority=1,
            subjects=["role:admin"],
            resources=["*"],
            actions=list(AccessPermission),
            created_by="system"
        )

        # Analyst read access to internal data
        analyst_read = PolicyRule(
            rule_id="analyst_read_internal",
            name="Analyst Read Internal Data",
            effect=AccessDecision.ALLOW,
            priority=100,
            subjects=["role:analyst"],
            resources=["data:internal", "data:public"],
            actions=[AccessPermission.READ],
            created_by="system"
        )

        # Restrict access to sensitive data outside business hours
        sensitive_hours = PolicyRule(
            rule_id="sensitive_business_hours",
            name="Sensitive Data - Business Hours Only",
            effect=AccessDecision.DENY,
            priority=50,
            subjects=["*"],
            resources=["data:confidential", "data:restricted"],
            actions=[AccessPermission.READ, AccessPermission.WRITE],
            contexts={
                ContextType.TIME_BASED: {
                    "business_hours_only": True
                }
            },
            created_by="system"
        )

        self._policies.extend([admin_allow, analyst_read, sensitive_hours])

    def add_policy(self, policy: PolicyRule) -> bool:
        """Add a new policy rule"""
        try:
            # Validate policy
            if not policy.rule_id or not policy.name:
                raise ValueError("Policy must have rule_id and name")

            # Check for duplicate rule_id
            existing_ids = {p.rule_id for p in self._policies}
            if policy.rule_id in existing_ids:
                raise ValueError(f"Policy with rule_id '{policy.rule_id}' already exists")

            self._policies.append(policy)

            # Sort policies by priority
            self._policies.sort(key=lambda p: p.priority)

            # Clear evaluation cache
            self._evaluation_cache.clear()

            logger.info(f"Added policy: {policy.rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add policy: {e}")
            return False

    def remove_policy(self, rule_id: str) -> bool:
        """Remove a policy rule"""
        try:
            self._policies = [p for p in self._policies if p.rule_id != rule_id]
            self._evaluation_cache.clear()
            logger.info(f"Removed policy: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove policy: {e}")
            return False

    def assign_role_to_user(self, user_id: str, role: str) -> bool:
        """Assign a role to a user"""
        try:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()

            self._user_roles[user_id].add(role)

            # Clear cache for this user
            self._clear_user_cache(user_id)

            logger.info(f"Assigned role '{role}' to user '{user_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return False

    def revoke_role_from_user(self, user_id: str, role: str) -> bool:
        """Revoke a role from a user"""
        try:
            if user_id in self._user_roles:
                self._user_roles[user_id].discard(role)
                if not self._user_roles[user_id]:
                    del self._user_roles[user_id]

            self._clear_user_cache(user_id)

            logger.info(f"Revoked role '{role}' from user '{user_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke role: {e}")
            return False

    def _clear_user_cache(self, user_id: str):
        """Clear cache entries for a specific user"""
        keys_to_remove = [k for k in self._evaluation_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self._evaluation_cache[key]

    def evaluate_access(self,
                        user: User,
                        resource_id: str,
                        action: AccessPermission,
                        context: AccessContext) -> AccessEvaluation:
        """
        Evaluate access request and return decision
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(user.id, resource_id, action, context)

            # Check cache
            cached_result = self._get_cached_evaluation(cache_key)
            if cached_result:
                cached_result.cached = True
                return cached_result

            # Perform evaluation
            evaluation = self._perform_evaluation(user, resource_id, action, context)

            # Cache result
            self._cache_evaluation(cache_key, evaluation)

            # Calculate evaluation time
            evaluation.evaluation_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info(f"Access evaluation: user={user.id}, resource={resource_id}, "
                        f"action={action}, decision={evaluation.decision.value}")

            return evaluation

        except Exception as e:
            logger.error(f"Access evaluation failed: {e}")
            # Fail securely - deny access on error
            return AccessEvaluation(
                decision=AccessDecision.DENY,
                reason=f"Evaluation error: {str(e)}",
                confidence=1.0,
                applied_rules=["error_fallback"],
                evaluation_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

    def _perform_evaluation(self,
                            user: User,
                            resource_id: str,
                            action: AccessPermission,
                            context: AccessContext) -> AccessEvaluation:
        """Perform the actual access evaluation"""

        applicable_rules = []
        applied_rules = []

        # Get user roles
        user_roles = self._user_roles.get(user.id, set())
        # Also include the user's primary role from their profile
        if hasattr(user, 'role') and user.role:
            user_roles.add(user.role)

        user_subjects = {user.id} | {f"role:{role}" for role in user_roles}

        # Find applicable policies
        for policy in self._policies:
            if self._is_policy_applicable(policy, user_subjects, resource_id, action, context):
                applicable_rules.append(policy)

        # If no specific policies matched, apply default deny
        if not applicable_rules:
            return AccessEvaluation(
                decision=AccessDecision.DENY,
                reason="No applicable policies found - default deny",
                confidence=1.0,
                applied_rules=["default_deny"]
            )
        final_decision = AccessDecision.DENY  # Default deny
        decision_reason = "No applicable allow rules found"
        conditions = []
        masked_fields = []
        row_limit = None
        confidence = 1.0

        for policy in applicable_rules:
            applied_rules.append(policy.rule_id)

            if policy.effect == AccessDecision.DENY:
                # Explicit deny always wins immediately
                final_decision = AccessDecision.DENY
                decision_reason = f"Denied by policy: {policy.name}"
                break  # Explicit deny overrides everything

            elif policy.effect == AccessDecision.ALLOW:
                final_decision = AccessDecision.ALLOW
                decision_reason = f"Allowed by policy: {policy.name}"

                # Apply conditional constraints
                if self._has_conditional_constraints(policy, context):
                    final_decision = AccessDecision.CONDITIONAL
                    conditions, masked_fields, row_limit = self._extract_conditions(policy, context)
                    decision_reason = f"Conditionally allowed by policy: {policy.name}"

                # Continue evaluating to check for deny policies (don't break here)
                # Only break if we hit a deny policy

        return AccessEvaluation(
            decision=final_decision,
            reason=decision_reason,
            confidence=confidence,
            applied_rules=applied_rules,
            conditions=conditions,
            masked_fields=masked_fields,
            row_limit=row_limit
        )

    def _is_policy_applicable(self,
                              policy: PolicyRule,
                              user_subjects: Set[str],
                              resource_id: str,
                              action: AccessPermission,
                              context: AccessContext) -> bool:
        """Check if a policy applies to the current request"""

        # Check if policy is active and within validity period
        if not policy.is_active:
            return False

        now = datetime.now(timezone.utc)
        if policy.valid_from and now < policy.valid_from:
            return False
        if policy.valid_until and now > policy.valid_until:
            return False

        # Check subjects (users/roles)
        if policy.subjects and not self._matches_subjects(policy.subjects, user_subjects):
            return False

        # Check resources
        if policy.resources and not self._matches_resources(policy.resources, resource_id):
            return False

        # Check actions
        if policy.actions and action not in policy.actions:
            return False

        # Check context conditions
        if policy.contexts and not self._matches_context(policy.contexts, context):
            return False

        return True

    def _matches_subjects(self, policy_subjects: List[str], user_subjects: Set[str]) -> bool:
        """Check if user subjects match policy subjects"""
        # Empty subjects list means this policy doesn't match based on subjects
        if not policy_subjects:
            return False

        if "*" in policy_subjects:
            return True

        return bool(set(policy_subjects) & user_subjects)

    def _matches_resources(self, policy_resources: List[str], resource_id: str) -> bool:
        """Check if resource matches policy resource patterns"""
        # Empty resources list means this policy doesn't match based on resources
        if not policy_resources:
            return False

        if "*" in policy_resources:
            return True

        for pattern in policy_resources:
            if self._resource_matches_pattern(resource_id, pattern):
                return True

        return False

    def _resource_matches_pattern(self, resource_id: str, pattern: str) -> bool:
        """Check if resource ID matches a pattern"""
        # Support wildcards and data classification patterns
        if pattern == "*":
            return True

        if pattern.startswith("data:"):
            # Data classification pattern (e.g., "data:confidential")
            classification = pattern.split(":", 1)[1]
            return self._resource_has_classification(resource_id, classification)

        # Exact match or simple wildcard
        if "*" in pattern:
            # Simple wildcard matching
            import fnmatch
            return fnmatch.fnmatch(resource_id, pattern)

        return resource_id == pattern

    def _resource_has_classification(self, resource_id: str, classification: str) -> bool:
        """Check if resource has a specific data classification"""
        # This would typically query the data source metadata
        # For now, implement based on resource naming conventions

        classification_map = {
            "public": DataSensitivityLevel.PUBLIC,
            "internal": DataSensitivityLevel.INTERNAL,
            "confidential": DataSensitivityLevel.CONFIDENTIAL,
            "restricted": DataSensitivityLevel.RESTRICTED,
            "top_secret": DataSensitivityLevel.TOP_SECRET
        }

        target_level = classification_map.get(classification.lower())
        if not target_level:
            return False

        # In production, this would query the actual resource metadata
        # For now, assume classification is embedded in resource_id
        return classification.lower() in resource_id.lower()

    def _matches_context(self, policy_contexts: Dict[ContextType, Any], context: AccessContext) -> bool:
        """Check if context matches policy context conditions"""

        for context_type, conditions in policy_contexts.items():
            if context_type == ContextType.TIME_BASED:
                if not self._matches_time_context(conditions, context):
                    return False

            elif context_type == ContextType.PURPOSE_BASED:
                if not self._matches_purpose_context(conditions, context):
                    return False

            # Add more context type handlers as needed

        return True

    def _matches_time_context(self, conditions: Dict[str, Any], context: AccessContext) -> bool:
        """Check time-based context conditions"""

        if "business_hours_only" in conditions:
            business_hours_required = conditions["business_hours_only"]
            is_business_hours = not self._is_outside_business_hours(context.time_of_request)

            # If business hours are required but it's not business hours, deny
            if business_hours_required and not is_business_hours:
                return True  # Match the deny policy

        if "outside_business_hours" in conditions:
            is_outside_hours = self._is_outside_business_hours(context.time_of_request)
            expected_outside = conditions["outside_business_hours"]

            if expected_outside and not is_outside_hours:
                return False
            if not expected_outside and is_outside_hours:
                return False

        if "allowed_hours" in conditions:
            allowed_ranges = conditions["allowed_hours"]
            current_hour = context.time_of_request.hour

            for start_hour, end_hour in allowed_ranges:
                if start_hour <= current_hour <= end_hour:
                    return True
            return False

        return True

    def _is_outside_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is outside business hours"""
        # Business hours: 9 AM to 6 PM, Monday to Friday
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        hour = timestamp.hour

        if weekday >= 5:  # Weekend
            return True

        if hour < 9 or hour >= 18:  # Outside 9 AM - 6 PM
            return True

        return False

    def _matches_purpose_context(self, conditions: Dict[str, Any], context: AccessContext) -> bool:
        """Check purpose-based context conditions"""

        if "allowed_purposes" in conditions:
            allowed_purposes = conditions["allowed_purposes"]
            return context.purpose in allowed_purposes

        if "forbidden_purposes" in conditions:
            forbidden_purposes = conditions["forbidden_purposes"]
            return context.purpose not in forbidden_purposes

        return True

    def _has_conditional_constraints(self, policy: PolicyRule, context: AccessContext) -> bool:
        """Check if policy has conditional constraints that should be applied"""

        # Check for data masking requirements
        if "mask_fields" in policy.contexts.get(ContextType.DATA_CLASSIFICATION, {}):
            return True

        # Check for row limiting requirements
        if "max_rows" in policy.contexts.get(ContextType.AGGREGATION_LEVEL, {}):
            return True

        return False

    def _extract_conditions(self, policy: PolicyRule, context: AccessContext) -> Tuple[
        List[str], List[str], Optional[int]]:
        """Extract conditional constraints from policy"""

        conditions = []
        masked_fields = []
        row_limit = None

        # Extract masking requirements
        data_class_context = policy.contexts.get(ContextType.DATA_CLASSIFICATION, {})
        if "mask_fields" in data_class_context:
            masked_fields = data_class_context["mask_fields"]
            conditions.append("Data masking applied")

        # Extract row limiting
        agg_context = policy.contexts.get(ContextType.AGGREGATION_LEVEL, {})
        if "max_rows" in agg_context:
            row_limit = agg_context["max_rows"]
            conditions.append(f"Results limited to {row_limit} rows")

        return conditions, masked_fields, row_limit

    def _generate_cache_key(self,
                            user_id: str,
                            resource_id: str,
                            action: AccessPermission,
                            context: AccessContext) -> str:
        """Generate cache key for access evaluation"""

        # Create a hash of the evaluation parameters
        key_data = {
            "user_id": user_id,
            "resource_id": resource_id,
            "action": action.value,
            "purpose": context.purpose,
            "hour": context.time_of_request.hour,  # Hour-level caching for time-based rules
            "day_of_week": context.time_of_request.weekday()
        }

        key_string = json.dumps(key_data, sort_keys=True)
        hash_key = hashlib.md5(key_string.encode()).hexdigest()

        return f"{user_id}:{hash_key}"

    def _get_cached_evaluation(self, cache_key: str) -> Optional[AccessEvaluation]:
        """Get cached evaluation result"""

        if cache_key not in self._evaluation_cache:
            return None

        # Check if cache entry is still valid
        # For simplicity, we're not storing cache timestamps here
        # In production, you'd want proper cache expiration

        return self._evaluation_cache[cache_key]

    def _cache_evaluation(self, cache_key: str, evaluation: AccessEvaluation):
        """Cache an evaluation result"""
        self._evaluation_cache[cache_key] = evaluation

        # Simple cache size management - remove oldest entries if cache gets too large
        if len(self._evaluation_cache) > 1000:
            # Remove 100 oldest entries
            keys_to_remove = list(self._evaluation_cache.keys())[:100]
            for key in keys_to_remove:
                del self._evaluation_cache[key]


# ===============================
# Decorators and Utilities
# ===============================

def require_permission(resource_pattern: str, action: AccessPermission, purpose: Optional[str] = None):
    """Decorator to enforce access control on functions/methods"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user context - this would depend on your authentication system
            # For now, we'll assume the user is passed as an argument or available in context

            # This is a simplified example - in practice you'd get the user from
            # request context, JWT token, session, etc.
            user = kwargs.get('user') or getattr(args[0], 'current_user', None)
            if not user:
                raise Exception("No user context available for access control")

            # Create access context
            context = AccessContext(
                user_id=user.id,
                purpose=purpose
            )

            # Get the access control engine (would typically be injected/singleton)
            access_engine = AccessControlEngine()

            # Evaluate access
            evaluation = access_engine.evaluate_access(
                user=user,
                resource_id=resource_pattern,
                action=action,
                context=context
            )

            if evaluation.decision == AccessDecision.DENY:
                raise Exception(f"Access denied: {evaluation.reason}")

            # For conditional access, you might want to modify the function behavior
            if evaluation.decision == AccessDecision.CONDITIONAL:
                # Could modify kwargs to include masking requirements, row limits, etc.
                kwargs['_access_conditions'] = evaluation.conditions
                kwargs['_masked_fields'] = evaluation.masked_fields
                kwargs['_row_limit'] = evaluation.row_limit

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ===============================
# Example Usage and Testing
# ===============================

# ===============================
# Example Usage and Testing
# ===============================

def demo_access_control():
    """Demonstrate the access control system"""

    # Create access control engine
    engine = AccessControlEngine()

    # Create test users
    admin_user = User(id="admin1", username="admin", email="admin@company.com", role="admin")
    analyst_user = User(id="analyst1", username="analyst", email="analyst@company.com", role="analyst")

    # Assign roles (this adds to the user_roles dict)
    engine.assign_role_to_user("admin1", "admin")
    engine.assign_role_to_user("analyst1", "analyst")

    print("DEBUG: User roles assigned:")
    print(f"  admin1 roles: {engine._user_roles.get('admin1', set())}")
    print(f"  analyst1 roles: {engine._user_roles.get('analyst1', set())}")
    print()

    # Test access scenarios
    scenarios = [
        # Admin should have access to everything
        (admin_user, "data:confidential:customer_data", AccessPermission.READ, "Admin read confidential"),
        (admin_user, "data:public:reports", AccessPermission.WRITE, "Admin write public"),

        # Analyst should have read access to internal data
        (analyst_user, "data:internal:sales_data", AccessPermission.READ, "Analyst read internal"),
        (analyst_user, "data:public:reports", AccessPermission.READ, "Analyst read public"),

        # Analyst should NOT have write access
        (analyst_user, "data:internal:sales_data", AccessPermission.WRITE, "Analyst write internal (should deny)"),

        # Nobody should access confidential data outside business hours
        (analyst_user, "data:confidential:customer_data", AccessPermission.READ,
         "Analyst read confidential (should deny)"),
    ]

    print("Access Control Demo Results:")
    print("=" * 50)

    for user, resource, action, description in scenarios:
        context = AccessContext(user_id=user.id)
        evaluation = engine.evaluate_access(user, resource, action, context)

        print(f"{description}:")
        print(f"  Decision: {evaluation.decision.value}")
        print(f"  Reason: {evaluation.reason}")
        print(f"  Applied Rules: {evaluation.applied_rules}")
        if evaluation.conditions:
            print(f"  Conditions: {evaluation.conditions}")
        print()


if __name__ == "__main__":
    demo_access_control()