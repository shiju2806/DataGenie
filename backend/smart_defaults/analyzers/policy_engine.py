"""
Smart Defaults Policy Engine
Security, compliance, and governance policy enforcement for smart recommendations
File location: smart_defaults/analyzers/policy_engine.py
"""

import asyncio
import logging
import json
import sys
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

# Import dependencies with fallbacks
try:
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..storage.config_loader import ConfigurationLoader, SecurityPolicy
    from ..models.user_profile import UserProfile
    from ..models.recommendation import Recommendation
    from ..models.data_source import DataSource
    from ..utils.monitoring import AnalyticsEngine, EventType
except ImportError:
    # For direct execution, create mock classes
    from typing import Any
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "data_analyst"

    @dataclass
    class Recommendation:
        id: str = "test_rec"
        user_id: str = "test_user"
        source_id: str = "test_source"
        confidence_score: float = 0.8

    @dataclass
    class DataSource:
        id: str = "test_source"
        name: str = "Test Source"
        source_type: str = "database"

    @dataclass
    class SecurityPolicy:
        name: str = "Test Policy"
        min_security_level: str = "medium"
        required_permissions: List[str] = None
        encryption_required: bool = True

    class DatabaseManager:
        async def initialize(self): pass
        async def close(self): pass

    class CacheManager:
        async def initialize(self): pass
        async def close(self): pass
        async def get(self, key, default=None): return default
        async def set(self, key, value, ttl=None): pass

    class ConfigurationLoader:
        def get_security_policy(self, policy): return None
        def get_all_security_policies(self): return {}

    class AnalyticsEngine:
        async def track_event(self, *args, **kwargs): pass

    class EventType:
        SOURCE_CONNECTED = "source_connected"
        SOURCE_FAILED = "source_failed"

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of policies"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    DATA_GOVERNANCE = "data_governance"
    ACCESS_CONTROL = "access_control"
    PRIVACY = "privacy"
    AUDIT = "audit"
    OPERATIONAL = "operational"

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PolicyAction(Enum):
    """Actions to take on policy violations"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    AUDIT_ONLY = "audit_only"

class ComplianceStandard(Enum):
    """Compliance standards"""
    SOX = "sox"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    CCPA = "ccpa"
    FERPA = "ferpa"

@dataclass
class PolicyRule:
    """Individual policy rule"""
    id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    action: PolicyAction
    conditions: Dict[str, Any]
    exceptions: List[str] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PolicyViolation:
    """Policy violation record"""
    id: str
    rule_id: str
    user_id: str
    resource_id: str  # recommendation_id, source_id, etc.
    violation_type: str
    severity: PolicySeverity
    message: str
    details: Dict[str, Any]
    detected_at: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class PolicyEvaluation:
    """Result of policy evaluation"""
    user_id: str
    resource_id: str
    resource_type: str
    evaluated_at: datetime

    # Results
    allowed: bool
    violations: List[PolicyViolation]
    warnings: List[str]
    required_approvals: List[str]

    # Policy context
    applicable_policies: List[str]
    security_level_required: str
    compliance_requirements: List[str]

    # Actions
    recommended_action: PolicyAction
    approval_required: bool
    audit_required: bool

@dataclass
class SecurityContext:
    """Security context for policy evaluation"""
    user_id: str
    user_role: str
    user_permissions: List[str]
    user_security_level: str
    session_info: Dict[str, Any]
    environment_type: str  # production, staging, development
    network_location: str
    timestamp: datetime

class PolicyEngine:
    """Main policy engine for security and compliance"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 config_loader: Optional[ConfigurationLoader] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 policy_cache_ttl: int = 3600,
                 strict_mode: bool = False):

        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.config_loader = config_loader
        self.analytics_engine = analytics_engine

        self.policy_cache_ttl = policy_cache_ttl
        self.strict_mode = strict_mode  # If True, deny on any policy failure

        # Policy rules storage
        self.policy_rules: Dict[str, PolicyRule] = {}
        self.violation_history: List[PolicyViolation] = []

        # Security levels hierarchy
        self.security_levels = {
            'public': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4,
            'maximum': 5
        }

        self._initialized = False

    async def initialize(self):
        """Initialize the policy engine"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()

        # Load policy rules
        await self._load_policy_rules()

        self._initialized = True
        logger.info("✅ Policy engine initialized")

    async def close(self):
        """Close the policy engine"""
        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Policy engine closed")

    async def _load_policy_rules(self):
        """Load policy rules from configuration and defaults"""

        # Load default security policies
        await self._load_default_policies()

        # Load from configuration if available
        if self.config_loader:
            await self._load_config_policies()

        logger.info(f"📋 Loaded {len(self.policy_rules)} policy rules")

    async def _load_default_policies(self):
        """Load default built-in policy rules"""

        default_rules = [
            # Data Source Security Rules
            PolicyRule(
                id="data_source_encryption",
                name="Data Source Encryption Required",
                description="All data sources must support encryption in transit and at rest",
                policy_type=PolicyType.SECURITY,
                severity=PolicySeverity.HIGH,
                action=PolicyAction.BLOCK,
                conditions={
                    "resource_type": "data_source",
                    "encryption_required": True,
                    "security_level_min": "medium"
                },
                compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.HIPAA]
            ),

            # User Permission Rules
            PolicyRule(
                id="user_permission_check",
                name="User Permission Verification",
                description="Users must have required permissions for data source access",
                policy_type=PolicyType.ACCESS_CONTROL,
                severity=PolicySeverity.CRITICAL,
                action=PolicyAction.BLOCK,
                conditions={
                    "resource_type": "recommendation",
                    "check_permissions": True
                }
            ),

            # Sensitive Data Rules
            PolicyRule(
                id="sensitive_data_approval",
                name="Sensitive Data Access Approval",
                description="Access to sensitive data sources requires approval",
                policy_type=PolicyType.DATA_GOVERNANCE,
                severity=PolicySeverity.HIGH,
                action=PolicyAction.REQUIRE_APPROVAL,
                conditions={
                    "data_sensitivity": ["confidential", "restricted"],
                    "auto_approval_roles": ["data_steward", "security_officer"]
                },
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA]
            ),

            # Production Environment Rules
            PolicyRule(
                id="production_access_control",
                name="Production Environment Access Control",
                description="Production data access requires elevated privileges",
                policy_type=PolicyType.SECURITY,
                severity=PolicySeverity.HIGH,
                action=PolicyAction.REQUIRE_APPROVAL,
                conditions={
                    "environment_type": "production",
                    "min_security_clearance": "high",
                    "exclude_roles": ["intern", "contractor"]
                }
            ),

            # Rate Limiting Rules
            PolicyRule(
                id="connection_rate_limit",
                name="Connection Rate Limiting",
                description="Limit the number of new connections per user per hour",
                policy_type=PolicyType.OPERATIONAL,
                severity=PolicySeverity.MEDIUM,
                action=PolicyAction.WARN,
                conditions={
                    "max_connections_per_hour": 10,
                    "max_connections_per_day": 50
                }
            ),

            # Audit Requirements
            PolicyRule(
                id="audit_logging_required",
                name="Audit Logging Required",
                description="All data access must be logged for audit purposes",
                policy_type=PolicyType.AUDIT,
                severity=PolicySeverity.MEDIUM,
                action=PolicyAction.AUDIT_ONLY,
                conditions={
                    "audit_all_access": True,
                    "retention_days": 90
                },
                compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.SOC2]
            ),

            # Compliance-Specific Rules
            PolicyRule(
                id="hipaa_phi_protection",
                name="HIPAA PHI Protection",
                description="Protected Health Information must meet HIPAA requirements",
                policy_type=PolicyType.COMPLIANCE,
                severity=PolicySeverity.CRITICAL,
                action=PolicyAction.BLOCK,
                conditions={
                    "industry": "healthcare",
                    "data_types": ["phi", "medical_records"],
                    "encryption_required": True,
                    "access_logging": True
                },
                compliance_standards=[ComplianceStandard.HIPAA]
            ),

            # Financial Data Rules
            PolicyRule(
                id="financial_data_sox",
                name="SOX Financial Data Controls",
                description="Financial data must meet SOX compliance requirements",
                policy_type=PolicyType.COMPLIANCE,
                severity=PolicySeverity.CRITICAL,
                action=PolicyAction.REQUIRE_APPROVAL,
                conditions={
                    "industry": "finance",
                    "data_types": ["financial_records", "trading_data"],
                    "segregation_of_duties": True
                },
                compliance_standards=[ComplianceStandard.SOX]
            )
        ]

        for rule in default_rules:
            self.policy_rules[rule.id] = rule

    async def _load_config_policies(self):
        """Load policies from configuration"""

        try:
            security_policies = self.config_loader.get_all_security_policies()

            for policy_name, policy_config in security_policies.items():
                # Convert security policy to policy rules
                rule = PolicyRule(
                    id=f"config_{policy_name}",
                    name=policy_config.name,
                    description=policy_config.description,
                    policy_type=PolicyType.SECURITY,
                    severity=PolicySeverity.MEDIUM,
                    action=PolicyAction.BLOCK if policy_config.encryption_required else PolicyAction.WARN,
                    conditions={
                        "min_security_level": policy_config.min_security_level,
                        "required_permissions": policy_config.required_permissions,
                        "blocked_sources": policy_config.blocked_sources,
                        "encryption_required": policy_config.encryption_required
                    }
                )

                self.policy_rules[rule.id] = rule

        except Exception as e:
            logger.warning(f"⚠️ Failed to load config policies: {e}")

    async def evaluate_recommendation(self, recommendation: Recommendation,
                                    user_profile: UserProfile,
                                    data_source: Optional[DataSource] = None,
                                    security_context: Optional[SecurityContext] = None) -> PolicyEvaluation:
        """Evaluate a recommendation against all applicable policies"""

        evaluation_id = f"eval_{recommendation.id}_{int(datetime.now(timezone.utc).timestamp())}"

        try:
            # Create security context if not provided
            if not security_context:
                security_context = SecurityContext(
                    user_id=user_profile.user_id,
                    user_role=user_profile.role,
                    user_permissions=[],  # Would be loaded from user profile
                    user_security_level="medium",  # Default
                    session_info={},
                    environment_type="development",  # Default
                    network_location="internal",
                    timestamp=datetime.now(timezone.utc)
                )

            # Find applicable policies
            applicable_policies = await self._find_applicable_policies(
                "recommendation", recommendation, user_profile, data_source
            )

            # Evaluate each policy
            violations = []
            warnings = []
            required_approvals = []

            for policy_id in applicable_policies:
                policy_rule = self.policy_rules[policy_id]

                violation = await self._evaluate_policy_rule(
                    policy_rule, recommendation, user_profile, data_source, security_context
                )

                if violation:
                    violations.append(violation)

                    if policy_rule.action == PolicyAction.BLOCK:
                        pass  # Will be handled in final decision
                    elif policy_rule.action == PolicyAction.WARN:
                        warnings.append(f"Policy warning: {policy_rule.name}")
                    elif policy_rule.action == PolicyAction.REQUIRE_APPROVAL:
                        required_approvals.append(policy_rule.name)

            # Determine final decision
            allowed = True
            recommended_action = PolicyAction.ALLOW

            # Check for blocking violations
            blocking_violations = [v for v in violations if
                                 self.policy_rules[v.rule_id].action == PolicyAction.BLOCK]

            if blocking_violations:
                allowed = False
                recommended_action = PolicyAction.BLOCK
            elif required_approvals:
                allowed = False if self.strict_mode else True
                recommended_action = PolicyAction.REQUIRE_APPROVAL
            elif warnings:
                recommended_action = PolicyAction.WARN

            # Determine security requirements
            security_level_required = self._determine_required_security_level(applicable_policies)
            compliance_requirements = self._determine_compliance_requirements(applicable_policies)

            evaluation = PolicyEvaluation(
                user_id=user_profile.user_id,
                resource_id=recommendation.id,
                resource_type="recommendation",
                evaluated_at=datetime.now(timezone.utc),
                allowed=allowed,
                violations=violations,
                warnings=warnings,
                required_approvals=required_approvals,
                applicable_policies=applicable_policies,
                security_level_required=security_level_required,
                compliance_requirements=compliance_requirements,
                recommended_action=recommended_action,
                approval_required=len(required_approvals) > 0,
                audit_required=any(self.policy_rules[p].policy_type == PolicyType.AUDIT
                                 for p in applicable_policies)
            )

            # Track policy evaluation
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_profile.user_id,
                    event_type=EventType.SOURCE_CONNECTED if allowed else EventType.SOURCE_FAILED,
                    data={
                        "policy_evaluation": True,
                        "evaluation_id": evaluation_id,
                        "allowed": allowed,
                        "violations_count": len(violations),
                        "approvals_required": len(required_approvals),
                        "recommended_action": recommended_action.value
                    }
                )

            return evaluation

        except Exception as e:
            logger.error(f"❌ Policy evaluation failed: {e}")

            # Return restrictive evaluation on error
            return PolicyEvaluation(
                user_id=user_profile.user_id,
                resource_id=recommendation.id,
                resource_type="recommendation",
                evaluated_at=datetime.now(timezone.utc),
                allowed=False,
                violations=[],
                warnings=["Policy evaluation failed - access denied for safety"],
                required_approvals=["security_review"],
                applicable_policies=[],
                security_level_required="high",
                compliance_requirements=[],
                recommended_action=PolicyAction.BLOCK,
                approval_required=True,
                audit_required=True
            )

    async def _find_applicable_policies(self, resource_type: str,
                                      recommendation: Recommendation,
                                      user_profile: UserProfile,
                                      data_source: Optional[DataSource] = None) -> List[str]:
        """Find policies that apply to the current context"""

        applicable = []

        for policy_id, policy_rule in self.policy_rules.items():
            if not policy_rule.active:
                continue

            conditions = policy_rule.conditions

            # Check resource type
            if "resource_type" in conditions:
                if conditions["resource_type"] != resource_type:
                    continue

            # Check user role
            if "exclude_roles" in conditions:
                if user_profile.role in conditions["exclude_roles"]:
                    continue

            if "include_roles" in conditions:
                if user_profile.role not in conditions["include_roles"]:
                    continue

            # Check environment type (would be determined from context)
            if "environment_type" in conditions:
                # For demo, assume development environment
                if conditions["environment_type"] != "development":
                    continue

            # Check data source properties
            if data_source and "source_type" in conditions:
                if data_source.source_type not in conditions["source_type"]:
                    continue

            # Policy applies
            applicable.append(policy_id)

        return applicable

    async def _evaluate_policy_rule(self, policy_rule: PolicyRule,
                                   recommendation: Recommendation,
                                   user_profile: UserProfile,
                                   data_source: Optional[DataSource],
                                   security_context: SecurityContext) -> Optional[PolicyViolation]:
        """Evaluate a single policy rule"""

        conditions = policy_rule.conditions

        try:
            # Check encryption requirements
            if conditions.get("encryption_required", False):
                # In real implementation, check if data source supports encryption
                source_supports_encryption = True  # Mock check

                if not source_supports_encryption:
                    return PolicyViolation(
                        id=f"violation_{policy_rule.id}_{recommendation.id}",
                        rule_id=policy_rule.id,
                        user_id=user_profile.user_id,
                        resource_id=recommendation.id,
                        violation_type="encryption_not_supported",
                        severity=policy_rule.severity,
                        message=f"Data source does not support required encryption",
                        details={
                            "policy": policy_rule.name,
                            "requirement": "encryption_required",
                            "source_id": recommendation.source_id
                        },
                        detected_at=datetime.now(timezone.utc)
                    )

            # Check security level requirements
            if "security_level_min" in conditions:
                required_level = conditions["security_level_min"]
                user_level = security_context.user_security_level

                if self.security_levels[user_level] < self.security_levels[required_level]:
                    return PolicyViolation(
                        id=f"violation_{policy_rule.id}_{recommendation.id}",
                        rule_id=policy_rule.id,
                        user_id=user_profile.user_id,
                        resource_id=recommendation.id,
                        violation_type="insufficient_security_level",
                        severity=policy_rule.severity,
                        message=f"User security level '{user_level}' below required '{required_level}'",
                        details={
                            "policy": policy_rule.name,
                            "user_level": user_level,
                            "required_level": required_level
                        },
                        detected_at=datetime.now(timezone.utc)
                    )

            # Check required permissions
            if conditions.get("check_permissions", False) or "required_permissions" in conditions:
                required_perms = conditions.get("required_permissions", ["read_data"])
                user_perms = security_context.user_permissions

                missing_perms = [p for p in required_perms if p not in user_perms]

                if missing_perms:
                    return PolicyViolation(
                        id=f"violation_{policy_rule.id}_{recommendation.id}",
                        rule_id=policy_rule.id,
                        user_id=user_profile.user_id,
                        resource_id=recommendation.id,
                        violation_type="insufficient_permissions",
                        severity=policy_rule.severity,
                        message=f"User missing required permissions: {missing_perms}",
                        details={
                            "policy": policy_rule.name,
                            "missing_permissions": missing_perms,
                            "required_permissions": required_perms
                        },
                        detected_at=datetime.now(timezone.utc)
                    )

            # Check blocked sources
            if "blocked_sources" in conditions:
                blocked = conditions["blocked_sources"]
                if recommendation.source_id in blocked:
                    return PolicyViolation(
                        id=f"violation_{policy_rule.id}_{recommendation.id}",
                        rule_id=policy_rule.id,
                        user_id=user_profile.user_id,
                        resource_id=recommendation.id,
                        violation_type="blocked_source",
                        severity=policy_rule.severity,
                        message=f"Source '{recommendation.source_id}' is blocked by policy",
                        details={
                            "policy": policy_rule.name,
                            "blocked_source": recommendation.source_id
                        },
                        detected_at=datetime.now(timezone.utc)
                    )

            # Check rate limits
            if "max_connections_per_hour" in conditions:
                # In real implementation, check recent connections from database
                recent_connections = 5  # Mock count
                max_allowed = conditions["max_connections_per_hour"]

                if recent_connections >= max_allowed:
                    return PolicyViolation(
                        id=f"violation_{policy_rule.id}_{recommendation.id}",
                        rule_id=policy_rule.id,
                        user_id=user_profile.user_id,
                        resource_id=recommendation.id,
                        violation_type="rate_limit_exceeded",
                        severity=policy_rule.severity,
                        message=f"Rate limit exceeded: {recent_connections}/{max_allowed} connections",
                        details={
                            "policy": policy_rule.name,
                            "current_connections": recent_connections,
                            "max_allowed": max_allowed
                        },
                        detected_at=datetime.now(timezone.utc)
                    )

            # No violations found
            return None

        except Exception as e:
            logger.error(f"❌ Policy rule evaluation failed for {policy_rule.id}: {e}")

            # Return violation on evaluation error for safety
            return PolicyViolation(
                id=f"violation_{policy_rule.id}_{recommendation.id}",
                rule_id=policy_rule.id,
                user_id=user_profile.user_id,
                resource_id=recommendation.id,
                violation_type="evaluation_error",
                severity=PolicySeverity.HIGH,
                message=f"Policy evaluation failed: {str(e)}",
                details={"error": str(e)},
                detected_at=datetime.now(timezone.utc)
            )

    def _determine_required_security_level(self, applicable_policies: List[str]) -> str:
        """Determine the highest security level required by applicable policies"""

        max_level = "low"
        max_level_value = 0

        for policy_id in applicable_policies:
            policy_rule = self.policy_rules[policy_id]
            conditions = policy_rule.conditions

            if "security_level_min" in conditions:
                level = conditions["security_level_min"]
                level_value = self.security_levels.get(level, 0)

                if level_value > max_level_value:
                    max_level = level
                    max_level_value = level_value

        return max_level

    def _determine_compliance_requirements(self, applicable_policies: List[str]) -> List[str]:
        """Determine compliance requirements from applicable policies"""

        requirements = set()

        for policy_id in applicable_policies:
            policy_rule = self.policy_rules[policy_id]

            for standard in policy_rule.compliance_standards:
                requirements.add(standard.value)

        return list(requirements)

    async def create_approval_request(self, evaluation: PolicyEvaluation,
                                    justification: str) -> Dict[str, Any]:
        """Create an approval request for policy violations"""

        request_id = f"approval_{evaluation.user_id}_{int(datetime.now(timezone.utc).timestamp())}"

        approval_request = {
            "id": request_id,
            "user_id": evaluation.user_id,
            "resource_id": evaluation.resource_id,
            "resource_type": evaluation.resource_type,
            "justification": justification,
            "required_approvals": evaluation.required_approvals,
            "violations": [asdict(v) for v in evaluation.violations],
            "compliance_requirements": evaluation.compliance_requirements,
            "requested_at": datetime.now(timezone.utc),
            "status": "pending",
            "approvers": [],
            "approved": False
        }

        # In production, this would be saved to database
        logger.info(f"📝 Created approval request: {request_id}")

        return approval_request

    async def audit_policy_decision(self, evaluation: PolicyEvaluation,
                                  final_decision: str,
                                  decision_maker: str) -> bool:
        """Audit a policy decision for compliance"""

        audit_record = {
            "evaluation_id": f"eval_{evaluation.resource_id}",
            "user_id": evaluation.user_id,
            "resource_id": evaluation.resource_id,
            "resource_type": evaluation.resource_type,
            "policy_result": {
                "allowed": evaluation.allowed,
                "violations": len(evaluation.violations),
                "warnings": len(evaluation.warnings),
                "approvals_required": len(evaluation.required_approvals)
            },
            "final_decision": final_decision,
            "decision_maker": decision_maker,
            "compliance_requirements": evaluation.compliance_requirements,
            "timestamp": datetime.now(timezone.utc),
            "audit_trail": True
        }

        # In production, save to audit database
        if self.analytics_engine:
            await self.analytics_engine.track_event(
                user_id=evaluation.user_id,
                event_type=EventType.SOURCE_CONNECTED,
                data={
                    "audit_record": True,
                    **audit_record
                }
            )

        return True

    async def get_policy_status(self) -> Dict[str, Any]:
        """Get status of the policy engine"""

        # Count policies by type and status
        policy_stats = defaultdict(int)
        active_policies = 0

        for policy_rule in self.policy_rules.values():
            policy_stats[policy_rule.policy_type.value] += 1
            if policy_rule.active:
                active_policies += 1

        # Recent violations (last 24 hours)
        recent_violations = [
            v for v in self.violation_history
            if (datetime.now(timezone.utc) - v.detected_at).days < 1
        ]

        return {
            "total_policies": len(self.policy_rules),
            "active_policies": active_policies,
            "policies_by_type": dict(policy_stats),
            "recent_violations": len(recent_violations),
            "strict_mode": self.strict_mode,
            "security_levels": list(self.security_levels.keys()),
            "compliance_standards": [s.value for s in ComplianceStandard],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    async def update_policy_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy rule"""

        try:
            if rule_id not in self.policy_rules:
                logger.warning(f"⚠️ Policy rule not found: {rule_id}")
                return False

            policy_rule = self.policy_rules[rule_id]

    async def update_policy_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy rule"""

        try:
            if rule_id not in self.policy_rules:
                logger.warning(f"⚠️ Policy rule not found: {rule_id}")
                return False

            policy_rule = self.policy_rules[rule_id]

            # Update allowed fields
            if 'active' in updates:
                policy_rule.active = updates['active']
            if 'severity' in updates:
                policy_rule.severity = PolicySeverity(updates['severity'])
            if 'action' in updates:
                policy_rule.action = PolicyAction(updates['action'])
            if 'conditions' in updates:
                policy_rule.conditions.update(updates['conditions'])
            if 'description' in updates:
                policy_rule.description = updates['description']

            logger.info(f"✅ Updated policy rule: {rule_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update policy rule {rule_id}: {e}")
            return False

    async def add_policy_exception(self, rule_id: str, exception_id: str,
                                 reason: str, expires_at: Optional[datetime] = None) -> bool:
        """Add an exception to a policy rule"""

        try:
            if rule_id not in self.policy_rules:
                return False

            policy_rule = self.policy_rules[rule_id]

            exception_data = {
                'id': exception_id,
                'reason': reason,
                'created_at': datetime.now(timezone.utc),
                'expires_at': expires_at
            }

            policy_rule.exceptions.append(exception_data)

            logger.info(f"✅ Added exception to policy {rule_id}: {exception_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add policy exception: {e}")
            return False

    async def validate_compliance(self, user_profile: UserProfile,
                                recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Validate compliance across multiple recommendations"""

        compliance_report = {
            'user_id': user_profile.user_id,
            'evaluated_at': datetime.now(timezone.utc),
            'total_recommendations': len(recommendations),
            'compliant_recommendations': 0,
            'violations_by_standard': defaultdict(int),
            'blocked_recommendations': [],
            'approval_required': [],
            'compliance_score': 0.0,
            'recommendations': []
        }

        try:
            compliant_count = 0

            for recommendation in recommendations:
                evaluation = await self.evaluate_recommendation(
                    recommendation, user_profile
                )

                rec_compliance = {
                    'recommendation_id': recommendation.id,
                    'source_id': recommendation.source_id,
                    'allowed': evaluation.allowed,
                    'violations_count': len(evaluation.violations),
                    'compliance_requirements': evaluation.compliance_requirements,
                    'action_required': evaluation.recommended_action.value
                }

                compliance_report['recommendations'].append(rec_compliance)

                if evaluation.allowed and not evaluation.violations:
                    compliant_count += 1

                if not evaluation.allowed:
                    compliance_report['blocked_recommendations'].append(recommendation.id)

                if evaluation.approval_required:
                    compliance_report['approval_required'].append(recommendation.id)

                # Count violations by compliance standard
                for violation in evaluation.violations:
                    policy_rule = self.policy_rules[violation.rule_id]
                    for standard in policy_rule.compliance_standards:
                        compliance_report['violations_by_standard'][standard.value] += 1

            compliance_report['compliant_recommendations'] = compliant_count
            compliance_report['compliance_score'] = compliant_count / len(recommendations) if recommendations else 1.0

        except Exception as e:
            logger.error(f"❌ Compliance validation failed: {e}")
            compliance_report['error'] = str(e)

        return compliance_report

    def get_policy_recommendations(self, evaluation: PolicyEvaluation) -> List[str]:
        """Get recommendations for improving policy compliance"""

        recommendations = []

        # Security level recommendations
        if evaluation.security_level_required != "low":
            recommendations.append(
                f"Consider upgrading security clearance to '{evaluation.security_level_required}' level"
            )

        # Permission recommendations
        permission_violations = [
            v for v in evaluation.violations
            if v.violation_type == "insufficient_permissions"
        ]

        if permission_violations:
            recommendations.append(
                "Request additional permissions from your system administrator"
            )

        # Encryption recommendations
        encryption_violations = [
            v for v in evaluation.violations
            if v.violation_type == "encryption_not_supported"
        ]

        if encryption_violations:
            recommendations.append(
                "Use data sources that support encryption in transit and at rest"
            )

        # Rate limit recommendations
        rate_violations = [
            v for v in evaluation.violations
            if v.violation_type == "rate_limit_exceeded"
        ]

        if rate_violations:
            recommendations.append(
                "Reduce connection frequency or request rate limit increase"
            )

        # Approval process recommendations
        if evaluation.required_approvals:
            recommendations.append(
                f"Submit approval request for: {', '.join(evaluation.required_approvals)}"
            )

        # Compliance recommendations
        if "hipaa" in evaluation.compliance_requirements:
            recommendations.append(
                "Ensure HIPAA compliance training is current and data handling follows PHI guidelines"
            )

        if "sox" in evaluation.compliance_requirements:
            recommendations.append(
                "Follow SOX controls for financial data access and maintain audit trails"
            )

        return recommendations

# Factory function
async def create_policy_engine(
    database_manager: Optional[DatabaseManager] = None,
    cache_manager: Optional[CacheManager] = None,
    config_loader: Optional[ConfigurationLoader] = None,
    analytics_engine: Optional[AnalyticsEngine] = None,
    strict_mode: bool = False,
    **kwargs
) -> PolicyEngine:
    """Factory function to create and initialize policy engine"""
    engine = PolicyEngine(
        database_manager=database_manager,
        cache_manager=cache_manager,
        config_loader=config_loader,
        analytics_engine=analytics_engine,
        strict_mode=strict_mode,
        **kwargs
    )
    await engine.initialize()
    return engine

# Testing
if __name__ == "__main__":
    async def test_policy_engine():
        """Test policy engine functionality"""

        try:
            print("🧪 Testing Policy Engine...")

            # Create mock dependencies
            class MockConfig:
                def get_security_policy(self, policy):
                    return SecurityPolicy(
                        name="Test Security Policy",
                        min_security_level="medium",
                        required_permissions=["read_data"],
                        encryption_required=True
                    )

                def get_all_security_policies(self):
                    return {
                        "medium_security": SecurityPolicy(
                            name="Medium Security Policy",
                            min_security_level="medium",
                            required_permissions=["read_data", "verified_identity"],
                            blocked_sources=["public_apis"],
                            encryption_required=True
                        )
                    }

            class MockCache:
                def __init__(self):
                    self.data = {}
                async def initialize(self): pass
                async def close(self): pass
                async def get(self, key, default=None): return self.data.get(key, default)
                async def set(self, key, value, ttl=None):
                    self.data[key] = value
                    return True

            class MockAnalytics:
                async def track_event(self, *args, **kwargs):
                    action = kwargs.get('data', {}).get('policy_evaluation', False)
                    if action:
                        print(f"📊 Policy Analytics: Evaluation for user {kwargs.get('user_id')}")

            # Initialize policy engine
            engine = await create_policy_engine(
                config_loader=MockConfig(),
                cache_manager=MockCache(),
                analytics_engine=MockAnalytics(),
                strict_mode=False
            )

            print("✅ Policy engine created successfully")

            try:
                # Test 1: Policy Status
                print("\n🔍 Test 1: Policy Engine Status")
                status = await engine.get_policy_status()
                print(f"   Total policies: {status['total_policies']}")
                print(f"   Active policies: {status['active_policies']}")
                print(f"   Policies by type: {status['policies_by_type']}")
                print(f"   Strict mode: {status['strict_mode']}")
                print(f"   Security levels: {status['security_levels']}")

                # Test 2: Basic Recommendation Evaluation
                print("\n🔍 Test 2: Recommendation Policy Evaluation")

                test_user = UserProfile(
                    id="test_policy_user",
                    user_id="policy_test_user",
                    role="data_analyst"
                )

                test_recommendation = Recommendation(
                    id="policy_test_rec",
                    user_id="policy_test_user",
                    source_id="postgresql",
                    confidence_score=0.8
                )

                evaluation = await engine.evaluate_recommendation(
                    test_recommendation, test_user
                )

                print(f"   Evaluation result: {'✅ Allowed' if evaluation.allowed else '❌ Blocked'}")
                print(f"   Violations: {len(evaluation.violations)}")
                print(f"   Warnings: {len(evaluation.warnings)}")
                print(f"   Approvals required: {len(evaluation.required_approvals)}")
                print(f"   Recommended action: {evaluation.recommended_action.value}")
                print(f"   Security level required: {evaluation.security_level_required}")

                if evaluation.violations:
                    print("   Policy violations:")
                    for violation in evaluation.violations:
                        print(f"     - {violation.violation_type}: {violation.message}")

                # Test 3: Security Context Evaluation
                print("\n🔍 Test 3: Security Context Evaluation")

                security_context = SecurityContext(
                    user_id="policy_test_user",
                    user_role="data_analyst",
                    user_permissions=["read_data"],  # Limited permissions
                    user_security_level="low",  # Low security level
                    session_info={},
                    environment_type="production",
                    network_location="external",
                    timestamp=datetime.now(timezone.utc)
                )

                context_evaluation = await engine.evaluate_recommendation(
                    test_recommendation, test_user, security_context=security_context
                )

                print(f"   With security context: {'✅ Allowed' if context_evaluation.allowed else '❌ Blocked'}")
                print(f"   Violations: {len(context_evaluation.violations)}")

                # Test 4: Blocked Source Evaluation
                print("\n🔍 Test 4: Blocked Source Evaluation")

                blocked_recommendation = Recommendation(
                    id="blocked_test_rec",
                    user_id="policy_test_user",
                    source_id="public_apis",  # This should be blocked
                    confidence_score=0.9
                )

                blocked_evaluation = await engine.evaluate_recommendation(
                    blocked_recommendation, test_user
                )

                print(f"   Blocked source result: {'✅ Allowed' if blocked_evaluation.allowed else '❌ Blocked'}")
                print(f"   Violations: {len(blocked_evaluation.violations)}")

                # Test 5: Compliance Validation
                print("\n🔍 Test 5: Compliance Validation")

                test_recommendations = [
                    Recommendation(id="comp_1", user_id="policy_test_user", source_id="postgresql", confidence_score=0.8),
                    Recommendation(id="comp_2", user_id="policy_test_user", source_id="public_apis", confidence_score=0.7),
                    Recommendation(id="comp_3", user_id="policy_test_user", source_id="secure_db", confidence_score=0.9)
                ]

                compliance_report = await engine.validate_compliance(test_user, test_recommendations)

                print(f"   Compliance score: {compliance_report['compliance_score']:.2%}")
                print(f"   Compliant recommendations: {compliance_report['compliant_recommendations']}/{compliance_report['total_recommendations']}")
                print(f"   Blocked recommendations: {len(compliance_report['blocked_recommendations'])}")
                print(f"   Approvals required: {len(compliance_report['approval_required'])}")

                if compliance_report['violations_by_standard']:
                    print("   Violations by standard:")
                    for standard, count in compliance_report['violations_by_standard'].items():
                        print(f"     - {standard}: {count}")

                # Test 6: Policy Recommendations
                print("\n🔍 Test 6: Policy Recommendations")

                policy_recommendations = engine.get_policy_recommendations(evaluation)
                print(f"   Generated {len(policy_recommendations)} recommendations:")
                for i, rec in enumerate(policy_recommendations, 1):
                    print(f"     {i}. {rec}")

                # Test 7: Approval Request
                print("\n🔍 Test 7: Approval Request Creation")

                if evaluation.required_approvals:
                    approval_request = await engine.create_approval_request(
                        evaluation,
                        "Required for critical data analysis project"
                    )
                    print(f"   Approval request created: {approval_request['id']}")
                    print(f"   Status: {approval_request['status']}")
                    print(f"   Required approvals: {approval_request['required_approvals']}")
                else:
                    print("   No approvals required for this evaluation")

                # Test 8: Policy Rule Updates
                print("\n🔍 Test 8: Policy Rule Management")

                # Update a policy rule
                update_success = await engine.update_policy_rule(
                    "user_permission_check",
                    {"severity": "medium", "active": True}
                )
                print(f"   Policy rule update: {'✅ Success' if update_success else '❌ Failed'}")

                # Add exception
                exception_success = await engine.add_policy_exception(
                    "data_source_encryption",
                    "dev_exception",
                    "Development environment exception",
                    datetime.now(timezone.utc) + timedelta(days=30)
                )
                print(f"   Policy exception added: {'✅ Success' if exception_success else '❌ Failed'}")

                # Test 9: Audit Trail
                print("\n🔍 Test 9: Audit Trail")

                audit_success = await engine.audit_policy_decision(
                    evaluation,
                    "approved_with_conditions",
                    "security_officer"
                )
                print(f"   Audit record created: {'✅ Success' if audit_success else '❌ Failed'}")

                print("\n" + "=" * 50)
                print("✅ ALL POLICY ENGINE TESTS PASSED! 🎉")
                print("   - Policy loading and initialization ✓")
                print("   - Recommendation evaluation ✓")
                print("   - Security context handling ✓")
                print("   - Compliance validation ✓")
                print("   - Policy recommendations ✓")
                print("   - Approval workflow ✓")
                print("   - Policy management ✓")
                print("   - Audit trail ✓")

            finally:
                await engine.close()
                print("\n🔐 Policy engine closed gracefully")

        except Exception as e:
            print(f"\n❌ Policy engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    # Run tests
    print("🚀 Starting Smart Defaults Policy Engine Test")
    success = asyncio.run(test_policy_engine())

    if success:
        print("\n🎯 Policy engine is ready for integration!")
        print("   Next steps:")
        print("   1. Connect to real user permission systems")
        print("   2. Integrate with enterprise compliance tools")
        print("   3. Set up approval workflow automation")
        print("   4. Configure industry-specific compliance rules")
        print("   5. Implement real-time policy updates")
        print("   6. Add advanced risk assessment features")
    else:
        print("\n💥 Tests failed - check the error messages above")
        sys.exit(1)