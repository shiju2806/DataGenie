# smart_defaults/models/policy.py
"""
Comprehensive Policy Management for Smart Defaults

This module handles all policy-related models including:
- Security policies and access control rules
- Compliance frameworks and regulations
- Data governance and classification policies
- Approval workflows and audit requirements
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from pathlib import Path


class PolicyType(Enum):
    """Types of policies supported"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    DATA_GOVERNANCE = "data_governance"
    ACCESS_CONTROL = "access_control"
    PRIVACY = "privacy"
    OPERATIONAL = "operational"
    BUSINESS = "business"


class PolicyScope(Enum):
    """Scope of policy application"""
    GLOBAL = "global"  # Applies to entire organization
    DEPARTMENT = "department"  # Applies to specific department
    ROLE = "role"  # Applies to specific roles
    USER = "user"  # Applies to specific users
    DATA_TYPE = "data_type"  # Applies to specific data types
    SYSTEM = "system"  # Applies to specific systems


class PolicySeverity(Enum):
    """Severity levels for policy violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PolicyAction(Enum):
    """Actions that can be taken when policy is triggered"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    LOG_ONLY = "log_only"
    WARN = "warn"
    MASK_DATA = "mask_data"
    REDIRECT = "redirect"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOC_2 = "soc_2"
    ISO_27001 = "iso_27001"
    BASEL_III = "basel_iii"
    MIFID_II = "mifid_ii"
    CCPA = "ccpa"
    OSHA = "osha"
    ISO_9001 = "iso_9001"
    ISO_14001 = "iso_14001"


@dataclass
class PolicyCondition:
    """Represents a condition that triggers a policy"""
    # Condition identification
    condition_id: str
    condition_type: str  # user_role, data_type, system_name, time, location, etc.

    # Condition logic
    operator: str  # equals, contains, matches, in, not_in, greater_than, etc.
    value: Union[str, List[str], Dict[str, Any]]
    case_sensitive: bool = False

    # Advanced matching
    regex_pattern: Optional[str] = None
    custom_function: Optional[str] = None  # Reference to custom validation function

    # Metadata
    description: str = ""
    examples: List[str] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if this condition matches the given context"""
        if self.condition_type not in context:
            return False

        context_value = context[self.condition_type]
        condition_value = self.value

        # Handle case sensitivity
        if isinstance(context_value, str) and isinstance(condition_value, str) and not self.case_sensitive:
            context_value = context_value.lower()
            condition_value = condition_value.lower()

        # Apply operator logic
        try:
            if self.operator == "equals":
                return context_value == condition_value
            elif self.operator == "not_equals":
                return context_value != condition_value
            elif self.operator == "contains":
                return str(condition_value) in str(context_value)
            elif self.operator == "not_contains":
                return str(condition_value) not in str(context_value)
            elif self.operator == "in":
                return context_value in condition_value
            elif self.operator == "not_in":
                return context_value not in condition_value
            elif self.operator == "starts_with":
                return str(context_value).startswith(str(condition_value))
            elif self.operator == "ends_with":
                return str(context_value).endswith(str(condition_value))
            elif self.operator == "matches" and self.regex_pattern:
                return bool(re.match(self.regex_pattern, str(context_value)))
            elif self.operator == "greater_than":
                return float(context_value) > float(condition_value)
            elif self.operator == "less_than":
                return float(context_value) < float(condition_value)
            elif self.operator == "exists":
                return context_value is not None
            elif self.operator == "not_exists":
                return context_value is None
            else:
                return False
        except (ValueError, TypeError, AttributeError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary"""
        return {
            'condition_id': self.condition_id,
            'condition_type': self.condition_type,
            'operator': self.operator,
            'value': self.value,
            'case_sensitive': self.case_sensitive,
            'regex_pattern': self.regex_pattern,
            'custom_function': self.custom_function,
            'description': self.description,
            'examples': self.examples
        }


@dataclass
class PolicyRule:
    """Represents a complete policy rule with conditions and actions"""
    # Rule identification
    rule_id: str
    name: str
    description: str
    version: str = "1.0"

    # Rule logic
    conditions: List[PolicyCondition] = field(default_factory=list)
    condition_logic: str = "AND"  # AND, OR, CUSTOM
    custom_logic: Optional[str] = None  # Custom boolean expression

    # Actions and responses
    action: PolicyAction = PolicyAction.LOG_ONLY
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Approval workflow
    approval_required: bool = False
    approval_roles: List[str] = field(default_factory=list)
    approval_timeout: Optional[timedelta] = None
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Notifications and alerting
    notify_on_trigger: bool = True
    notification_recipients: List[str] = field(default_factory=list)
    notification_template: Optional[str] = None
    alert_severity: PolicySeverity = PolicySeverity.MEDIUM

    # Rule metadata
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True

    # Temporal settings
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    schedule: Optional[Dict[str, Any]] = None  # Time-based activation

    # Compliance and audit
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    audit_required: bool = False
    retention_period: Optional[timedelta] = None

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the rule against given context"""
        result = {
            'rule_id': self.rule_id,
            'triggered': False,
            'action': None,
            'message': '',
            'conditions_met': [],
            'conditions_failed': [],
            'evaluation_time': datetime.now()
        }

        # Check if rule is active
        if not self.enabled:
            result['message'] = 'Rule is disabled'
            return result

        # Check effective dates
        now = datetime.now()
        if self.effective_date and now < self.effective_date:
            result['message'] = 'Rule not yet effective'
            return result

        if self.expiration_date and now > self.expiration_date:
            result['message'] = 'Rule has expired'
            return result

        # Evaluate conditions
        condition_results = []
        for condition in self.conditions:
            is_met = condition.evaluate(context)
            condition_results.append(is_met)

            if is_met:
                result['conditions_met'].append(condition.condition_id)
            else:
                result['conditions_failed'].append(condition.condition_id)

        # Apply condition logic
        if self.condition_logic == "AND":
            rule_triggered = all(condition_results)
        elif self.condition_logic == "OR":
            rule_triggered = any(condition_results)
        elif self.condition_logic == "CUSTOM" and self.custom_logic:
            # PLACEHOLDER: Implement custom logic evaluation
            rule_triggered = any(condition_results)  # Fallback to OR
        else:
            rule_triggered = all(condition_results)  # Default to AND

        result['triggered'] = rule_triggered

        if rule_triggered:
            result['action'] = self.action.value
            result['message'] = f"Rule '{self.name}' triggered"

            # Add action parameters
            if self.action_parameters:
                result['action_parameters'] = self.action_parameters

            # Add approval information if required
            if self.approval_required:
                result['approval_required'] = True
                result['approval_roles'] = self.approval_roles
                result['approval_timeout'] = self.approval_timeout.total_seconds() if self.approval_timeout else None

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'conditions': [c.to_dict() for c in self.conditions],
            'condition_logic': self.condition_logic,
            'custom_logic': self.custom_logic,
            'action': self.action.value,
            'action_parameters': self.action_parameters,
            'approval_required': self.approval_required,
            'approval_roles': self.approval_roles,
            'approval_timeout': self.approval_timeout.total_seconds() if self.approval_timeout else None,
            'notify_on_trigger': self.notify_on_trigger,
            'notification_recipients': self.notification_recipients,
            'alert_severity': self.alert_severity.value,
            'tags': self.tags,
            'category': self.category,
            'priority': self.priority,
            'enabled': self.enabled,
            'effective_date': self.effective_date.isoformat() if self.effective_date else None,
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'compliance_frameworks': [cf.value for cf in self.compliance_frameworks],
            'audit_required': self.audit_required
        }


@dataclass
class Policy:
    """Comprehensive policy definition with multiple rules"""
    # Policy identification
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope

    # Policy rules
    rules: List[PolicyRule] = field(default_factory=list)
    rule_execution_order: str = "priority"  # priority, sequence, parallel

    # Policy metadata
    version: str = "1.0"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_by: str = ""
    updated_at: datetime = field(default_factory=datetime.now)

    # Approval and governance
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    review_frequency: Optional[timedelta] = None

    # Application scope
    applies_to_roles: List[str] = field(default_factory=list)
    applies_to_departments: List[str] = field(default_factory=list)
    applies_to_data_types: List[str] = field(default_factory=list)
    applies_to_systems: List[str] = field(default_factory=list)

    # Compliance and legal
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)
    business_justification: str = ""

    # Operational settings
    enabled: bool = True
    enforcement_mode: str = "enforce"  # enforce, monitor, test
    exception_handling: str = "strict"  # strict, flexible, custom

    # Documentation
    documentation_links: List[str] = field(default_factory=list)
    training_materials: List[str] = field(default_factory=list)
    faq: List[Dict[str, str]] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all rules in this policy"""
        policy_result = {
            'policy_id': self.policy_id,
            'policy_name': self.name,
            'applicable': self._is_applicable(context),
            'rules_evaluated': [],
            'triggered_rules': [],
            'final_action': PolicyAction.ALLOW.value,
            'requires_approval': False,
            'approval_roles': [],
            'messages': [],
            'evaluation_time': datetime.now()
        }

        if not policy_result['applicable']:
            policy_result['messages'].append(f"Policy '{self.name}' not applicable to current context")
            return policy_result

        if not self.enabled:
            policy_result['messages'].append(f"Policy '{self.name}' is disabled")
            return policy_result

        # Sort rules by priority if needed
        rules_to_evaluate = self.rules
        if self.rule_execution_order == "priority":
            rules_to_evaluate = sorted(self.rules, key=lambda r: r.priority)

        # Evaluate each rule
        for rule in rules_to_evaluate:
            if not rule.enabled:
                continue

            rule_result = rule.evaluate(context)
            policy_result['rules_evaluated'].append(rule_result)

            if rule_result['triggered']:
                policy_result['triggered_rules'].append(rule_result)

                # Update final action based on most restrictive rule
                current_action = PolicyAction(rule_result['action'])
                if self._is_more_restrictive(current_action, PolicyAction(policy_result['final_action'])):
                    policy_result['final_action'] = current_action.value

                # Aggregate approval requirements
                if rule_result.get('approval_required'):
                    policy_result['requires_approval'] = True
                    policy_result['approval_roles'].extend(rule_result.get('approval_roles', []))

                policy_result['messages'].append(rule_result['message'])

                # If enforcement mode is not parallel and rule denies, stop evaluation
                if current_action == PolicyAction.DENY and self.rule_execution_order != "parallel":
                    break

        # Remove duplicate approval roles
        policy_result['approval_roles'] = list(set(policy_result['approval_roles']))

        return policy_result

    def _is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this policy applies to the given context"""
        # Check role applicability
        if self.applies_to_roles:
            user_role = context.get('user_role', '')
            if user_role not in self.applies_to_roles:
                return False

        # Check department applicability
        if self.applies_to_departments:
            user_department = context.get('user_department', '')
            if user_department not in self.applies_to_departments:
                return False

        # Check data type applicability
        if self.applies_to_data_types:
            data_type = context.get('data_type', '')
            if data_type not in self.applies_to_data_types:
                return False

        # Check system applicability
        if self.applies_to_systems:
            system_name = context.get('system_name', '')
            if system_name not in self.applies_to_systems:
                return False

        return True

    def _is_more_restrictive(self, action1: PolicyAction, action2: PolicyAction) -> bool:
        """Check if action1 is more restrictive than action2"""
        restrictiveness_order = {
            PolicyAction.ALLOW: 0,
            PolicyAction.LOG_ONLY: 1,
            PolicyAction.WARN: 2,
            PolicyAction.MASK_DATA: 3,
            PolicyAction.REQUIRE_APPROVAL: 4,
            PolicyAction.DENY: 5
        }

        return restrictiveness_order.get(action1, 0) > restrictiveness_order.get(action2, 0)

    def add_rule(self, rule: PolicyRule):
        """Add a new rule to this policy"""
        self.rules.append(rule)
        self.updated_at = datetime.now()

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from this policy"""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

        if len(self.rules) < initial_count:
            self.updated_at = datetime.now()
            return True
        return False

    def get_applicable_rules(self, context: Dict[str, Any]) -> List[PolicyRule]:
        """Get all rules that apply to the given context"""
        if not self._is_applicable(context):
            return []

        applicable_rules = []
        for rule in self.rules:
            if rule.enabled:
                applicable_rules.append(rule)

        return applicable_rules

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'policy_type': self.policy_type.value,
            'scope': self.scope.value,
            'rules': [r.to_dict() for r in self.rules],
            'rule_execution_order': self.rule_execution_order,
            'version': self.version,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_by': self.updated_by,
            'updated_at': self.updated_at.isoformat(),
            'approved_by': self.approved_by,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'next_review_date': self.next_review_date.isoformat() if self.next_review_date else None,
            'applies_to_roles': self.applies_to_roles,
            'applies_to_departments': self.applies_to_departments,
            'applies_to_data_types': self.applies_to_data_types,
            'applies_to_systems': self.applies_to_systems,
            'compliance_frameworks': [cf.value for cf in self.compliance_frameworks],
            'legal_references': self.legal_references,
            'business_justification': self.business_justification,
            'enabled': self.enabled,
            'enforcement_mode': self.enforcement_mode,
            'exception_handling': self.exception_handling,
            'documentation_links': self.documentation_links
        }


@dataclass
class PolicyViolation:
    """Record of a policy violation"""
    # Violation identification
    violation_id: str
    policy_id: str
    rule_id: str

    # Violation details
    violation_time: datetime = field(default_factory=datetime.now)
    user_id: str = ""
    action_attempted: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Violation metadata
    severity: PolicySeverity = PolicySeverity.MEDIUM
    status: str = "open"  # open, investigating, resolved, false_positive
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # Impact assessment
    impact_level: str = "medium"  # low, medium, high, critical
    affected_systems: List[str] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)

    # Response and remediation
    immediate_action_taken: Optional[str] = None
    remediation_steps: List[str] = field(default_factory=list)
    escalated_to: List[str] = field(default_factory=list)

    # Compliance and reporting
    reportable_incident: bool = False
    reported_to_authorities: bool = False
    compliance_officer_notified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary"""
        return {
            'violation_id': self.violation_id,
            'policy_id': self.policy_id,
            'rule_id': self.rule_id,
            'violation_time': self.violation_time.isoformat(),
            'user_id': self.user_id,
            'action_attempted': self.action_attempted,
            'context': self.context,
            'severity': self.severity.value,
            'status': self.status,
            'resolution': self.resolution,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'impact_level': self.impact_level,
            'affected_systems': self.affected_systems,
            'affected_data': self.affected_data,
            'immediate_action_taken': self.immediate_action_taken,
            'remediation_steps': self.remediation_steps,
            'escalated_to': self.escalated_to,
            'reportable_incident': self.reportable_incident,
            'reported_to_authorities': self.reported_to_authorities,
            'compliance_officer_notified': self.compliance_officer_notified
        }


class PolicyEngine:
    """Central policy engine for evaluating and enforcing policies"""

    def __init__(self, storage_backend=None):
        self.storage_backend = storage_backend
        self.policies: Dict[str, Policy] = {}
        self.violations: List[PolicyViolation] = []
        self.cached_evaluations: Dict[str, Any] = {}

    async def load_policies(self) -> bool:
        """Load all policies from storage"""
        try:
            # PLACEHOLDER: Load from database or configuration files
            # policies_data = await self.storage_backend.get_all_policies()
            # for policy_data in policies_data:
            #     policy = self._create_policy_from_dict(policy_data)
            #     self.policies[policy.policy_id] = policy

            return True
        except Exception as e:
            print(f"Error loading policies: {e}")
            return False

    async def evaluate_policies(self, context: Dict[str, Any], policy_types: Optional[List[PolicyType]] = None) -> Dict[
        str, Any]:
        """Evaluate all applicable policies for given context"""
        evaluation_result = {
            'evaluation_id': f"eval_{datetime.now().timestamp()}",
            'context': context,
            'policies_evaluated': [],
            'violations': [],
            'final_decision': PolicyAction.ALLOW.value,
            'requires_approval': False,
            'approval_roles': [],
            'messages': [],
            'evaluation_time': datetime.now()
        }

        # Filter policies by type if specified
        policies_to_evaluate = self.policies.values()
        if policy_types:
            policies_to_evaluate = [p for p in policies_to_evaluate if p.policy_type in policy_types]

        # Evaluate each applicable policy
        for policy in policies_to_evaluate:
            if not policy.enabled:
                continue

            policy_result = policy.evaluate(context)
            evaluation_result['policies_evaluated'].append(policy_result)

            # Check for violations
            if policy_result['triggered_rules']:
                for triggered_rule in policy_result['triggered_rules']:
                    if triggered_rule['action'] in ['deny', 'require_approval']:
                        violation = self._create_violation_record(policy, triggered_rule, context)
                        evaluation_result['violations'].append(violation.to_dict())

            # Update final decision
            policy_action = PolicyAction(policy_result['final_action'])
            current_decision = PolicyAction(evaluation_result['final_decision'])

            if policy._is_more_restrictive(policy_action, current_decision):
                evaluation_result['final_decision'] = policy_action.value

            # Aggregate approval requirements
            if policy_result['requires_approval']:
                evaluation_result['requires_approval'] = True
                evaluation_result['approval_roles'].extend(policy_result['approval_roles'])

            evaluation_result['messages'].extend(policy_result['messages'])

        # Remove duplicate approval roles
        evaluation_result['approval_roles'] = list(set(evaluation_result['approval_roles']))

        return evaluation_result

    def _create_violation_record(self, policy: Policy, triggered_rule: Dict[str, Any],
                                 context: Dict[str, Any]) -> PolicyViolation:
        """Create a policy violation record"""
        violation = PolicyViolation(
            violation_id=f"viol_{datetime.now().timestamp()}",
            policy_id=policy.policy_id,
            rule_id=triggered_rule['rule_id'],
            user_id=context.get('user_id', ''),
            action_attempted=context.get('action', ''),
            context=context,
            severity=PolicySeverity.MEDIUM  # Default, should be determined by rule
        )

        self.violations.append(violation)
        return violation

    async def add_policy(self, policy: Policy) -> bool:
        """Add a new policy to the engine"""
        try:
            self.policies[policy.policy_id] = policy

            # PLACEHOLDER: Save to storage
            # await self.storage_backend.save_policy(policy.to_dict())

            return True
        except Exception as e:
            print(f"Error adding policy: {e}")
            return False

    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy"""
        if policy_id not in self.policies:
            return False

        try:
            policy = self.policies[policy_id]

            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)

            policy.updated_at = datetime.now()

            # PLACEHOLDER: Save to storage
            # await self.storage_backend.update_policy(policy_id, policy.to_dict())

            return True
        except Exception as e:
            print(f"Error updating policy: {e}")
            return False

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy"""
        if policy_id not in self.policies:
            return False

        try:
            del self.policies[policy_id]

            # PLACEHOLDER: Delete from storage
            # await self.storage_backend.delete_policy(policy_id)

            return True
        except Exception as e:
            print(f"Error deleting policy: {e}")
            return False

    def get_policies_by_type(self, policy_type: PolicyType) -> List[Policy]:
        """Get all policies of a specific type"""
        return [p for p in self.policies.values() if p.policy_type == policy_type]

    def get_policies_for_context(self, context: Dict[str, Any]) -> List[Policy]:
        """Get all policies applicable to a given context"""
        applicable_policies = []

        for policy in self.policies.values():
            if policy._is_applicable(context):
                applicable_policies.append(policy)

        return applicable_policies

    def get_violation_summary(self, time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of policy violations"""
        cutoff_time = datetime.now() - time_period if time_period else datetime.min

        recent_violations = [v for v in self.violations if v.violation_time >= cutoff_time]

        return {
            'total_violations': len(recent_violations),
            'violations_by_severity': self._group_violations_by_severity(recent_violations),
            'violations_by_policy': self._group_violations_by_policy(recent_violations),
            'top_violating_users': self._get_top_violating_users(recent_violations),
            'trend_analysis': self._analyze_violation_trends(recent_violations)
        }

    def _group_violations_by_severity(self, violations: List[PolicyViolation]) -> Dict[str, int]:
        """Group violations by severity level"""
        severity_counts = {}
        for violation in violations:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts

    def _group_violations_by_policy(self, violations: List[PolicyViolation]) -> Dict[str, int]:
        """Group violations by policy"""
        policy_counts = {}
        for violation in violations:
            policy_id = violation.policy_id
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
        return policy_counts

    def _get_top_violating_users(self, violations: List[PolicyViolation], limit: int = 10) -> List[Dict[str, Any]]:
        """Get users with most violations"""
        user_counts = {}
        for violation in violations:
            user_id = violation.user_id
            if user_id:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'user_id': user_id, 'violation_count': count} for user_id, count in sorted_users[:limit]]

    def _analyze_violation_trends(self, violations: List[PolicyViolation]) -> Dict[str, Any]:
        """Analyze violation trends over time"""
        # PLACEHOLDER: Implement trend analysis
        # This would analyze violations over time to identify patterns
        return {
            'trend_direction': 'stable',  # increasing, decreasing, stable
            'trend_percentage': 0.0,
            'peak_violation_times': [],
            'seasonal_patterns': {}
        }