# Evaluate policies in priority order
decision_reason = "No applicable allow rules found"
conditions = []
masked_fields = []
row_limit = None
confidence = 1.0  # phase1_launcher.py - Complete Phase 1 Multi-Source DataGenie with Integrated Components
"""
Production-ready Phase 1 implementation combining:
- Data Governance (RBAC + ABAC)
- Smart Caching System
- Conflict Resolution Engine
- Multi-Source Integration

Standalone launcher with FastAPI integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import pickle
from functools import wraps
from collections import defaultdict, Counter
import statistics

# Core imports
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DataGenie Phase 1 - Multi-Source Analytics",
    description="Production-ready multi-source data analytics with governance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# PHASE 1: CORE MODELS & TYPES
# ===============================

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


class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


@dataclass
class SimpleUser:
    """Simplified user model for Phase 1"""
    id: str
    username: str
    email: str
    role: str = "analyst"
    expertise_level: str = "intermediate"
    permissions: List[AccessPermission] = field(default_factory=list)
    data_access_level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL


@dataclass
class SimpleDataSource:
    """Simplified data source model for Phase 1"""
    id: str
    name: str
    source_type: str
    sensitivity_level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL
    owner: str = "system"
    is_active: bool = True


@dataclass
class AccessContext:
    """Context for access decisions"""
    user_id: str
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    purpose: Optional[str] = None
    time_of_request: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessEvaluation:
    """Result of access evaluation"""
    decision: AccessDecision
    reason: str
    confidence: float
    applied_rules: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    masked_fields: List[str] = field(default_factory=list)
    row_limit: Optional[int] = None
    evaluation_time_ms: float = 0.0


@dataclass
class ConflictingValue:
    """A conflicting value from a source"""
    source_id: str
    value: Any
    timestamp: datetime
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConflict:
    """Data conflict between sources"""
    id: str
    field_name: str
    record_identifier: str
    conflicting_values: List[ConflictingValue]
    severity: str = "medium"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Resolution info
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_value: Optional[Any] = None
    resolution_confidence: Optional[float] = None
    resolved_at: Optional[datetime] = None


# ===============================
# PHASE 1: ACCESS CONTROL ENGINE
# ===============================

class AccessControlEngine:
    """Production RBAC + ABAC Engine"""

    def __init__(self):
        self._user_roles: Dict[str, Set[str]] = {}
        self._source_permissions: Dict[str, Dict[str, Set[AccessPermission]]] = {}
        self._evaluation_cache: Dict[str, AccessEvaluation] = {}

        # Initialize default permissions
        self._setup_default_permissions()

    def _setup_default_permissions(self):
        """Setup default role-based permissions"""

        # Admin can access everything
        self.assign_role_to_user("admin", "admin")

        # Analyst can read internal and public data
        self.assign_role_to_user("analyst", "analyst")

        # Viewer can only read public data with limitations
        self.assign_role_to_user("viewer", "viewer")

    def assign_role_to_user(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
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

    def _clear_user_cache(self, user_id: str):
        """Clear cache for user"""
        keys_to_remove = [k for k in self._evaluation_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self._evaluation_cache[key]

    def evaluate_access(self,
                        user: SimpleUser,
                        resource_id: str,
                        action: AccessPermission,
                        context: AccessContext) -> AccessEvaluation:
        """Evaluate access request"""

        start_time = datetime.now(timezone.utc)

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(user.id, resource_id, action, context)

            # Check cache
            if cache_key in self._evaluation_cache:
                cached = self._evaluation_cache[cache_key]
                cached.evaluation_time_ms = 0.0  # Cached
                return cached

            # Perform evaluation
            evaluation = self._perform_evaluation(user, resource_id, action, context)

            # Cache result
            self._evaluation_cache[cache_key] = evaluation

            # Calculate timing
            evaluation.evaluation_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return evaluation

        except Exception as e:
            logger.error(f"Access evaluation failed: {e}")
            return AccessEvaluation(
                decision=AccessDecision.DENY,
                reason=f"Evaluation error: {str(e)}",
                confidence=1.0,
                applied_rules=["error_fallback"]
            )

    def _perform_evaluation(self,
                            user: SimpleUser,
                            resource_id: str,
                            action: AccessPermission,
                            context: AccessContext) -> AccessEvaluation:
        """Perform access evaluation logic"""

        user_roles = self._user_roles.get(user.id, set())

        # Admin gets everything
        if "admin" in user_roles:
            return AccessEvaluation(
                decision=AccessDecision.ALLOW,
                reason="Admin access",
                confidence=1.0,
                applied_rules=["admin_rule"]
            )

        # Parse resource for sensitivity level
        sensitivity = self._extract_sensitivity_from_resource(resource_id)

        # Analyst rules
        if "analyst" in user_roles:
            if sensitivity in [DataSensitivityLevel.PUBLIC, DataSensitivityLevel.INTERNAL]:
                if action == AccessPermission.READ:
                    return AccessEvaluation(
                        decision=AccessDecision.ALLOW,
                        reason="Analyst read access to internal/public data",
                        confidence=0.9,
                        applied_rules=["analyst_read_rule"]
                    )

        # Viewer rules with conditions
        if "viewer" in user_roles:
            if sensitivity == DataSensitivityLevel.PUBLIC and action == AccessPermission.READ:
                return AccessEvaluation(
                    decision=AccessDecision.CONDITIONAL,
                    reason="Viewer access with data masking",
                    confidence=0.8,
                    applied_rules=["viewer_conditional_rule"],
                    conditions=["Data masking applied", "Row limit applied"],
                    masked_fields=["email", "phone", "ssn"],
                    row_limit=1000
                )

        # Business hours restriction for sensitive data
        if sensitivity in [DataSensitivityLevel.CONFIDENTIAL, DataSensitivityLevel.RESTRICTED]:
            if self._is_outside_business_hours(context.time_of_request):
                return AccessEvaluation(
                    decision=AccessDecision.DENY,
                    reason="Sensitive data access restricted outside business hours",
                    confidence=1.0,
                    applied_rules=["business_hours_rule"]
                )

        # Default deny
        return AccessEvaluation(
            decision=AccessDecision.DENY,
            reason="No applicable allow rules",
            confidence=1.0,
            applied_rules=["default_deny"]
        )

    def _extract_sensitivity_from_resource(self, resource_id: str) -> DataSensitivityLevel:
        """Extract sensitivity level from resource ID"""
        if ":confidential" in resource_id.lower():
            return DataSensitivityLevel.CONFIDENTIAL
        elif ":restricted" in resource_id.lower():
            return DataSensitivityLevel.RESTRICTED
        elif ":public" in resource_id.lower():
            return DataSensitivityLevel.PUBLIC
        else:
            return DataSensitivityLevel.INTERNAL

    def _is_outside_business_hours(self, timestamp: datetime) -> bool:
        """Check if outside business hours (9 AM - 6 PM, Mon-Fri)"""
        weekday = timestamp.weekday()
        hour = timestamp.hour

        if weekday >= 5:  # Weekend
            return True
        if hour < 9 or hour >= 18:  # Outside 9-6
            return True
        return False

    def _generate_cache_key(self, user_id: str, resource_id: str, action: AccessPermission,
                            context: AccessContext) -> str:
        """Generate cache key"""
        key_data = {
            "user_id": user_id,
            "resource_id": resource_id,
            "action": action.value,
            "purpose": context.purpose,
            "hour": context.time_of_request.hour
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# ===============================
# PHASE 1: SMART CACHE MANAGER
# ===============================

class SmartCacheManager:
    """Production cache manager with multiple strategies"""

    def __init__(self):
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.max_memory_size = 1000

        # Different TTL for different data types
        self.ttl_configs = {
            "query_results": 3600,  # 1 hour
            "data_source_metadata": 86400,  # 24 hours
            "access_evaluations": 300,  # 5 minutes
            "conflict_resolutions": 7200,  # 2 hours
            "join_results": 1800  # 30 minutes
        }

    async def get(self, key: str, data_type: str = "query_results") -> Optional[Any]:
        """Get from cache with automatic expiration"""

        # Record access
        self._record_access(key)

        if key in self._memory_cache:
            value, expires_at = self._memory_cache[key]

            if datetime.now(timezone.utc) <= expires_at:
                self._cache_stats["hits"] += 1
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                # Expired
                del self._memory_cache[key]

        self._cache_stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    async def set(self, key: str, value: Any, data_type: str = "query_results",
                  custom_ttl: Optional[int] = None) -> bool:
        """Set cache with intelligent TTL"""

        try:
            ttl = custom_ttl or self.ttl_configs.get(data_type, 3600)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

            # Check if we need to evict
            if len(self._memory_cache) >= self.max_memory_size:
                await self._evict_lru()

            self._memory_cache[key] = (value, expires_at)

            # Promote frequently accessed items to longer TTL
            if self._is_frequently_accessed(key):
                promotion_factor = 2.0
                new_expires = datetime.now(timezone.utc) + timedelta(seconds=int(ttl * promotion_factor))
                self._memory_cache[key] = (value, new_expires)
                logger.debug(f"Cache promotion: {key}")

            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        if key in self._memory_cache:
            del self._memory_cache[key]
            return True
        return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        invalidated = 0
        keys_to_remove = []

        for key in self._memory_cache.keys():
            if pattern in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            await self.delete(key)
            invalidated += 1
            self._cache_stats["evictions"] += 1

        return invalidated

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_ops = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_ops if total_ops > 0 else 0.0

        return {
            "entries_count": len(self._memory_cache),
            "max_size": self.max_memory_size,
            "utilization": len(self._memory_cache) / self.max_memory_size,
            "hit_rate": hit_rate,
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "evictions": self._cache_stats["evictions"],
            "access_patterns": len(self._access_patterns)
        }

    def _record_access(self, key: str):
        """Record access pattern"""
        self._access_patterns[key].append(datetime.now(timezone.utc))

        # Keep only recent accesses
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self._access_patterns[key] = [dt for dt in self._access_patterns[key] if dt > cutoff]

    def _is_frequently_accessed(self, key: str) -> bool:
        """Check if key is frequently accessed"""
        accesses = self._access_patterns.get(key, [])
        return len(accesses) >= 5  # 5+ accesses in last hour

    async def _evict_lru(self):
        """Evict least recently used items"""
        # Find LRU key
        lru_key = None
        oldest_access = datetime.now(timezone.utc)

        for key in self._memory_cache.keys():
            accesses = self._access_patterns.get(key, [])
            last_access = accesses[-1] if accesses else datetime.min.replace(tzinfo=timezone.utc)

            if last_access < oldest_access:
                oldest_access = last_access
                lru_key = key

        if lru_key:
            await self.delete(lru_key)
            self._cache_stats["evictions"] += 1
            logger.debug(f"Evicted LRU: {lru_key}")


# ===============================
# PHASE 1: CONFLICT RESOLUTION ENGINE
# ===============================

class ConflictResolutionEngine:
    """Production conflict resolution for multi-source data"""

    def __init__(self):
        self._source_authority_scores: Dict[str, float] = {}
        self._resolution_feedback: Dict[str, List[Dict]] = defaultdict(list)
        self._trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)

        # Business rules for different fields
        self.business_rules = {
            "price": ConflictResolutionStrategy.RECENCY_BASED,
            "revenue": ConflictResolutionStrategy.AUTHORITY_BASED,
            "customer_id": ConflictResolutionStrategy.CONSENSUS_BASED,
            "status": ConflictResolutionStrategy.RECENCY_BASED
        }

    def set_source_authority(self, source_id: str, authority_score: float):
        """Set authority score for a source (0.0 - 1.0)"""
        self._source_authority_scores[source_id] = max(0.0, min(1.0, authority_score))

    async def resolve_conflict(self,
                               conflict: DataConflict,
                               strategy: Optional[ConflictResolutionStrategy] = None) -> DataConflict:
        """Resolve a data conflict"""

        start_time = datetime.now(timezone.utc)

        # Determine strategy
        if not strategy:
            strategy = self._determine_strategy(conflict)

        conflict.resolution_strategy = strategy

        # Apply resolution
        if strategy == ConflictResolutionStrategy.AUTHORITY_BASED:
            resolved = await self._resolve_by_authority(conflict)
        elif strategy == ConflictResolutionStrategy.RECENCY_BASED:
            resolved = await self._resolve_by_recency(conflict)
        elif strategy == ConflictResolutionStrategy.CONSENSUS_BASED:
            resolved = await self._resolve_by_consensus(conflict)
        elif strategy == ConflictResolutionStrategy.BUSINESS_RULE_BASED:
            resolved = await self._resolve_by_business_rules(conflict)
        elif strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
            resolved = await self._resolve_by_confidence(conflict)
        else:
            # USER_CHOICE - return conflict as is for user to decide
            return conflict

        conflict.resolved_at = datetime.now(timezone.utc)

        # Learn from resolution
        self._update_trust_scores(conflict)

        logger.info(f"Resolved conflict {conflict.id} using {strategy.value} in "
                    f"{(datetime.now(timezone.utc) - start_time).total_seconds() * 1000:.2f}ms")

        return conflict

    def _determine_strategy(self, conflict: DataConflict) -> ConflictResolutionStrategy:
        """Determine best strategy based on field and context"""

        # Check business rules first
        if conflict.field_name in self.business_rules:
            return self.business_rules[conflict.field_name]

        # Default strategies based on field patterns
        field_lower = conflict.field_name.lower()

        if any(word in field_lower for word in ["price", "cost", "amount", "rate"]):
            return ConflictResolutionStrategy.RECENCY_BASED
        elif any(word in field_lower for word in ["id", "code", "key"]):
            return ConflictResolutionStrategy.CONSENSUS_BASED
        elif any(word in field_lower for word in ["name", "description", "title"]):
            return ConflictResolutionStrategy.AUTHORITY_BASED
        else:
            return ConflictResolutionStrategy.CONFIDENCE_WEIGHTED

    async def _resolve_by_authority(self, conflict: DataConflict) -> DataConflict:
        """Resolve by source authority"""
        best_value = None
        best_score = -1.0

        for cv in conflict.conflicting_values:
            authority = self._source_authority_scores.get(cv.source_id, 0.5)
            trust = self._trust_scores[cv.source_id]
            combined_score = (authority * 0.7 + trust * 0.3)

            if combined_score > best_score:
                best_score = combined_score
                best_value = cv

        if best_value:
            conflict.resolved_value = best_value.value
            conflict.resolution_confidence = best_score

        return conflict

    async def _resolve_by_recency(self, conflict: DataConflict) -> DataConflict:
        """Resolve by most recent value"""
        sorted_values = sorted(conflict.conflicting_values,
                               key=lambda cv: cv.timestamp,
                               reverse=True)

        if sorted_values:
            most_recent = sorted_values[0]
            conflict.resolved_value = most_recent.value

            # Confidence based on time difference
            if len(sorted_values) > 1:
                time_diff = (most_recent.timestamp - sorted_values[1].timestamp).total_seconds()
                confidence = min(1.0, time_diff / 3600)  # Higher confidence if >1 hour difference
            else:
                confidence = 1.0

            conflict.resolution_confidence = confidence

        return conflict

    async def _resolve_by_consensus(self, conflict: DataConflict) -> DataConflict:
        """Resolve by consensus (majority vote)"""
        value_counts = Counter()
        value_sources = defaultdict(list)

        for cv in conflict.conflicting_values:
            value_counts[cv.value] += 1
            value_sources[cv.value].append(cv.source_id)

        if value_counts:
            # Get most common value
            most_common_value, count = value_counts.most_common(1)[0]
            total_sources = len(conflict.conflicting_values)

            conflict.resolved_value = most_common_value
            conflict.resolution_confidence = count / total_sources

        return conflict

    async def _resolve_by_business_rules(self, conflict: DataConflict) -> DataConflict:
        """Apply specific business rules"""

        # Example business rules
        if conflict.field_name == "revenue":
            # For revenue, take the maximum conservative estimate
            numeric_values = []
            for cv in conflict.conflicting_values:
                try:
                    numeric_values.append((float(cv.value), cv))
                except:
                    pass

            if numeric_values:
                # Take median as conservative estimate
                sorted_values = sorted(numeric_values, key=lambda x: x[0])
                median_idx = len(sorted_values) // 2
                conflict.resolved_value = sorted_values[median_idx][1].value
                conflict.resolution_confidence = 0.8

        elif conflict.field_name == "status":
            # For status, prefer active/open statuses
            active_statuses = ["active", "open", "enabled", "live"]
            for cv in conflict.conflicting_values:
                if str(cv.value).lower() in active_statuses:
                    conflict.resolved_value = cv.value
                    conflict.resolution_confidence = 0.9
                    break

        return conflict

    async def _resolve_by_confidence(self, conflict: DataConflict) -> DataConflict:
        """Resolve by confidence scores"""
        best_value = None
        best_confidence = -1.0

        for cv in conflict.conflicting_values:
            # Combine source confidence with trust score
            trust = self._trust_scores[cv.source_id]
            combined_confidence = cv.confidence * 0.6 + trust * 0.4

            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_value = cv

        if best_value:
            conflict.resolved_value = best_value.value
            conflict.resolution_confidence = best_confidence

        return conflict

    def _update_trust_scores(self, conflict: DataConflict):
        """Update trust scores based on resolution"""
        if conflict.resolved_value is None:
            return

        # Sources that had the resolved value get trust boost
        for cv in conflict.conflicting_values:
            if cv.value == conflict.resolved_value:
                # Increase trust
                self._trust_scores[cv.source_id] = min(1.0,
                                                       self._trust_scores[cv.source_id] * 1.1)
            else:
                # Slight decrease
                self._trust_scores[cv.source_id] = max(0.1,
                                                       self._trust_scores[cv.source_id] * 0.95)

    async def record_user_feedback(self,
                                   conflict_id: str,
                                   chosen_value: Any,
                                   source_id: str):
        """Record user's choice for learning"""
        feedback = {
            "conflict_id": conflict_id,
            "chosen_value": chosen_value,
            "source_id": source_id,
            "timestamp": datetime.now(timezone.utc)
        }

        self._resolution_feedback[conflict_id].append(feedback)

        # Boost trust for chosen source
        self._trust_scores[source_id] = min(1.0, self._trust_scores[source_id] * 1.2)

        logger.info(f"Recorded user feedback for conflict {conflict_id}")


# ===============================
# PHASE 1: MULTI-SOURCE QUERY ENGINE
# ===============================

class MultiSourceQueryEngine:
    """Execute queries across multiple data sources with governance"""

    def __init__(self,
                 access_engine: AccessControlEngine,
                 cache_manager: SmartCacheManager,
                 conflict_engine: ConflictResolutionEngine):
        self.access_engine = access_engine
        self.cache_manager = cache_manager
        self.conflict_engine = conflict_engine
        self._data_sources: Dict[str, SimpleDataSource] = {}
        self._source_connections: Dict[str, Any] = {}

    def register_source(self, source: SimpleDataSource, connection: Any = None):
        """Register a data source"""
        self._data_sources[source.id] = source
        if connection:
            self._source_connections[source.id] = connection
        logger.info(f"Registered data source: {source.id}")

    async def execute_query(self,
                            query: str,
                            user: SimpleUser,
                            context: AccessContext,
                            sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute query across multiple sources with full governance"""

        start_time = datetime.now(timezone.utc)

        # Determine which sources to query
        if not sources:
            sources = list(self._data_sources.keys())

        # Check access for each source
        accessible_sources = []
        access_decisions = {}

        for source_id in sources:
            source = self._data_sources.get(source_id)
            if not source:
                continue

            # Evaluate access
            resource_id = f"source:{source_id}:{source.sensitivity_level.value}"
            evaluation = self.access_engine.evaluate_access(
                user, resource_id, AccessPermission.READ, context
            )

            access_decisions[source_id] = evaluation

            if evaluation.decision in [AccessDecision.ALLOW, AccessDecision.CONDITIONAL]:
                accessible_sources.append(source_id)
                logger.info(f"Access granted to source {source_id} for user {user.id}")
            else:
                logger.warning(f"Access denied to source {source_id} for user {user.id}: {evaluation.reason}")

        if not accessible_sources:
            return {
                "status": "error",
                "error": "No accessible data sources",
                "access_decisions": access_decisions
            }

        # Check cache
        cache_key = self._generate_query_cache_key(query, accessible_sources, user.id)
        cached_result = await self.cache_manager.get(cache_key, "query_results")

        if cached_result:
            logger.info(f"Returning cached result for query")
            return cached_result

        # Execute query on each accessible source
        source_results = {}
        conflicts = []

        for source_id in accessible_sources:
            try:
                # Execute on source (mock for Phase 1)
                result = await self._execute_on_source(source_id, query)

                # Apply access conditions (masking, row limits)
                evaluation = access_decisions[source_id]
                if evaluation.decision == AccessDecision.CONDITIONAL:
                    result = self._apply_access_conditions(result, evaluation)

                source_results[source_id] = result

            except Exception as e:
                logger.error(f"Query failed on source {source_id}: {e}")
                source_results[source_id] = {"error": str(e)}

        # Detect and resolve conflicts
        if len(source_results) > 1:
            conflicts = await self._detect_conflicts(source_results)

            for conflict in conflicts:
                resolved = await self.conflict_engine.resolve_conflict(conflict)
                logger.info(f"Resolved conflict in field '{conflict.field_name}' "
                            f"with {conflict.resolution_strategy.value} strategy")

        # Combine results
        combined_result = await self._combine_results(source_results, conflicts)

        # Cache the result
        await self.cache_manager.set(cache_key, combined_result, "query_results")

        # Add metadata
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        final_result = {
            "status": "success",
            "data": combined_result,
            "metadata": {
                "sources_queried": len(accessible_sources),
                "sources_succeeded": len([r for r in source_results.values() if "error" not in r]),
                "conflicts_detected": len(conflicts),
                "conflicts_resolved": len([c for c in conflicts if c.resolved_value is not None]),
                "execution_time_ms": execution_time,
                "cache_hit": False,
                "access_decisions": {
                    sid: {
                        "decision": dec.decision.value,
                        "conditions": dec.conditions
                    } for sid, dec in access_decisions.items()
                }
            }
        }

        return final_result

    async def _execute_on_source(self, source_id: str, query: str) -> Any:
        """Execute query on a specific source (mock for Phase 1)"""
        # This is where you'd integrate with real data sources
        # For now, return mock data

        mock_data = {
            "postgres_main": pd.DataFrame({
                "customer_id": [1, 2, 3],
                "revenue": [1000, 2000, 1500],
                "status": ["active", "active", "inactive"],
                "updated_at": [datetime.now(timezone.utc) - timedelta(days=1),
                               datetime.now(timezone.utc) - timedelta(days=2),
                               datetime.now(timezone.utc) - timedelta(hours=1)]
            }),
            "mysql_backup": pd.DataFrame({
                "customer_id": [1, 2, 3],
                "revenue": [1100, 1950, 1500],  # Different values
                "status": ["active", "inactive", "inactive"],  # Different status
                "updated_at": [datetime.now(timezone.utc) - timedelta(hours=12),
                               datetime.now(timezone.utc) - timedelta(days=1),
                               datetime.now(timezone.utc) - timedelta(hours=2)]
            }),
            "api_realtime": pd.DataFrame({
                "customer_id": [1, 2],
                "revenue": [1050, 2000],
                "status": ["active", "active"],
                "updated_at": [datetime.now(timezone.utc),
                               datetime.now(timezone.utc)]
            })
        }

        return mock_data.get(source_id, pd.DataFrame())

    async def _detect_conflicts(self, source_results: Dict[str, Any]) -> List[DataConflict]:
        """Detect conflicts between source results"""
        conflicts = []

        # Convert to comparable format
        dfs = {}
        for source_id, result in source_results.items():
            if isinstance(result, pd.DataFrame) and not result.empty:
                dfs[source_id] = result

        if len(dfs) < 2:
            return conflicts

        # Compare values across sources
        # Get common columns
        common_columns = set.intersection(*[set(df.columns) for df in dfs.values()])

        # For each record, check for conflicts
        for idx in range(max(len(df) for df in dfs.values())):
            for col in common_columns:
                if col in ["updated_at", "created_at"]:  # Skip timestamp columns
                    continue

                values_by_source = {}
                for source_id, df in dfs.items():
                    if idx < len(df):
                        values_by_source[source_id] = df.iloc[idx][col]

                # Check if values differ
                unique_values = set(values_by_source.values())
                if len(unique_values) > 1:
                    # Create conflict
                    conflict = DataConflict(
                        id=f"conflict_{col}_{idx}_{datetime.now(timezone.utc).timestamp()}",
                        field_name=col,
                        record_identifier=f"record_{idx}",
                        conflicting_values=[
                            ConflictingValue(
                                source_id=source_id,
                                value=value,
                                timestamp=dfs[source_id].iloc[idx].get("updated_at", datetime.now(timezone.utc)),
                                confidence=self.conflict_engine._trust_scores[source_id]
                            )
                            for source_id, value in values_by_source.items()
                        ]
                    )
                    conflicts.append(conflict)

        return conflicts

    async def _combine_results(self,
                               source_results: Dict[str, Any],
                               resolved_conflicts: List[DataConflict]) -> Any:
        """Combine results from multiple sources"""

        # For Phase 1, simple combination strategy:
        # Use resolved values where conflicts exist, otherwise take most recent

        combined_df = pd.DataFrame()

        # Get all dataframes
        dfs = [(sid, res) for sid, res in source_results.items()
               if isinstance(res, pd.DataFrame) and not res.empty]

        if not dfs:
            return pd.DataFrame()

        # Start with first dataframe
        combined_df = dfs[0][1].copy()

        # Apply conflict resolutions
        for conflict in resolved_conflicts:
            if conflict.resolved_value is not None:
                # Parse record identifier
                try:
                    idx = int(conflict.record_identifier.split("_")[1])
                    if idx < len(combined_df):
                        combined_df.at[idx, conflict.field_name] = conflict.resolved_value
                except:
                    pass

        return combined_df

    def _apply_access_conditions(self, data: Any, evaluation: AccessEvaluation) -> Any:
        """Apply access conditions like masking and row limits"""

        if not isinstance(data, pd.DataFrame):
            return data

        df = data.copy()

        # Apply field masking
        for field in evaluation.masked_fields:
            if field in df.columns:
                df[field] = "***MASKED***"

        # Apply row limit
        if evaluation.row_limit and len(df) > evaluation.row_limit:
            df = df.head(evaluation.row_limit)

        return df

    def _generate_query_cache_key(self, query: str, sources: List[str], user_id: str) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query,
            "sources": sorted(sources),
            "user_id": user_id
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about data sources and usage"""

        cache_stats = await self.cache_manager.get_stats()

        source_stats = {}
        for source_id, source in self._data_sources.items():
            source_stats[source_id] = {
                "name": source.name,
                "type": source.source_type,
                "sensitivity": source.sensitivity_level.value,
                "active": source.is_active,
                "authority_score": self.conflict_engine._source_authority_scores.get(source_id, 0.5),
                "trust_score": self.conflict_engine._trust_scores[source_id]
            }

        return {
            "total_sources": len(self._data_sources),
            "active_sources": len([s for s in self._data_sources.values() if s.is_active]),
            "source_details": source_stats,
            "cache_statistics": cache_stats,
            "conflict_resolution_strategies": {
                strategy.value: strategy.value for strategy in ConflictResolutionStrategy
            }
        }


# ===============================
# PHASE 1: SYSTEM ORCHESTRATOR
# ===============================

class Phase1DataGenieSystem:
    """Complete Phase 1 Multi-Source DataGenie System"""

    def __init__(self):
        # Initialize all components
        self.access_engine = AccessControlEngine()
        self.cache_manager = SmartCacheManager()
        self.conflict_engine = ConflictResolutionEngine()
        self.query_engine = MultiSourceQueryEngine(
            self.access_engine,
            self.cache_manager,
            self.conflict_engine
        )

        # Setup default data sources
        self._setup_default_sources()

        logger.info("🚀 Phase 1 Multi-Source DataGenie System initialized")

    def _setup_default_sources(self):
        """Setup default data sources for demo"""

        sources = [
            SimpleDataSource(
                id="postgres_main",
                name="Main PostgreSQL Database",
                source_type="postgresql",
                sensitivity_level=DataSensitivityLevel.INTERNAL,
                owner="data_team"
            ),
            SimpleDataSource(
                id="mysql_backup",
                name="MySQL Backup Database",
                source_type="mysql",
                sensitivity_level=DataSensitivityLevel.INTERNAL,
                owner="data_team"
            ),
            SimpleDataSource(
                id="api_realtime",
                name="Real-time API",
                source_type="rest_api",
                sensitivity_level=DataSensitivityLevel.PUBLIC,
                owner="api_team"
            )
        ]

        # Register sources
        for source in sources:
            self.query_engine.register_source(source)

        # Set authority scores
        self.conflict_engine.set_source_authority("postgres_main", 0.9)
        self.conflict_engine.set_source_authority("mysql_backup", 0.7)
        self.conflict_engine.set_source_authority("api_realtime", 0.8)

        logger.info(f"Registered {len(sources)} default data sources")

    async def analyze_with_governance(self,
                                      query: str,
                                      user_id: str,
                                      user_role: str = "analyst",
                                      purpose: str = "analysis") -> Dict[str, Any]:
        """Main entry point for governed multi-source analysis"""

        # Create user
        user = SimpleUser(
            id=user_id,
            username=user_id,
            email=f"{user_id}@example.com",
            role=user_role
        )

        # Assign role
        self.access_engine.assign_role_to_user(user_id, user_role)

        # Create context
        context = AccessContext(
            user_id=user_id,
            purpose=purpose
        )

        # Execute query
        result = await self.query_engine.execute_query(query, user, context)

        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""

        source_stats = await self.query_engine.get_source_statistics()

        return {
            "system": "Phase 1 Multi-Source DataGenie",
            "version": "1.0.0",
            "components": {
                "access_control": "operational",
                "smart_cache": "operational",
                "conflict_resolution": "operational",
                "multi_source_query": "operational"
            },
            "features": {
                "rbac_abac": True,
                "intelligent_caching": True,
                "conflict_detection": True,
                "multi_source_join": True,
                "audit_logging": True,
                "learning_engine": True
            },
            "statistics": source_stats
        }


# ===============================
# PHASE 1: FASTAPI INTEGRATION
# ===============================

# Initialize Phase 1 system
phase1_system = Phase1DataGenieSystem()

# Create security
security = HTTPBearer()


# Add these endpoints to your existing FastAPI app

@app.post("/phase1/analyze")
async def phase1_multi_source_analyze(
        query: str = Form(...),
        purpose: str = Form("analysis"),
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Phase 1: Multi-source analysis with full governance"""

    # Extract user info from token (simplified for demo)
    user_id = f"user_{hash(credentials.credentials) % 1000}"
    user_role = "analyst"  # In production, extract from JWT

    try:
        result = await phase1_system.analyze_with_governance(
            query=query,
            user_id=user_id,
            user_role=user_role,
            purpose=purpose
        )

        return result

    except Exception as e:
        logger.error(f"Phase 1 analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/phase1/sources")
async def get_data_sources(
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get available data sources with access info"""

    user_id = f"user_{hash(credentials.credentials) % 1000}"

    # Get sources and check access
    sources = []
    for source_id, source in phase1_system.query_engine._data_sources.items():
        # Check access
        user = SimpleUser(id=user_id, username=user_id, email=f"{user_id}@example.com", role="analyst")
        context = AccessContext(user_id=user_id)
        resource_id = f"source:{source_id}:{source.sensitivity_level.value}"

        evaluation = phase1_system.access_engine.evaluate_access(
            user, resource_id, AccessPermission.READ, context
        )

        sources.append({
            "id": source.id,
            "name": source.name,
            "type": source.source_type,
            "sensitivity": source.sensitivity_level.value,
            "access": {
                "granted": evaluation.decision.value != "deny",
                "decision": evaluation.decision.value,
                "conditions": evaluation.conditions
            }
        })

    return {"sources": sources}


@app.post("/phase1/conflicts/feedback")
async def record_conflict_feedback(
        conflict_id: str = Form(...),
        chosen_value: str = Form(...),
        source_id: str = Form(...),
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Record user feedback on conflict resolution"""

    try:
        await phase1_system.conflict_engine.record_user_feedback(
            conflict_id=conflict_id,
            chosen_value=chosen_value,
            source_id=source_id
        )

        return {"status": "success", "message": "Feedback recorded"}

    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/phase1/status")
async def get_phase1_status():
    """Get Phase 1 system status"""

    status = await phase1_system.get_system_status()
    return status


@app.get("/phase1/cache/stats")
async def get_cache_statistics(
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get cache statistics"""

    stats = await phase1_system.cache_manager.get_stats()
    return stats


@app.post("/phase1/cache/invalidate")
async def invalidate_cache(
        pattern: str = Form(...),
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Invalidate cache entries matching pattern"""

    # Check admin access
    user_id = f"user_{hash(credentials.credentials) % 1000}"
    if user_id not in phase1_system.access_engine._user_roles or \
            "admin" not in phase1_system.access_engine._user_roles[user_id]:
        raise HTTPException(status_code=403, detail="Admin access required")

    count = await phase1_system.cache_manager.invalidate_pattern(pattern)
    return {"invalidated": count}


@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "system": "DataGenie Phase 1 - Multi-Source Analytics",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/phase1/analyze",
            "sources": "/phase1/sources",
            "status": "/phase1/status",
            "cache_stats": "/phase1/cache/stats"
        }
    }


# ===============================
# PHASE 1: EXAMPLE USAGE
# ===============================

async def demo_phase1_usage():
    """Demo of Phase 1 capabilities"""

    # Example 1: Analyst querying data
    result1 = await phase1_system.analyze_with_governance(
        query="SELECT revenue FROM customers WHERE status='active'",
        user_id="john_analyst",
        user_role="analyst",
        purpose="quarterly_report"
    )
    print("Analyst Query Result:", result1["metadata"])

    # Example 2: Viewer with limited access
    result2 = await phase1_system.analyze_with_governance(
        query="SELECT * FROM customers",
        user_id="jane_viewer",
        user_role="viewer",
        purpose="customer_review"
    )
    print("Viewer Query Result:", result2["metadata"])

    # Example 3: Admin with full access
    result3 = await phase1_system.analyze_with_governance(
        query="SELECT * FROM sensitive_financial_data",
        user_id="admin_user",
        user_role="admin",
        purpose="audit"
    )
    print("Admin Query Result:", result3["metadata"])


if __name__ == "__main__":
    # Run standalone
    print("🚀 Starting DataGenie Phase 1 Multi-Source Analytics System")

    # Run the demo
    asyncio.run(demo_phase1_usage())

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

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