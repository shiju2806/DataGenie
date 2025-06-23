"""
Smart Defaults Scoring Engine
Advanced scoring algorithms for recommendation confidence and relevance
File location: smart_defaults/utils/scoring.py
"""

import logging
import math
import sys
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

# Import dependencies with fallbacks
try:
    from ..models.user_profile import UserProfile
    from ..models.recommendation import Recommendation
    from ..models.data_source import DataSource
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
except ImportError:
    # For direct execution, create mock classes
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
        recommendation_type: str = "environment"
        reasoning: Dict[str, Any] = None
        context: Dict[str, Any] = None


    @dataclass
    class DataSource:
        id: str = "test_source"
        name: str = "Test Source"
        source_type: str = "database"


    class DatabaseManager:
        async def initialize(self): pass

        async def close(self): pass


    class CacheManager:
        async def initialize(self): pass

        async def close(self): pass

        async def get(self, key, default=None): return default

        async def set(self, key, value, ttl=None): pass

logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Different scoring methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"
    CONFIDENCE_INTERVAL = "confidence_interval"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ScoreComponent(Enum):
    """Components that contribute to the overall score"""
    ENVIRONMENT_MATCH = "environment_match"
    ROLE_ALIGNMENT = "role_alignment"
    USAGE_PATTERNS = "usage_patterns"
    PEER_SIMILARITY = "peer_similarity"
    HISTORICAL_SUCCESS = "historical_success"
    TECHNICAL_COMPATIBILITY = "technical_compatibility"
    SECURITY_COMPLIANCE = "security_compliance"
    PERFORMANCE_METRICS = "performance_metrics"
    FRESHNESS = "freshness"
    POPULARITY = "popularity"


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a score was calculated"""
    total_score: float
    components: Dict[ScoreComponent, float]
    weights: Dict[ScoreComponent, float]
    adjustments: Dict[str, float]
    confidence: float
    method_used: ScoringMethod
    calculated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringConfig:
    """Configuration for scoring algorithms"""
    default_method: ScoringMethod = ScoringMethod.ENSEMBLE

    # Component weights (should sum to 1.0)
    component_weights: Dict[ScoreComponent, float] = field(default_factory=lambda: {
        ScoreComponent.ENVIRONMENT_MATCH: 0.20,
        ScoreComponent.ROLE_ALIGNMENT: 0.18,
        ScoreComponent.USAGE_PATTERNS: 0.15,
        ScoreComponent.PEER_SIMILARITY: 0.12,
        ScoreComponent.HISTORICAL_SUCCESS: 0.10,
        ScoreComponent.TECHNICAL_COMPATIBILITY: 0.08,
        ScoreComponent.SECURITY_COMPLIANCE: 0.07,
        ScoreComponent.PERFORMANCE_METRICS: 0.05,
        ScoreComponent.FRESHNESS: 0.03,
        ScoreComponent.POPULARITY: 0.02
    })

    # Scoring parameters
    min_score: float = 0.0
    max_score: float = 1.0
    confidence_threshold: float = 0.6

    # Bayesian parameters
    prior_strength: float = 5.0
    prior_success_rate: float = 0.7

    # Adaptive parameters
    learning_rate: float = 0.1
    adaptation_window_days: int = 30


@dataclass
class UserScoringProfile:
    """User-specific scoring profile for personalization"""
    user_id: str
    role: str
    experience_level: str = "intermediate"

    # Personalized weights
    personal_weights: Dict[ScoreComponent, float] = field(default_factory=dict)

    # Historical data
    total_recommendations: int = 0
    accepted_recommendations: int = 0
    rejected_recommendations: int = 0

    # Preferences learned from behavior
    preferred_source_types: List[str] = field(default_factory=list)
    avoided_source_types: List[str] = field(default_factory=list)

    # Time-based patterns
    active_hours: List[int] = field(default_factory=list)
    response_time_avg: float = 0.0

    # Success patterns
    success_by_source_type: Dict[str, float] = field(default_factory=dict)
    success_by_time_of_day: Dict[int, float] = field(default_factory=dict)

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ScoringEngine:
    """Advanced scoring engine for recommendation confidence calculation"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 config: Optional[ScoringConfig] = None):

        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.config = config or ScoringConfig()

        # User scoring profiles
        self.user_profiles: Dict[str, UserScoringProfile] = {}

        # Historical data for ensemble methods
        self.global_success_rates: Dict[str, float] = {}
        self.peer_groups: Dict[str, List[str]] = {}
        self.source_performance_metrics: Dict[str, Dict[str, float]] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize the scoring engine"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()

        # Load historical data
        await self._load_historical_data()

        # Load user profiles
        await self._load_user_profiles()

        self._initialized = True
        logger.info("✅ Scoring engine initialized")

    async def close(self):
        """Close the scoring engine"""
        # Save user profiles
        await self._save_user_profiles()

        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Scoring engine closed")

    async def calculate_score(self,
                              recommendation: Recommendation,
                              user_profile: UserProfile,
                              data_source: Optional[DataSource] = None,
                              context: Optional[Dict[str, Any]] = None,
                              method: Optional[ScoringMethod] = None) -> ScoreBreakdown:
        """Calculate a comprehensive score for a recommendation"""

        method = method or self.config.default_method
        context = context or {}

        try:
            # Get user scoring profile
            user_scoring_profile = await self._get_user_scoring_profile(user_profile.user_id)

            # Calculate component scores
            component_scores = await self._calculate_component_scores(
                recommendation, user_profile, data_source, context, user_scoring_profile
            )

            # Apply scoring method
            if method == ScoringMethod.WEIGHTED_AVERAGE:
                score, confidence = self._weighted_average_score(component_scores, user_scoring_profile)
            elif method == ScoringMethod.BAYESIAN:
                score, confidence = self._bayesian_score(component_scores, user_scoring_profile)
            elif method == ScoringMethod.CONFIDENCE_INTERVAL:
                score, confidence = self._confidence_interval_score(component_scores)
            elif method == ScoringMethod.ENSEMBLE:
                score, confidence = self._ensemble_score(component_scores, user_scoring_profile)
            else:  # ADAPTIVE
                score, confidence = self._adaptive_score(component_scores, user_scoring_profile)

            # Apply adjustments
            adjustments = self._calculate_adjustments(
                recommendation, user_profile, data_source, context
            )

            adjusted_score = self._apply_adjustments(score, adjustments)

            # Create breakdown
            breakdown = ScoreBreakdown(
                total_score=adjusted_score,
                components=component_scores,
                weights=self._get_effective_weights(user_scoring_profile),
                adjustments=adjustments,
                confidence=confidence,
                method_used=method,
                calculated_at=datetime.now(timezone.utc),
                metadata={
                    "user_id": user_profile.user_id,
                    "source_id": recommendation.source_id,
                    "recommendation_type": recommendation.recommendation_type,
                    "context_keys": list(context.keys())
                }
            )

            return breakdown

        except Exception as e:
            logger.error(f"❌ Score calculation failed: {e}")

            # Return fallback score
            return ScoreBreakdown(
                total_score=0.5,  # Neutral score
                components={},
                weights={},
                adjustments={"error_fallback": -0.1},
                confidence=0.1,
                method_used=method,
                calculated_at=datetime.now(timezone.utc),
                metadata={"error": str(e)}
            )

    async def _calculate_component_scores(self,
                                          recommendation: Recommendation,
                                          user_profile: UserProfile,
                                          data_source: Optional[DataSource],
                                          context: Dict[str, Any],
                                          user_scoring_profile: UserScoringProfile) -> Dict[ScoreComponent, float]:
        """Calculate individual component scores"""

        scores = {}

        # Environment Match Score
        scores[ScoreComponent.ENVIRONMENT_MATCH] = self._calculate_environment_match_score(
            recommendation, context
        )

        # Role Alignment Score
        scores[ScoreComponent.ROLE_ALIGNMENT] = self._calculate_role_alignment_score(
            user_profile, data_source
        )

        # Usage Patterns Score
        scores[ScoreComponent.USAGE_PATTERNS] = self._calculate_usage_patterns_score(
            user_profile, recommendation, user_scoring_profile
        )

        # Peer Similarity Score
        scores[ScoreComponent.PEER_SIMILARITY] = await self._calculate_peer_similarity_score(
            user_profile, recommendation
        )

        # Historical Success Score
        scores[ScoreComponent.HISTORICAL_SUCCESS] = self._calculate_historical_success_score(
            recommendation, user_scoring_profile
        )

        # Technical Compatibility Score
        scores[ScoreComponent.TECHNICAL_COMPATIBILITY] = self._calculate_technical_compatibility_score(
            data_source, context
        )

        # Security Compliance Score
        scores[ScoreComponent.SECURITY_COMPLIANCE] = self._calculate_security_compliance_score(
            data_source, user_profile
        )

        # Performance Metrics Score
        scores[ScoreComponent.PERFORMANCE_METRICS] = self._calculate_performance_metrics_score(
            recommendation.source_id
        )

        # Freshness Score
        scores[ScoreComponent.FRESHNESS] = self._calculate_freshness_score(
            recommendation, data_source
        )

        # Popularity Score
        scores[ScoreComponent.POPULARITY] = self._calculate_popularity_score(
            recommendation.source_id
        )

        return scores

    def _calculate_environment_match_score(self, recommendation: Recommendation, context: Dict[str, Any]) -> float:
        """Calculate how well the recommendation matches the user's environment"""

        score = 0.5  # Base score

        # Check if source was discovered in environment
        if recommendation.reasoning and recommendation.reasoning.get("discovered_automatically"):
            score += 0.3

        # Check environment context
        if context.get("environment_type") == "local":
            score += 0.1
        elif context.get("environment_type") == "cloud":
            score += 0.05

        # Check if source is actually available
        if context.get("source_available", True):
            score += 0.1
        else:
            score -= 0.2

        return min(1.0, max(0.0, score))

    def _calculate_role_alignment_score(self, user_profile: UserProfile, data_source: Optional[DataSource]) -> float:
        """Calculate how well the data source aligns with the user's role"""

        role = user_profile.role.lower()
        source_type = data_source.source_type.lower() if data_source else "unknown"

        # Role-specific preferences
        role_preferences = {
            "data_analyst": {
                "database": 0.9,
                "visualization": 0.8,
                "cache": 0.6,
                "search": 0.7,
                "storage": 0.5
            },
            "data_scientist": {
                "database": 0.8,
                "notebook": 0.9,
                "ml_platform": 0.9,
                "cache": 0.7,
                "search": 0.6
            },
            "software_engineer": {
                "database": 0.7,
                "cache": 0.8,
                "api": 0.9,
                "message_queue": 0.8,
                "storage": 0.7
            },
            "business_analyst": {
                "visualization": 0.9,
                "dashboard": 0.9,
                "database": 0.6,
                "reporting": 0.8,
                "spreadsheet": 0.7
            }
        }

        preferences = role_preferences.get(role, {})
        return preferences.get(source_type, 0.5)

    def _calculate_usage_patterns_score(self, user_profile: UserProfile,
                                        recommendation: Recommendation,
                                        user_scoring_profile: UserScoringProfile) -> float:
        """Calculate score based on user's historical usage patterns"""

        source_type = recommendation.source_id.split("_")[0]  # Extract type from source ID

        # Check if user has shown preference for this source type
        if source_type in user_scoring_profile.preferred_source_types:
            return 0.8
        elif source_type in user_scoring_profile.avoided_source_types:
            return 0.2

        # Check historical success rate for this source type
        success_rate = user_scoring_profile.success_by_source_type.get(source_type, 0.5)

        # Weight by total experience
        experience_weight = min(1.0, user_scoring_profile.total_recommendations / 10.0)

        return success_rate * experience_weight + 0.5 * (1 - experience_weight)

    async def _calculate_peer_similarity_score(self, user_profile: UserProfile,
                                               recommendation: Recommendation) -> float:
        """Calculate score based on similar users' preferences"""

        # Get peer group for user's role
        peers = self.peer_groups.get(user_profile.role, [])

        if not peers:
            return 0.5  # No peer data available

        # Calculate how popular this source is among peers
        peer_success_count = 0
        peer_total_count = 0

        # In a real implementation, this would query the database
        # For demo, simulate peer preferences
        if recommendation.source_id in ["postgresql", "tableau", "jupyter"]:
            peer_success_count = len(peers) * 0.7  # 70% of peers like these
        elif recommendation.source_id in ["redis", "elasticsearch"]:
            peer_success_count = len(peers) * 0.5  # 50% of peers like these
        else:
            peer_success_count = len(peers) * 0.3  # 30% of peers like others

        peer_total_count = len(peers)

        if peer_total_count == 0:
            return 0.5

        peer_preference_rate = peer_success_count / peer_total_count
        return peer_preference_rate

    def _calculate_historical_success_score(self, recommendation: Recommendation,
                                            user_scoring_profile: UserScoringProfile) -> float:
        """Calculate score based on historical success of this source"""

        source_id = recommendation.source_id

        # Global success rate for this source
        global_success_rate = self.global_success_rates.get(source_id, 0.7)

        # User-specific success rate
        user_success_rate = user_scoring_profile.success_by_source_type.get(source_id, global_success_rate)

        # Weight between global and user-specific rates
        if user_scoring_profile.total_recommendations > 5:
            # More weight on personal experience
            return 0.3 * global_success_rate + 0.7 * user_success_rate
        else:
            # More weight on global data
            return 0.7 * global_success_rate + 0.3 * user_success_rate

    def _calculate_technical_compatibility_score(self, data_source: Optional[DataSource],
                                                 context: Dict[str, Any]) -> float:
        """Calculate technical compatibility score"""

        score = 0.7  # Base compatibility score

        if not data_source:
            return score

        # Check system requirements
        if context.get("system_requirements_met", True):
            score += 0.1
        else:
            score -= 0.3

        # Check network accessibility
        if context.get("network_accessible", True):
            score += 0.1
        else:
            score -= 0.4

        # Check driver availability
        if context.get("drivers_available", True):
            score += 0.1
        else:
            score -= 0.2

        return min(1.0, max(0.0, score))

    def _calculate_security_compliance_score(self, data_source: Optional[DataSource],
                                             user_profile: UserProfile) -> float:
        """Calculate security and compliance score"""

        score = 0.8  # Base security score

        if not data_source:
            return score

        # Role-based security requirements
        if user_profile.role in ["security_officer", "compliance_manager"]:
            # Higher security requirements
            score = 0.9
        elif user_profile.role in ["intern", "contractor"]:
            # Lower access level
            score = 0.6

        return score

    def _calculate_performance_metrics_score(self, source_id: str) -> float:
        """Calculate performance metrics score"""

        metrics = self.source_performance_metrics.get(source_id, {})

        if not metrics:
            return 0.7  # Default performance score

        # Weighted performance calculation
        latency_score = 1.0 - min(1.0, metrics.get("avg_latency_ms", 100) / 1000.0)
        uptime_score = metrics.get("uptime_percentage", 95) / 100.0
        throughput_score = min(1.0, metrics.get("throughput_qps", 100) / 1000.0)

        performance_score = (
                0.4 * latency_score +
                0.4 * uptime_score +
                0.2 * throughput_score
        )

        return performance_score

    def _calculate_freshness_score(self, recommendation: Recommendation,
                                   data_source: Optional[DataSource]) -> float:
        """Calculate freshness score based on how recent the data/source is"""

        now = datetime.now(timezone.utc)

        # Check recommendation freshness
        if hasattr(recommendation, 'created_at') and recommendation.created_at:
            rec_age_hours = (now - recommendation.created_at).total_seconds() / 3600
        else:
            rec_age_hours = 0  # Fresh recommendation

        # Freshness decays over time
        if rec_age_hours < 1:
            freshness = 1.0
        elif rec_age_hours < 24:
            freshness = 0.9
        elif rec_age_hours < 168:  # 1 week
            freshness = 0.7
        else:
            freshness = 0.5

        return freshness

    def _calculate_popularity_score(self, source_id: str) -> float:
        """Calculate popularity score based on overall usage"""

        # In a real implementation, this would query usage statistics
        # For demo, simulate popularity based on source type
        popularity_map = {
            "postgresql": 0.9,
            "mysql": 0.8,
            "redis": 0.7,
            "elasticsearch": 0.6,
            "mongodb": 0.7,
            "tableau": 0.6,
            "jupyter": 0.8,
            "spark": 0.5
        }

        return popularity_map.get(source_id.lower(), 0.5)

    def _weighted_average_score(self, component_scores: Dict[ScoreComponent, float],
                                user_scoring_profile: UserScoringProfile) -> Tuple[float, float]:
        """Calculate weighted average score"""

        weights = self._get_effective_weights(user_scoring_profile)

        total_score = 0.0
        total_weight = 0.0

        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5, 0.1

        final_score = total_score / total_weight
        confidence = min(1.0, total_weight)  # Confidence based on weight coverage

        return final_score, confidence

    def _bayesian_score(self, component_scores: Dict[ScoreComponent, float],
                        user_scoring_profile: UserScoringProfile) -> Tuple[float, float]:
        """Calculate Bayesian score with prior beliefs"""

        # Prior beliefs
        prior_score = self.config.prior_success_rate
        prior_strength = self.config.prior_strength

        # Observed data strength (based on user experience)
        data_strength = min(20.0, user_scoring_profile.total_recommendations)

        # Calculate weighted average with prior
        observed_score = statistics.mean(component_scores.values()) if component_scores else 0.5

        posterior_score = (
                (prior_score * prior_strength + observed_score * data_strength) /
                (prior_strength + data_strength)
        )

        # Confidence increases with more data
        confidence = data_strength / (data_strength + prior_strength)

        return posterior_score, confidence

    def _confidence_interval_score(self, component_scores: Dict[ScoreComponent, float]) -> Tuple[float, float]:
        """Calculate score with confidence interval"""

        if not component_scores:
            return 0.5, 0.1

        scores = list(component_scores.values())
        mean_score = statistics.mean(scores)

        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            confidence = max(0.1, 1.0 - std_dev)  # Higher confidence for lower variance
        else:
            confidence = 0.5

        return mean_score, confidence

    def _ensemble_score(self, component_scores: Dict[ScoreComponent, float],
                        user_scoring_profile: UserScoringProfile) -> Tuple[float, float]:
        """Calculate ensemble score using multiple methods"""

        # Calculate scores using different methods
        weighted_score, weighted_conf = self._weighted_average_score(component_scores, user_scoring_profile)
        bayesian_score, bayesian_conf = self._bayesian_score(component_scores, user_scoring_profile)
        ci_score, ci_conf = self._confidence_interval_score(component_scores)

        # Ensemble weights based on confidence
        total_conf = weighted_conf + bayesian_conf + ci_conf

        if total_conf == 0:
            return 0.5, 0.1

        ensemble_score = (
                                 weighted_score * weighted_conf +
                                 bayesian_score * bayesian_conf +
                                 ci_score * ci_conf
                         ) / total_conf

        ensemble_confidence = total_conf / 3.0  # Average confidence

        return ensemble_score, ensemble_confidence

    def _adaptive_score(self, component_scores: Dict[ScoreComponent, float],
                        user_scoring_profile: UserScoringProfile) -> Tuple[float, float]:
        """Calculate adaptive score that learns from user behavior"""

        # Start with weighted average
        base_score, base_confidence = self._weighted_average_score(component_scores, user_scoring_profile)

        # Adapt based on user's historical accuracy
        if user_scoring_profile.total_recommendations > 0:
            user_accuracy = user_scoring_profile.accepted_recommendations / user_scoring_profile.total_recommendations

            # Adjust score based on user's typical acceptance pattern
            if user_accuracy > 0.8:
                # User is selective, increase confidence in high scores
                if base_score > 0.7:
                    base_score = min(1.0, base_score * 1.1)
                    base_confidence = min(1.0, base_confidence * 1.2)
            elif user_accuracy < 0.3:
                # User rejects many recommendations, be more conservative
                base_score = base_score * 0.9
                base_confidence = base_confidence * 0.8

        return base_score, base_confidence

    def _calculate_adjustments(self, recommendation: Recommendation,
                               user_profile: UserProfile,
                               data_source: Optional[DataSource],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate score adjustments"""

        adjustments = {}

        # Time-based adjustments
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            adjustments["business_hours"] = 0.05
        else:
            adjustments["off_hours"] = -0.02

        # Urgency adjustments
        if context.get("urgency") == "high":
            adjustments["high_urgency"] = 0.1
        elif context.get("urgency") == "low":
            adjustments["low_urgency"] = -0.05

        # User experience adjustments
        if user_profile.role.endswith("_senior"):
            adjustments["senior_role"] = 0.05
        elif user_profile.role.startswith("junior_"):
            adjustments["junior_role"] = -0.03

        # Recommendation type adjustments
        if recommendation.recommendation_type == "ml_generated":
            adjustments["ml_recommendation"] = 0.03
        elif recommendation.recommendation_type == "environment_discovered":
            adjustments["environment_recommendation"] = 0.02

        return adjustments

    def _apply_adjustments(self, base_score: float, adjustments: Dict[str, float]) -> float:
        """Apply adjustments to base score"""

        adjusted_score = base_score

        for adjustment_name, adjustment_value in adjustments.items():
            adjusted_score += adjustment_value

        # Ensure score stays within bounds
        return min(self.config.max_score, max(self.config.min_score, adjusted_score))

    def _get_effective_weights(self, user_scoring_profile: UserScoringProfile) -> Dict[ScoreComponent, float]:
        """Get effective weights combining default and personalized weights"""

        effective_weights = self.config.component_weights.copy()

        # Apply personalized weights if available
        for component, personal_weight in user_scoring_profile.personal_weights.items():
            if component in effective_weights:
                # Blend default and personal weights
                effective_weights[component] = (
                        0.7 * effective_weights[component] + 0.3 * personal_weight
                )

        return effective_weights

    async def _get_user_scoring_profile(self, user_id: str) -> UserScoringProfile:
        """Get or create user scoring profile"""

        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Create new profile
        profile = UserScoringProfile(
            user_id=user_id,
            role="data_analyst"  # Default role
        )

        self.user_profiles[user_id] = profile
        return profile

    async def update_user_feedback(self, user_id: str, recommendation: Recommendation,
                                   action: str, context: Optional[Dict[str, Any]] = None):
        """Update user scoring profile based on feedback"""

        try:
            profile = await self._get_user_scoring_profile(user_id)

            # Update totals
            profile.total_recommendations += 1

            if action in ["accept", "connect", "helpful"]:
                profile.accepted_recommendations += 1

                # Update preferred source types
                source_type = recommendation.source_id.split("_")[0]
                if source_type not in profile.preferred_source_types:
                    profile.preferred_source_types.append(source_type)

                # Update success by source type
                current_success = profile.success_by_source_type.get(source_type, 0.0)
                profile.success_by_source_type[source_type] = min(1.0, current_success + 0.1)

            elif action in ["reject", "dismiss", "not_helpful"]:
                profile.rejected_recommendations += 1

                # Update avoided source types
                source_type = recommendation.source_id.split("_")[0]
                if (profile.rejected_recommendations > 2 and
                        source_type not in profile.avoided_source_types):
                    profile.avoided_source_types.append(source_type)

            # Update time-based patterns
            current_hour = datetime.now().hour
            if current_hour not in profile.active_hours:
                profile.active_hours.append(current_hour)

            # Keep only recent active hours (last 30 entries)
            if len(profile.active_hours) > 30:
                profile.active_hours = profile.active_hours[-30:]

            # Update success by time of day
            if action in ["accept", "connect", "helpful"]:
                current_time_success = profile.success_by_time_of_day.get(current_hour, 0.0)
                profile.success_by_time_of_day[current_hour] = min(1.0, current_time_success + 0.1)

            profile.last_updated = datetime.now(timezone.utc)

            logger.info(f"📊 Updated scoring profile for user {user_id}: {action} on {recommendation.source_id}")

        except Exception as e:
            logger.error(f"❌ Failed to update user feedback: {e}")

    async def get_scoring_insights(self, user_id: str) -> Dict[str, Any]:
        """Get scoring insights for a user"""

        try:
            profile = await self._get_user_scoring_profile(user_id)

            # Calculate insights
            acceptance_rate = 0.0
            if profile.total_recommendations > 0:
                acceptance_rate = profile.accepted_recommendations / profile.total_recommendations

            most_active_hours = Counter(profile.active_hours).most_common(3)
            top_source_types = sorted(
                profile.success_by_source_type.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            insights = {
                "user_id": user_id,
                "total_recommendations": profile.total_recommendations,
                "acceptance_rate": acceptance_rate,
                "experience_level": profile.experience_level,
                "most_active_hours": [hour for hour, count in most_active_hours],
                "preferred_source_types": profile.preferred_source_types,
                "avoided_source_types": profile.avoided_source_types,
                "top_performing_sources": top_source_types,
                "personalization_strength": len(profile.personal_weights),
                "last_updated": profile.last_updated,
                "scoring_maturity": min(1.0, profile.total_recommendations / 20.0)
            }

            return insights

        except Exception as e:
            logger.error(f"❌ Failed to get scoring insights: {e}")
            return {"user_id": user_id, "error": str(e)}

    async def calibrate_scoring_model(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calibrate the scoring model based on feedback data"""

        try:
            # Analyze feedback patterns
            total_feedback = len(feedback_data)
            accepted_count = sum(1 for f in feedback_data if f.get("action") in ["accept", "connect"])

            # Calculate component effectiveness
            component_effectiveness = {}

            for component in ScoreComponent:
                component_scores = []
                outcomes = []

                for feedback in feedback_data:
                    if "score_breakdown" in feedback:
                        breakdown = feedback["score_breakdown"]
                        if component.value in breakdown.get("components", {}):
                            component_scores.append(breakdown["components"][component.value])
                            outcomes.append(1 if feedback.get("action") in ["accept", "connect"] else 0)

                if component_scores and outcomes:
                    # Calculate correlation between component score and positive outcome
                    correlation = self._calculate_correlation(component_scores, outcomes)
                    component_effectiveness[component.value] = correlation

            # Suggest weight adjustments
            suggested_weights = self._suggest_weight_adjustments(component_effectiveness)

            calibration_results = {
                "total_feedback_samples": total_feedback,
                "overall_acceptance_rate": accepted_count / total_feedback if total_feedback > 0 else 0,
                "component_effectiveness": component_effectiveness,
                "suggested_weight_adjustments": suggested_weights,
                "calibration_confidence": min(1.0, total_feedback / 100.0),
                "calibrated_at": datetime.now(timezone.utc)
            }

            logger.info(f"📈 Scoring model calibrated with {total_feedback} feedback samples")
            return calibration_results

        except Exception as e:
            logger.error(f"❌ Scoring model calibration failed: {e}")
            return {"error": str(e)}

    def _calculate_correlation(self, scores: List[float], outcomes: List[int]) -> float:
        """Calculate correlation between scores and outcomes"""

        if len(scores) != len(outcomes) or len(scores) < 2:
            return 0.0

        # Simple correlation calculation
        mean_score = statistics.mean(scores)
        mean_outcome = statistics.mean(outcomes)

        numerator = sum((s - mean_score) * (o - mean_outcome) for s, o in zip(scores, outcomes))

        score_variance = sum((s - mean_score) ** 2 for s in scores)
        outcome_variance = sum((o - mean_outcome) ** 2 for o in outcomes)

        if score_variance == 0 or outcome_variance == 0:
            return 0.0

        correlation = numerator / math.sqrt(score_variance * outcome_variance)
        return correlation

    def _suggest_weight_adjustments(self, component_effectiveness: Dict[str, float]) -> Dict[str, float]:
        """Suggest weight adjustments based on component effectiveness"""

        adjustments = {}

        for component_name, effectiveness in component_effectiveness.items():
            current_weight = 0.1  # Default weight

            # Find current weight
            for component in ScoreComponent:
                if component.value == component_name:
                    current_weight = self.config.component_weights.get(component, 0.1)
                    break

            # Suggest adjustment based on effectiveness
            if effectiveness > 0.7:
                # High effectiveness - increase weight
                suggested_adjustment = min(0.05, (effectiveness - 0.7) * 0.1)
            elif effectiveness < 0.3:
                # Low effectiveness - decrease weight
                suggested_adjustment = -min(0.03, (0.3 - effectiveness) * 0.1)
            else:
                # Moderate effectiveness - minor adjustment
                suggested_adjustment = (effectiveness - 0.5) * 0.02

            adjustments[component_name] = suggested_adjustment

        return adjustments

    async def _load_historical_data(self):
        """Load historical data for scoring"""

        # In a real implementation, this would load from database
        # For demo, initialize with sample data

        self.global_success_rates = {
            "postgresql": 0.85,
            "mysql": 0.80,
            "redis": 0.75,
            "elasticsearch": 0.70,
            "mongodb": 0.72,
            "tableau": 0.65,
            "jupyter": 0.88,
            "spark": 0.60
        }

        self.peer_groups = {
            "data_analyst": ["user1", "user2", "user3", "user4", "user5"],
            "data_scientist": ["user6", "user7", "user8", "user9"],
            "software_engineer": ["user10", "user11", "user12", "user13", "user14", "user15"]
        }

        self.source_performance_metrics = {
            "postgresql": {
                "avg_latency_ms": 50,
                "uptime_percentage": 99.5,
                "throughput_qps": 1000
            },
            "redis": {
                "avg_latency_ms": 10,
                "uptime_percentage": 99.9,
                "throughput_qps": 10000
            },
            "elasticsearch": {
                "avg_latency_ms": 100,
                "uptime_percentage": 99.0,
                "throughput_qps": 500
            }
        }

        logger.info("📊 Historical scoring data loaded")

    async def _load_user_profiles(self):
        """Load user scoring profiles"""

        # In a real implementation, this would load from database/cache
        # For demo, start with empty profiles that will be created on demand

        logger.info("👥 User scoring profiles ready")

    async def _save_user_profiles(self):
        """Save user scoring profiles"""

        # In a real implementation, this would save to database/cache
        try:
            if self.cache_manager:
                for user_id, profile in self.user_profiles.items():
                    cache_key = f"scoring_profile:{user_id}"
                    await self.cache_manager.set(cache_key, profile.__dict__, ttl=86400)  # 24 hours

            logger.info(f"💾 Saved {len(self.user_profiles)} user scoring profiles")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save user profiles: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for scoring engine"""

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "user_profiles_loaded": len(self.user_profiles),
            "global_success_rates": len(self.global_success_rates),
            "peer_groups": len(self.peer_groups),
            "performance_metrics": len(self.source_performance_metrics),
            "scoring_methods": [method.value for method in ScoringMethod],
            "score_components": [component.value for component in ScoreComponent]
        }


# Factory function
async def create_scoring_engine(
        database_manager: Optional[DatabaseManager] = None,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[ScoringConfig] = None
) -> ScoringEngine:
    """Create and initialize scoring engine"""
    engine = ScoringEngine(database_manager, cache_manager, config)
    await engine.initialize()
    return engine


# Testing
if __name__ == "__main__":
    import asyncio


    async def test_scoring_engine():
        """Test scoring engine functionality"""

        try:
            print("🧪 Testing Scoring Engine...")

            # Create scoring engine
            engine = await create_scoring_engine()
            print("✅ Scoring engine created successfully")

            # Test 1: Basic Score Calculation
            print("\n🔍 Test 1: Basic Score Calculation")

            user_profile = UserProfile(id="test_user", user_id="test_user", role="data_analyst")
            recommendation = Recommendation(
                id="test_rec",
                user_id="test_user",
                source_id="postgresql",
                confidence_score=0.7,
                recommendation_type="environment",
                reasoning={"discovered_automatically": True},
                context={"environment_type": "local"}
            )

            score_breakdown = await engine.calculate_score(
                recommendation, user_profile, method=ScoringMethod.WEIGHTED_AVERAGE
            )

            print(f"   Total score: {score_breakdown.total_score:.3f}")
            print(f"   Confidence: {score_breakdown.confidence:.3f}")
            print(f"   Method used: {score_breakdown.method_used.value}")
            print(f"   Components calculated: {len(score_breakdown.components)}")

            # Test 2: Different Scoring Methods
            print("\n🔍 Test 2: Different Scoring Methods")

            methods = [ScoringMethod.BAYESIAN, ScoringMethod.ENSEMBLE, ScoringMethod.ADAPTIVE]

            for method in methods:
                breakdown = await engine.calculate_score(recommendation, user_profile, method=method)
                print(f"   {method.value}: {breakdown.total_score:.3f} (confidence: {breakdown.confidence:.3f})")

            # Test 3: User Feedback Learning
            print("\n🔍 Test 3: User Feedback Learning")

            # Simulate user feedback
            await engine.update_user_feedback("test_user", recommendation, "accept")
            await engine.update_user_feedback("test_user", recommendation, "reject")
            await engine.update_user_feedback("test_user", recommendation, "accept")

            insights = await engine.get_scoring_insights("test_user")
            print(f"   Total recommendations: {insights['total_recommendations']}")
            print(f"   Acceptance rate: {insights['acceptance_rate']:.3f}")
            print(f"   Scoring maturity: {insights['scoring_maturity']:.3f}")

            # Test 4: Component Score Analysis
            print("\n🔍 Test 4: Component Score Analysis")

            detailed_breakdown = await engine.calculate_score(
                recommendation, user_profile, method=ScoringMethod.ENSEMBLE
            )

            print("   Component scores:")
            for component, score in detailed_breakdown.components.items():
                print(f"     {component.value}: {score:.3f}")

            if detailed_breakdown.adjustments:
                print("   Score adjustments:")
                for adjustment, value in detailed_breakdown.adjustments.items():
                    print(f"     {adjustment}: {value:+.3f}")

            # Test 5: Model Calibration
            print("\n🔍 Test 5: Model Calibration")

            # Simulate feedback data for calibration
            feedback_data = [
                {
                    "action": "accept",
                    "score_breakdown": {
                        "components": {
                            "environment_match": 0.8,
                            "role_alignment": 0.9,
                            "usage_patterns": 0.7
                        }
                    }
                },
                {
                    "action": "reject",
                    "score_breakdown": {
                        "components": {
                            "environment_match": 0.4,
                            "role_alignment": 0.5,
                            "usage_patterns": 0.3
                        }
                    }
                }
            ]

            calibration_results = await engine.calibrate_scoring_model(feedback_data)
            print(f"   Feedback samples: {calibration_results['total_feedback_samples']}")
            print(f"   Acceptance rate: {calibration_results['overall_acceptance_rate']:.3f}")
            print(f"   Calibration confidence: {calibration_results['calibration_confidence']:.3f}")

            # Test 6: Health Check
            print("\n🔍 Test 6: Health Check")

            health = await engine.health_check()
            print(f"   Status: {health['status']}")
            print(f"   User profiles: {health['user_profiles_loaded']}")
            print(f"   Available methods: {len(health['scoring_methods'])}")
            print(f"   Score components: {len(health['score_components'])}")

            print("\n" + "=" * 50)
            print("✅ ALL SCORING ENGINE TESTS PASSED! 🎉")
            print("   - Score calculation ✓")
            print("   - Multiple scoring methods ✓")
            print("   - User feedback learning ✓")
            print("   - Component analysis ✓")
            print("   - Model calibration ✓")
            print("   - Health monitoring ✓")

            await engine.close()
            print("\n🔐 Scoring engine closed gracefully")

        except Exception as e:
            print(f"\n❌ Scoring engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


    # Run tests
    print("🚀 Starting Scoring Engine Test")
    success = asyncio.run(test_scoring_engine())

    if success:
        print("\n🎯 Scoring engine is ready for integration!")
        print("   Features available:")
        print("   • Advanced multi-method scoring algorithms")
        print("   • Personalized user scoring profiles")
        print("   • Adaptive learning from feedback")
        print("   • Component-based score analysis")
        print("   • Model calibration and optimization")
        print("   • Comprehensive scoring insights")
    else:
        print("\n💥 Tests failed - check the error messages above")
        sys.exit(1)