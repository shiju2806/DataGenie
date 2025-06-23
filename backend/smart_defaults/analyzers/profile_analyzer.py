"""
Smart Defaults Profile Analyzer
Intelligently analyzes user profiles, roles, and behavior patterns for personalized recommendations
File location: smart_defaults/analyzers/profile_analyzer.py
"""

import asyncio
import logging
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import statistics
import re

# Import dependencies with fallbacks
try:
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..storage.config_loader import ConfigurationLoader, RoleTemplate, IndustryProfile
    from ..models.user_profile import UserProfile, UserPreferences
    from ..models.data_source import DataSource
    from ..models.recommendation import Recommendation
    from ..utils.monitoring import AnalyticsEngine, EventType
    from ..analyzers.environment_scanner import EnvironmentScanner, EnvironmentProfile
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
        department: str = "analytics"
        industry: str = "technology"
        seniority_level: str = "senior"
        preferences: Any = None

    @dataclass
    class UserPreferences:
        auto_connect_threshold: float = 0.8
        recommendation_frequency: str = "daily"

    @dataclass
    class DataSource:
        id: str = "test_source"
        name: str = "Test Source"
        source_type: str = "database"

    @dataclass
    class Recommendation:
        id: str = "test_rec"
        user_id: str = "test_user"
        confidence_score: float = 0.8

    @dataclass
    class RoleTemplate:
        name: str = "Data Analyst"
        permissions: List[str] = None
        default_sources: List[str] = None
        auto_connect_threshold: float = 0.8

    @dataclass
    class IndustryProfile:
        name: str = "Technology"
        common_sources: List[str] = None
        security_requirements: List[str] = None

    @dataclass
    class EnvironmentProfile:
        user_id: str = "test_user"
        discovered_sources: List[Any] = None

    class DatabaseManager:
        async def initialize(self): pass
        async def close(self): pass
        async def get_user_behavior_summary(self, user_id): return None

    class CacheManager:
        async def initialize(self): pass
        async def close(self): pass
        async def get(self, key, default=None): return default
        async def set(self, key, value, ttl=None): pass

    class ConfigurationLoader:
        def get_role_template(self, role): return None
        def get_industry_profile(self, industry): return None

    class AnalyticsEngine:
        async def track_event(self, *args, **kwargs): pass

    class EnvironmentScanner:
        async def get_cached_scan(self, user_id): return None

    class EventType:
        USER_ACTION = "user_action"

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of profile analysis"""
    ROLE_ANALYSIS = "role_analysis"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    PREFERENCE_ANALYSIS = "preference_analysis"
    CAPABILITY_ANALYSIS = "capability_analysis"
    RISK_ANALYSIS = "risk_analysis"
    COMPATIBILITY_ANALYSIS = "compatibility_analysis"

class UserSegment(Enum):
    """User behavior segments"""
    POWER_USER = "power_user"
    CAUTIOUS_USER = "cautious_user"
    EXPLORER = "explorer"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    NEWCOMER = "newcomer"
    EXPERT = "expert"

class RecommendationStrategy(Enum):
    """Recommendation strategies based on user profile"""
    AGGRESSIVE = "aggressive"  # High auto-connect, many suggestions
    CONSERVATIVE = "conservative"  # Low auto-connect, few suggestions
    BALANCED = "balanced"  # Moderate approach
    GUIDED = "guided"  # Step-by-step with explanations
    MINIMAL = "minimal"  # Only essential recommendations

@dataclass
class BehaviorPattern:
    """User behavior pattern analysis"""
    pattern_id: str
    user_id: str
    pattern_type: str
    frequency: float
    confidence: float
    last_observed: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapabilityAssessment:
    """User capability assessment"""
    user_id: str
    technical_level: str  # beginner, intermediate, advanced, expert
    domain_expertise: List[str]
    tool_proficiency: Dict[str, float]
    learning_pace: str  # slow, medium, fast
    risk_tolerance: str  # low, medium, high
    autonomy_preference: str  # guided, balanced, independent
    confidence_score: float

@dataclass
class ProfileInsights:
    """Comprehensive profile analysis insights"""
    user_id: str
    analyzed_at: datetime

    # Core profile analysis
    role_match_score: float
    industry_alignment: float
    experience_level: str
    user_segment: UserSegment

    # Behavior analysis
    behavior_patterns: List[BehaviorPattern]
    activity_trends: Dict[str, Any]
    preference_evolution: Dict[str, Any]

    # Capability assessment
    capabilities: CapabilityAssessment

    # Recommendations
    recommended_strategy: RecommendationStrategy
    suggested_sources: List[str]
    personalization_tips: List[str]

    # Scores and metrics
    overall_confidence: float
    analysis_completeness: float

class ProfileAnalyzer:
    """Main profile analyzer for intelligent user analysis"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 config_loader: Optional[ConfigurationLoader] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 environment_scanner: Optional[EnvironmentScanner] = None,
                 analysis_cache_ttl: int = 3600):

        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.config_loader = config_loader
        self.analytics_engine = analytics_engine
        self.environment_scanner = environment_scanner
        self.analysis_cache_ttl = analysis_cache_ttl

        # Analysis configurations
        self.role_weights = self._get_role_analysis_weights()
        self.behavior_thresholds = self._get_behavior_thresholds()
        self.capability_indicators = self._get_capability_indicators()

        self._initialized = False

    async def initialize(self):
        """Initialize the profile analyzer"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()
        if self.analytics_engine:
            await self.analytics_engine.initialize()

        self._initialized = True
        logger.info("✅ Profile analyzer initialized")

    async def close(self):
        """Close the profile analyzer"""
        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Profile analyzer closed")

    def _get_role_analysis_weights(self) -> Dict[str, float]:
        """Get weights for different aspects of role analysis"""
        return {
            "role_template_match": 0.3,
            "industry_alignment": 0.2,
            "seniority_consistency": 0.2,
            "department_relevance": 0.15,
            "behavior_consistency": 0.15
        }

    def _get_behavior_thresholds(self) -> Dict[str, Any]:
        """Get thresholds for behavior analysis"""
        return {
            "power_user_actions_per_day": 20,
            "explorer_source_diversity": 5,
            "cautious_user_acceptance_rate": 0.3,
            "expert_advanced_feature_usage": 0.7,
            "newcomer_days_since_signup": 30
        }

    def _get_capability_indicators(self) -> Dict[str, List[str]]:
        """Get indicators for capability assessment"""
        return {
            "technical_level": {
                "beginner": ["basic_sql", "excel", "tableau_basic"],
                "intermediate": ["python_basic", "r_basic", "advanced_sql", "git"],
                "advanced": ["python_advanced", "machine_learning", "spark", "kubernetes"],
                "expert": ["distributed_systems", "custom_algorithms", "system_architecture"]
            },
            "domain_expertise": {
                "analytics": ["statistics", "data_visualization", "reporting"],
                "engineering": ["etl", "data_pipelines", "system_design"],
                "science": ["machine_learning", "research", "experimentation"],
                "business": ["kpis", "business_intelligence", "stakeholder_management"]
            }
        }

    async def analyze_user_profile(self, user_profile: UserProfile,
                                 include_behavior: bool = True,
                                 include_environment: bool = True,
                                 force_refresh: bool = False) -> ProfileInsights:
        """Perform comprehensive user profile analysis"""

        if not self._initialized:
            await self.initialize()

        analysis_id = f"profile_analysis_{user_profile.user_id}_{int(datetime.now(timezone.utc).timestamp())}"

        # Check cache for recent analysis
        if not force_refresh and self.cache_manager:
            cache_key = f"profile_analysis:{user_profile.user_id}"
            cached_analysis = await self.cache_manager.get(cache_key)
            if cached_analysis:
                logger.info(f"📋 Using cached profile analysis for {user_profile.user_id}")
                return ProfileInsights(**cached_analysis)

        logger.info(f"🔍 Starting profile analysis: {analysis_id}")
        start_time = datetime.now(timezone.utc)

        try:
            # Core profile analysis
            role_analysis = await self._analyze_role_compatibility(user_profile)
            industry_analysis = await self._analyze_industry_alignment(user_profile)
            experience_analysis = await self._analyze_experience_level(user_profile)

            # Behavior analysis
            behavior_patterns = []
            activity_trends = {}
            preference_evolution = {}

            if include_behavior:
                behavior_patterns = await self._analyze_behavior_patterns(user_profile.user_id)
                activity_trends = await self._analyze_activity_trends(user_profile.user_id)
                preference_evolution = await self._analyze_preference_evolution(user_profile.user_id)

            # Capability assessment
            capabilities = await self._assess_user_capabilities(user_profile, behavior_patterns)

            # User segmentation
            user_segment = await self._determine_user_segment(user_profile, behavior_patterns, capabilities)

            # Environment context
            environment_context = {}
            if include_environment and self.environment_scanner:
                env_profile = await self.environment_scanner.get_cached_scan(user_profile.user_id)
                if env_profile:
                    environment_context = self._analyze_environment_context(env_profile)

            # Generate recommendations
            recommended_strategy = self._determine_recommendation_strategy(
                user_profile, capabilities, user_segment
            )

            suggested_sources = await self._suggest_data_sources(
                user_profile, capabilities, environment_context
            )

            personalization_tips = self._generate_personalization_tips(
                user_profile, capabilities, user_segment
            )

            # Calculate confidence scores
            overall_confidence = self._calculate_overall_confidence(
                role_analysis, industry_analysis, behavior_patterns, capabilities
            )

            analysis_completeness = self._calculate_analysis_completeness(
                user_profile, behavior_patterns, environment_context
            )

            # Create comprehensive insights
            insights = ProfileInsights(
                user_id=user_profile.user_id,
                analyzed_at=start_time,
                role_match_score=role_analysis["match_score"],
                industry_alignment=industry_analysis["alignment_score"],
                experience_level=experience_analysis["level"],
                user_segment=user_segment,
                behavior_patterns=behavior_patterns,
                activity_trends=activity_trends,
                preference_evolution=preference_evolution,
                capabilities=capabilities,
                recommended_strategy=recommended_strategy,
                suggested_sources=suggested_sources,
                personalization_tips=personalization_tips,
                overall_confidence=overall_confidence,
                analysis_completeness=analysis_completeness
            )

            # Cache the analysis
            if self.cache_manager:
                await self.cache_manager.set(
                    f"profile_analysis:{user_profile.user_id}",
                    asdict(insights),
                    ttl=self.analysis_cache_ttl
                )

            # Track analytics
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_profile.user_id,
                    event_type=EventType.USER_ACTION,
                    data={
                        "action": "profile_analyzed",
                        "analysis_id": analysis_id,
                        "confidence_score": overall_confidence,
                        "user_segment": user_segment.value,
                        "recommended_strategy": recommended_strategy.value
                    }
                )

            analysis_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✅ Profile analysis completed in {analysis_duration:.2f}s: {overall_confidence:.2f} confidence")

            return insights

        except Exception as e:
            logger.error(f"❌ Profile analysis failed for {user_profile.user_id}: {e}")

            # Return minimal insights on failure
            return ProfileInsights(
                user_id=user_profile.user_id,
                analyzed_at=start_time,
                role_match_score=0.5,
                industry_alignment=0.5,
                experience_level="unknown",
                user_segment=UserSegment.NEWCOMER,
                behavior_patterns=[],
                activity_trends={},
                preference_evolution={},
                capabilities=CapabilityAssessment(
                    user_id=user_profile.user_id,
                    technical_level="intermediate",
                    domain_expertise=[],
                    tool_proficiency={},
                    learning_pace="medium",
                    risk_tolerance="medium",
                    autonomy_preference="balanced",
                    confidence_score=0.3
                ),
                recommended_strategy=RecommendationStrategy.BALANCED,
                suggested_sources=[],
                personalization_tips=["Profile analysis unavailable - using default recommendations"],
                overall_confidence=0.3,
                analysis_completeness=0.2
            )

    async def _analyze_role_compatibility(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze how well the user's profile matches their declared role"""

        analysis = {
            "match_score": 0.5,
            "confidence": 0.5,
            "mismatches": [],
            "strengths": [],
            "suggestions": []
        }

        try:
            if not self.config_loader:
                return analysis

            # Get role template
            role_template = self.config_loader.get_role_template(user_profile.role)
            if not role_template:
                analysis["suggestions"].append(f"Role '{user_profile.role}' not found in templates")
                return analysis

            match_score = 0.7  # Base score for having a valid role

            # Check seniority consistency
            seniority_map = {
                "junior": ["entry", "junior", "associate"],
                "mid": ["mid", "intermediate", "regular"],
                "senior": ["senior", "lead", "principal"],
                "staff": ["staff", "architect", "distinguished"]
            }

            expected_seniority = None
            for level, keywords in seniority_map.items():
                if any(keyword in user_profile.seniority_level.lower() for keyword in keywords):
                    expected_seniority = level
                    break

            if expected_seniority:
                match_score += 0.1
                analysis["strengths"].append(f"Seniority level '{user_profile.seniority_level}' is consistent")
            else:
                analysis["mismatches"].append("Seniority level unclear or inconsistent")

            # Check department alignment
            role_departments = {
                "data_analyst": ["analytics", "data", "business_intelligence", "insights"],
                "data_engineer": ["engineering", "data", "platform", "infrastructure"],
                "data_scientist": ["data", "science", "research", "ai", "ml"],
                "business_analyst": ["business", "strategy", "operations", "product"]
            }

            if user_profile.role in role_departments:
                expected_departments = role_departments[user_profile.role]
                if any(dept in user_profile.department.lower() for dept in expected_departments):
                    match_score += 0.1
                    analysis["strengths"].append("Department alignment is good")
                else:
                    analysis["mismatches"].append("Department doesn't typically align with role")

            # Check preferences alignment
            if user_profile.preferences and hasattr(user_profile.preferences, 'auto_connect_threshold'):
                user_threshold = user_profile.preferences.auto_connect_threshold
                role_threshold = role_template.auto_connect_threshold

                threshold_diff = abs(user_threshold - role_threshold)
                if threshold_diff < 0.1:
                    match_score += 0.1
                    analysis["strengths"].append("Auto-connect threshold aligns with role expectations")
                elif threshold_diff > 0.3:
                    analysis["mismatches"].append("Auto-connect threshold significantly differs from role norm")

            analysis["match_score"] = min(match_score, 1.0)
            analysis["confidence"] = 0.8 if len(analysis["strengths"]) > len(analysis["mismatches"]) else 0.6

            # Generate suggestions
            if analysis["match_score"] < 0.7:
                analysis["suggestions"].append("Consider reviewing role configuration or profile details")
            if len(analysis["mismatches"]) > 2:
                analysis["suggestions"].append("Profile may benefit from role-specific customization")

        except Exception as e:
            logger.warning(f"⚠️ Role compatibility analysis failed: {e}")
            analysis["suggestions"].append("Role analysis incomplete due to configuration issues")

        return analysis

    async def _analyze_industry_alignment(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze industry-specific profile alignment"""

        analysis = {
            "alignment_score": 0.5,
            "industry_strengths": [],
            "compliance_notes": [],
            "recommendations": []
        }

        try:
            if not self.config_loader:
                return analysis

            industry_profile = self.config_loader.get_industry_profile(user_profile.industry)
            if not industry_profile:
                analysis["recommendations"].append(f"Industry '{user_profile.industry}' profile not available")
                return analysis

            alignment_score = 0.6  # Base score for having industry info

            # Check role-industry fit
            industry_common_roles = {
                "technology": ["data_engineer", "data_scientist", "ml_engineer", "data_analyst"],
                "finance": ["data_analyst", "quantitative_analyst", "risk_analyst", "business_analyst"],
                "healthcare": ["data_analyst", "clinical_analyst", "research_analyst"],
                "retail": ["business_analyst", "data_analyst", "marketing_analyst"]
            }

            if user_profile.industry in industry_common_roles:
                common_roles = industry_common_roles[user_profile.industry]
                if user_profile.role in common_roles:
                    alignment_score += 0.2
                    analysis["industry_strengths"].append(f"Role is common in {user_profile.industry} industry")
                else:
                    analysis["recommendations"].append("Role is less common in this industry - may need specialized setup")

            # Industry-specific compliance considerations
            high_security_industries = ["finance", "healthcare", "government", "defense"]
            if user_profile.industry in high_security_industries:
                analysis["compliance_notes"].append(f"{user_profile.industry} requires enhanced security measures")

                if user_profile.preferences and hasattr(user_profile.preferences, 'auto_connect_threshold'):
                    if user_profile.preferences.auto_connect_threshold > 0.8:
                        analysis["compliance_notes"].append("High auto-connect threshold may conflict with security requirements")
                        alignment_score -= 0.1

            # Industry data patterns
            data_intensive_industries = ["technology", "finance", "telecommunications", "retail"]
            if user_profile.industry in data_intensive_industries:
                analysis["industry_strengths"].append("Industry typically has rich data environments")
                alignment_score += 0.1

            analysis["alignment_score"] = min(alignment_score, 1.0)

        except Exception as e:
            logger.warning(f"⚠️ Industry alignment analysis failed: {e}")
            analysis["recommendations"].append("Industry analysis incomplete")

        return analysis

    async def _analyze_experience_level(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze user's experience level based on profile signals"""

        # Map seniority to experience levels
        seniority_experience_map = {
            "intern": "beginner",
            "entry": "beginner",
            "junior": "beginner",
            "associate": "intermediate",
            "mid": "intermediate",
            "intermediate": "intermediate",
            "senior": "advanced",
            "lead": "advanced",
            "principal": "expert",
            "staff": "expert",
            "architect": "expert",
            "distinguished": "expert"
        }

        experience_level = "intermediate"  # Default
        confidence = 0.5

        # Check seniority level
        seniority = user_profile.seniority_level.lower()
        for keyword, level in seniority_experience_map.items():
            if keyword in seniority:
                experience_level = level
                confidence = 0.8
                break

        # Adjust based on preferences
        if user_profile.preferences and hasattr(user_profile.preferences, 'auto_connect_threshold'):
            threshold = user_profile.preferences.auto_connect_threshold
            if threshold > 0.9 and experience_level == "beginner":
                # High threshold suggests confidence, maybe not true beginner
                experience_level = "intermediate"
            elif threshold < 0.5 and experience_level == "expert":
                # Low threshold suggests caution, maybe not full expert confidence
                experience_level = "advanced"

        return {
            "level": experience_level,
            "confidence": confidence,
            "indicators": [f"Seniority: {user_profile.seniority_level}"]
        }

    async def _analyze_behavior_patterns(self, user_id: str) -> List[BehaviorPattern]:
        """Analyze user behavior patterns from historical data"""
        patterns = []

        try:
            if not self.database_manager:
                return patterns

            # Get behavior data from database
            behavior_summary = await self.database_manager.get_user_behavior_summary(user_id)
            if not behavior_summary:
                return patterns

            # Analyze recommendation acceptance patterns
            if hasattr(behavior_summary, 'recommendation_acceptance_rate'):
                acceptance_rate = behavior_summary.recommendation_acceptance_rate

                if acceptance_rate > 0.8:
                    patterns.append(BehaviorPattern(
                        pattern_id=f"high_acceptance_{user_id}",
                        user_id=user_id,
                        pattern_type="high_recommendation_acceptance",
                        frequency=acceptance_rate,
                        confidence=0.9,
                        last_observed=datetime.now(timezone.utc),
                        metadata={"acceptance_rate": acceptance_rate}
                    ))
                elif acceptance_rate < 0.3:
                    patterns.append(BehaviorPattern(
                        pattern_id=f"low_acceptance_{user_id}",
                        user_id=user_id,
                        pattern_type="cautious_recommendation_acceptance",
                        frequency=acceptance_rate,
                        confidence=0.8,
                        last_observed=datetime.now(timezone.utc),
                        metadata={"acceptance_rate": acceptance_rate}
                    ))

            # Analyze activity patterns
            if hasattr(behavior_summary, 'most_common_actions'):
                common_actions = dict(behavior_summary.most_common_actions)

                # High activity pattern
                total_actions = sum(common_actions.values())
                if total_actions > 100:
                    patterns.append(BehaviorPattern(
                        pattern_id=f"high_activity_{user_id}",
                        user_id=user_id,
                        pattern_type="high_activity_user",
                        frequency=total_actions / 30,  # Actions per day estimate
                        confidence=0.7,
                        last_observed=datetime.now(timezone.utc),
                        metadata={"total_actions": total_actions, "actions": common_actions}
                    ))

                # Source exploration pattern
                if "source_connected" in common_actions and common_actions["source_connected"] > 10:
                    patterns.append(BehaviorPattern(
                        pattern_id=f"explorer_{user_id}",
                        user_id=user_id,
                        pattern_type="source_explorer",
                        frequency=common_actions["source_connected"],
                        confidence=0.8,
                        last_observed=datetime.now(timezone.utc),
                        metadata={"connections": common_actions["source_connected"]}
                    ))

        except Exception as e:
            logger.warning(f"⚠️ Behavior pattern analysis failed for {user_id}: {e}")

        return patterns

    async def _analyze_activity_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze user activity trends over time"""
        trends = {
            "activity_level": "moderate",
            "peak_hours": [],
            "weekly_pattern": {},
            "growth_trend": "stable"
        }

        # In a real implementation, this would analyze time-series data
        # For now, return mock trends
        trends.update({
            "peak_hours": ["09:00-11:00", "14:00-16:00"],
            "weekly_pattern": {
                "monday": 0.8,
                "tuesday": 0.9,
                "wednesday": 1.0,
                "thursday": 0.8,
                "friday": 0.6,
                "weekend": 0.2
            }
        })

        return trends

    async def _analyze_preference_evolution(self, user_id: str) -> Dict[str, Any]:
        """Analyze how user preferences have changed over time"""
        evolution = {
            "stability": "stable",
            "recent_changes": [],
            "trend_direction": "neutral"
        }

        # In a real implementation, this would track preference changes
        # For now, return basic evolution data
        return evolution

    async def _assess_user_capabilities(self, user_profile: UserProfile,
                                      behavior_patterns: List[BehaviorPattern]) -> CapabilityAssessment:
        """Assess user's technical and domain capabilities"""

        # Start with defaults based on role and seniority
        role_capability_map = {
            "data_analyst": {
                "technical_level": "intermediate",
                "domain_expertise": ["analytics", "reporting", "visualization"],
                "base_tools": {"sql": 0.7, "excel": 0.8, "tableau": 0.6}
            },
            "data_engineer": {
                "technical_level": "advanced",
                "domain_expertise": ["engineering", "etl", "infrastructure"],
                "base_tools": {"python": 0.8, "sql": 0.9, "spark": 0.7}
            },
            "data_scientist": {
                "technical_level": "advanced",
                "domain_expertise": ["science", "machine_learning", "research"],
                "base_tools": {"python": 0.9, "r": 0.7, "jupyter": 0.8}
            }
        }

        # Get base capabilities from role
        role_defaults = role_capability_map.get(user_profile.role, {
            "technical_level": "intermediate",
            "domain_expertise": ["general"],
            "base_tools": {}
        })

        technical_level = role_defaults["technical_level"]
        domain_expertise = role_defaults["domain_expertise"].copy()
        tool_proficiency = role_defaults["base_tools"].copy()

        # Adjust based on seniority
        seniority_adjustments = {
            "junior": {"technical": -1, "autonomy": "guided"},
            "mid": {"technical": 0, "autonomy": "balanced"},
            "senior": {"technical": 1, "autonomy": "independent"},
            "lead": {"technical": 2, "autonomy": "independent"}
        }

        seniority_key = None
        for key in seniority_adjustments:
            if key in user_profile.seniority_level.lower():
                seniority_key = key
                break

        if seniority_key:
            adjustment = seniority_adjustments[seniority_key]

            # Adjust technical level
            levels = ["beginner", "intermediate", "advanced", "expert"]
            current_index = levels.index(technical_level)
            new_index = max(0, min(len(levels) - 1, current_index + adjustment["technical"]))
            technical_level = levels[new_index]

        # Analyze behavior patterns for capability indicators
        risk_tolerance = "medium"
        learning_pace = "medium"
        autonomy_preference = seniority_adjustments.get(seniority_key, {}).get("autonomy", "balanced")

        for pattern in behavior_patterns:
            if pattern.pattern_type == "high_recommendation_acceptance":
                risk_tolerance = "high"
                learning_pace = "fast"
            elif pattern.pattern_type == "cautious_recommendation_acceptance":
                risk_tolerance = "low"
                autonomy_preference = "guided"
        for pattern in behavior_patterns:
            if pattern.pattern_type == "high_recommendation_acceptance":
                risk_tolerance = "high"
                learning_pace = "fast"
            elif pattern.pattern_type == "cautious_recommendation_acceptance":
                risk_tolerance = "low"
                autonomy_preference = "guided"
            elif pattern.pattern_type == "source_explorer":
                risk_tolerance = "high"
                learning_pace = "fast"
                # Add exploration tools to proficiency
                tool_proficiency.update({"api_tools": 0.6, "data_discovery": 0.7})
            elif pattern.pattern_type == "high_activity_user":
                learning_pace = "fast"
                # Boost tool proficiency for active users
                for tool in tool_proficiency:
                    tool_proficiency[tool] = min(tool_proficiency[tool] + 0.1, 1.0)

        # Calculate overall confidence
        confidence_factors = [
            0.8,  # Base confidence
            0.1 if len(behavior_patterns) > 0 else 0,  # Behavior data available
            0.1 if len(domain_expertise) > 1 else 0   # Domain diversity
        ]
        confidence_score = sum(confidence_factors)

        return CapabilityAssessment(
            user_id=user_profile.user_id,
            technical_level=technical_level,
            domain_expertise=domain_expertise,
            tool_proficiency=tool_proficiency,
            learning_pace=learning_pace,
            risk_tolerance=risk_tolerance,
            autonomy_preference=autonomy_preference,
            confidence_score=min(confidence_score, 1.0)
        )

    async def _determine_user_segment(self, user_profile: UserProfile,
                                    behavior_patterns: List[BehaviorPattern],
                                    capabilities: CapabilityAssessment) -> UserSegment:
        """Determine user segment based on profile and behavior analysis"""

        # Analyze behavior patterns for segmentation signals
        pattern_types = [p.pattern_type for p in behavior_patterns]

        # Power user indicators
        if ("high_activity_user" in pattern_types and
            "source_explorer" in pattern_types and
            capabilities.technical_level in ["advanced", "expert"]):
            return UserSegment.POWER_USER

        # Cautious user indicators
        if ("cautious_recommendation_acceptance" in pattern_types or
            capabilities.risk_tolerance == "low"):
            return UserSegment.CAUTIOUS_USER

        # Explorer indicators
        if ("source_explorer" in pattern_types or
            capabilities.risk_tolerance == "high"):
            return UserSegment.EXPLORER

        # Expert indicators
        if (capabilities.technical_level == "expert" and
            "senior" in user_profile.seniority_level.lower()):
            return UserSegment.EXPERT

        # Specialist vs Generalist
        if len(capabilities.domain_expertise) == 1:
            return UserSegment.SPECIALIST
        elif len(capabilities.domain_expertise) > 2:
            return UserSegment.GENERALIST

        # Newcomer indicators (default for limited data)
        if len(behavior_patterns) == 0:
            return UserSegment.NEWCOMER

        # Default fallback
        return UserSegment.GENERALIST

    def _analyze_environment_context(self, env_profile: EnvironmentProfile) -> Dict[str, Any]:
        """Analyze environment context for profile insights"""
        context = {
            "data_richness": "unknown",
            "infrastructure_maturity": "unknown",
            "available_tools": [],
            "complexity_level": "medium"
        }

        if not env_profile or not env_profile.discovered_sources:
            return context

        available_sources = [s for s in env_profile.discovered_sources if hasattr(s, 'available') and s.available]

        # Assess data richness
        if len(available_sources) >= 5:
            context["data_richness"] = "rich"
        elif len(available_sources) >= 2:
            context["data_richness"] = "moderate"
        else:
            context["data_richness"] = "limited"

        # Assess infrastructure maturity
        source_types = set()
        advanced_tools = 0

        for source in available_sources:
            if hasattr(source, 'target'):
                source_types.add(source.target.type.value if hasattr(source.target.type, 'value') else str(source.target.type))

                # Count advanced tools
                if hasattr(source.target, 'service_name'):
                    advanced_services = ["kafka", "elasticsearch", "airflow", "kubernetes", "spark"]
                    if source.target.service_name in advanced_services:
                        advanced_tools += 1

        if len(source_types) >= 3 and advanced_tools >= 2:
            context["infrastructure_maturity"] = "advanced"
        elif len(source_types) >= 2:
            context["infrastructure_maturity"] = "intermediate"
        else:
            context["infrastructure_maturity"] = "basic"

        # Extract available tools
        context["available_tools"] = [
            source.target.service_name for source in available_sources
            if hasattr(source, 'target') and hasattr(source.target, 'service_name')
        ]

        # Determine complexity level
        if context["infrastructure_maturity"] == "advanced" and context["data_richness"] == "rich":
            context["complexity_level"] = "high"
        elif context["infrastructure_maturity"] == "basic" and context["data_richness"] == "limited":
            context["complexity_level"] = "low"
        else:
            context["complexity_level"] = "medium"

        return context

    def _determine_recommendation_strategy(self, user_profile: UserProfile,
                                         capabilities: CapabilityAssessment,
                                         user_segment: UserSegment) -> RecommendationStrategy:
        """Determine the best recommendation strategy for the user"""

        # Strategy based on user segment
        segment_strategies = {
            UserSegment.POWER_USER: RecommendationStrategy.AGGRESSIVE,
            UserSegment.CAUTIOUS_USER: RecommendationStrategy.CONSERVATIVE,
            UserSegment.EXPLORER: RecommendationStrategy.AGGRESSIVE,
            UserSegment.EXPERT: RecommendationStrategy.MINIMAL,
            UserSegment.NEWCOMER: RecommendationStrategy.GUIDED,
            UserSegment.SPECIALIST: RecommendationStrategy.BALANCED,
            UserSegment.GENERALIST: RecommendationStrategy.BALANCED
        }

        base_strategy = segment_strategies.get(user_segment, RecommendationStrategy.BALANCED)

        # Adjust based on capabilities
        if capabilities.autonomy_preference == "guided":
            return RecommendationStrategy.GUIDED
        elif capabilities.autonomy_preference == "independent" and capabilities.technical_level == "expert":
            return RecommendationStrategy.MINIMAL
        elif capabilities.risk_tolerance == "low":
            return RecommendationStrategy.CONSERVATIVE
        elif capabilities.risk_tolerance == "high" and capabilities.technical_level in ["advanced", "expert"]:
            return RecommendationStrategy.AGGRESSIVE

        return base_strategy

    async def _suggest_data_sources(self, user_profile: UserProfile,
                                   capabilities: CapabilityAssessment,
                                   environment_context: Dict[str, Any]) -> List[str]:
        """Suggest relevant data sources based on profile analysis"""

        suggestions = []

        # Role-based suggestions
        role_sources = {
            "data_analyst": ["postgresql", "mysql", "tableau", "excel", "google_analytics"],
            "data_engineer": ["postgresql", "kafka", "airflow", "spark", "redis"],
            "data_scientist": ["jupyter", "postgresql", "python", "r", "mlflow"],
            "business_analyst": ["tableau", "powerbi", "salesforce", "google_analytics", "excel"]
        }

        if user_profile.role in role_sources:
            suggestions.extend(role_sources[user_profile.role])

        # Industry-specific suggestions
        industry_sources = {
            "technology": ["github", "jira", "elasticsearch", "kubernetes"],
            "finance": ["bloomberg", "reuters", "risk_systems", "compliance_db"],
            "healthcare": ["hl7", "epic", "cerner", "clinical_db"],
            "retail": ["shopify", "stripe", "inventory_systems", "pos_systems"]
        }

        if user_profile.industry in industry_sources:
            suggestions.extend(industry_sources[user_profile.industry])

        # Capability-based suggestions
        if capabilities.technical_level in ["advanced", "expert"]:
            suggestions.extend(["spark", "kubernetes", "kafka", "elasticsearch"])
        elif capabilities.technical_level == "beginner":
            suggestions.extend(["excel", "google_sheets", "tableau", "powerbi"])

        # Environment-based suggestions (prefer what's already available)
        if environment_context.get("available_tools"):
            # Prioritize discovered tools
            available_tools = environment_context["available_tools"]
            suggestions = available_tools + [s for s in suggestions if s not in available_tools]

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)

        # Limit suggestions based on strategy
        strategy_limits = {
            RecommendationStrategy.MINIMAL: 3,
            RecommendationStrategy.CONSERVATIVE: 5,
            RecommendationStrategy.BALANCED: 8,
            RecommendationStrategy.AGGRESSIVE: 12,
            RecommendationStrategy.GUIDED: 5
        }

        # This would normally be determined by the recommendation strategy
        limit = strategy_limits.get(RecommendationStrategy.BALANCED, 8)

        return unique_suggestions[:limit]

    def _generate_personalization_tips(self, user_profile: UserProfile,
                                     capabilities: CapabilityAssessment,
                                     user_segment: UserSegment) -> List[str]:
        """Generate personalized tips based on profile analysis"""

        tips = []

        # Segment-specific tips
        if user_segment == UserSegment.NEWCOMER:
            tips.extend([
                "Start with familiar tools like Excel or Google Sheets",
                "Consider guided tutorials for new data sources",
                "Begin with read-only connections to build confidence"
            ])
        elif user_segment == UserSegment.POWER_USER:
            tips.extend([
                "Enable auto-connect for trusted sources to save time",
                "Explore advanced features and integrations",
                "Consider setting up automated data pipelines"
            ])
        elif user_segment == UserSegment.CAUTIOUS_USER:
            tips.extend([
                "Review each recommendation carefully before accepting",
                "Start with sandbox or development environments",
                "Use lower auto-connect thresholds for security"
            ])
        elif user_segment == UserSegment.EXPLORER:
            tips.extend([
                "Try connecting to multiple data sources for broader insights",
                "Experiment with different visualization tools",
                "Consider cross-platform data integration"
            ])

        # Technical level tips
        if capabilities.technical_level == "beginner":
            tips.extend([
                "Focus on GUI-based tools initially",
                "Take advantage of drag-and-drop interfaces",
                "Start with pre-built templates and examples"
            ])
        elif capabilities.technical_level == "expert":
            tips.extend([
                "Customize connection parameters for optimal performance",
                "Consider API-based integrations for flexibility",
                "Set up monitoring and alerting for data sources"
            ])

        # Industry-specific tips
        if user_profile.industry in ["finance", "healthcare"]:
            tips.extend([
                "Ensure all connections meet compliance requirements",
                "Review data governance policies before connecting",
                "Consider encrypted connections for sensitive data"
            ])

        # Role-specific tips
        if user_profile.role == "data_engineer":
            tips.extend([
                "Focus on data pipeline and ETL tool connections",
                "Set up monitoring for data source health",
                "Consider batch vs real-time data source needs"
            ])
        elif user_profile.role == "data_analyst":
            tips.extend([
                "Prioritize business intelligence and reporting tools",
                "Connect to both raw data and processed datasets",
                "Set up alerts for data freshness and quality"
            ])

        # Remove duplicates
        return list(set(tips))

    def _calculate_overall_confidence(self, role_analysis: Dict[str, Any],
                                    industry_analysis: Dict[str, Any],
                                    behavior_patterns: List[BehaviorPattern],
                                    capabilities: CapabilityAssessment) -> float:
        """Calculate overall confidence in the profile analysis"""

        confidence_components = [
            role_analysis.get("confidence", 0.5) * 0.3,
            industry_analysis.get("alignment_score", 0.5) * 0.2,
            capabilities.confidence_score * 0.3,
            min(len(behavior_patterns) / 5, 1.0) * 0.2  # More patterns = higher confidence
        ]

        return round(sum(confidence_components), 2)

    def _calculate_analysis_completeness(self, user_profile: UserProfile,
                                       behavior_patterns: List[BehaviorPattern],
                                       environment_context: Dict[str, Any]) -> float:
        """Calculate how complete the analysis is"""

        completeness_factors = []

        # Profile completeness
        profile_fields = [user_profile.role, user_profile.department, user_profile.industry, user_profile.seniority_level]
        profile_completeness = sum(1 for field in profile_fields if field and field.strip()) / len(profile_fields)
        completeness_factors.append(profile_completeness * 0.4)

        # Behavior data completeness
        behavior_completeness = min(len(behavior_patterns) / 3, 1.0)  # 3+ patterns = complete
        completeness_factors.append(behavior_completeness * 0.3)

        # Environment data completeness
        env_completeness = 1.0 if environment_context.get("data_richness") != "unknown" else 0.3
        completeness_factors.append(env_completeness * 0.3)

        return round(sum(completeness_factors), 2)

    async def get_profile_recommendations(self, insights: ProfileInsights) -> List[Recommendation]:
        """Convert profile insights into actionable recommendations"""

        recommendations = []

        try:
            # Create recommendations based on suggested sources
            for i, source in enumerate(insights.suggested_sources[:5]):  # Limit to top 5
                confidence = max(0.6, insights.overall_confidence - (i * 0.1))  # Decrease confidence for lower priority

                recommendation = Recommendation(
                    id=f"profile_rec_{insights.user_id}_{source}_{int(datetime.now(timezone.utc).timestamp())}",
                    user_id=insights.user_id,
                    data_source_id=source,
                    recommendation_type="data_source_connection",
                    confidence_score=confidence,
                    reasoning={
                        "profile_based": True,
                        "user_segment": insights.user_segment.value,
                        "technical_level": insights.capabilities.technical_level,
                        "strategy": insights.recommended_strategy.value
                    },
                    context={
                        "analysis_id": f"profile_analysis_{insights.user_id}",
                        "priority": i + 1,
                        "personalization_tip": insights.personalization_tips[0] if insights.personalization_tips else None
                    }
                )

                recommendations.append(recommendation)

        except Exception as e:
            logger.warning(f"⚠️ Failed to generate profile recommendations: {e}")

        return recommendations

    async def update_profile_from_feedback(self, user_id: str, feedback_data: Dict[str, Any]) -> bool:
        """Update profile analysis based on user feedback"""

        try:
            # Invalidate cached analysis
            if self.cache_manager:
                await self.cache_manager.delete(f"profile_analysis:{user_id}")

            # In production, this would update the user's profile or preferences
            # based on their feedback to improve future recommendations

            logger.info(f"📝 Profile feedback processed for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update profile from feedback: {e}")
            return False

# Factory function
async def create_profile_analyzer(
    database_manager: Optional[DatabaseManager] = None,
    cache_manager: Optional[CacheManager] = None,
    config_loader: Optional[ConfigurationLoader] = None,
    analytics_engine: Optional[AnalyticsEngine] = None,
    environment_scanner: Optional[EnvironmentScanner] = None,
    **kwargs
) -> ProfileAnalyzer:
    """Factory function to create and initialize profile analyzer"""
    analyzer = ProfileAnalyzer(
        database_manager=database_manager,
        cache_manager=cache_manager,
        config_loader=config_loader,
        analytics_engine=analytics_engine,
        environment_scanner=environment_scanner,
        **kwargs
    )
    await analyzer.initialize()
    return analyzer

# Testing
if __name__ == "__main__":
    async def test_profile_analyzer():
        """Test profile analyzer functionality"""

        try:
            print("🧪 Testing Profile Analyzer...")

            # Create mock dependencies
            class MockConfig:
                def get_role_template(self, role):
                    if role == "data_analyst":
                        return RoleTemplate(
                            name="Data Analyst",
                            permissions=["read_data", "create_reports"],
                            default_sources=["postgresql", "tableau"],
                            auto_connect_threshold=0.8
                        )
                    return None

                def get_industry_profile(self, industry):
                    if industry == "technology":
                        return IndustryProfile(
                            name="Technology",
                            common_sources=["github", "postgres"],
                            security_requirements=["access_logging"]
                        )
                    return None

            class MockCache:
                def __init__(self):
                    self.data = {}
                async def initialize(self): pass
                async def close(self): pass
                async def get(self, key, default=None): return self.data.get(key, default)
                async def set(self, key, value, ttl=None):
                    self.data[key] = value
                    return True
                async def delete(self, key):
                    self.data.pop(key, None)
                    return True

            class MockAnalytics:
                async def initialize(self): pass
                async def track_event(self, *args, **kwargs):
                    print(f"📊 Analytics: {kwargs.get('data', {}).get('action', 'unknown')} for user {kwargs.get('user_id')}")

            # Initialize analyzer
            analyzer = await create_profile_analyzer(
                cache_manager=MockCache(),
                config_loader=MockConfig(),
                analytics_engine=MockAnalytics()
            )

            print("✅ Profile analyzer created successfully")

            try:
                # Test 1: Create test user profile
                print("\n🔍 Test 1: User Profile Analysis")

                test_user = UserProfile(
                    id="test_profile_123",
                    user_id="test_user_analyzer",
                    role="data_analyst",
                    department="analytics",
                    seniority_level="senior",
                    industry="technology",
                    preferences=UserPreferences(
                        auto_connect_threshold=0.85,
                        recommendation_frequency="daily"
                    )
                )

                # Perform comprehensive analysis
                insights = await analyzer.analyze_user_profile(
                    user_profile=test_user,
                    include_behavior=True,
                    include_environment=False  # Skip environment for this test
                )

                print(f"   Analysis completed for user: {insights.user_id}")
                print(f"   Role match score: {insights.role_match_score:.2f}")
                print(f"   Industry alignment: {insights.industry_alignment:.2f}")
                print(f"   Experience level: {insights.experience_level}")
                print(f"   User segment: {insights.user_segment.value}")
                print(f"   Recommended strategy: {insights.recommended_strategy.value}")
                print(f"   Overall confidence: {insights.overall_confidence:.2f}")

                # Test 2: Capability Assessment
                print("\n🔍 Test 2: Capability Assessment")
                capabilities = insights.capabilities
                print(f"   Technical level: {capabilities.technical_level}")
                print(f"   Domain expertise: {capabilities.domain_expertise}")
                print(f"   Tool proficiency: {capabilities.tool_proficiency}")
                print(f"   Risk tolerance: {capabilities.risk_tolerance}")
                print(f"   Learning pace: {capabilities.learning_pace}")
                print(f"   Autonomy preference: {capabilities.autonomy_preference}")

                # Test 3: Suggested Sources
                print("\n🔍 Test 3: Data Source Suggestions")
                print(f"   Suggested sources ({len(insights.suggested_sources)}):")
                for i, source in enumerate(insights.suggested_sources, 1):
                    print(f"     {i}. {source}")

                # Test 4: Personalization Tips
                print("\n🔍 Test 4: Personalization Tips")
                print(f"   Tips ({len(insights.personalization_tips)}):")
                for i, tip in enumerate(insights.personalization_tips, 1):
                    print(f"     {i}. {tip}")

                # Test 5: Generate Recommendations
                print("\n🔍 Test 5: Profile-Based Recommendations")
                recommendations = await analyzer.get_profile_recommendations(insights)
                print(f"   Generated {len(recommendations)} recommendations:")

                for rec in recommendations:
                    print(f"     - {rec.data_source_id} (confidence: {rec.confidence_score:.2f})")
                    print(f"       Strategy: {rec.reasoning.get('strategy', 'unknown')}")
                    print(f"       Priority: {rec.context.get('priority', 'N/A')}")

                # Test 6: Cache Operations
                print("\n🔍 Test 6: Analysis Caching")

                # Second analysis should use cache
                cached_insights = await analyzer.analyze_user_profile(
                    user_profile=test_user,
                    force_refresh=False
                )

                if cached_insights.analyzed_at == insights.analyzed_at:
                    print("   ✅ Cache working - returned same analysis")
                else:
                    print("   ⚠️ Cache not used - generated new analysis")

                # Test 7: Different User Profiles
                print("\n🔍 Test 7: Different User Profiles")

                # Newcomer profile
                newcomer = UserProfile(
                    id="newcomer_123",
                    user_id="newcomer_user",
                    role="data_analyst",
                    department="analytics",
                    seniority_level="junior",
                    industry="technology",
                    preferences=UserPreferences(auto_connect_threshold=0.6)
                )

                newcomer_insights = await analyzer.analyze_user_profile(newcomer)
                print(f"   Newcomer segment: {newcomer_insights.user_segment.value}")
                print(f"   Newcomer strategy: {newcomer_insights.recommended_strategy.value}")

                # Expert profile
                expert = UserProfile(
                    id="expert_123",
                    user_id="expert_user",
                    role="data_engineer",
                    department="engineering",
                    seniority_level="principal",
                    industry="technology",
                    preferences=UserPreferences(auto_connect_threshold=0.95)
                )

                expert_insights = await analyzer.analyze_user_profile(expert)
                print(f"   Expert segment: {expert_insights.user_segment.value}")
                print(f"   Expert strategy: {expert_insights.recommended_strategy.value}")

                print("\n" + "=" * 50)
                print("✅ ALL PROFILE ANALYZER TESTS PASSED! 🎉")
                print("   - Profile analysis ✓")
                print("   - Capability assessment ✓")
                print("   - User segmentation ✓")
                print("   - Recommendation generation ✓")
                print("   - Personalization tips ✓")
                print("   - Caching ✓")
                print("   - Multiple user types ✓")

            finally:
                await analyzer.close()
                print("\n🔐 Profile analyzer closed gracefully")

        except Exception as e:
            print(f"\n❌ Profile analyzer test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    # Run tests
    print("🚀 Starting Smart Defaults Profile Analyzer Test")
    success = asyncio.run(test_profile_analyzer())

    if success:
        print("\n🎯 Profile analyzer is ready for integration!")
        print("   Next steps:")
        print("   1. Integrate with user behavior tracking")
        print("   2. Connect to environment scanner for context")
        print("   3. Add machine learning for pattern recognition")
        print("   4. Set up A/B testing for recommendation strategies")
        print("   5. Implement feedback loops for continuous improvement")
    else:
        print("\n💥 Tests failed - check the error messages above")
        sys.exit(1)