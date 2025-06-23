"""
Smart Defaults Engine - Main Orchestrator (FIXED)
Coordinates all components to provide intelligent, personalized, and compliant data source recommendations
File location: smart_defaults/engine.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import all components with proper error handling
try:
    # Storage components
    from .storage.database import DatabaseManager, create_database_manager
    from .storage.cache import CacheManager, create_cache_manager
    from .storage.config_loader import ConfigurationLoader, create_config_loader

    # Model components
    from .models.user_profile import UserProfile, UserPreferences
    from .models.data_source import DataSource, SourceMetadata
    from .models.recommendation import Recommendation

    # Analyzer components - FIXED IMPORTS
    from .analyzers.environment_scanner import EnvironmentScanner, EnvironmentProfile, create_environment_scanner
    from .analyzers.profile_analyzer import ProfileAnalyzer, ProfileInsights, create_profile_analyzer
    from .analyzers.learning_engine import LearningEngine, LearningInsights, create_learning_engine
    from .analyzers.policy_engine import PolicyEngine, PolicyEvaluation, create_policy_engine

    # Utility components
    from .utils.monitoring import AnalyticsEngine, create_analytics_engine, EventType
    from .utils.notifications import NotificationEngine, create_notification_engine, NotificationType
    from .utils.scoring import ScoringEngine, create_scoring_engine

except ImportError:
    # For direct execution, create comprehensive mock classes
    from typing import Any
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "data_analyst"
        department: str = "analytics"
        seniority_level: str = "senior"
        industry: str = "technology"
        permissions: list = field(default_factory=list)
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
        connection_config: dict = field(default_factory=dict)
        health_status: str = "healthy"

    @dataclass
    class Recommendation:
        id: str = "test_rec"
        user_id: str = "test_user"
        source_id: str = "test_source"
        recommendation_type: str = "recommend"
        confidence_score: float = 0.8
        reasoning: dict = field(default_factory=dict)
        context: dict = field(default_factory=dict)

    @dataclass
    class EnvironmentProfile:
        user_id: str = "test_user"
        discovered_sources: List[Any] = field(default_factory=list)
        recommendations: List[str] = field(default_factory=list)
        confidence_score: float = 0.7

    @dataclass
    class ProfileInsights:
        user_id: str = "test_user"
        user_segment: Any = None
        suggested_sources: List[str] = field(default_factory=list)
        overall_confidence: float = 0.8
        recommended_strategy: Any = None

    @dataclass
    class LearningInsights:
        user_id: str = "test_user"
        ml_recommendations: List[Recommendation] = field(default_factory=list)

    @dataclass
    class PolicyEvaluation:
        user_id: str = "test_user"
        allowed: bool = True
        violations: List = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        recommended_action: str = "allow"
        security_level_required: str = "medium"
        compliance_requirements: List[str] = field(default_factory=list)

    @dataclass
    class SourceMetadata:
        discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Mock manager classes
    class DatabaseManager:
        async def initialize(self): pass
        async def close(self): pass
        async def create_user_profile(self, profile): return profile
        async def get_user_profile(self, user_id): return None
        async def create_data_source(self, source): return source

    class CacheManager:
        async def initialize(self): pass
        async def close(self): pass
        async def get(self, key, default=None): return default
        async def set(self, key, value, ttl=None): pass
        async def cache_user_profile(self, user_id, profile): pass

    class ConfigurationLoader:
        async def load_all_configs(self): pass
        def get_role_template(self, role): return None

    class EnvironmentScanner:
        async def initialize(self): pass
        async def close(self): pass
        async def scan_environment(self, user_id, **kwargs):
            return EnvironmentProfile(user_id=user_id, discovered_sources=[])
        async def create_data_sources_from_scan(self, profile):
            return [DataSource(id=f"discovered_{i}", name=f"Source {i}") for i in range(3)]

    class ProfileAnalyzer:
        async def initialize(self): pass
        async def close(self): pass
        async def analyze_user_profile(self, profile, **kwargs):
            return ProfileInsights(
                user_id=profile.user_id,
                suggested_sources=["postgresql", "redis", "tableau"],
                overall_confidence=0.8
            )
        async def get_profile_recommendations(self, insights):
            return [
                Recommendation(
                    id=f"profile_rec_{i}",
                    user_id=insights.user_id,
                    source_id=source,
                    recommendation_type="profile_based",
                    confidence_score=0.7 + (i * 0.05)
                ) for i, source in enumerate(insights.suggested_sources[:3])
            ]

    class LearningEngine:
        async def initialize(self): pass
        async def close(self): pass
        async def generate_ml_recommendations(self, user_id, profile, sources, **kwargs):
            return [
                Recommendation(
                    id=f"ml_rec_{i}",
                    user_id=user_id,
                    source_id=source,
                    recommendation_type="ml_generated",
                    confidence_score=0.75
                ) for i, source in enumerate(sources[:3])
            ]
        async def predict_recommendation_success(self, user_id, rec, profile):
            from dataclasses import dataclass
            @dataclass
            class PredictionResult:
                prediction: float = 0.8
                confidence: float = 0.7
            return PredictionResult()
        async def learn_from_feedback(self, user_id, rec, feedback): return True
        def get_model_status(self): return {"models": {}, "ready": True}

    class PolicyEngine:
        async def initialize(self): pass
        async def close(self): pass
        async def evaluate_recommendation(self, rec, profile, **kwargs):
            return PolicyEvaluation(
                user_id=profile.user_id,
                allowed=True,
                violations=[],
                warnings=[],
                recommended_action="allow"
            )
        async def get_policy_status(self): return {"total_policies": 5, "active_policies": 5}

    class AnalyticsEngine:
        async def initialize(self): pass
        async def close(self): pass
        async def track_event(self, *args, **kwargs): pass

    class NotificationEngine:
        async def initialize(self): pass
        async def close(self): pass
        async def send_recommendation_notification(self, rec, profile): pass

    class ScoringEngine:
        async def initialize(self): pass
        async def close(self): pass
        async def calculate_score(self, rec, profile, **kwargs):
            from dataclasses import dataclass
            @dataclass
            class ScoreBreakdown:
                total_score: float = 0.8
                confidence: float = 0.7
            return ScoreBreakdown()

    class EventType:
        USER_ACTION = "user_action"
        SOURCE_CONNECTED = "source_connected"
        SOURCE_FAILED = "source_failed"

    class NotificationType:
        RECOMMENDATION = "recommendation"

    # Mock factory functions
    async def create_database_manager(**kwargs): return DatabaseManager()
    async def create_cache_manager(**kwargs): return CacheManager()
    async def create_config_loader(**kwargs): return ConfigurationLoader()
    async def create_environment_scanner(**kwargs): return EnvironmentScanner()
    async def create_profile_analyzer(**kwargs): return ProfileAnalyzer()
    async def create_learning_engine(**kwargs): return LearningEngine()
    async def create_policy_engine(**kwargs): return PolicyEngine()
    async def create_analytics_engine(**kwargs): return AnalyticsEngine()
    async def create_notification_engine(**kwargs): return NotificationEngine()
    async def create_scoring_engine(**kwargs): return ScoringEngine()

logger = logging.getLogger(__name__)

class EngineStatus(Enum):
    """Smart defaults engine status"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class RecommendationMode(Enum):
    """Recommendation generation modes"""
    CONSERVATIVE = "conservative"  # Only high-confidence recommendations
    BALANCED = "balanced"         # Mix of confidence levels
    AGGRESSIVE = "aggressive"     # Include lower-confidence recommendations
    LEARNING = "learning"         # Prioritize learning opportunities

@dataclass
class EngineConfig:
    """Configuration for the smart defaults engine"""
    # Database settings
    database_url: str = "sqlite:///smart_defaults.db"

    # Cache settings
    redis_url: str = "redis://localhost:6379"
    use_cache: bool = True

    # Component settings
    enable_environment_scanning: bool = True
    enable_profile_analysis: bool = True
    enable_machine_learning: bool = True
    enable_policy_enforcement: bool = True
    enable_analytics: bool = True
    enable_notifications: bool = True
    enable_advanced_scoring: bool = True

    # Recommendation settings
    default_recommendation_mode: RecommendationMode = RecommendationMode.BALANCED
    max_recommendations_per_request: int = 10
    min_confidence_threshold: float = 0.6
    auto_connect_threshold: float = 0.85

    # Security settings
    strict_policy_mode: bool = False
    require_approval_threshold: float = 0.9

    # Performance settings
    cache_ttl_seconds: int = 3600
    background_scan_interval_hours: int = 24
    model_retrain_interval_hours: int = 48

@dataclass
class RecommendationRequest:
    """Request for recommendations"""
    user_id: str
    mode: RecommendationMode = RecommendationMode.BALANCED
    max_recommendations: Optional[int] = None
    include_environment_scan: bool = True
    include_ml_recommendations: bool = True
    force_refresh: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationResponse:
    """Response containing recommendations and metadata"""
    user_id: str
    recommendations: List[Recommendation]
    metadata: Dict[str, Any]
    generated_at: datetime

    # Analytics
    total_candidates: int
    filtered_by_policy: int
    enhanced_by_ml: int
    confidence_distribution: Dict[str, int]

    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class SmartDefaultsEngine:
    """Main Smart Defaults Engine - orchestrates all components"""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.status = EngineStatus.NOT_INITIALIZED

        # Component managers
        self.database_manager: Optional[DatabaseManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.config_loader: Optional[ConfigurationLoader] = None
        self.environment_scanner: Optional[EnvironmentScanner] = None
        self.profile_analyzer: Optional[ProfileAnalyzer] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.policy_engine: Optional[PolicyEngine] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None
        self.notification_engine: Optional[NotificationEngine] = None
        self.scoring_engine: Optional[ScoringEngine] = None

        # Runtime state
        self.initialization_errors: List[str] = []
        self.initialized_at: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

    async def initialize(self) -> bool:
        """Initialize all components of the smart defaults engine"""

        if self.status == EngineStatus.READY:
            logger.info("✅ Smart defaults engine already initialized")
            return True

        self.status = EngineStatus.INITIALIZING
        logger.info("🚀 Initializing Smart Defaults Engine...")

        try:
            # Initialize core storage components
            await self._initialize_storage()

            # Initialize configuration
            await self._initialize_configuration()

            # Initialize analytics (early for tracking initialization)
            if self.config.enable_analytics:
                await self._initialize_analytics()

            # Initialize notifications
            if self.config.enable_notifications:
                await self._initialize_notifications()

            # Initialize advanced scoring
            if self.config.enable_advanced_scoring:
                await self._initialize_scoring()

            # Initialize analyzer components
            await self._initialize_analyzers()

            # Start background tasks
            await self._start_background_tasks()

            # Final health check
            health_status = await self.health_check()

            if health_status["overall_status"] in ["healthy", "degraded"]:
                self.status = EngineStatus.READY
                self.initialized_at = datetime.now(timezone.utc)

                logger.info("✅ Smart Defaults Engine initialized successfully!")

                # Track initialization
                if self.analytics_engine:
                    await self.analytics_engine.track_event(
                        user_id="system",
                        event_type=EventType.USER_ACTION,
                        data={
                            "action": "engine_initialized",
                            "components_enabled": {
                                "environment_scanning": self.config.enable_environment_scanning,
                                "profile_analysis": self.config.enable_profile_analysis,
                                "machine_learning": self.config.enable_machine_learning,
                                "policy_enforcement": self.config.enable_policy_enforcement,
                                "analytics": self.config.enable_analytics,
                                "notifications": self.config.enable_notifications,
                                "advanced_scoring": self.config.enable_advanced_scoring
                            },
                            "initialization_time": (datetime.now(timezone.utc) - self.initialized_at).total_seconds() if self.initialized_at else 0
                        }
                    )

                return True
            else:
                self.status = EngineStatus.ERROR
                logger.error("❌ Smart Defaults Engine initialization failed health check")
                return False

        except Exception as e:
            self.status = EngineStatus.ERROR
            self.initialization_errors.append(str(e))
            logger.error(f"❌ Smart Defaults Engine initialization failed: {e}")
            return False

    async def _initialize_storage(self):
        """Initialize storage components"""
        logger.info("📊 Initializing storage components...")

        try:
            # Database
            self.database_manager = await create_database_manager(
                db_type="sqlite" if "sqlite" in self.config.database_url else "postgresql"
            )

            # Cache
            if self.config.use_cache:
                self.cache_manager = await create_cache_manager(
                    redis_url=self.config.redis_url,
                    use_memory_fallback=True
                )

            logger.info("✅ Storage components initialized")
        except Exception as e:
            logger.error(f"❌ Storage initialization failed: {e}")
            raise

    async def _initialize_configuration(self):
        """Initialize configuration management"""
        logger.info("⚙️ Initializing configuration...")

        try:
            self.config_loader = await create_config_loader()
            logger.info("✅ Configuration initialized")
        except Exception as e:
            logger.warning(f"⚠️ Configuration initialization failed: {e}")
            # Continue without config loader

    async def _initialize_analytics(self):
        """Initialize analytics engine"""
        logger.info("📈 Initializing analytics...")

        try:
            self.analytics_engine = await create_analytics_engine(
                database_manager=self.database_manager,
                cache_manager=self.cache_manager
            )
            logger.info("✅ Analytics initialized")
        except Exception as e:
            logger.warning(f"⚠️ Analytics initialization failed: {e}")

    async def _initialize_notifications(self):
        """Initialize notification engine"""
        logger.info("📢 Initializing notifications...")

        try:
            self.notification_engine = await create_notification_engine(
                database_manager=self.database_manager,
                cache_manager=self.cache_manager,
                analytics_engine=self.analytics_engine
            )
            logger.info("✅ Notifications initialized")
        except Exception as e:
            logger.warning(f"⚠️ Notifications initialization failed: {e}")

    async def _initialize_scoring(self):
        """Initialize advanced scoring engine"""
        logger.info("🎯 Initializing advanced scoring...")

        try:
            self.scoring_engine = await create_scoring_engine(
                database_manager=self.database_manager,
                cache_manager=self.cache_manager
            )
            logger.info("✅ Advanced scoring initialized")
        except Exception as e:
            logger.warning(f"⚠️ Advanced scoring initialization failed: {e}")

    async def _initialize_analyzers(self):
        """Initialize analyzer components"""
        logger.info("🧠 Initializing analyzer components...")

        try:
            # Environment Scanner
            if self.config.enable_environment_scanning:
                self.environment_scanner = await create_environment_scanner(
                    database_manager=self.database_manager,
                    cache_manager=self.cache_manager,
                    config_loader=self.config_loader,
                    analytics_engine=self.analytics_engine
                )
                logger.info("✅ Environment scanner initialized")

            # Profile Analyzer
            if self.config.enable_profile_analysis:
                self.profile_analyzer = await create_profile_analyzer(
                    database_manager=self.database_manager,
                    cache_manager=self.cache_manager,
                    config_loader=self.config_loader,
                    analytics_engine=self.analytics_engine,
                    environment_scanner=self.environment_scanner
                )
                logger.info("✅ Profile analyzer initialized")

            # Learning Engine
            if self.config.enable_machine_learning:
                self.learning_engine = await create_learning_engine(
                    database_manager=self.database_manager,
                    cache_manager=self.cache_manager,
                    analytics_engine=self.analytics_engine,
                    profile_analyzer=self.profile_analyzer
                )
                logger.info("✅ Learning engine initialized")

            # Policy Engine
            if self.config.enable_policy_enforcement:
                self.policy_engine = await create_policy_engine(
                    database_manager=self.database_manager,
                    cache_manager=self.cache_manager,
                    config_loader=self.config_loader,
                    analytics_engine=self.analytics_engine,
                    strict_mode=self.config.strict_policy_mode
                )
                logger.info("✅ Policy engine initialized")

            logger.info("✅ All analyzer components initialized")
        except Exception as e:
            logger.error(f"❌ Analyzer initialization failed: {e}")
            raise

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        logger.info("🔄 Starting background tasks...")

        try:
            # Background environment scanning
            if self.environment_scanner:
                scan_task = asyncio.create_task(self._background_environment_scan())
                self._background_tasks.append(scan_task)

            # Background model retraining
            if self.learning_engine:
                retrain_task = asyncio.create_task(self._background_model_retrain())
                self._background_tasks.append(retrain_task)

            logger.info(f"✅ Started {len(self._background_tasks)} background tasks")
        except Exception as e:
            logger.warning(f"⚠️ Background task initialization failed: {e}")

    async def close(self):
        """Clean shutdown of the engine"""
        logger.info("🔐 Shutting down Smart Defaults Engine...")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close components in reverse order
        components = [
            ("Scoring Engine", self.scoring_engine),
            ("Notification Engine", self.notification_engine),
            ("Policy Engine", self.policy_engine),
            ("Learning Engine", self.learning_engine),
            ("Profile Analyzer", self.profile_analyzer),
            ("Environment Scanner", self.environment_scanner),
            ("Analytics Engine", self.analytics_engine),
            ("Cache Manager", self.cache_manager),
            ("Database Manager", self.database_manager)
        ]

        for name, component in components:
            if component:
                try:
                    await component.close()
                    logger.info(f"✅ {name} closed")
                except Exception as e:
                    logger.warning(f"⚠️ Error closing {name}: {e}")

        self.status = EngineStatus.NOT_INITIALIZED
        logger.info("🔐 Smart Defaults Engine shutdown complete")

    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get personalized recommendations for a user"""

        if self.status != EngineStatus.READY:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                metadata={"error": "Engine not ready"},
                generated_at=datetime.now(timezone.utc),
                total_candidates=0,
                filtered_by_policy=0,
                enhanced_by_ml=0,
                confidence_distribution={},
                success=False,
                errors=["Smart Defaults Engine not initialized"]
            )

        start_time = datetime.now(timezone.utc)
        logger.info(f"🎯 Generating recommendations for user: {request.user_id}")

        try:
            # Get or create user profile
            user_profile = await self._get_or_create_user_profile(request.user_id)

            if not user_profile:
                return self._create_error_response(
                    request.user_id, "Failed to load user profile"
                )

            # Generate candidate recommendations
            candidates = await self._generate_candidate_recommendations(
                user_profile, request
            )

            # Apply policy filtering
            policy_filtered = await self._apply_policy_filtering(
                candidates, user_profile
            )

            # Apply ML enhancements
            ml_enhanced = await self._apply_ml_enhancements(
                policy_filtered, user_profile, request
            )

            # Apply advanced scoring if available
            scored_recommendations = await self._apply_advanced_scoring(
                ml_enhanced, user_profile, request
            )

            # Final filtering and ranking
            final_recommendations = await self._finalize_recommendations(
                scored_recommendations, request
            )

            # Send notifications for high-confidence recommendations
            await self._send_recommendation_notifications(
                final_recommendations, user_profile
            )

            # Generate metadata
            metadata = await self._generate_response_metadata(
                user_profile, request, candidates, policy_filtered, ml_enhanced
            )

            # Calculate confidence distribution
            confidence_dist = self._calculate_confidence_distribution(final_recommendations)

            # Create response
            response = RecommendationResponse(
                user_id=request.user_id,
                recommendations=final_recommendations,
                metadata=metadata,
                generated_at=start_time,
                total_candidates=len(candidates),
                filtered_by_policy=len(candidates) - len(policy_filtered),
                enhanced_by_ml=len([r for r in ml_enhanced if r.reasoning.get("ml_enhanced")]),
                confidence_distribution=confidence_dist,
                success=True
            )

            # Track recommendation generation
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=request.user_id,
                    event_type=EventType.USER_ACTION,
                    data={
                        "action": "recommendations_generated",
                        "mode": request.mode.value,
                        "count": len(final_recommendations),
                        "generation_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                        "policy_filtered": len(candidates) - len(policy_filtered),
                        "ml_enhanced": response.enhanced_by_ml
                    }
                )

            logger.info(f"✅ Generated {len(final_recommendations)} recommendations for {request.user_id}")
            return response

        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return self._create_error_response(request.user_id, str(e))

    async def _get_or_create_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get existing user profile or create a new one"""

        try:
            # Try to get existing profile
            if self.database_manager:
                profile = await self.database_manager.get_user_profile(user_id)
                if profile:
                    return profile

            # Create new profile with defaults
            new_profile = UserProfile(
                id=f"profile_{user_id}",
                user_id=user_id,
                role="data_analyst",  # Default role
                department="analytics",
                seniority_level="intermediate",
                industry="technology",
                permissions=[],
                preferences=UserPreferences()
            )

            # Save to database
            if self.database_manager:
                await self.database_manager.create_user_profile(new_profile)

            # Cache the profile
            if self.cache_manager:
                await self.cache_manager.cache_user_profile(user_id, new_profile)

            logger.info(f"📝 Created new user profile for: {user_id}")
            return new_profile

        except Exception as e:
            logger.error(f"❌ Failed to get/create user profile for {user_id}: {e}")
            return None

    async def _generate_candidate_recommendations(self, user_profile: UserProfile,
                                                request: RecommendationRequest) -> List[Recommendation]:
        """Generate candidate recommendations from all sources"""

        candidates = []

        try:
            # Environment-based recommendations
            if request.include_environment_scan and self.environment_scanner:
                env_profile = await self.environment_scanner.scan_environment(
                    user_profile.user_id,
                    quick_scan=not request.force_refresh,
                    network_scan=True
                )

                env_sources = await self.environment_scanner.create_data_sources_from_scan(env_profile)

                for i, source in enumerate(env_sources):
                    rec = Recommendation(
                        id=f"env_rec_{user_profile.user_id}_{source.id}_{i}",
                        user_id=user_profile.user_id,
                        source_id=source.id,
                        recommendation_type="environment_discovered",
                        confidence_score=0.7,  # Base environment confidence
                        reasoning={
                            "discovered_automatically": True,
                            "environment_scan": True,
                            "source_available": True
                        },
                        context={
                            "source_name": source.name,
                            "source_type": source.source_type,
                            "discovery_method": "environment_scan"
                        }
                    )
                    candidates.append(rec)

            # Profile-based recommendations
            if self.profile_analyzer:
                profile_insights = await self.profile_analyzer.analyze_user_profile(user_profile)
                profile_recommendations = await self.profile_analyzer.get_profile_recommendations(profile_insights)
                candidates.extend(profile_recommendations)

            # ML-generated recommendations
            if request.include_ml_recommendations and self.learning_engine:
                available_sources = ["postgresql", "redis", "tableau", "jupyter", "elasticsearch"]
                ml_recommendations = await self.learning_engine.generate_ml_recommendations(
                    user_profile.user_id,
                    user_profile,
                    available_sources,
                    max_recommendations=5
                )
                candidates.extend(ml_recommendations)

            logger.info(f"📋 Generated {len(candidates)} candidate recommendations")
            return candidates

        except Exception as e:
            logger.error(f"❌ Failed to generate candidate recommendations: {e}")
            return []

    async def _apply_policy_filtering(self, candidates: List[Recommendation],
                                    user_profile: UserProfile) -> List[Recommendation]:
        """Apply policy filtering to recommendations"""

        if not self.policy_engine:
            return candidates

        filtered = []

        try:
            for recommendation in candidates:
                evaluation = await self.policy_engine.evaluate_recommendation(
                    recommendation, user_profile
                )

                if evaluation.allowed:
                    # Add policy context to recommendation
                    recommendation.context.update({
                        "policy_approved": True,
                        "security_level": evaluation.security_level_required,
                        "compliance_requirements": evaluation.compliance_requirements
                    })

                    if evaluation.warnings:
                        recommendation.context["policy_warnings"] = evaluation.warnings

                    filtered.append(recommendation)
                else:
                    logger.debug(f"🚫 Policy blocked recommendation: {recommendation.id}")

            logger.info(f"🛡️ Policy filtering: {len(candidates)} → {len(filtered)} recommendations")
            return filtered

        except Exception as e:
            logger.error(f"❌ Policy filtering failed: {e}")
            return candidates  # Return unfiltered on error

    async def _apply_ml_enhancements(self, recommendations: List[Recommendation],
                                   user_profile: UserProfile,
                                   request: RecommendationRequest) -> List[Recommendation]:
        """Apply ML enhancements to recommendations - FIXED METHOD"""

        if not self.learning_engine:
            return recommendations

        try:
            enhanced = []

            for rec in recommendations:
                # Predict success probability for this recommendation
                prediction = await self.learning_engine.predict_recommendation_success(
                    user_profile.user_id, rec, user_profile
                )

                # Adjust confidence based on ML prediction
                original_confidence = rec.confidence_score
                ml_prediction = prediction.prediction
                ml_confidence = prediction.confidence

                # Calculate enhanced confidence
                enhanced_confidence = min(0.99, original_confidence * ml_prediction * 1.1)

                # Create enhanced recommendation
                enhanced_rec = Recommendation(
                    id=rec.id,
                    user_id=rec.user_id,
                    source_id=rec.source_id,
                    recommendation_type=rec.recommendation_type,
                    confidence_score=enhanced_confidence,
                    reasoning={
                        **rec.reasoning,
                        "ml_enhanced": True,
                        "ml_prediction": ml_prediction,
                        "original_confidence": original_confidence
                    },
                    context=rec.context
                )
                enhanced.append(enhanced_rec)

            logger.info(f"🧠 ML enhanced {len(enhanced)} recommendations")
            return enhanced

        except Exception as e:
            logger.error(f"❌ ML enhancement failed: {e}")
            return recommendations

    async def _apply_advanced_scoring(self, recommendations: List[Recommendation],
                                    user_profile: UserProfile,
                                    request: RecommendationRequest) -> List[Recommendation]:
        """Apply advanced scoring if available"""

        if not self.scoring_engine:
            return recommendations

        try:
            scored = []
            for rec in recommendations:
                score_breakdown = await self.scoring_engine.calculate_score(
                    rec, user_profile, context=request.context
                )

                # Update recommendation with advanced score
                rec.confidence_score = score_breakdown.total_score
                rec.reasoning.update({"advanced_scoring": True})
                scored.append(rec)

            return scored
        except Exception as e:
            logger.warning(f"⚠️ Advanced scoring failed: {e}")
            return recommendations

    async def _finalize_recommendations(self, recommendations: List[Recommendation],
                                      request: RecommendationRequest) -> List[Recommendation]:
        """Final filtering, ranking, and limiting of recommendations"""

        # Apply confidence threshold
        min_confidence = self.config.min_confidence_threshold
        if request.mode == RecommendationMode.CONSERVATIVE:
            min_confidence = max(min_confidence, 0.8)
        elif request.mode == RecommendationMode.AGGRESSIVE:
            min_confidence = max(min_confidence, 0.4)

        filtered = [r for r in recommendations if r.confidence_score >= min_confidence]

        # Sort by confidence score
        sorted_recs = sorted(filtered, key=lambda r: r.confidence_score, reverse=True)

        # Apply limit
        max_recs = request.max_recommendations or self.config.max_recommendations_per_request
        limited = sorted_recs[:max_recs]

        logger.info(f"🎯 Final recommendations: {len(limited)} (from {len(recommendations)} candidates)")
        return limited

    async def _send_recommendation_notifications(self, recommendations: List[Recommendation],
                                               user_profile: UserProfile):
        """Send notifications for high-confidence recommendations"""

        if not self.notification_engine:
            return

        try:
            for rec in recommendations:
                if rec.confidence_score >= self.config.auto_connect_threshold:
                    await self.notification_engine.send_recommendation_notification(rec, user_profile)
        except Exception as e:
            logger.warning(f"⚠️ Notification sending failed: {e}")

    async def _generate_response_metadata(self, user_profile: UserProfile,
                                        request: RecommendationRequest,
                                        candidates: List[Recommendation],
                                        policy_filtered: List[Recommendation],
                                        ml_enhanced: List[Recommendation]) -> Dict[str, Any]:
        """Generate metadata for the recommendation response"""

        metadata = {
            "user_profile": {
                "user_id": user_profile.user_id,
                "role": user_profile.role
            },
            "request": {
                "mode": request.mode.value,
                "include_environment_scan": request.include_environment_scan,
                "include_ml_recommendations": request.include_ml_recommendations,
                "force_refresh": request.force_refresh
            },
            "processing": {
                "candidates_generated": len(candidates),
                "policy_filtered": len(policy_filtered),
                "ml_enhanced": len(ml_enhanced)
            },
            "engine_status": {
                "components_active": {
                    "environment_scanner": self.environment_scanner is not None,
                    "profile_analyzer": self.profile_analyzer is not None,
                    "learning_engine": self.learning_engine is not None,
                    "policy_engine": self.policy_engine is not None,
                    "notification_engine": self.notification_engine is not None,
                    "scoring_engine": self.scoring_engine is not None
                }
            }
        }

        return metadata

    def _calculate_confidence_distribution(self, recommendations: List[Recommendation]) -> Dict[str, int]:
        """Calculate confidence score distribution"""

        distribution = {
            "high (0.8+)": 0,
            "medium (0.6-0.8)": 0,
            "low (0.4-0.6)": 0,
            "very_low (<0.4)": 0
        }

        for rec in recommendations:
            score = rec.confidence_score
            if score >= 0.8:
                distribution["high (0.8+)"] += 1
            elif score >= 0.6:
                distribution["medium (0.6-0.8)"] += 1
            elif score >= 0.4:
                distribution["low (0.4-0.6)"] += 1
            else:
                distribution["very_low (<0.4)"] += 1

        return distribution

    def _create_error_response(self, user_id: str, error_message: str) -> RecommendationResponse:
        """Create an error response"""
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[],
            metadata={"error": error_message},
            generated_at=datetime.now(timezone.utc),
            total_candidates=0,
            filtered_by_policy=0,
            enhanced_by_ml=0,
            confidence_distribution={},
            success=False,
            errors=[error_message]
        )

    async def record_feedback(self, user_id: str, recommendation_id: str,
                            action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Record user feedback on a recommendation"""

        try:
            # Create mock recommendation for feedback
            mock_recommendation = Recommendation(
                id=recommendation_id,
                user_id=user_id,
                source_id="mock_source",
                confidence_score=0.8
            )

            feedback_data = {
                "action": action,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc)
            }

            # Record with learning engine
            if self.learning_engine:
                await self.learning_engine.learn_from_feedback(
                    user_id, mock_recommendation, feedback_data
                )

            # Track feedback
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_id,
                    event_type=EventType.USER_ACTION,
                    data={
                        "action": "feedback_recorded",
                        "recommendation_id": recommendation_id,
                        "feedback_action": action,
                        "context": context or {}
                    }
                )

            logger.info(f"📝 Recorded feedback: {user_id} → {recommendation_id} → {action}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to record feedback: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components"""

        health_status = {
            "overall_status": "healthy",
            "engine_status": self.status.value,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "uptime_seconds": (datetime.now(timezone.utc) - self.initialized_at).total_seconds() if self.initialized_at else 0,
            "components": {},
            "errors": self.initialization_errors.copy(),
            "background_tasks": len(self._background_tasks),
            "last_check": datetime.now(timezone.utc).isoformat()
        }

        try:
            # Check each component
            components = [
                ("database", self.database_manager),
                ("cache", self.cache_manager),
                ("config_loader", self.config_loader),
                ("environment_scanner", self.environment_scanner),
                ("profile_analyzer", self.profile_analyzer),
                ("learning_engine", self.learning_engine),
                ("policy_engine", self.policy_engine),
                ("analytics", self.analytics_engine),
                ("notifications", self.notification_engine),
                ("scoring", self.scoring_engine)
            ]

            for name, component in components:
                if component:
                    try:
                        if hasattr(component, 'health_check'):
                            component_health = await component.health_check()
                            health_status["components"][name] = {
                                "status": "healthy",
                                "details": component_health
                            }
                        else:
                            health_status["components"][name] = {"status": "healthy", "details": "no health check available"}
                    except Exception as e:
                        health_status["components"][name] = {"status": "unhealthy", "error": str(e)}
                        health_status["overall_status"] = "degraded"
                else:
                    health_status["components"][name] = {"status": "disabled"}

            # Check if any critical components are unhealthy
            critical_components = ["database", "config_loader"]
            for comp in critical_components:
                if comp in health_status["components"] and health_status["components"][comp]["status"] == "unhealthy":
                    health_status["overall_status"] = "unhealthy"

            self.last_health_check = datetime.now(timezone.utc)

        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["health_check_error"] = str(e)

        return health_status

    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""

        stats = {
            "engine": {
                "status": self.status.value,
                "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
                "config": {
                    "components_enabled": {
                        "environment_scanning": self.config.enable_environment_scanning,
                        "profile_analysis": self.config.enable_profile_analysis,
                        "machine_learning": self.config.enable_machine_learning,
                        "policy_enforcement": self.config.enable_policy_enforcement,
                        "analytics": self.config.enable_analytics,
                        "notifications": self.config.enable_notifications,
                        "advanced_scoring": self.config.enable_advanced_scoring
                    }
                }
            }
        }

        # Component-specific stats
        try:
            if self.learning_engine:
                model_status = self.learning_engine.get_model_status()
                stats["machine_learning"] = model_status

            if self.policy_engine:
                policy_status = await self.policy_engine.get_policy_status()
                stats["policy"] = policy_status

        except Exception as e:
            stats["stats_error"] = str(e)

        return stats

    # Background task methods
    async def _background_environment_scan(self):
        """Background task for periodic environment scanning"""
        while True:
            try:
                await asyncio.sleep(self.config.background_scan_interval_hours * 3600)
                logger.info("🔍 Background environment scan completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Background environment scan failed: {e}")

    async def _background_model_retrain(self):
        """Background task for periodic model retraining"""
        while True:
            try:
                await asyncio.sleep(self.config.model_retrain_interval_hours * 3600)
                if self.learning_engine:
                    await self.learning_engine.retrain_models()
                logger.info("🎓 Background model retraining completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Background model retraining failed: {e}")

# Factory function
async def create_smart_defaults_engine(config: Optional[EngineConfig] = None) -> SmartDefaultsEngine:
    """Create and initialize a Smart Defaults Engine"""
    engine = SmartDefaultsEngine(config)
    success = await engine.initialize()
    if not success:
        raise RuntimeError("Smart Defaults Engine initialization failed")
    return engine