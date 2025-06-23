"""
Smart Defaults Monitoring and Analytics System
Tracks user behavior, measures recommendation effectiveness, and provides insights
File location: smart_defaults/utils/monitoring.py
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from enum import Enum
import statistics
import uuid

# Import dependencies with fallbacks
try:
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..models.user_profile import UserProfile
    from ..models.recommendation import Recommendation
    from ..models.data_source import DataSource
except ImportError:
    # For direct execution, create mock classes
    from typing import Any
    from dataclasses import dataclass

    @dataclass
    class UserProfile:
        id: str = "test_id"
        user_id: str = "test_user"
        role: str = "test_role"

    @dataclass
    class Recommendation:
        id: str = "test_rec_id"
        user_id: str = "test_user"
        confidence_score: float = 0.8

    @dataclass
    class DataSource:
        id: str = "test_source_id"
        name: str = "Test Source"

    class DatabaseManager:
        async def initialize(self): pass
        async def close(self): pass
        async def record_user_behavior(self, *args, **kwargs): pass
        async def record_user_feedback(self, *args, **kwargs): pass

    class CacheManager:
        async def initialize(self): pass
        async def close(self): pass
        async def set(self, key, value, ttl=None): pass
        async def get(self, key, default=None): return default
        async def increment(self, key, amount=1): return amount

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of analytics events"""
    USER_ACTION = "user_action"
    RECOMMENDATION_SHOWN = "recommendation_shown"
    RECOMMENDATION_ACCEPTED = "recommendation_accepted"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    SOURCE_CONNECTED = "source_connected"
    SOURCE_FAILED = "source_failed"
    PREFERENCE_CHANGED = "preference_changed"
    ERROR_OCCURRED = "error_occurred"
    SESSION_START = "session_start"
    SESSION_END = "session_end"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class AnalyticsEvent:
    """Individual analytics event"""
    id: str
    user_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    source_id: Optional[str] = None
    recommendation_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

@dataclass
class MetricValue:
    """Metric measurement"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

@dataclass
class UserBehaviorSummary:
    """Summary of user behavior patterns"""
    user_id: str
    total_events: int
    recommendation_acceptance_rate: float
    most_common_actions: List[Tuple[str, int]]
    source_usage_patterns: Dict[str, int]
    session_duration_avg: float
    last_activity: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RecommendationMetrics:
    """Metrics for recommendation effectiveness"""
    total_shown: int
    total_accepted: int
    total_rejected: int
    acceptance_rate: float
    confidence_vs_acceptance: List[Tuple[float, bool]]
    source_effectiveness: Dict[str, float]
    user_segment_performance: Dict[str, float]
    time_to_decision_avg: float

@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    total_users: int
    active_users_24h: int
    total_recommendations: int
    overall_acceptance_rate: float
    avg_session_duration: float
    error_rate: float
    source_health_scores: Dict[str, float]
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class AnalyticsEngine:
    """Main analytics engine for smart defaults system"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 batch_size: int = 100,
                 flush_interval: int = 60):
        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # In-memory storage for batching
        self.event_buffer: List[AnalyticsEvent] = []
        self.metric_buffer: List[MetricValue] = []

        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Real-time counters (will be persisted periodically)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}

        self._initialized = False
        self._flush_task = None

    async def initialize(self):
        """Initialize analytics engine"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()

        # Start background flush task
        self._flush_task = asyncio.create_task(self._background_flush())

        self._initialized = True
        logger.info("✅ Analytics engine initialized")

    async def close(self):
        """Close analytics engine and flush remaining data"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining data
        await self._flush_buffers()

        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Analytics engine closed")

    async def track_event(self,
                         user_id: str,
                         event_type: EventType,
                         data: Optional[Dict[str, Any]] = None,
                         session_id: Optional[str] = None,
                         source_id: Optional[str] = None,
                         recommendation_id: Optional[str] = None) -> str:
        """Track an analytics event"""

        event = AnalyticsEvent(
            id=str(uuid.uuid4()),
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data or {},
            session_id=session_id,
            source_id=source_id,
            recommendation_id=recommendation_id
        )

        # Add to buffer
        self.event_buffer.append(event)

        # Update real-time counters
        self._update_counters(event)

        # Flush if buffer is full
        if len(self.event_buffer) >= self.batch_size:
            await self._flush_events()

        logger.debug(f"📊 Tracked event: {event_type.value} for user {user_id}")
        return event.id

    async def record_metric(self,
                          name: str,
                          value: Union[int, float],
                          metric_type: MetricType,
                          tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""

        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )

        self.metric_buffer.append(metric)

        # Update in-memory metrics
        if metric_type == MetricType.COUNTER:
            self.counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = value

        # Cache recent metrics
        if self.cache_manager:
            cache_key = f"metric:{name}:{int(metric.timestamp.timestamp())}"
            await self.cache_manager.set(cache_key, asdict(metric), ttl=3600)

        # Flush if buffer is full
        if len(self.metric_buffer) >= self.batch_size:
            await self._flush_metrics()

    async def start_session(self, user_id: str, session_data: Optional[Dict[str, Any]] = None) -> str:
        """Start a user session"""
        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(timezone.utc),
            'data': session_data or {},
            'event_count': 0
        }

        await self.track_event(
            user_id=user_id,
            event_type=EventType.SESSION_START,
            session_id=session_id,
            data=session_data or {}
        )

        return session_id

    async def end_session(self, session_id: str, end_data: Optional[Dict[str, Any]] = None):
        """End a user session"""
        if session_id not in self.active_sessions:
            logger.warning(f"⚠️ Attempted to end unknown session: {session_id}")
            return

        session = self.active_sessions[session_id]
        end_time = datetime.now(timezone.utc)
        duration = (end_time - session['start_time']).total_seconds()

        await self.track_event(
            user_id=session['user_id'],
            event_type=EventType.SESSION_END,
            session_id=session_id,
            data={
                'duration_seconds': duration,
                'event_count': session['event_count'],
                **(end_data or {})
            }
        )

        await self.record_metric(
            name="session_duration",
            value=duration,
            metric_type=MetricType.TIMER,
            tags={'user_id': session['user_id']}
        )

        del self.active_sessions[session_id]

    async def track_recommendation_shown(self,
                                       user_id: str,
                                       recommendation: Recommendation,
                                       context: Optional[Dict[str, Any]] = None):
        """Track when a recommendation is shown to user"""
        await self.track_event(
            user_id=user_id,
            event_type=EventType.RECOMMENDATION_SHOWN,
            recommendation_id=recommendation.id,
            source_id=getattr(recommendation, 'source_id', None),
            data={
                'confidence_score': getattr(recommendation, 'confidence_score', 0.0),
                'recommendation_type': getattr(recommendation, 'recommendation_type', 'unknown'),
                'context': context or {}
            }
        )

        self.counters['recommendations_shown'] += 1

    async def track_recommendation_feedback(self,
                                          user_id: str,
                                          recommendation_id: str,
                                          accepted: bool,
                                          feedback_data: Optional[Dict[str, Any]] = None):
        """Track user feedback on a recommendation"""
        event_type = EventType.RECOMMENDATION_ACCEPTED if accepted else EventType.RECOMMENDATION_REJECTED

        await self.track_event(
            user_id=user_id,
            event_type=event_type,
            recommendation_id=recommendation_id,
            data=feedback_data or {}
        )

        if accepted:
            self.counters['recommendations_accepted'] += 1
        else:
            self.counters['recommendations_rejected'] += 1

        # Record to database for long-term storage
        if self.database_manager:
            try:
                await self.database_manager.record_user_feedback(
                    user_id=user_id,
                    recommendation_id=recommendation_id,
                    action='accept' if accepted else 'reject',
                    context=feedback_data or {}
                )
            except Exception as e:
                logger.error(f"❌ Failed to record feedback to database: {e}")

    async def get_user_behavior_summary(self, user_id: str, days: int = 30) -> Optional[UserBehaviorSummary]:
        """Get behavior summary for a user"""
        try:
            # Try cache first
            cache_key = f"user_behavior_summary:{user_id}:{days}"
            if self.cache_manager:
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    return UserBehaviorSummary(**cached)

            # Calculate from events (in production, this would query the database)
            user_events = [e for e in self.event_buffer if e.user_id == user_id]

            if not user_events:
                return None

            # Calculate metrics
            total_events = len(user_events)

            # Recommendation acceptance rate
            shown = sum(1 for e in user_events if e.event_type == EventType.RECOMMENDATION_SHOWN)
            accepted = sum(1 for e in user_events if e.event_type == EventType.RECOMMENDATION_ACCEPTED)
            acceptance_rate = (accepted / shown) if shown > 0 else 0.0

            # Most common actions
            action_counts = Counter(e.event_type.value for e in user_events)
            most_common_actions = action_counts.most_common(5)

            # Source usage patterns
            source_usage = Counter(e.source_id for e in user_events if e.source_id)

            # Session duration (simplified)
            session_durations = []
            for session_id in set(e.session_id for e in user_events if e.session_id):
                session_events = [e for e in user_events if e.session_id == session_id]
                if len(session_events) >= 2:
                    duration = (max(e.timestamp for e in session_events) -
                              min(e.timestamp for e in session_events)).total_seconds()
                    session_durations.append(duration)

            avg_session_duration = statistics.mean(session_durations) if session_durations else 0.0
            last_activity = max(e.timestamp for e in user_events)

            summary = UserBehaviorSummary(
                user_id=user_id,
                total_events=total_events,
                recommendation_acceptance_rate=acceptance_rate,
                most_common_actions=most_common_actions,
                source_usage_patterns=dict(source_usage),
                session_duration_avg=avg_session_duration,
                last_activity=last_activity
            )

            # Cache the summary
            if self.cache_manager:
                await self.cache_manager.set(cache_key, asdict(summary), ttl=1800)

            return summary

        except Exception as e:
            logger.error(f"❌ Failed to get user behavior summary: {e}")
            return None

    async def get_recommendation_metrics(self, days: int = 7) -> RecommendationMetrics:
        """Get recommendation effectiveness metrics"""
        try:
            # Calculate from recent events
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            recent_events = [e for e in self.event_buffer if e.timestamp >= cutoff_time]

            shown_events = [e for e in recent_events if e.event_type == EventType.RECOMMENDATION_SHOWN]
            accepted_events = [e for e in recent_events if e.event_type == EventType.RECOMMENDATION_ACCEPTED]
            rejected_events = [e for e in recent_events if e.event_type == EventType.RECOMMENDATION_REJECTED]

            total_shown = len(shown_events)
            total_accepted = len(accepted_events)
            total_rejected = len(rejected_events)

            acceptance_rate = (total_accepted / total_shown) if total_shown > 0 else 0.0

            # Confidence vs acceptance analysis
            confidence_vs_acceptance = []
            for event in shown_events:
                rec_id = event.recommendation_id
                confidence = event.data.get('confidence_score', 0.0)
                was_accepted = any(e.recommendation_id == rec_id for e in accepted_events)
                confidence_vs_acceptance.append((confidence, was_accepted))

            # Source effectiveness
            source_effectiveness = {}
            for source_id in set(e.source_id for e in shown_events if e.source_id):
                source_shown = [e for e in shown_events if e.source_id == source_id]
                source_accepted = [e for e in accepted_events if e.source_id == source_id]
                if source_shown:
                    source_effectiveness[source_id] = len(source_accepted) / len(source_shown)

            # Simplified metrics (in production, these would be more sophisticated)
            user_segment_performance = {'all_users': acceptance_rate}
            time_to_decision_avg = 30.0  # Placeholder

            return RecommendationMetrics(
                total_shown=total_shown,
                total_accepted=total_accepted,
                total_rejected=total_rejected,
                acceptance_rate=acceptance_rate,
                confidence_vs_acceptance=confidence_vs_acceptance,
                source_effectiveness=source_effectiveness,
                user_segment_performance=user_segment_performance,
                time_to_decision_avg=time_to_decision_avg
            )

        except Exception as e:
            logger.error(f"❌ Failed to get recommendation metrics: {e}")
            return RecommendationMetrics(0, 0, 0, 0.0, [], {}, {}, 0.0)

    async def get_system_metrics(self) -> SystemMetrics:
        """Get overall system performance metrics"""
        try:
            # Calculate from current data
            unique_users = len(set(e.user_id for e in self.event_buffer))

            # Active users in last 24h
            cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_events = [e for e in self.event_buffer if e.timestamp >= cutoff_24h]
            active_users_24h = len(set(e.user_id for e in recent_events))

            # Recommendation metrics
            total_recommendations = self.counters.get('recommendations_shown', 0)
            accepted = self.counters.get('recommendations_accepted', 0)
            overall_acceptance_rate = (accepted / total_recommendations) if total_recommendations > 0 else 0.0

            # Session metrics
            session_events = [e for e in self.event_buffer if e.event_type == EventType.SESSION_END]
            session_durations = [e.data.get('duration_seconds', 0) for e in session_events]
            avg_session_duration = statistics.mean(session_durations) if session_durations else 0.0

            # Error rate
            error_events = [e for e in self.event_buffer if e.event_type == EventType.ERROR_OCCURRED]
            error_rate = len(error_events) / len(self.event_buffer) if self.event_buffer else 0.0

            # Placeholder metrics (would be real in production)
            source_health_scores = {'default_source': 0.95}
            cache_hit_rate = 0.85

            return SystemMetrics(
                total_users=unique_users,
                active_users_24h=active_users_24h,
                total_recommendations=total_recommendations,
                overall_acceptance_rate=overall_acceptance_rate,
                avg_session_duration=avg_session_duration,
                error_rate=error_rate,
                source_health_scores=source_health_scores,
                cache_hit_rate=cache_hit_rate
            )

        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0.0, 0.0, 0.0, {}, 0.0)

    def _update_counters(self, event: AnalyticsEvent):
        """Update real-time counters based on event"""
        # Update session event count
        if event.session_id and event.session_id in self.active_sessions:
            self.active_sessions[event.session_id]['event_count'] += 1

        # Update global counters
        self.counters[f'events_{event.event_type.value}'] += 1
        self.counters['total_events'] += 1

        # Update user-specific counters
        self.counters[f'user_{event.user_id}_events'] += 1

    async def _flush_buffers(self):
        """Flush all buffers to persistent storage"""
        await self._flush_events()
        await self._flush_metrics()

    async def _flush_events(self):
        """Flush event buffer to database"""
        if not self.event_buffer:
            return

        try:
            # In production, this would batch insert to database
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()

            if self.database_manager:
                for event in events_to_flush:
                    await self.database_manager.record_user_behavior(
                        user_id=event.user_id,
                        action_type=event.event_type.value,
                        source_id=event.source_id,
                        context=event.data
                    )

            logger.debug(f"💾 Flushed {len(events_to_flush)} events to database")

        except Exception as e:
            logger.error(f"❌ Failed to flush events: {e}")
            # In production, you might want to retry or save to backup storage

    async def _flush_metrics(self):
        """Flush metric buffer to storage"""
        if not self.metric_buffer:
            return

        try:
            metrics_to_flush = self.metric_buffer.copy()
            self.metric_buffer.clear()

            # Cache aggregated metrics
            if self.cache_manager:
                for metric in metrics_to_flush:
                    cache_key = f"metric_history:{metric.name}"
                    # In production, this would append to a time series
                    await self.cache_manager.set(cache_key, asdict(metric), ttl=86400)

            logger.debug(f"💾 Flushed {len(metrics_to_flush)} metrics to storage")

        except Exception as e:
            logger.error(f"❌ Failed to flush metrics: {e}")

    async def _background_flush(self):
        """Background task to periodically flush buffers"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Background flush error: {e}")

    # Analytics query methods
    async def get_top_sources(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular data sources"""
        source_counts = Counter(e.source_id for e in self.event_buffer if e.source_id)
        return source_counts.most_common(limit)

    async def get_user_segments(self) -> Dict[str, List[str]]:
        """Analyze user segments based on behavior"""
        # Simplified segmentation (in production, this would be more sophisticated)
        segments = {
            'active_users': [],
            'recommendation_lovers': [],
            'recommendation_skeptics': []
        }

        for user_id in set(e.user_id for e in self.event_buffer):
            user_events = [e for e in self.event_buffer if e.user_id == user_id]

            if len(user_events) > 10:
                segments['active_users'].append(user_id)

            # Check recommendation acceptance
            shown = sum(1 for e in user_events if e.event_type == EventType.RECOMMENDATION_SHOWN)
            accepted = sum(1 for e in user_events if e.event_type == EventType.RECOMMENDATION_ACCEPTED)

            if shown > 0:
                acceptance_rate = accepted / shown
                if acceptance_rate > 0.7:
                    segments['recommendation_lovers'].append(user_id)
                elif acceptance_rate < 0.3:
                    segments['recommendation_skeptics'].append(user_id)

        return segments

# Factory function
async def create_analytics_engine(
    database_manager: Optional[DatabaseManager] = None,
    cache_manager: Optional[CacheManager] = None,
    **kwargs
) -> AnalyticsEngine:
    """Factory function to create and initialize analytics engine"""
    engine = AnalyticsEngine(database_manager, cache_manager, **kwargs)
    await engine.initialize()
    return engine

# Example usage and testing
if __name__ == "__main__":
    async def test_analytics():
        """Test analytics engine functionality"""

        try:
            # Create mock dependencies - import the actual classes
            try:
                from smart_defaults.storage.database import DatabaseManager, create_database_manager
                from smart_defaults.storage.cache import CacheManager, create_cache_manager

                # Use the factory functions
                db_manager = await create_database_manager()
                cache_manager = await create_cache_manager()
            except ImportError:
                # Fallback to mock classes for testing
                db_manager = DatabaseManager()

                # Create a mock cache manager
                class MockCacheManager:
                    async def initialize(self): pass
                    async def close(self): pass
                    async def set(self, key, value, ttl=None): return True
                    async def get(self, key, default=None): return default
                    async def increment(self, key, amount=1): return amount

                cache_manager = MockCacheManager()

            # Initialize analytics engine
            analytics = await create_analytics_engine(
                database_manager=db_manager,
                cache_manager=cache_manager,
                batch_size=5,  # Small batch for testing
                flush_interval=2  # Quick flush for testing
            )

            try:
                print("🧪 Testing Analytics Engine...")

                # Test session tracking
                session_id = await analytics.start_session(
                    user_id="test_user_1",
                    session_data={"device": "desktop", "browser": "chrome"}
                )
                print(f"✅ Started session: {session_id}")

                # Test recommendation tracking
                test_rec = Recommendation(id="rec_1", user_id="test_user_1", confidence_score=0.85)

                await analytics.track_recommendation_shown(
                    user_id="test_user_1",
                    recommendation=test_rec,
                    context={"source": "database_integration"}
                )
                print("✅ Tracked recommendation shown")

                await analytics.track_recommendation_feedback(
                    user_id="test_user_1",
                    recommendation_id="rec_1",
                    accepted=True,
                    feedback_data={"time_to_decision": 15.2}
                )
                print("✅ Tracked recommendation feedback")

                # Test custom events
                await analytics.track_event(
                    user_id="test_user_1",
                    event_type=EventType.SOURCE_CONNECTED,
                    data={"source_type": "postgres", "connection_time": 2.3},
                    session_id=session_id
                )
                print("✅ Tracked custom event")

                # Test metrics
                await analytics.record_metric(
                    name="response_time",
                    value=0.25,
                    metric_type=MetricType.TIMER,
                    tags={"endpoint": "recommendations"}
                )
                print("✅ Recorded metric")

                # Wait for potential background flush
                await asyncio.sleep(1)

                # Test analytics queries
                behavior_summary = await analytics.get_user_behavior_summary("test_user_1")
                if behavior_summary:
                    print(f"✅ User behavior summary: {behavior_summary.total_events} events, "
                          f"{behavior_summary.recommendation_acceptance_rate:.2%} acceptance rate")

                rec_metrics = await analytics.get_recommendation_metrics()
                print(f"✅ Recommendation metrics: {rec_metrics.total_shown} shown, "
                      f"{rec_metrics.acceptance_rate:.2%} acceptance rate")

                system_metrics = await analytics.get_system_metrics()
                print(f"✅ System metrics: {system_metrics.total_users} users, "
                      f"{system_metrics.overall_acceptance_rate:.2%} overall acceptance")

                # Test user segmentation
                segments = await analytics.get_user_segments()
                print(f"✅ User segments: {len(segments)} segments identified")

                # Test top sources
                top_sources = await analytics.get_top_sources()
                print(f"✅ Top sources: {top_sources}")

                # End session
                await analytics.end_session(session_id, {"reason": "test_complete"})
                print("✅ Session ended")

                print("\n✅ All analytics tests passed!")

            finally:
                await analytics.close()

        except Exception as e:
            print(f"❌ Analytics test failed: {e}")
            import traceback
            traceback.print_exc()

    # Run tests
    print("🔬 Testing Analytics Engine...")
    asyncio.run(test_analytics())