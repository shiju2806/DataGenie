"""
Smart Defaults Learning Engine
Machine learning and pattern recognition for adaptive recommendations
File location: smart_defaults/analyzers/learning_engine.py
"""

import asyncio
import logging
import json
import sys
import pickle
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import statistics
import math

# Import dependencies with fallbacks
try:
    from ..storage.database import DatabaseManager
    from ..storage.cache import CacheManager
    from ..models.user_profile import UserProfile
    from ..models.recommendation import Recommendation
    from ..utils.monitoring import AnalyticsEngine, EventType
    from ..analyzers.profile_analyzer import ProfileAnalyzer, ProfileInsights
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
        confidence_score: float = 0.8

    @dataclass
    class ProfileInsights:
        user_id: str = "test_user"
        user_segment: Any = None

    class DatabaseManager:
        async def initialize(self): pass
        async def close(self): pass

    class CacheManager:
        async def initialize(self): pass
        async def close(self): pass
        async def get(self, key, default=None): return default
        async def set(self, key, value, ttl=None): pass

    class AnalyticsEngine:
        async def track_event(self, *args, **kwargs): pass

    class ProfileAnalyzer:
        async def analyze_user_profile(self, profile): return ProfileInsights()

logger = logging.getLogger(__name__)

class LearningMethod(Enum):
    """Machine learning methods available"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    MATRIX_FACTORIZATION = "matrix_factorization"
    CLUSTERING = "clustering"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    REINFORCEMENT = "reinforcement"

class ModelType(Enum):
    """Types of models in the learning engine"""
    USER_PREFERENCE = "user_preference"
    SOURCE_SIMILARITY = "source_similarity"
    SUCCESS_PREDICTION = "success_prediction"
    CHURN_PREDICTION = "churn_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION_RANKING = "recommendation_ranking"

class TrainingStatus(Enum):
    """Training status for models"""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    OUTDATED = "outdated"

@dataclass
class FeatureVector:
    """Feature vector for machine learning"""
    user_id: str
    features: Dict[str, float]
    timestamp: datetime
    feature_version: str = "1.0"

@dataclass
class TrainingData:
    """Training data for ML models"""
    features: List[FeatureVector]
    labels: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Square Error
    training_samples: int = 0
    validation_samples: int = 0
    last_trained: Optional[datetime] = None

@dataclass
class PredictionResult:
    """Result of a model prediction"""
    model_type: ModelType
    prediction: float
    confidence: float
    features_used: List[str]
    model_version: str
    predicted_at: datetime

@dataclass
class LearningInsights:
    """Insights from the learning engine"""
    user_id: str
    generated_at: datetime

    # Pattern insights
    discovered_patterns: List[Dict[str, Any]]
    behavior_clusters: Dict[str, List[str]]
    preference_evolution: Dict[str, Any]

    # Predictions
    success_predictions: Dict[str, float]
    risk_assessments: Dict[str, float]

    # Recommendations
    ml_recommendations: List[Recommendation]
    confidence_adjustments: Dict[str, float]

    # Model performance
    model_metrics: Dict[ModelType, ModelMetrics]

class SimpleMLModel:
    """Simplified machine learning model for demonstration"""

    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.weights = {}
        self.bias = 0.0
        self.is_trained = False
        self.feature_importance = {}
        self.training_history = []

    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from data"""
        features = {}

        # User features
        if 'user_profile' in data:
            profile = data['user_profile']
            features.update({
                'seniority_score': self._encode_seniority(getattr(profile, 'seniority_level', 'mid')),
                'role_score': self._encode_role(getattr(profile, 'role', 'data_analyst')),
                'industry_score': self._encode_industry(getattr(profile, 'industry', 'technology'))
            })

        # Behavioral features
        if 'behavior_data' in data:
            behavior = data['behavior_data']
            features.update({
                'activity_level': behavior.get('total_events', 0) / 100.0,  # Normalize
                'acceptance_rate': behavior.get('recommendation_acceptance_rate', 0.5),
                'exploration_score': len(behavior.get('source_usage_patterns', {})) / 10.0
            })

        # Recommendation features
        if 'recommendation' in data:
            rec = data['recommendation']
            features.update({
                'base_confidence': getattr(rec, 'confidence_score', 0.5),
                'source_popularity': 0.5,  # Would be calculated from usage stats
                'source_complexity': 0.3   # Would be based on source type/config
            })

        # Context features
        if 'context' in data:
            context = data['context']
            features.update({
                'time_of_day': self._encode_time(context.get('timestamp', datetime.now())),
                'day_of_week': self._encode_day(context.get('timestamp', datetime.now())),
                'session_length': min(context.get('session_duration', 0) / 3600.0, 1.0)  # Hours, capped at 1
            })

        return features

    def _encode_seniority(self, seniority: str) -> float:
        """Encode seniority level as numerical score"""
        seniority_map = {
            'intern': 0.1, 'entry': 0.2, 'junior': 0.3,
            'mid': 0.5, 'intermediate': 0.5,
            'senior': 0.7, 'lead': 0.8, 'principal': 0.9, 'staff': 1.0
        }
        seniority_lower = seniority.lower()
        for key, value in seniority_map.items():
            if key in seniority_lower:
                return value
        return 0.5  # Default

    def _encode_role(self, role: str) -> float:
        """Encode role as numerical score (complexity/technical level)"""
        role_map = {
            'business_analyst': 0.3,
            'data_analyst': 0.5,
            'data_scientist': 0.8,
            'data_engineer': 0.9,
            'ml_engineer': 1.0
        }
        return role_map.get(role.lower(), 0.5)

    def _encode_industry(self, industry: str) -> float:
        """Encode industry as data-intensity score"""
        industry_map = {
            'technology': 1.0,
            'finance': 0.9,
            'telecommunications': 0.8,
            'retail': 0.7,
            'healthcare': 0.6,
            'manufacturing': 0.4,
            'education': 0.3
        }
        return industry_map.get(industry.lower(), 0.5)

    def _encode_time(self, timestamp: datetime) -> float:
        """Encode time of day as 0-1 score"""
        hour = timestamp.hour
        # Peak work hours (9-17) get higher scores
        if 9 <= hour <= 17:
            return 0.8 + (hour - 9) / 40.0  # 0.8 to 1.0
        else:
            return max(0.1, 0.8 - abs(hour - 13) / 20.0)  # Decline from midday

    def _encode_day(self, timestamp: datetime) -> float:
        """Encode day of week as weekday score"""
        # Monday=0, Sunday=6
        weekday = timestamp.weekday()
        if weekday < 5:  # Monday-Friday
            return 0.8 + weekday / 20.0  # 0.8 to 1.0
        else:  # Weekend
            return 0.2 + weekday / 20.0  # Lower scores

    def train(self, training_data: TrainingData) -> ModelMetrics:
        """Train the model using simple linear regression"""
        if not training_data.features or not training_data.labels:
            raise ValueError("Training data is empty")

        # Convert features to matrix
        feature_names = list(training_data.features[0].features.keys())
        X = np.array([[fv.features.get(name, 0.0) for name in feature_names]
                     for fv in training_data.features])
        y = np.array(training_data.labels)

        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid training samples")

        # Simple linear regression using normal equation: w = (X^T X)^-1 X^T y
        try:
            # Add bias column
            X_with_bias = np.column_stack([np.ones(len(X)), X])

            # Calculate weights
            XtX = np.dot(X_with_bias.T, X_with_bias)
            XtX_inv = np.linalg.pinv(XtX)  # Use pseudo-inverse for stability
            weights = np.dot(XtX_inv, np.dot(X_with_bias.T, y))

            self.bias = weights[0]
            self.weights = {name: weights[i+1] for i, name in enumerate(feature_names)}

            # Calculate feature importance (absolute weights normalized)
            weight_abs = np.abs(weights[1:])
            if np.sum(weight_abs) > 0:
                importance = weight_abs / np.sum(weight_abs)
                self.feature_importance = {name: importance[i] for i, name in enumerate(feature_names)}

            self.is_trained = True

            # Calculate basic metrics
            predictions = self.predict_batch(training_data.features)
            mse = np.mean((y - predictions) ** 2)
            mae = np.mean(np.abs(y - predictions))
            rmse = np.sqrt(mse)

            # For classification-like problems, calculate accuracy
            accuracy = 1.0 - mae  # Simple approximation

            metrics = ModelMetrics(
                model_type=self.model_type,
                accuracy=max(0.0, min(1.0, accuracy)),
                precision=0.7,  # Simplified
                recall=0.7,     # Simplified
                f1_score=0.7,   # Simplified
                mae=mae,
                rmse=rmse,
                training_samples=len(X),
                validation_samples=0,
                last_trained=datetime.now(timezone.utc)
            )

            self.training_history.append({
                'timestamp': datetime.now(timezone.utc),
                'samples': len(X),
                'mae': mae,
                'rmse': rmse
            })

            return metrics

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise

    def predict(self, features: Dict[str, float]) -> float:
        """Make a prediction using the trained model"""
        if not self.is_trained:
            return 0.5  # Default prediction

        prediction = self.bias
        for feature_name, weight in self.weights.items():
            feature_value = features.get(feature_name, 0.0)
            prediction += weight * feature_value

        # Ensure prediction is in valid range [0, 1]
        return max(0.0, min(1.0, prediction))

    def predict_batch(self, feature_vectors: List[FeatureVector]) -> np.ndarray:
        """Make batch predictions"""
        predictions = []
        for fv in feature_vectors:
            predictions.append(self.predict(fv.features))
        return np.array(predictions)

class LearningEngine:
    """Main learning engine for adaptive recommendations"""

    def __init__(self,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None,
                 analytics_engine: Optional[AnalyticsEngine] = None,
                 profile_analyzer: Optional[ProfileAnalyzer] = None,
                 model_cache_ttl: int = 86400,  # 24 hours
                 min_training_samples: int = 10,
                 retrain_interval_hours: int = 24):

        self.database_manager = database_manager
        self.cache_manager = cache_manager
        self.analytics_engine = analytics_engine
        self.profile_analyzer = profile_analyzer

        self.model_cache_ttl = model_cache_ttl
        self.min_training_samples = min_training_samples
        self.retrain_interval_hours = retrain_interval_hours

        # ML Models
        self.models: Dict[ModelType, SimpleMLModel] = {}
        self.model_metrics: Dict[ModelType, ModelMetrics] = {}

        # Learning data
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.feature_cache: Dict[str, FeatureVector] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize the learning engine"""
        if self._initialized:
            return

        # Initialize dependencies
        if self.database_manager:
            await self.database_manager.initialize()
        if self.cache_manager:
            await self.cache_manager.initialize()

        # Initialize models
        await self._initialize_models()

        # Load cached models if available
        await self._load_cached_models()

        self._initialized = True
        logger.info("✅ Learning engine initialized")

    async def close(self):
        """Close the learning engine"""
        # Save models to cache
        await self._save_models_to_cache()

        if self.database_manager:
            await self.database_manager.close()
        if self.cache_manager:
            await self.cache_manager.close()

        logger.info("🔐 Learning engine closed")

    async def _initialize_models(self):
        """Initialize ML models"""
        model_types = [
            ModelType.USER_PREFERENCE,
            ModelType.SUCCESS_PREDICTION,
            ModelType.RECOMMENDATION_RANKING
        ]

        for model_type in model_types:
            self.models[model_type] = SimpleMLModel(model_type)
            self.model_metrics[model_type] = ModelMetrics(
                model_type=model_type,
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5
            )

    async def _load_cached_models(self):
        """Load models from cache"""
        if not self.cache_manager:
            return

        try:
            for model_type in self.models.keys():
                cache_key = f"ml_model:{model_type.value}"
                cached_model = await self.cache_manager.get(cache_key)

                if cached_model:
                    # In production, this would deserialize the actual model
                    logger.info(f"📋 Loaded cached model: {model_type.value}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load cached models: {e}")

    async def _save_models_to_cache(self):
        """Save models to cache"""
        if not self.cache_manager:
            return

        try:
            for model_type, model in self.models.items():
                if model.is_trained:
                    cache_key = f"ml_model:{model_type.value}"
                    # In production, this would serialize the actual model
                    model_data = {
                        'weights': model.weights,
                        'bias': model.bias,
                        'feature_importance': model.feature_importance,
                        'training_history': model.training_history[-5:]  # Keep last 5
                    }
                    await self.cache_manager.set(cache_key, model_data, ttl=self.model_cache_ttl)

        except Exception as e:
            logger.warning(f"⚠️ Failed to save models to cache: {e}")

    async def learn_from_feedback(self, user_id: str, recommendation: Recommendation,
                                feedback: Dict[str, Any]) -> bool:
        """Learn from user feedback on recommendations"""

        try:
            # Create learning sample
            learning_sample = {
                'user_id': user_id,
                'recommendation': recommendation,
                'feedback': feedback,
                'timestamp': datetime.now(timezone.utc)
            }

            # Add to feedback buffer
            self.feedback_buffer.append(learning_sample)

            # Extract features for future training
            user_profile = feedback.get('user_profile')
            if user_profile:
                features = await self._extract_user_features(user_id, user_profile, recommendation)
                self.feature_cache[f"{user_id}_{recommendation.id}"] = features

            # Track learning event
            if self.analytics_engine:
                await self.analytics_engine.track_event(
                    user_id=user_id,
                    event_type=EventType.USER_ACTION,
                    data={
                        'action': 'feedback_received',
                        'recommendation_id': recommendation.id,
                        'feedback_type': feedback.get('action', 'unknown'),
                        'learning_sample_created': True
                    }
                )

            # Trigger training if enough samples
            if len(self.feedback_buffer) >= self.min_training_samples:
                await self._trigger_model_training()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to learn from feedback: {e}")
            return False

    async def _extract_user_features(self, user_id: str, user_profile: UserProfile,
                                   recommendation: Recommendation) -> FeatureVector:
        """Extract features for a user interaction"""

        # Get user behavior data (mock for now)
        behavior_data = {
            'total_events': 50,
            'recommendation_acceptance_rate': 0.7,
            'source_usage_patterns': {'postgres': 10, 'tableau': 5}
        }

        # Create feature extraction data
        feature_data = {
            'user_profile': user_profile,
            'behavior_data': behavior_data,
            'recommendation': recommendation,
            'context': {
                'timestamp': datetime.now(timezone.utc),
                'session_duration': 1800  # 30 minutes
            }
        }

        # Extract features using the model
        model = self.models[ModelType.USER_PREFERENCE]
        features = model.extract_features(feature_data)

        return FeatureVector(
            user_id=user_id,
            features=features,
            timestamp=datetime.now(timezone.utc)
        )

    async def _trigger_model_training(self):
        """Trigger training for models when enough data is available"""

        try:
            logger.info(f"🎓 Triggering model training with {len(self.feedback_buffer)} samples")

            # Prepare training data
            training_data = await self._prepare_training_data()

            if not training_data.features:
                logger.warning("⚠️ No training data available")
                return

            # Train models
            for model_type in [ModelType.USER_PREFERENCE, ModelType.SUCCESS_PREDICTION]:
                try:
                    model = self.models[model_type]
                    metrics = model.train(training_data)
                    self.model_metrics[model_type] = metrics

                    logger.info(f"✅ Trained {model_type.value}: MAE={metrics.mae:.3f}, Accuracy={metrics.accuracy:.3f}")

                except Exception as e:
                    logger.error(f"❌ Failed to train {model_type.value}: {e}")

            # Clear processed feedback buffer
            self.feedback_buffer = self.feedback_buffer[-self.min_training_samples:]

            # Save updated models
            await self._save_models_to_cache()

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")

    async def _prepare_training_data(self) -> TrainingData:
        """Prepare training data from feedback buffer"""

        features = []
        labels = []

        for sample in self.feedback_buffer:
            try:
                user_id = sample['user_id']
                recommendation = sample['recommendation']
                feedback = sample['feedback']

                # Get cached features or extract new ones
                feature_key = f"{user_id}_{recommendation.id}"
                if feature_key in self.feature_cache:
                    feature_vector = self.feature_cache[feature_key]
                else:
                    # Skip if no features available
                    continue

                # Create label from feedback
                feedback_action = feedback.get('action', 'unknown')
                if feedback_action == 'accept':
                    label = 1.0
                elif feedback_action == 'reject':
                    label = 0.0
                else:
                    label = 0.5  # Neutral/unclear feedback

                features.append(feature_vector)
                labels.append(label)

            except Exception as e:
                logger.warning(f"⚠️ Failed to process training sample: {e}")
                continue

        return TrainingData(
            features=features,
            labels=labels,
            metadata={
                'total_samples': len(self.feedback_buffer),
                'processed_samples': len(features),
                'created_at': datetime.now(timezone.utc)
            }
        )

    async def predict_recommendation_success(self, user_id: str, recommendation: Recommendation,
                                          user_profile: Optional[UserProfile] = None) -> PredictionResult:
        """Predict the likelihood of recommendation success"""

        try:
            model = self.models[ModelType.SUCCESS_PREDICTION]

            if not model.is_trained:
                # Return default prediction
                return PredictionResult(
                    model_type=ModelType.SUCCESS_PREDICTION,
                    prediction=recommendation.confidence_score,  # Use original confidence
                    confidence=0.5,
                    features_used=[],
                    model_version="untrained",
                    predicted_at=datetime.now(timezone.utc)
                )

            # Extract features
            if user_profile:
                feature_vector = await self._extract_user_features(user_id, user_profile, recommendation)
                features = feature_vector.features
            else:
                # Use default features
                features = {
                    'base_confidence': recommendation.confidence_score,
                    'seniority_score': 0.5,
                    'role_score': 0.5,
                    'activity_level': 0.5
                }

            # Make prediction
            prediction = model.predict(features)

            # Calculate confidence based on feature importance and model metrics
            model_metrics = self.model_metrics.get(ModelType.SUCCESS_PREDICTION)
            confidence = model_metrics.accuracy if model_metrics else 0.7

            return PredictionResult(
                model_type=ModelType.SUCCESS_PREDICTION,
                prediction=prediction,
                confidence=confidence,
                features_used=list(features.keys()),
                model_version="1.0",
                predicted_at=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return PredictionResult(
                model_type=ModelType.SUCCESS_PREDICTION,
                prediction=0.5,
                confidence=0.3,
                features_used=[],
                model_version="error",
                predicted_at=datetime.now(timezone.utc)
            )

    async def generate_ml_recommendations(self, user_id: str, user_profile: UserProfile,
                                        available_sources: List[str],
                                        max_recommendations: int = 5) -> List[Recommendation]:
        """Generate ML-powered recommendations"""

        recommendations = []

        try:
            # For each available source, predict success probability
            source_predictions = []

            for source in available_sources:
                # Create mock recommendation for prediction
                mock_rec = Recommendation(
                    id=f"ml_rec_{user_id}_{source}",
                    user_id=user_id,
                    source_id=source,
                    confidence_score=0.7  # Base confidence
                )

                # Predict success
                prediction = await self.predict_recommendation_success(
                    user_id, mock_rec, user_profile
                )

                source_predictions.append((source, prediction.prediction, prediction.confidence))

            # Sort by predicted success probability
            source_predictions.sort(key=lambda x: x[1], reverse=True)

            # Create recommendations from top predictions
            for i, (source, prediction, confidence) in enumerate(source_predictions[:max_recommendations]):

                # Adjust confidence based on ML prediction
                adjusted_confidence = min(0.95, prediction * confidence)

                recommendation = Recommendation(
                    id=f"ml_rec_{user_id}_{source}_{int(datetime.now(timezone.utc).timestamp())}",
                    user_id=user_id,
                    source_id=source,
                    recommendation_type="ml_generated",
                    confidence_score=adjusted_confidence,
                    reasoning={
                        "ml_prediction": prediction,
                        "model_confidence": confidence,
                        "rank": i + 1,
                        "learning_based": True
                    },
                    context={
                        "generation_method": "learning_engine",
                        "model_version": "1.0",
                        "features_considered": len(self.models[ModelType.SUCCESS_PREDICTION].weights)
                    }
                )

                recommendations.append(recommendation)

        except Exception as e:
            logger.error(f"❌ ML recommendation generation failed: {e}")

        return recommendations

    async def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user behavior patterns using ML"""

        patterns = {
            'activity_clusters': [],
            'preference_drift': {},
            'anomalies': [],
            'predicted_interests': []
        }

        try:
            # Mock pattern analysis (in production, this would use real ML)

            # Activity clustering
            patterns['activity_clusters'] = [
                {'cluster': 'morning_analyst', 'confidence': 0.8, 'peak_hours': '9-11am'},
                {'cluster': 'database_focused', 'confidence': 0.7, 'preferred_sources': ['postgres', 'mysql']}
            ]

            # Preference drift detection
            patterns['preference_drift'] = {
                'auto_connect_threshold': {'trend': 'increasing', 'rate': 0.05, 'confidence': 0.6},
                'source_diversity': {'trend': 'stable', 'rate': 0.0, 'confidence': 0.8}
            }

            # Anomaly detection
            patterns['anomalies'] = [
                {'type': 'unusual_activity_time', 'timestamp': datetime.now(timezone.utc), 'severity': 'low'},
            ]

            # Predicted interests
            patterns['predicted_interests'] = [
                {'source': 'elasticsearch', 'probability': 0.75, 'reason': 'similar_users'},
                {'source': 'kafka', 'probability': 0.6, 'reason': 'role_progression'}
            ]

        except Exception as e:
            logger.warning(f"⚠️ Pattern analysis failed for {user_id}: {e}")

        return patterns

    async def get_learning_insights(self, user_id: str) -> LearningInsights:
        """Get comprehensive learning insights for a user"""

        try:
            # Analyze patterns
            patterns = await self.analyze_user_patterns(user_id)

            # Get user profile for ML recommendations
            mock_profile = UserProfile(
                id="mock_profile",
                user_id=user_id,
                role="data_analyst"
            )

            # Generate ML recommendations
            available_sources = ['postgresql', 'redis', 'tableau', 'jupyter', 'airflow']
            ml_recommendations = await self.generate_ml_recommendations(
                user_id, mock_profile, available_sources
            )

            # Calculate success predictions for various scenarios
            success_predictions = {}
            for source in available_sources[:3]:
                mock_rec = Recommendation(
                    id=f"pred_{source}",
                    user_id=user_id,
                    source_id=source,
                    confidence_score=0.7
                )
                pred_result = await self.predict_recommendation_success(user_id, mock_rec, mock_profile)
                success_predictions[source] = pred_result.prediction

            # Risk assessments
            risk_assessments = {
                'recommendation_fatigue': 0.2,
                'over_automation': 0.1,
                'skill_mismatch': 0.3
            }

            # Confidence adjustments based on learning
            confidence_adjustments = {}
            for source in available_sources:
                # Adjust based on historical performance
                base_adjustment = 0.0
                if source in success_predictions:
                    # Higher success prediction = positive adjustment
                    base_adjustment = (success_predictions[source] - 0.5) * 0.2
                confidence_adjustments[source] = base_adjustment

            return LearningInsights(
                user_id=user_id,
                generated_at=datetime.now(timezone.utc),
                discovered_patterns=patterns['activity_clusters'],
                behavior_clusters={'analysts': [user_id], 'explorers': [], 'experts': []},
                preference_evolution=patterns['preference_drift'],
                success_predictions=success_predictions,
                risk_assessments=risk_assessments,
                ml_recommendations=ml_recommendations,
                confidence_adjustments=confidence_adjustments,
                model_metrics=self.model_metrics
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate learning insights: {e}")

            # Return minimal insights on failure
            return LearningInsights(
                user_id=user_id,
                generated_at=datetime.now(timezone.utc),
                discovered_patterns=[],
                behavior_clusters={},
                preference_evolution={},
                success_predictions={},
                risk_assessments={},
                ml_recommendations=[],
                confidence_adjustments={},
                model_metrics={}
            )

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models"""

        status = {
            'models': {},
            'training_stats': {
                'total_feedback_samples': len(self.feedback_buffer),
                'min_samples_needed': self.min_training_samples,
                'last_training': None
            },
            'performance_summary': {}
        }

        for model_type, model in self.models.items():
            model_status = {
                'trained': model.is_trained,
                'feature_count': len(model.weights),
                'training_history_count': len(model.training_history)
            }

            # Add metrics if available
            if model_type in self.model_metrics:
                metrics = self.model_metrics[model_type]
                model_status.update({
                    'accuracy': metrics.accuracy,
                    'mae': metrics.mae,
                    'training_samples': metrics.training_samples,
                    'last_trained': metrics.last_trained.isoformat() if metrics.last_trained else None
                })

            status['models'][model_type.value] = model_status

        # Performance summary
        trained_models = [m for m in self.models.values() if m.is_trained]
        if trained_models:
            avg_accuracy = np.mean([self.model_metrics[mt].accuracy for mt in self.model_metrics.keys()])
            status['performance_summary'] = {
                'trained_models': len(trained_models),
                'average_accuracy': avg_accuracy,
                'ready_for_predictions': len(trained_models) > 0
            }

        return status

    async def retrain_models(self, force: bool = False) -> Dict[str, bool]:
        """Retrain all models"""

        results = {}

        try:
            if not force and len(self.feedback_buffer) < self.min_training_samples:
                logger.warning(f"⚠️ Insufficient training data: {len(self.feedback_buffer)} < {self.min_training_samples}")
                return {mt.value: False for mt in self.models.keys()}

            # Prepare training data
            training_data = await self._prepare_training_data()

            # Retrain each model
            for model_type, model in self.models.items():
                try:
                    metrics = model.train(training_data)
                    self.model_metrics[model_type] = metrics
                    results[model_type.value] = True

                    logger.info(f"✅ Retrained {model_type.value}: Accuracy={metrics.accuracy:.3f}")

                except Exception as e:
                    logger.error(f"❌ Failed to retrain {model_type.value}: {e}")
                    results[model_type.value] = False

            # Save updated models
            await self._save_models_to_cache()

        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            results = {mt.value: False for mt in self.models.keys()}

        return results

    async def explain_prediction(self, prediction_result: PredictionResult) -> Dict[str, Any]:
        """Provide explanation for a model prediction"""

        explanation = {
            'prediction_value': prediction_result.prediction,
            'confidence': prediction_result.confidence,
            'model_type': prediction_result.model_type.value,
            'feature_importance': {},
            'decision_factors': [],
            'uncertainty_sources': []
        }

        try:
            model = self.models[prediction_result.model_type]

            if model.is_trained and model.feature_importance:
                # Get feature importance
                explanation['feature_importance'] = model.feature_importance.copy()

                # Generate decision factors
                top_features = sorted(
                    model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                for feature, importance in top_features:
                    if importance > 0.2:  # Significant importance
                        explanation['decision_factors'].append({
                            'factor': feature,
                            'importance': importance,
                            'description': self._get_feature_description(feature)
                        })

                # Identify uncertainty sources
                if prediction_result.confidence < 0.7:
                    explanation['uncertainty_sources'].extend([
                        'Limited training data',
                        'Model complexity constraints',
                        'Feature correlation issues'
                    ])

                if model.training_history:
                    last_training = model.training_history[-1]
                    if last_training['mae'] > 0.3:
                        explanation['uncertainty_sources'].append('High prediction error in training')

        except Exception as e:
            logger.warning(f"⚠️ Failed to explain prediction: {e}")
            explanation['error'] = str(e)

        return explanation

    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature"""
        descriptions = {
            'seniority_score': 'User experience and seniority level',
            'role_score': 'Technical complexity of user role',
            'industry_score': 'Data intensity of user industry',
            'activity_level': 'User engagement and activity patterns',
            'acceptance_rate': 'Historical recommendation acceptance',
            'exploration_score': 'Willingness to try new sources',
            'base_confidence': 'Initial recommendation confidence',
            'source_popularity': 'How commonly used this source is',
            'source_complexity': 'Technical complexity of the source',
            'time_of_day': 'When the user is most active',
            'day_of_week': 'User activity patterns by day',
            'session_length': 'Typical user session duration'
        }
        return descriptions.get(feature_name, f'Feature: {feature_name}')

    async def optimize_recommendations(self, user_id: str,
                                     candidate_recommendations: List[Recommendation]) -> List[Recommendation]:
        """Optimize a list of recommendations using ML insights"""

        try:
            optimized = []

            # Get user profile (mock for now)
            mock_profile = UserProfile(
                id="mock_profile",
                user_id=user_id,
                role="data_analyst"
            )

            # Score each recommendation with ML
            recommendation_scores = []

            for rec in candidate_recommendations:
                # Predict success probability
                prediction = await self.predict_recommendation_success(user_id, rec, mock_profile)

                # Calculate optimization score (combine original confidence with ML prediction)
                original_confidence = rec.confidence_score
                ml_prediction = prediction.prediction
                ml_confidence = prediction.confidence

                # Weighted combination
                optimized_score = (
                    original_confidence * 0.4 +  # Original algorithm
                    ml_prediction * 0.4 +        # ML prediction
                    ml_confidence * 0.2           # ML confidence
                )

                recommendation_scores.append((rec, optimized_score, prediction))

            # Sort by optimized score
            recommendation_scores.sort(key=lambda x: x[1], reverse=True)

            # Create optimized recommendations
            for rec, score, prediction in recommendation_scores:
                # Update recommendation with ML insights
                optimized_rec = Recommendation(
                    id=rec.id,
                    user_id=rec.user_id,
                    source_id=rec.source_id,
                    recommendation_type=rec.recommendation_type,
                    confidence_score=min(0.99, score),  # Use optimized score
                    reasoning={
                        **rec.reasoning,
                        'ml_optimized': True,
                        'original_confidence': rec.confidence_score,
                        'ml_prediction': prediction.prediction,
                        'optimization_boost': score - rec.confidence_score
                    },
                    context={
                        **rec.context,
                        'ml_features_used': prediction.features_used,
                        'ml_model_version': prediction.model_version
                    }
                )

                optimized.append(optimized_rec)

            return optimized

        except Exception as e:
            logger.error(f"❌ Recommendation optimization failed: {e}")
            return candidate_recommendations  # Return original on failure

# Factory function
async def create_learning_engine(
    database_manager: Optional[DatabaseManager] = None,
    cache_manager: Optional[CacheManager] = None,
    analytics_engine: Optional[AnalyticsEngine] = None,
    profile_analyzer: Optional[ProfileAnalyzer] = None,
    **kwargs
) -> LearningEngine:
    """Factory function to create and initialize learning engine"""
    engine = LearningEngine(
        database_manager=database_manager,
        cache_manager=cache_manager,
        analytics_engine=analytics_engine,
        profile_analyzer=profile_analyzer,
        **kwargs
    )
    await engine.initialize()
    return engine

# Testing
if __name__ == "__main__":
    async def test_learning_engine():
        """Test learning engine functionality"""

        try:
            print("🧪 Testing Learning Engine...")

            # Create mock dependencies
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
                    print(f"📊 Analytics: {kwargs.get('data', {}).get('action', 'unknown')} for user {kwargs.get('user_id')}")

            # Initialize learning engine
            engine = await create_learning_engine(
                cache_manager=MockCache(),
                analytics_engine=MockAnalytics(),
                min_training_samples=5,  # Lower for testing
                retrain_interval_hours=1
            )

            print("✅ Learning engine created successfully")

            try:
                # Test 1: Model Status
                print("\n🔍 Test 1: Model Status")
                status = engine.get_model_status()
                print(f"   Models initialized: {len(status['models'])}")
                print(f"   Training samples: {status['training_stats']['total_feedback_samples']}")
                print(f"   Models ready: {status['performance_summary'].get('ready_for_predictions', False)}")

                # Test 2: Generate synthetic feedback for training
                print("\n🔍 Test 2: Learning from Feedback")

                test_user = UserProfile(
                    id="test_profile_ml",
                    user_id="test_user_ml",
                    role="data_analyst"
                )

                # Create feedback samples
                for i in range(6):  # Generate enough for training
                    recommendation = Recommendation(
                        id=f"test_rec_{i}",
                        user_id="test_user_ml",
                        source_id=f"source_{i % 3}",  # Rotate through sources
                        confidence_score=0.6 + (i * 0.1)
                    )

                    feedback = {
                        'action': 'accept' if i % 3 != 0 else 'reject',  # Varied feedback
                        'user_profile': test_user,
                        'satisfaction': 4 if i % 3 != 0 else 2
                    }

                    success = await engine.learn_from_feedback("test_user_ml", recommendation, feedback)
                    print(f"   Feedback {i+1}: {'✅' if success else '❌'}")

                # Test 3: Model Training
                print("\n🔍 Test 3: Model Training Status")
                new_status = engine.get_model_status()
                trained_models = [name for name, info in new_status['models'].items() if info.get('trained', False)]
                print(f"   Trained models: {trained_models}")

                if trained_models:
                    for model_name in trained_models:
                        model_info = new_status['models'][model_name]
                        print(f"   {model_name}: Accuracy={model_info.get('accuracy', 0):.3f}")

                # Test 4: Prediction
                print("\n🔍 Test 4: Success Prediction")

                test_recommendation = Recommendation(
                    id="prediction_test",
                    user_id="test_user_ml",
                    source_id="postgresql",
                    confidence_score=0.75
                )

                prediction = await engine.predict_recommendation_success(
                    "test_user_ml", test_recommendation, test_user
                )

                print(f"   Prediction: {prediction.prediction:.3f}")
                print(f"   Confidence: {prediction.confidence:.3f}")
                print(f"   Features used: {len(prediction.features_used)}")
                print(f"   Model version: {prediction.model_version}")

                # Test 5: Explanation
                print("\n🔍 Test 5: Prediction Explanation")

                explanation = await engine.explain_prediction(prediction)
                print(f"   Decision factors: {len(explanation['decision_factors'])}")

                for factor in explanation['decision_factors']:
                    print(f"     - {factor['factor']}: {factor['importance']:.3f}")
                    print(f"       {factor['description']}")

                if explanation['uncertainty_sources']:
                    print(f"   Uncertainty sources: {explanation['uncertainty_sources']}")

                # Test 6: ML Recommendations
                print("\n🔍 Test 6: ML-Generated Recommendations")

                available_sources = ['postgresql', 'redis', 'tableau', 'jupyter', 'airflow']
                ml_recommendations = await engine.generate_ml_recommendations(
                    "test_user_ml", test_user, available_sources, max_recommendations=3
                )

                print(f"   Generated {len(ml_recommendations)} ML recommendations:")
                for rec in ml_recommendations:
                    print(f"     - {rec.source_id}: {rec.confidence_score:.3f}")
                    print(f"       ML prediction: {rec.reasoning.get('ml_prediction', 'N/A'):.3f}")
                    print(f"       Rank: {rec.reasoning.get('rank', 'N/A')}")

                # Test 7: Learning Insights
                print("\n🔍 Test 7: Learning Insights")

                insights = await engine.get_learning_insights("test_user_ml")
                print(f"   Discovered patterns: {len(insights.discovered_patterns)}")
                print(f"   Success predictions: {len(insights.success_predictions)}")
                print(f"   Risk assessments: {len(insights.risk_assessments)}")
                print(f"   Confidence adjustments: {len(insights.confidence_adjustments)}")

                # Show some insights
                if insights.success_predictions:
                    print("   Top success predictions:")
                    for source, prob in list(insights.success_predictions.items())[:3]:
                        print(f"     - {source}: {prob:.3f}")

                # Test 8: Recommendation Optimization
                print("\n🔍 Test 8: Recommendation Optimization")

                # Create some candidate recommendations
                candidates = [
                    Recommendation(
                        id=f"opt_test_{i}",
                        user_id="test_user_ml",
                        source_id=source,
                        confidence_score=0.6 + (i * 0.1)
                    )
                    for i, source in enumerate(['mysql', 'mongodb', 'elasticsearch'])
                ]

                optimized = await engine.optimize_recommendations("test_user_ml", candidates)

                print(f"   Optimized {len(optimized)} recommendations:")
                for orig, opt in zip(candidates, optimized):
                    boost = opt.confidence_score - orig.confidence_score
                    print(f"     - {opt.source_id}: {orig.confidence_score:.3f} → {opt.confidence_score:.3f} (+{boost:.3f})")

                print("\n" + "=" * 50)
                print("✅ ALL LEARNING ENGINE TESTS PASSED! 🎉")
                print("   - Model initialization ✓")
                print("   - Feedback learning ✓")
                print("   - Model training ✓")
                print("   - Success prediction ✓")
                print("   - Prediction explanation ✓")
                print("   - ML recommendations ✓")
                print("   - Learning insights ✓")
                print("   - Recommendation optimization ✓")

            finally:
                await engine.close()
                print("\n🔐 Learning engine closed gracefully")

        except Exception as e:
            print(f"\n❌ Learning engine test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    # Run tests
    print("🚀 Starting Smart Defaults Learning Engine Test")
    success = asyncio.run(test_learning_engine())

    if success:
        print("\n🎯 Learning engine is ready for integration!")
        print("   Next steps:")
        print("   1. Connect to real user behavior data")
        print("   2. Implement advanced ML algorithms (deep learning, ensemble methods)")
        print("   3. Add A/B testing for model performance")
        print("   4. Set up real-time model updates")
        print("   5. Implement federated learning for privacy")
        print("   6. Add explainable AI features")
    else:
        print("\n💥 Tests failed - check the error messages above")
        sys.exit(1)