# backend/main.py - Enhanced Analytics Backend with Advanced Charts Integration
"""
Enhanced Analytics Backend - Production Ready with Advanced Chart Support

This module integrates all enhanced analytics components including:
- Unified Smart Query Engine
- Adaptive Query Interpreter
- Mathematical Knowledge Engine
- Hybrid Knowledge Framework
- Enhanced Analytics Parts 1-4
- Advanced Chart Suggestion Engine (waterfall, funnel, gauge)
- Smart Defaults Engine

Author: Enhanced Analytics Team
Version: 5.1.1
Status: Production Ready with Advanced Chart Support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime, date, timedelta
import io
import os
import logging
import traceback
import warnings
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import re
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Import core analytics components
try:
    from mathematical_engine import MathematicalKnowledgeEngine
    from knowledge_framework import HybridKnowledgeFramework
    from adaptive_interpreter import (
        AdaptiveQueryInterpreter,
        create_adaptive_query_processor
    )
    from enhanced_analytics_part1 import (
        EnhancedAnalysisResult,
        IntelligentDataProcessor
    )
    from enhanced_analytics_part2 import IntelligentAggregator
    from enhanced_analytics_part3 import (
        IntelligentAnalysisEngine,
        BusinessAggregationEngine
    )
    from enhanced_analytics_part4a import AdvancedInsightGenerator
    from enhanced_analytics_part4b import (
        ComprehensiveReportBuilder,
        EnhancedAnalyticsSystem,
        ReportSection,
        ComprehensiveReport
    )

    # Enhanced Chart Suggestion Engine - Import with error handling
    try:
        import chart_suggestion_engine

        CHART_ENGINE_AVAILABLE = True
        logger.info("✅ Enhanced Chart Suggestion Engine available")
    except ImportError as chart_error:
        logger.warning(f"⚠️ Chart Suggestion Engine not available: {chart_error}")
        CHART_ENGINE_AVAILABLE = False

    logger.info("✅ Core analytics modules imported successfully")
    ENHANCED_ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import core modules: {e}")
    ENHANCED_ANALYTICS_AVAILABLE = False
    raise ImportError(f"Missing required analytics modules: {e}")

# Unified Smart Query Engine
try:
    from unified_smart_query_engine import smart_analyze_query

    SMART_ENGINE_AVAILABLE = True
    logger.info("✅ Unified Smart Query Engine available")
except ImportError as e:
    logger.warning(f"⚠️ Unified Smart Query Engine not available: {e}")
    SMART_ENGINE_AVAILABLE = False

# Smart Defaults Engine
try:
    from smart_defaults.engine import (
        SmartDefaultsEngine,
        EngineConfig,
        RecommendationRequest,
        RecommendationMode,
        create_smart_defaults_engine
    )

    SMART_DEFAULTS_AVAILABLE = True
    logger.info("✅ Smart Defaults Engine available")
except ImportError as e:
    logger.warning(f"⚠️ Smart Defaults Engine not available: {e}")
    SMART_DEFAULTS_AVAILABLE = False

    # ===== PHASE 1 IMPORTS =====
    from phase1_launcher import (
        Phase1DataGenieSystem,
        DataSensitivityLevel,
        AccessPermission,
        ConflictResolutionStrategy,
        SimpleDataSource,
        SimpleUser,
        AccessContext
    )

# Security
security = HTTPBearer()


# Chart utility functions - defined early to avoid unresolved references
def get_intent_metadata(prompt: str) -> Dict[str, Any]:
    """Get intent metadata with fallback"""
    if CHART_ENGINE_AVAILABLE:
        try:
            return chart_suggestion_engine.classify_intent(prompt)
        except Exception as e:
            logger.warning(f"Chart engine classify_intent failed: {e}")

    # Fallback intent detection
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ['waterfall', 'bridge', 'cumulative']):
        return {"intent": "waterfall_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['funnel', 'pipeline', 'conversion']):
        return {"intent": "funnel_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['gauge', 'meter', 'progress', 'kpi']):
        return {"intent": "gauge_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['pie', 'distribution', 'share']):
        return {"intent": "distribution_analysis", "confidence": 0.7}
    elif any(word in prompt_lower for word in ['trend', 'time', 'line']):
        return {"intent": "trend_analysis", "confidence": 0.7}
    else:
        return {"intent": "comparison_analysis", "confidence": 0.5}


def get_data_structure(chart_data: List[Dict]) -> Dict[str, Any]:
    """Get data structure analysis with fallback"""
    if CHART_ENGINE_AVAILABLE:
        try:
            return chart_suggestion_engine.analyze_data_structure(chart_data)
        except Exception as e:
            logger.warning(f"Chart engine analyze_data_structure failed: {e}")

    # Fallback data structure analysis
    if not chart_data:
        return {"numeric_fields": [], "categorical_fields": [], "datetime_fields": [], "row_count": 0}

    sample = chart_data[0]
    numeric_fields = []
    categorical_fields = []
    datetime_fields = []

    for key, value in sample.items():
        if isinstance(value, (int, float)):
            numeric_fields.append(key)
        elif isinstance(value, str):
            if any(pattern in key.lower() for pattern in ['date', 'time', 'year', 'month', 'day']):
                datetime_fields.append(key)
            else:
                categorical_fields.append(key)

    return {
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "datetime_fields": datetime_fields,
        "row_count": len(chart_data)
    }


def generate_waterfall_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate waterfall chart data"""
    waterfall_data = []
    for i, item in enumerate(chart_data[:8]):
        x_val = list(item.values())[0] if item else f"Step {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (100 - i * 10)

        # Make some values negative for waterfall effect
        if i % 3 == 2:
            y_val = -abs(float(y_val)) if isinstance(y_val, (int, float)) else -20

        waterfall_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (100 - i * 10)
        })

    return [{
        "id": f"waterfall_{datetime.now().timestamp()}",
        "title": "Waterfall Analysis",
        "type": "waterfall",
        "data": waterfall_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Cumulative effect analysis",
        "reasoning": "Waterfall chart showing progressive changes"
    }]


def generate_funnel_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate funnel chart data"""
    funnel_data = []
    for i, item in enumerate(chart_data[:5]):
        name = list(item.values())[0] if item else f"Stage {i + 1}"
        value = list(item.values())[1] if len(item.values()) > 1 else (1000 - i * 200)

        funnel_data.append({
            "name": str(name),
            "value": float(value) if isinstance(value, (int, float)) else (1000 - i * 200)
        })

    # Sort by value descending for funnel effect
    funnel_data.sort(key=lambda x: x['value'], reverse=True)

    return [{
        "id": f"funnel_{datetime.now().timestamp()}",
        "title": "Funnel Analysis",
        "type": "funnel",
        "data": funnel_data,
        "category_field": "name",
        "value_field": "value",
        "description": "Process stages visualization",
        "reasoning": "Funnel chart showing conversion rates"
    }]


def generate_gauge_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate gauge chart data"""
    # Find first numeric value
    value = 75  # default
    if chart_data:
        for item_value in chart_data[0].values():
            if isinstance(item_value, (int, float)):
                value = float(item_value)
                break

    return [{
        "id": f"gauge_{datetime.now().timestamp()}",
        "title": "Performance Gauge",
        "type": "gauge",
        "data": [{"value": value}],
        "value_field": "value",
        "description": "Key performance indicator",
        "reasoning": "Gauge chart showing single metric performance"
    }]


def generate_pie_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate pie chart data"""
    pie_data = []
    for i, item in enumerate(chart_data[:6]):
        name = list(item.values())[0] if item else f"Category {i + 1}"
        value = list(item.values())[1] if len(item.values()) > 1 else (100 - i * 15)

        pie_data.append({
            "name": str(name),
            "value": float(value) if isinstance(value, (int, float)) else (100 - i * 15)
        })

    return [{
        "id": f"pie_{datetime.now().timestamp()}",
        "title": "Distribution Analysis",
        "type": "pie",
        "data": pie_data,
        "category_field": "name",
        "value_field": "value",
        "description": "Data distribution breakdown",
        "reasoning": "Pie chart showing proportional distribution"
    }]


def generate_line_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate line chart data"""
    line_data = []
    for i, item in enumerate(chart_data[:15]):
        x_val = list(item.values())[0] if item else f"Point {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (50 + i * 5)

        line_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (50 + i * 5)
        })

    return [{
        "id": f"line_{datetime.now().timestamp()}",
        "title": "Trend Analysis",
        "type": "line",
        "data": line_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Data trends over time",
        "reasoning": "Line chart showing progression patterns"
    }]


def generate_bar_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate bar chart data"""
    bar_data = []
    for i, item in enumerate(chart_data[:10]):
        x_val = list(item.values())[0] if item else f"Category {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (30 + i * 8)

        bar_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (30 + i * 8)
        })

    return [{
        "id": f"bar_{datetime.now().timestamp()}",
        "title": "Comparison Analysis",
        "type": "bar",
        "data": bar_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Comparative data analysis",
        "reasoning": "Bar chart for comparing values across categories"
    }]


def generate_fallback_charts(chart_data: List[Dict], prompt: str) -> List[Dict]:
    """Generate fallback charts when advanced engine is not available"""
    prompt_lower = prompt.lower()

    # Basic intent detection
    if any(word in prompt_lower for word in ['waterfall', 'bridge', 'cumulative']):
        return generate_waterfall_chart(chart_data)
    elif any(word in prompt_lower for word in ['funnel', 'pipeline', 'conversion']):
        return generate_funnel_chart(chart_data)
    elif any(word in prompt_lower for word in ['gauge', 'meter', 'progress', 'kpi']):
        return generate_gauge_chart(chart_data)
    elif any(word in prompt_lower for word in ['pie', 'distribution', 'share']):
        return generate_pie_chart(chart_data)
    elif any(word in prompt_lower for word in ['trend', 'time', 'line']):
        return generate_line_chart(chart_data)
    else:
        return generate_bar_chart(chart_data)


def enhance_charts_with_advanced_types(chart_data: List[Dict], prompt: str) -> List[Dict]:
    """
    Enhanced chart processing that generates advanced chart types
    based on user intent and data characteristics
    """
    if not chart_data:
        return []

    try:
        if not CHART_ENGINE_AVAILABLE:
            # Fallback chart generation
            return generate_fallback_charts(chart_data, prompt)

        # Use the enhanced chart suggestion engine
        chart_result = chart_suggestion_engine.handle_query(prompt, chart_data)

        if 'suggested_charts' in chart_result:
            enhanced_charts = []

            for chart in chart_result['suggested_charts']:
                # Ensure chart has proper structure for frontend
                enhanced_chart = {
                    "id": f"chart_{datetime.now().timestamp()}_{len(enhanced_charts)}",
                    "title": chart.get('title', 'Data Visualization'),
                    "type": chart.get('type', 'bar'),
                    "chart_type": chart.get('chart_type', chart.get('type', 'bar')),
                    "data": chart.get('data', chart_data[:20]),
                    "x_field": chart.get('x_field', 'x'),
                    "y_field": chart.get('y_field', 'value'),
                    "category_field": chart.get('category_field', 'name'),
                    "value_field": chart.get('value_field', 'value'),
                    "description": chart.get('description', 'Data visualization'),
                    "reasoning": chart.get('reasoning', 'Generated based on data analysis'),
                    "colors": chart.get('colors', None),
                    "customConfig": chart.get('config', {})
                }
                enhanced_charts.append(enhanced_chart)

            logger.info(f"📊 Generated {len(enhanced_charts)} enhanced charts")
            return enhanced_charts

        return generate_fallback_charts(chart_data, prompt)

    except Exception as e:
        logger.warning(f"⚠️ Enhanced chart generation failed: {e}")
        return generate_fallback_charts(chart_data, prompt)


class AnalyticsSystemManager:
    """Manages all analytics system components"""

    def __init__(self):
        self.mathematical_engine = None
        self.knowledge_framework = None
        self.query_interpreter = None
        self.analysis_engine = None
        self.insight_generator = None
        self.report_builder = None
        self.enhanced_system = None
        self.adaptive_processor = None
        self.smart_defaults_engine = None
        self.is_initialized = False
        self.phase1_system = None  # Phase 1 Multi-Source system
        self.is_initialized = False
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Complete Analytics System...")

            self.mathematical_engine = MathematicalKnowledgeEngine()
            logger.info("📊 Mathematical engine initialized")

            self.knowledge_framework = HybridKnowledgeFramework(self.mathematical_engine)
            logger.info("🧠 Knowledge framework initialized")

            openai_key = os.getenv("OPENAI_API_KEY")
            self.query_interpreter = AdaptiveQueryInterpreter(openai_key)
            logger.info("🔍 Query interpreter initialized")

            self.insight_generator = AdvancedInsightGenerator(self.knowledge_framework)
            self.report_builder = ComprehensiveReportBuilder(self.insight_generator)
            logger.info("📋 Insight and reporting components initialized")

            self.enhanced_system = EnhancedAnalyticsSystem()
            logger.info("🎯 Enhanced analytics system initialized")

            if SMART_DEFAULTS_AVAILABLE:
                await self._initialize_smart_defaults()
                await self._initialize_phase1()
            self.is_initialized = True
            logger.info("✅ Complete Analytics System initialized successfully")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    async def _initialize_phase1(self):
        """Initialize Phase 1 Multi-Source DataGenie"""
        try:
            logger.info("🔗 Initializing Phase 1 Multi-Source DataGenie...")

            global phase1_system
            phase1_system = Phase1DataGenieSystem()
            self.phase1_system = phase1_system

            # Register additional real data sources if configured
            await self._register_configured_sources()

            logger.info("✅ Phase 1 Multi-Source DataGenie initialized")
        except Exception as e:
            logger.warning(f"⚠️ Phase 1 initialization failed: {e}")
            self.phase1_system = None

    async def _register_configured_sources(self):
        """Register data sources from environment configuration"""
        if not self.phase1_system:
            return

        # Example: Register PostgreSQL if configured
        if os.getenv("POSTGRES_PROD_URL"):
            source = SimpleDataSource(
                id="postgres_prod",
                name="Production PostgreSQL",
                source_type="postgresql",
                sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
                owner="data_team"
            )
            self.phase1_system.query_engine.register_source(source)
            self.phase1_system.conflict_engine.set_source_authority("postgres_prod", 0.95)

    async def _initialize_smart_defaults(self):
        """Initialize Smart Defaults Engine"""
        try:
            logger.info("🤖 Initializing Smart Defaults Engine...")
            smart_config = EngineConfig(
                enable_environment_scanning=True,
                enable_profile_analysis=True,
                enable_machine_learning=True,
                enable_policy_enforcement=True,
                enable_analytics=True,
                enable_notifications=True,
                default_recommendation_mode=RecommendationMode.BALANCED,
                min_confidence_threshold=0.6,
                auto_connect_threshold=0.8
            )
            self.smart_defaults_engine = await create_smart_defaults_engine(smart_config)
            logger.info("✅ Smart Defaults Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Smart Defaults initialization failed: {e}")
            self.smart_defaults_engine = None

    def create_adaptive_processor(self, datasets: Dict[str, pd.DataFrame]) -> tuple:
        """Create adaptive processor for datasets"""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            cache_path = "cache/adaptive_profile.pkl"
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            processor, interpreter = create_adaptive_query_processor(
                openai_api_key=openai_key,
                datasets=datasets,
                cache_path=cache_path
            )
            self.adaptive_processor = processor
            return processor, interpreter
        except Exception as e:
            logger.error(f"Failed to create adaptive processor: {e}")
            return None, None


# Global system manager
system_manager = AnalyticsSystemManager()
phase1_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Complete Analytics Backend...")
    await system_manager.initialize()
    yield
    logger.info("🛑 Shutting down Complete Analytics Backend...")
    if system_manager.smart_defaults_engine:
        await system_manager.smart_defaults_engine.close()


# FastAPI app setup
app = FastAPI(
    title="Complete Analytics Backend with Advanced Charts",
    description="Full analytics system with mathematical intelligence, domain knowledge, adaptive query processing, and advanced chart visualization support",
    version="5.1.1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
def log_timing(label: str, start_time: datetime) -> float:
    """Log execution timing"""
    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"⏱️ {label}: {elapsed_ms:.2f}ms")
    return elapsed_ms


def safe_json_convert(obj: Any) -> Any:
    """Safely convert objects to JSON-serializable format"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, '__dict__'):
        return {k: safe_json_convert(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def create_error_response(error: str, details: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": "error",
        "error": error,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "suggestions": [
            "Check data format and quality",
            "Verify query parameters",
            "Review system logs for details"
        ]
    }


def verify_openai_setup() -> tuple[bool, str]:
    """Verify OpenAI API key is available and valid"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment variables"
        if not api_key.startswith("sk-"):
            return False, "OPENAI_API_KEY appears to be invalid (should start with 'sk-')"
        if len(api_key) < 20:
            return False, "OPENAI_API_KEY appears to be too short"
        return True, "OpenAI API key found and appears valid"
    except Exception as e:
        return False, f"Error checking OpenAI setup: {e}"


def get_intent_metadata(prompt: str) -> Dict[str, Any]:
    """Get intent metadata with fallback"""
    if CHART_ENGINE_AVAILABLE:
        try:
            return chart_suggestion_engine.classify_intent(prompt)
        except Exception as e:
            logger.warning(f"Chart engine classify_intent failed: {e}")

    # Fallback intent detection
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ['waterfall', 'bridge', 'cumulative']):
        return {"intent": "waterfall_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['funnel', 'pipeline', 'conversion']):
        return {"intent": "funnel_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['gauge', 'meter', 'progress', 'kpi']):
        return {"intent": "gauge_analysis", "confidence": 0.8}
    elif any(word in prompt_lower for word in ['pie', 'distribution', 'share']):
        return {"intent": "distribution_analysis", "confidence": 0.7}
    elif any(word in prompt_lower for word in ['trend', 'time', 'line']):
        return {"intent": "trend_analysis", "confidence": 0.7}
    else:
        return {"intent": "comparison_analysis", "confidence": 0.5}


def get_data_structure(chart_data: List[Dict]) -> Dict[str, Any]:
    """Get data structure analysis with fallback"""
    if CHART_ENGINE_AVAILABLE:
        try:
            return chart_suggestion_engine.analyze_data_structure(chart_data)
        except Exception as e:
            logger.warning(f"Chart engine analyze_data_structure failed: {e}")

    # Fallback data structure analysis
    if not chart_data:
        return {"numeric_fields": [], "categorical_fields": [], "datetime_fields": [], "row_count": 0}

    sample = chart_data[0]
    numeric_fields = []
    categorical_fields = []
    datetime_fields = []

    for key, value in sample.items():
        if isinstance(value, (int, float)):
            numeric_fields.append(key)
        elif isinstance(value, str):
            if any(pattern in key.lower() for pattern in ['date', 'time', 'year', 'month', 'day']):
                datetime_fields.append(key)
            else:
                categorical_fields.append(key)

    return {
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "datetime_fields": datetime_fields,
        "row_count": len(chart_data)
    }
    """
    Enhanced chart processing that generates advanced chart types
    based on user intent and data characteristics
    """
    if not chart_data:
        return []

    try:
        if not CHART_ENGINE_AVAILABLE:
            # Fallback chart generation
            return generate_fallback_charts(chart_data, prompt)

        # Use the enhanced chart suggestion engine
        chart_result = chart_suggestion_engine.handle_query(prompt, chart_data)

        if 'suggested_charts' in chart_result:
            enhanced_charts = []

            for chart in chart_result['suggested_charts']:
                # Ensure chart has proper structure for frontend
                enhanced_chart = {
                    "id": f"chart_{datetime.now().timestamp()}_{len(enhanced_charts)}",
                    "title": chart.get('title', 'Data Visualization'),
                    "type": chart.get('type', 'bar'),
                    "chart_type": chart.get('chart_type', chart.get('type', 'bar')),
                    "data": chart.get('data', chart_data[:20]),
                    "x_field": chart.get('x_field', 'x'),
                    "y_field": chart.get('y_field', 'value'),
                    "category_field": chart.get('category_field', 'name'),
                    "value_field": chart.get('value_field', 'value'),
                    "description": chart.get('description', 'Data visualization'),
                    "reasoning": chart.get('reasoning', 'Generated based on data analysis'),
                    "colors": chart.get('colors', None),
                    "customConfig": chart.get('config', {})
                }
                enhanced_charts.append(enhanced_chart)

            logger.info(f"📊 Generated {len(enhanced_charts)} enhanced charts")
            return enhanced_charts

        return generate_fallback_charts(chart_data, prompt)

    except Exception as e:
        logger.warning(f"⚠️ Enhanced chart generation failed: {e}")
        return generate_fallback_charts(chart_data, prompt)


def enhance_charts_with_advanced_types(chart_data: List[Dict], prompt: str) -> List[Dict]:
    """Generate fallback charts when advanced engine is not available"""
    prompt_lower = prompt.lower()

    # Basic intent detection
    if any(word in prompt_lower for word in ['waterfall', 'bridge', 'cumulative']):
        return generate_waterfall_chart(chart_data)
    elif any(word in prompt_lower for word in ['funnel', 'pipeline', 'conversion']):
        return generate_funnel_chart(chart_data)
    elif any(word in prompt_lower for word in ['gauge', 'meter', 'progress', 'kpi']):
        return generate_gauge_chart(chart_data)
    elif any(word in prompt_lower for word in ['pie', 'distribution', 'share']):
        return generate_pie_chart(chart_data)
    elif any(word in prompt_lower for word in ['trend', 'time', 'line']):
        return generate_line_chart(chart_data)
    else:
        return generate_bar_chart(chart_data)


def generate_waterfall_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate waterfall chart data"""
    waterfall_data = []
    for i, item in enumerate(chart_data[:8]):
        x_val = list(item.values())[0] if item else f"Step {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (100 - i * 10)

        # Make some values negative for waterfall effect
        if i % 3 == 2:
            y_val = -abs(float(y_val)) if isinstance(y_val, (int, float)) else -20

        waterfall_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (100 - i * 10)
        })

    return [{
        "id": f"waterfall_{datetime.now().timestamp()}",
        "title": "Waterfall Analysis",
        "type": "waterfall",
        "data": waterfall_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Cumulative effect analysis",
        "reasoning": "Waterfall chart showing progressive changes"
    }]


def generate_funnel_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate funnel chart data"""
    funnel_data = []
    for i, item in enumerate(chart_data[:5]):
        name = list(item.values())[0] if item else f"Stage {i + 1}"
        value = list(item.values())[1] if len(item.values()) > 1 else (1000 - i * 200)

        funnel_data.append({
            "name": str(name),
            "value": float(value) if isinstance(value, (int, float)) else (1000 - i * 200)
        })

    # Sort by value descending for funnel effect
    funnel_data.sort(key=lambda x: x['value'], reverse=True)

    return [{
        "id": f"funnel_{datetime.now().timestamp()}",
        "title": "Funnel Analysis",
        "type": "funnel",
        "data": funnel_data,
        "category_field": "name",
        "value_field": "value",
        "description": "Process stages visualization",
        "reasoning": "Funnel chart showing conversion rates"
    }]


def generate_gauge_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate gauge chart data"""
    # Find first numeric value
    value = 75  # default
    if chart_data:
        for item_value in chart_data[0].values():
            if isinstance(item_value, (int, float)):
                value = float(item_value)
                break

    return [{
        "id": f"gauge_{datetime.now().timestamp()}",
        "title": "Performance Gauge",
        "type": "gauge",
        "data": [{"value": value}],
        "value_field": "value",
        "description": "Key performance indicator",
        "reasoning": "Gauge chart showing single metric performance"
    }]


def generate_pie_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate pie chart data"""
    pie_data = []
    for i, item in enumerate(chart_data[:6]):
        name = list(item.values())[0] if item else f"Category {i + 1}"
        value = list(item.values())[1] if len(item.values()) > 1 else (100 - i * 15)

        pie_data.append({
            "name": str(name),
            "value": float(value) if isinstance(value, (int, float)) else (100 - i * 15)
        })

    return [{
        "id": f"pie_{datetime.now().timestamp()}",
        "title": "Distribution Analysis",
        "type": "pie",
        "data": pie_data,
        "category_field": "name",
        "value_field": "value",
        "description": "Data distribution breakdown",
        "reasoning": "Pie chart showing proportional distribution"
    }]


def generate_line_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate line chart data"""
    line_data = []
    for i, item in enumerate(chart_data[:15]):
        x_val = list(item.values())[0] if item else f"Point {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (50 + i * 5)

        line_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (50 + i * 5)
        })

    return [{
        "id": f"line_{datetime.now().timestamp()}",
        "title": "Trend Analysis",
        "type": "line",
        "data": line_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Data trends over time",
        "reasoning": "Line chart showing progression patterns"
    }]


def generate_bar_chart(chart_data: List[Dict]) -> List[Dict]:
    """Generate bar chart data"""
    bar_data = []
    for i, item in enumerate(chart_data[:10]):
        x_val = list(item.values())[0] if item else f"Category {i + 1}"
        y_val = list(item.values())[1] if len(item.values()) > 1 else (30 + i * 8)

        bar_data.append({
            "x": str(x_val),
            "value": float(y_val) if isinstance(y_val, (int, float)) else (30 + i * 8)
        })

    return [{
        "id": f"bar_{datetime.now().timestamp()}",
        "title": "Comparison Analysis",
        "type": "bar",
        "data": bar_data,
        "x_field": "x",
        "y_field": "value",
        "description": "Comparative data analysis",
        "reasoning": "Bar chart for comparing values across categories"
    }]


def enhance_charts_with_advanced_types(chart_data: List[Dict], prompt: str) -> List[Dict]:
    """
    Enhanced chart processing that generates advanced chart types
    based on user intent and data characteristics
    """
    if not chart_data:
        return []

    try:
        if not CHART_ENGINE_AVAILABLE:
            # Fallback chart generation
            return generate_fallback_charts(chart_data, prompt)

        # Use the enhanced chart suggestion engine
        chart_result = chart_suggestion_engine.handle_query(prompt, chart_data)

        if 'suggested_charts' in chart_result:
            enhanced_charts = []

            for chart in chart_result['suggested_charts']:
                # Ensure chart has proper structure for frontend
                enhanced_chart = {
                    "id": f"chart_{datetime.now().timestamp()}_{len(enhanced_charts)}",
                    "title": chart.get('title', 'Data Visualization'),
                    "type": chart.get('type', 'bar'),
                    "chart_type": chart.get('chart_type', chart.get('type', 'bar')),
                    "data": chart.get('data', chart_data[:20]),
                    "x_field": chart.get('x_field', 'x'),
                    "y_field": chart.get('y_field', 'value'),
                    "category_field": chart.get('category_field', 'name'),
                    "value_field": chart.get('value_field', 'value'),
                    "description": chart.get('description', 'Data visualization'),
                    "reasoning": chart.get('reasoning', 'Generated based on data analysis'),
                    "colors": chart.get('colors', None),
                    "customConfig": chart.get('config', {})
                }
                enhanced_charts.append(enhanced_chart)

            logger.info(f"📊 Generated {len(enhanced_charts)} enhanced charts")
            return enhanced_charts

        return generate_fallback_charts(chart_data, prompt)

    except Exception as e:
        logger.warning(f"⚠️ Enhanced chart generation failed: {e}")
        return generate_fallback_charts(chart_data, prompt)


async def smart_query_handler(prompt: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Smart handler using unified query engine"""
    if not SMART_ENGINE_AVAILABLE:
        return None

    openai_available, openai_status = verify_openai_setup()
    if not openai_available:
        logger.warning(f"⚠️ OpenAI not available: {openai_status}")
        return None

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        smart_result = await smart_analyze_query(prompt, df, openai_api_key)

        if smart_result:
            confidence = smart_result.get('analysis', {}).get('metadata', {}).get('confidence', 0)
            logger.info(f"🎯 Smart engine confidence: {confidence:.2f}")

            if confidence >= 0.7:
                return smart_result
            else:
                logger.info("🔄 Smart engine confidence too low, using fallback")
                return None
        else:
            return None

    except Exception as e:
        logger.warning(f"⚠️ Smart engine failed: {e}")
        return None


# Security dependency
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Extract user ID from token - now optional"""
    if credentials:
        return "demo_user_" + str(hash(credentials.credentials))[:8]
    else:
        return f"anonymous_{int(datetime.now().timestamp())}"


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Complete Analytics Backend with Advanced Charts",
        "version": "5.1.1",
        "status": "operational" if system_manager.is_initialized else "initializing",
        "capabilities": {
            "adaptive_query_processing": True,
            "mathematical_analysis": True,
            "domain_knowledge": True,
            "intelligent_insights": True,
            "comprehensive_reporting": True,
            "advanced_chart_intelligence": True,
            "waterfall_charts": True,
            "funnel_charts": True,
            "gauge_charts": True,
            "smart_data_source_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "automated_recommendations": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE,
            "multi_source_queries": system_manager.phase1_system is not None,
            "data_governance": system_manager.phase1_system is not None,
            "smart_caching": system_manager.phase1_system is not None,
            "conflict_resolution": system_manager.phase1_system is not None,
            "rbac_abac_access_control": system_manager.phase1_system is not None
        },
        "supported_chart_types": [
            "line", "area", "bar", "pie", "scatter", "histogram",
            "waterfall", "funnel", "gauge"
        ],
        "endpoints": {
            "analyze": "/analyze/",
            "quick_analyze": "/quick-analyze/",
            "chart_suggestions": "/chart-suggestions/",
            "discover_sources": "/discover-sources/",
            "get_recommendations": "/recommendations/",
            "connect_source": "/connect-source/",
            "test_openai": "/test-openai/",
            "health": "/health/",
            "status": "/status/",
            "data_sources": "/data-sources/",
            "resolve_conflicts": "/conflicts/resolve/",
            "multi_source_status": "/multi-source/status/",
            "cache_stats": "/cache/stats/",
            "cache_invalidate": "/cache/invalidate/",
        },
        # ADD THIS:
            "multi_source_info": {
            "enabled": system_manager.phase1_system is not None,
            "sources_count": len(
            system_manager.phase1_system.query_engine._data_sources) if system_manager.phase1_system else 0,
            "conflict_strategies": [s.value for s in ConflictResolutionStrategy] if system_manager.phase1_system else []
        }
    }
                             # Smart Defaults Endpoints
@app.get("/discover-sources/")
async def discover_data_sources(
        mode: str = "balanced",
        include_environment_scan: bool = True,
        max_recommendations: int = 10,
        user_id: Optional[str] = None
):
    """Discover available data sources automatically"""
    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        return {
            "status": "success",
            "message": "Smart Defaults Engine not available - using fallback data sources",
            "discovered_sources": 4,
            "recommendations": [
                {
                    "id": "postgres_prod",
                    "source_id": "postgres_prod",
                    "type": "database",
                    "confidence": 0.92,
                    "reasoning": "Production PostgreSQL database detected",
                    "context": {"type": "PostgreSQL", "host": "localhost", "port": 5432}
                },
                {
                    "id": "sales_csv",
                    "source_id": "sales_csv",
                    "type": "file",
                    "confidence": 0.87,
                    "reasoning": "CSV files found in data directory",
                    "context": {"type": "CSV", "location": "/data/sales/"}
                },
                {
                    "id": "tableau_server",
                    "source_id": "tableau_server",
                    "type": "bi_tool",
                    "confidence": 0.78,
                    "reasoning": "Tableau Server connection available",
                    "context": {"type": "Tableau", "server": "tableau.company.com"}
                },
                {
                    "id": "api_crm",
                    "source_id": "api_crm",
                    "type": "api",
                    "confidence": 0.85,
                    "reasoning": "CRM API endpoint accessible",
                    "context": {"type": "REST API", "endpoint": "api.crm.company.com"}
                }
            ],
            "metadata": {
                "total_candidates": 4,
                "policy_filtered": 0,
                "ml_enhanced": False,
                "confidence_distribution": {"high": 2, "medium": 2, "low": 0},
                "generated_at": datetime.now().isoformat()
            }
        }

    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        rec_mode = RecommendationMode.BALANCED
        if mode.lower() == "conservative":
            rec_mode = RecommendationMode.CONSERVATIVE
        elif mode.lower() == "aggressive":
            rec_mode = RecommendationMode.AGGRESSIVE

        request = RecommendationRequest(
            user_id=user_id,
            mode=rec_mode,
            max_recommendations=max_recommendations,
            include_environment_scan=include_environment_scan,
            include_ml_recommendations=True,
            force_refresh=False
        )

        response = await system_manager.smart_defaults_engine.get_recommendations(request)

        return {
            "status": "success",
            "user_id": user_id,
            "discovered_sources": len(response.recommendations),
            "recommendations": [
                {
                    "id": rec.id,
                    "source_id": rec.source_id,
                    "type": rec.recommendation_type,
                    "confidence": rec.confidence_score,
                    "reasoning": rec.reasoning,
                    "context": rec.context
                } for rec in response.recommendations
            ],
            "metadata": {
                "total_candidates": response.total_candidates,
                "policy_filtered": response.filtered_by_policy,
                "ml_enhanced": response.enhanced_by_ml,
                "confidence_distribution": response.confidence_distribution,
                "generated_at": response.generated_at.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Source discovery failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "discovered_sources": [],
            "recommendations": []
        }


@app.post("/connect-source/")
async def connect_data_source(
        source_id: str,
        connection_params: Dict[str, Any],
        user_id: Optional[str] = None
):
    """Connect to a discovered data source"""
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
            await system_manager.smart_defaults_engine.record_feedback(
                user_id=user_id,
                recommendation_id=f"connect_{source_id}",
                action="connected",
                context={
                    "source_id": source_id,
                    "connection_params": connection_params,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return {
            "status": "success",
            "source_id": source_id,
            "connected": True,
            "message": f"Successfully connected to {source_id}",
            "connection_details": {
                "source_type": connection_params.get("type", "unknown"),
                "connected_at": datetime.now().isoformat(),
                "user_id": user_id
            }
        }
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "connected": False
        }


# MAIN ANALYZE ENDPOINT - ENHANCED WITH ADVANCED CHARTS
@app.post("/analyze/")
async def comprehensive_analyze(
        prompt: str = Form(...),
        file: UploadFile = File(None),
        data_source_id: Optional[str] = Form(None),
        use_multi_source: bool = Form(False),  # Enable multi-source query
        sources: Optional[str] = Form(None),    # Comma-separated source IDs
        user_id: Optional[str] = Form(None),
        domain: Optional[str] = Form(None),
        use_adaptive: bool = Form(True),
        include_charts: bool = Form(True),
        auto_discover: bool = Form(True),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    PRODUCTION: Comprehensive analysis with advanced chart support (waterfall, funnel, gauge)
    """
    if not system_manager.is_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")

    start_time = datetime.now()

    try:
        logger.info(f"🧠 Starting comprehensive analysis: '{prompt}'")
        # Extract user info from credentials
        if credentials:
            user_id = user_id or f"user_{hash(credentials.credentials) % 10000}"
            user_role = "analyst"  # In production, extract from JWT
        else:
            user_id = user_id or "anonymous"
            user_role = "viewer"

        # ===== MULTI-SOURCE PATH =====
        if use_multi_source and not file and system_manager.phase1_system:
            logger.info("🔗 Using Phase 1 Multi-Source Query Engine")

            # Parse sources list
            source_list = None
            if sources:
                source_list = [s.strip() for s in sources.split(",")]

            # Execute multi-source query with governance
            multi_source_result = await system_manager.phase1_system.analyze_with_governance(
                query=prompt,
                user_id=user_id,
                user_role=user_role,
                purpose=domain or "analysis",
                sources=source_list
            )

            # Extract DataFrame from multi-source result
            if multi_source_result["status"] == "success":
                df = multi_source_result.get("data", pd.DataFrame())

                # Store multi-source metadata
                multi_source_metadata = {
                    "sources_queried": multi_source_result["metadata"]["sources_queried"],
                    "sources_succeeded": multi_source_result["metadata"]["sources_succeeded"],
                    "conflicts_detected": multi_source_result["metadata"]["conflicts_detected"],
                    "conflicts_resolved": multi_source_result["metadata"]["conflicts_resolved"],
                    "access_decisions": multi_source_result["metadata"]["access_decisions"]
                }
            else:
                # Handle multi-source errors
                return JSONResponse(content={
                    "status": "error",
                    "error": multi_source_result.get("error", "Multi-source query failed"),
                    "details": multi_source_result,
                    "timestamp": datetime.now().isoformat()
                }, status_code=400)

        # ===== SINGLE SOURCE/FILE PATH (existing code) =====
        else:
            multi_source_metadata = None
            # ... rest of existing data loading logic ...
        # Data loading logic
        df = None
        discovery_metadata = {}

        if file:
            content = await file.read()
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(content))
                elif file.filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(content))
                else:
                    raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        elif data_source_id:
            if system_manager.smart_defaults_engine:
                await system_manager.smart_defaults_engine.record_feedback(
                    user_id=user_id or "anonymous",
                    recommendation_id=f"use_{data_source_id}",
                    action="used_for_analysis",
                    context={"prompt": prompt, "timestamp": datetime.now().isoformat()}
                )
            # Create sample data for demo
            df = pd.DataFrame({
                'value1': np.random.normal(100, 15, 50),
                'value2': np.random.normal(200, 25, 50),
                'category': np.random.choice(['A', 'B', 'C'], 50)
            })
            discovery_metadata["source"] = data_source_id
        elif auto_discover and system_manager.smart_defaults_engine:
            logger.info("🔍 Auto-discovering data sources...")
            request = RecommendationRequest(
                user_id=user_id or "anonymous",
                mode=RecommendationMode.BALANCED,
                max_recommendations=1,
                context={"analysis_prompt": prompt}
            )
            recommendations = await system_manager.smart_defaults_engine.get_recommendations(request)
            if recommendations.recommendations:
                best_rec = recommendations.recommendations[0]
                discovery_metadata = {
                    "auto_discovered": True,
                    "source_id": best_rec.source_id,
                    "confidence": best_rec.confidence_score,
                    "reasoning": best_rec.reasoning
                }
                df = pd.DataFrame({
                    'metric1': np.random.normal(50, 10, 100),
                    'metric2': np.random.normal(75, 15, 100),
                    'segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], 100),
                    'date': pd.date_range('2024-01-01', periods=100, freq='D')
                })
            else:
                raise HTTPException(status_code=400, detail="No suitable data sources found")
        else:
            raise HTTPException(status_code=400,
                                detail="Please provide a file, data_source_id, or enable auto_discover")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        logger.info(f"📊 Dataset ready: {df.shape[0]} rows × {df.shape[1]} columns")

        # 🤖 SMART QUERY ANALYSIS - Enhanced with advanced charts
        smart_result = await smart_query_handler(prompt, df)
        if smart_result:
            logger.info("🎯 Using unified smart query engine")

            # ENHANCED CHART PROCESSING
            if include_charts:
                try:
                    chart_data = df.head(1000).to_dict('records')

                    # Use enhanced chart suggestion engine
                    enhanced_charts = enhance_charts_with_advanced_types(chart_data, prompt)

                    smart_result['chart_intelligence'] = {
                        "suggested_charts": enhanced_charts,
                        "intent_metadata": get_intent_metadata(prompt),
                        "chart_count": len(enhanced_charts),
                        "data_structure": get_data_structure(chart_data),
                        "advanced_charts_supported": True
                    }

                    logger.info(f"📊 Enhanced charts generated: {len(enhanced_charts)} charts")

                except Exception as e:
                    logger.warning(f"Enhanced chart generation failed: {e}")

            # Add performance metadata
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            smart_result['performance'] = {
                'total_time_ms': total_time,
                'method': 'smart_engine_enhanced_charts',
                'data_stats': {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            smart_result['system_info'] = {
                'method': 'smart_engine',
                'openai_used': True,
                'confidence_threshold': 0.7,
                'version': '5.1.1',
                'advanced_charts': True
            }

            logger.info(f"🎉 Smart engine completed in {total_time:.2f}ms")
            return JSONResponse(content=smart_result)

        # Continue with existing analysis pipeline
        interpretation_start = datetime.now()

        if use_adaptive and system_manager.query_interpreter:
            try:
                datasets = {"main_dataset": df}
                adaptive_processor, adaptive_interpreter = system_manager.create_adaptive_processor(datasets)

                if adaptive_processor:
                    adaptive_result = adaptive_processor(prompt, "main_dataset")
                    query_interpretation = adaptive_result.get('interpretation', {})
                    logger.info(
                        f"🎯 Adaptive interpretation confidence: {query_interpretation.get('confidence', 0):.2f}")
                else:
                    query_interpretation = system_manager.query_interpreter.parse(prompt)
                    logger.info("📝 Using basic query interpretation")
            except Exception as e:
                logger.warning(f"Adaptive processing failed, using fallback: {e}")
                query_interpretation = {"intent": "summary", "metric": "custom", "granularity": "total"}
        else:
            query_interpretation = {"intent": "summary", "metric": "custom", "granularity": "total"}

        interpretation_time = log_timing("Query interpretation", interpretation_start)

        # Enhanced analytics processing
        analysis_start = datetime.now()

        try:
            data_processor = IntelligentDataProcessor(
                datasets={"main": df},
                knowledge_framework=system_manager.knowledge_framework
            )

            analysis_engine = IntelligentAnalysisEngine(
                knowledge_framework=system_manager.knowledge_framework,
                mathematical_engine=system_manager.mathematical_engine,
                data_processor=data_processor
            )

            analysis_result = analysis_engine.analyze_query(
                query=prompt,
                filters=query_interpretation.get('filters', {}),
                domain=domain
            )

            logger.info(f"✅ Analysis completed: {analysis_result.analysis_type}")

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            analysis_result = EnhancedAnalysisResult(
                analysis_type="basic_fallback",
                summary=f"Basic analysis for: {prompt}",
                data=df.head(100).to_dict('records'),
                insights=[f"Analysis completed with {len(df)} records",
                          f"Dataset contains {len(df.columns)} variables"],
                metadata={"fallback_used": True, "original_error": str(e)},
                performance_stats={"execution_time_seconds": 0}
            )

        analysis_time = log_timing("Enhanced analysis", analysis_start)

        # Enhanced insight generation
        insight_start = datetime.now()

        try:
            if system_manager.insight_generator:
                enhanced_insights = system_manager.insight_generator.generate_insights(
                    analysis_result, df, domain
                )
                logger.info(f"💡 Generated {len(enhanced_insights)} enhanced insights")
            else:
                enhanced_insights = analysis_result.insights
        except Exception as e:
            logger.warning(f"Insight generation failed: {e}")
            enhanced_insights = analysis_result.insights

        insight_time = log_timing("Insight generation", insight_start)

        # Comprehensive report generation
        report_start = datetime.now()

        try:
            if system_manager.report_builder:
                comprehensive_report = system_manager.report_builder.generate_comprehensive_report(
                    analysis_result, df, prompt, domain
                )
                logger.info(f"📄 Generated comprehensive report with {len(comprehensive_report.sections)} sections")
            else:
                comprehensive_report = None
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            comprehensive_report = None

        report_time = log_timing("Report generation", report_start)

        # ENHANCED CHART INTELLIGENCE WITH ADVANCED TYPES
        chart_start = datetime.now()
        chart_suggestions = {}

        if include_charts:
            try:
                chart_data = df.head(1000).to_dict('records')

                # Use enhanced chart suggestion engine
                enhanced_charts = enhance_charts_with_advanced_types(chart_data, prompt)

                # Get intent and data structure metadata
                intent_metadata = get_intent_metadata(prompt)
                data_structure = get_data_structure(chart_data)

                chart_suggestions = {
                    "suggested_charts": enhanced_charts,
                    "intent_metadata": intent_metadata,
                    "chart_count": len(enhanced_charts),
                    "data_structure": data_structure,
                    "advanced_charts_supported": CHART_ENGINE_AVAILABLE
                }

                logger.info(f"📊 Generated {chart_suggestions['chart_count']} enhanced chart suggestions")

            except Exception as e:
                logger.warning(f"Enhanced chart suggestion failed: {e}")
                chart_suggestions = {"error": str(e), "advanced_charts_supported": False}

        chart_time = log_timing("Enhanced chart suggestions", chart_start)

        # Package complete response
        total_time = log_timing("Total analysis", start_time)

        response = {
            "status": "success",
            "analysis": {
                "type": analysis_result.analysis_type,
                "summary": analysis_result.summary,
                "data": analysis_result.data[:500],
                "insights": enhanced_insights,
                "metadata": safe_json_convert(analysis_result.metadata)
            },
            "multi_source_info": multi_source_metadata,
            "query_interpretation": safe_json_convert(query_interpretation),
            # ... rest of existing response ...
            "system_info": {
                "version": "5.1.1",  # Update to "6.0.0"
                # ADD THESE:
            "multi_source_enabled": use_multi_source and system_manager.phase1_system is not None,
            "governance_applied": use_multi_source
            },
            "query_interpretation": safe_json_convert(query_interpretation),
            "comprehensive_report": safe_json_convert(comprehensive_report) if comprehensive_report else None,
            "chart_intelligence": chart_suggestions,
            "data_discovery": discovery_metadata,
            "performance": {
                "total_time_ms": total_time,
                "breakdown": {
                    "data_loading_ms": 0,
                    "interpretation_ms": interpretation_time,
                    "analysis_ms": analysis_time,
                    "insights_ms": insight_time,
                    "reporting_ms": report_time,
                    "charts_ms": chart_time
                },
                "data_stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            },
            "system_info": {
                "version": "5.1.1",
                "adaptive_processing_used": use_adaptive,
                "chart_intelligence_used": include_charts,
                "smart_defaults_enabled": system_manager.smart_defaults_engine is not None,
                "auto_discovery_used": bool(discovery_metadata),
                "smart_engine_available": SMART_ENGINE_AVAILABLE,
                "advanced_charts_supported": True,
                "domain": domain,
                "user_id": user_id
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"🎉 Comprehensive analysis completed in {total_time:.2f}ms")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Comprehensive analysis failed")
        logger.error(traceback.format_exc())

        error_response = create_error_response(
            error="Comprehensive analysis failed",
            details=str(e)
        )
        error_response["performance"] = {
            "total_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }

        return JSONResponse(content=error_response, status_code=500)


# Test endpoints
@app.get("/test-openai/")
async def test_openai_setup():
    """Test endpoint to verify OpenAI API key is working"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "OPENAI_API_KEY not found in environment variables",
                "suggestions": [
                    "Check your .env file exists",
                    "Verify .env contains OPENAI_API_KEY=sk-...",
                    "Restart your server after adding the key"
                ]
            }

        if not api_key.startswith("sk-"):
            return {
                "status": "error",
                "message": "OPENAI_API_KEY appears invalid (should start with 'sk-')",
                "key_preview": f"{api_key[:10]}..." if len(api_key) > 10 else "too_short"
            }

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'OpenAI connection test successful'"}],
            max_tokens=20,
            timeout=10
        )

        response_text = test_response.choices[0].message.content

        return {
            "status": "success",
            "message": "OpenAI API key is working correctly",
            "test_response": response_text,
            "key_preview": f"{api_key[:10]}...{api_key[-4:]}",
            "model_used": "gpt-3.5-turbo",
            "smart_engine_ready": SMART_ENGINE_AVAILABLE,
            "advanced_charts_ready": True
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI API test failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Check your API key is correct",
                "Verify you have OpenAI credits/quota",
                "Check your internet connection",
                "Try restarting the server"
            ]
        }


@app.post("/quick-analyze/")
async def quick_analyze(
        prompt: str,
        file: UploadFile = File(...),
        max_rows: int = 1000
):
    """Quick analysis for rapid insights with advanced chart support"""
    start_time = datetime.now()

    try:
        logger.info(f"⚡ Quick analysis: '{prompt}'")

        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=max_rows)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), nrows=max_rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        summary = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            },
            "data_types": {
                "numeric": len(df.select_dtypes(include=[np.number]).columns),
                "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(df.select_dtypes(include=['datetime']).columns)
            },
            "data_sample": df.head(5).to_dict('records'),
            "missing_data": df.isnull().sum().to_dict()
        }

        insights = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                insights.append(f"{col}: mean={mean_val:.2f}, std={std_val:.2f}")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:3]:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                insights.append(f"{col}: {unique_count} unique values, most common: {most_common}")

        # Enhanced chart processing
        chart_data = df.head(100).to_dict('records')
        enhanced_charts = enhance_charts_with_advanced_types(chart_data, prompt)

        response = {
            "status": "success",
            "analysis_type": "quick_summary",
            "summary": summary,
            "insights": insights,
            "chart_suggestions": enhanced_charts,
            "intent_metadata": get_intent_metadata(prompt),
            "advanced_charts_supported": CHART_ENGINE_AVAILABLE,
            "performance": {
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "rows_analyzed": len(df)
            },
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return JSONResponse(
            content=create_error_response("Quick analysis failed", str(e)),
            status_code=500
        )


@app.post("/chart-suggestions/")
async def get_chart_suggestions(
        prompt: str,
        file: UploadFile = File(...),
        max_rows: int = 1000
):
    """Get enhanced chart suggestions with advanced chart types"""
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=max_rows)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), nrows=max_rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        chart_data = df.to_dict('records')

        # Use enhanced chart suggestion engine
        enhanced_charts = enhance_charts_with_advanced_types(chart_data, prompt)

        # Get analysis data
        data_structure = get_data_structure(chart_data)
        intent_analysis = get_intent_metadata(prompt)

        response = {
            "status": "success",
            "chart_suggestions": enhanced_charts,
            "intent_analysis": intent_analysis,
            "data_structure": data_structure,
            "advanced_charts_supported": CHART_ENGINE_AVAILABLE,
            "supported_chart_types": [
                "line", "area", "bar", "pie", "scatter", "histogram",
                "waterfall", "funnel", "gauge"
            ],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }

        if len(data_structure["numeric_fields"]) > 1:
            response["recommendations"].append("Consider correlation analysis between numeric variables")
        if len(data_structure["categorical_fields"]) > 0:
            response["recommendations"].append("Categorical data suitable for grouping and segmentation")
        if len(data_structure["datetime_fields"]) > 0:
            response["recommendations"].append("Time series analysis possible with datetime data")

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Chart suggestions failed: {e}")
        return JSONResponse(
            content=create_error_response("Chart suggestions failed", str(e)),
            status_code=500
        )


@app.get("/health/")
async def health_check():
    """Enhanced health check"""
    health_status = {
        "status": "healthy" if system_manager.is_initialized else "initializing",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "mathematical_engine": system_manager.mathematical_engine is not None,
            "knowledge_framework": system_manager.knowledge_framework is not None,
            "query_interpreter": system_manager.query_interpreter is not None,
            "analysis_engine": system_manager.analysis_engine is not None,
            "insight_generator": system_manager.insight_generator is not None,
            "report_builder": system_manager.report_builder is not None,
            "enhanced_system": system_manager.enhanced_system is not None,
            "smart_defaults_engine": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE,
            "advanced_chart_engine": CHART_ENGINE_AVAILABLE,
            "phase1_multi_source": system_manager.phase1_system is not None,
            "phase1_governance": system_manager.phase1_system is not None if system_manager.phase1_system else False,
            "phase1_cache": system_manager.phase1_system is not None if system_manager.phase1_system else False,
        },
        "openai_status": {
            "available": verify_openai_setup()[0],
            "status": verify_openai_setup()[1]
        },
        "chart_capabilities": {
            "basic_charts": ["line", "bar", "area", "pie", "scatter"],
            "advanced_charts": ["waterfall", "funnel", "gauge", "histogram"],
            "chart_intelligence": True
        }
    }

    if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
        try:
            smart_health = await system_manager.smart_defaults_engine.health_check()
            health_status["smart_defaults_health"] = smart_health
        except Exception as e:
            health_status["smart_defaults_error"] = str(e)

    return health_status


@app.get("/status/")
async def system_status():
    """Enhanced system status"""
    if not system_manager.is_initialized:
        return {"status": "not_initialized", "message": "System is still initializing"}

    try:
        status = {
            "system_version": "5.1.1",
            "initialization_status": "complete",
            "components": {},
            "capabilities": {},
            "performance_metrics": {},
            "chart_engine": {
                "status": "operational",
                "supported_types": [
                    "line", "area", "bar", "pie", "scatter", "histogram",
                    "waterfall", "funnel", "gauge"
                ],
                "advanced_features": {
                    "intent_classification": True,
                    "data_structure_analysis": True,
                    "automatic_chart_selection": True,
                    "openai_integration": verify_openai_setup()[0]
                }
            }
        }

        if system_manager.enhanced_system:
            system_status_data = system_manager.enhanced_system.get_system_status()
            status["components"] = system_status_data.get("components", {})
            status["capabilities"] = system_status_data.get("capabilities", {})

        if system_manager.knowledge_framework:
            knowledge_summary = system_manager.knowledge_framework.get_knowledge_summary()
            status["knowledge_summary"] = knowledge_summary

        if system_manager.mathematical_engine:
            status["mathematical_methods"] = len(system_manager.mathematical_engine.methods_registry)

        status["smart_engine"] = {
            "available": SMART_ENGINE_AVAILABLE,
            "openai_configured": verify_openai_setup()[0],
            "openai_status": verify_openai_setup()[1]
        }

        if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
            try:
                smart_stats = await system_manager.smart_defaults_engine.get_engine_stats()
                status["smart_defaults"] = {
                    "engine_status": smart_stats.get("engine", {}).get("status"),
                    "components_enabled": smart_stats.get("engine", {}).get("config", {}).get("components_enabled", {}),
                    "machine_learning": smart_stats.get("machine_learning", {}),
                    "policy": smart_stats.get("policy", {})
                }
            except Exception as e:
                status["smart_defaults"] = {"error": str(e)}

        return status

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return create_error_response("Failed to get system status", str(e))


@app.get("/capabilities/")
async def get_capabilities():
    """Enhanced capabilities including advanced charts"""
    capabilities = {
        "analysis_types": [
            "trend_analysis", "distribution_analysis", "comparison_analysis",
            "correlation_analysis", "statistical_analysis", "business_aggregation",
            "predictive_modeling", "waterfall_analysis", "funnel_analysis",
            "gauge_analysis", "performance_analysis"
        ],
        "chart_types": [
            "line", "bar", "pie", "scatter", "heatmap", "histogram",
            "box", "area", "waterfall", "funnel", "gauge", "treemap", "radar"
        ],
        "advanced_chart_features": {
            "waterfall_charts": {
                "supported": True,
                "description": "Shows cumulative effect of sequential positive and negative values",
                "use_cases": ["Financial analysis", "Performance breakdown", "Variance analysis"]
            },
            "funnel_charts": {
                "supported": True,
                "description": "Visualizes stages in a process with progressive filtering",
                "use_cases": ["Sales pipeline", "Conversion analysis", "Process optimization"]
            },
            "gauge_charts": {
                "supported": True,
                "description": "Shows a single metric against a scale",
                "use_cases": ["KPI dashboards", "Performance monitoring", "Goal tracking"]
            }
        },
        "data_formats": ["CSV", "Excel (.xlsx, .xls)"],
        "data_sources": [
            "File Upload", "PostgreSQL", "MySQL", "Redis", "MongoDB",
            "Tableau", "Power BI", "Jupyter", "S3", "API Endpoints"
        ],
        "domains": [
            "insurance", "banking", "technology", "healthcare",
            "finance", "manufacturing", "retail", "general"
        ],
        "smart_features": {
            "auto_data_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "intelligent_recommendations": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE,
            "llm_powered_query_understanding": SMART_ENGINE_AVAILABLE and verify_openai_setup()[0],
            "advanced_chart_intelligence": CHART_ENGINE_AVAILABLE,
            "intent_based_chart_selection": CHART_ENGINE_AVAILABLE,
            "multi_source_governance": system_manager.phase1_system is not None,
            "conflict_resolution": system_manager.phase1_system is not None,
            "smart_caching": system_manager.phase1_system is not None,
            "role_based_access": system_manager.phase1_system is not None,
        },
        "features": {
            "adaptive_query_processing": True,
            "domain_knowledge_integration": True,
            "mathematical_analysis": True,
            "intelligent_insights": True,
            "comprehensive_reporting": True,
            "advanced_chart_intelligence": True,
            "real_time_processing": True,
            "multi_format_support": True,
            "automated_data_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_query_engine": SMART_ENGINE_AVAILABLE,
            "llm_powered_analysis": SMART_ENGINE_AVAILABLE and verify_openai_setup()[0]
        }
    }

    if system_manager.mathematical_engine and system_manager.is_initialized:
        try:
            methods = []
            for category, method_dict in system_manager.mathematical_engine.methods_registry.items():
                if isinstance(method_dict, dict):
                    methods.extend(list(method_dict.keys()))
            capabilities["mathematical_methods"] = methods
        except:
            pass

    if system_manager.knowledge_framework and system_manager.is_initialized:
        try:
            knowledge_summary = system_manager.knowledge_framework.get_knowledge_summary()
            capabilities["knowledge_domains"] = knowledge_summary.get("supported_domains", [])
        except:
            pass

    return capabilities


@app.get("/data-sources/")
async def list_data_sources(
        check_access: bool = True,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List available data sources with optional access check"""
    if not system_manager.phase1_system:
        return {"sources": [], "message": "Multi-source system not initialized"}

    user_id = "anonymous"
    user_role = "viewer"

    if credentials:
        user_id = f"user_{hash(credentials.credentials) % 10000}"
        user_role = "analyst"  # Extract from JWT in production

    sources = []
    for source_id, source in system_manager.phase1_system.query_engine._data_sources.items():
        source_info = {
            "id": source.id,
            "name": source.name,
            "type": source.source_type,
            "sensitivity": source.sensitivity_level.value,
            "active": source.is_active
        }

        if check_access and credentials:
            # Check user's access to this source
            user = SimpleUser(id=user_id, username=user_id, email=f"{user_id}@example.com", role=user_role)
            context = AccessContext(user_id=user_id)
            resource_id = f"source:{source_id}:{source.sensitivity_level.value}"

            evaluation = system_manager.phase1_system.access_engine.evaluate_access(
                user, resource_id, AccessPermission.READ, context
            )

            source_info["access"] = {
                "granted": evaluation.decision.value != "deny",
                "decision": evaluation.decision.value,
                "conditions": evaluation.conditions,
                "masked_fields": evaluation.masked_fields,
                "row_limit": evaluation.row_limit
            }

        sources.append(source_info)

    return {
        "sources": sources,
        "total": len(sources),
        "user_role": user_role if check_access else None
    }


@app.post("/conflicts/resolve/")
async def resolve_conflict_feedback(
        conflict_id: str = Form(...),
        chosen_value: str = Form(...),
        source_id: str = Form(...),
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Record user's conflict resolution choice"""
    if not system_manager.phase1_system:
        raise HTTPException(status_code=503, detail="Multi-source system not initialized")

    try:
        await system_manager.phase1_system.conflict_engine.record_user_feedback(
            conflict_id=conflict_id,
            chosen_value=chosen_value,
            source_id=source_id
        )

        return {
            "status": "success",
            "message": "Conflict resolution feedback recorded",
            "conflict_id": conflict_id
        }
    except Exception as e:
        logger.error(f"Failed to record conflict feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/multi-source/status/")
async def get_multi_source_status():
    """Get Phase 1 multi-source system status"""
    if not system_manager.phase1_system:
        return {
            "status": "not_initialized",
            "message": "Multi-source system is not available"
        }

    status = await system_manager.phase1_system.get_system_status()
    return status


@app.get("/cache/stats/")
async def get_cache_statistics(
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get smart cache statistics"""
    if not system_manager.phase1_system:
        return {"error": "Multi-source system not initialized"}

    stats = await system_manager.phase1_system.cache_manager.get_stats()
    return {
        "cache_stats": stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/cache/invalidate/")
async def invalidate_cache(
        pattern: str = Form(...),
        credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Invalidate cache entries matching pattern (admin only)"""
    if not system_manager.phase1_system:
        raise HTTPException(status_code=503, detail="Multi-source system not initialized")

    # In production, check if user has admin role from JWT
    user_id = f"user_{hash(credentials.credentials) % 10000}"

    count = await system_manager.phase1_system.cache_manager.invalidate_pattern(pattern)
    return {
        "status": "success",
        "invalidated_count": count,
        "pattern": pattern
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Complete Analytics Backend with Advanced Charts")
    print("📊 Version: 5.1.1")
    print("🎯 Features: Complete analytics pipeline with advanced chart support (waterfall, funnel, gauge)")
    print("\n🔗 Key Endpoints:")
    print("   • /analyze/ - Comprehensive analysis with advanced charts")
    print("   • /test-openai/ - Test OpenAI API setup")
    print("   • /discover-sources/ - Auto-discover data sources")
    print("   • /chart-suggestions/ - Advanced chart intelligence")
    print("   • /health/ - System health check")
    print("   • /status/ - System status")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )